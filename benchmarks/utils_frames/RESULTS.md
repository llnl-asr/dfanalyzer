# DFAnalyzer on dftracer-utils frames: pandas path against the port

Checkout: branch `refactor/port-dfanalyzer-helpers` at `7147c27`. dftracer-utils: `20839adf8`
(branch `perf/dataframe-io-memory`, not released). Python 3.12, pandas 2.3.3, NumPy 2.5.5,
pyarrow 26.0.1, dask 2026.11.0 on both machines.

- Mac: Apple M4 Pro, 14 cores, 48 GiB, macOS (Darwin 25.5.0). Load average 2.1 at the start
  of the sweeps and 5.7 at the end.
- Linux: AMD Ryzen 9 7900, 24 threads, Debian 13 (kernel 6.12), in an Incus container with a
  48 GiB memory limit and no CPU limit. Load average 0.8 at the start and 3.1 at the end.

Commands (one process per case; the scripts write JSON lines to `<out dir>`):

```
bash benchmarks/utils_frames/run.sh <this checkout> <venv python> <out dir>
bash benchmarks/utils_frames/run_views.sh <this checkout> <venv python> <out dir>
PYTHONPATH=python <venv python> -W ignore benchmarks/utils_frames/bench_main_view.py run|check N G [pandas|port]
PYTHONPATH=python <venv python> -W ignore benchmarks/utils_frames/bench_views.py run|check WORKLOAD N G [ENGINE]
```

Main-view workload: HLM with 6 index levels, 15 nullable metric columns (5 Float64, 10 Int64,
5% missing), grouped on `proc_name, time_range` (G groups), no derived metrics. `pandas` =
`groupby.agg` with the built-in `"sum"`, `set_main_metrics`, `replace(0, NA)`, `fix_dtypes`, from
`tests/pandas_reference.py`. `port` = `group_main_view`: one native group-by over the borrowed
HLM columns, keyed on the MultiIndex codes. The view workloads are described in the docstring
of `bench_views.py`.

Memory: a child process samples the RSS of the case and its children every 10 ms during the step
only (a sampler thread in the same process misses peaks while a C call holds the GIL). Step peak =
highest sample minus the RSS just before the step. Each case runs the step twice in one process;
the tables give both steps. Total peak = highest sample. Time is the best of the two steps.

Parity (`check` mode, every path on one frame): same columns, index, dtypes and missing values;
numbers within 1e-12 relative, zero absolute. Passed for every case on both machines, except
`derived` at 20M on Linux, where the pandas path did not fit (below).

## Library frames end to end: base `51d1e27` against this branch

Base `51d1e27` runs the analysis on dftracer-utils frames and converts to pandas at the stage borders.
This branch keeps library frames from the read to the output, with no pandas, numpy or pyarrow in the package.
dftracer-utils `0.0.14.post21.dev0`, pandas 2.3.3, NumPy 2.5.5 on both machines; Python 3.13 on Mac, 3.12 on Linux.
Mac: branch at `ca88a7f`, load average 2.4 to 3.4. Linux: branch at `2e3cc57` (later commits change only rare expression forms), load average 0.0 to 1.5.

End-to-end CLI, one process per run, a fresh copy of the trace (index rebuilt), `cluster=none`, `analyzer.checkpoint=False`, 5 runs, median and [range]:

```
python benchmarks/utils_frames/bench_cli.py <checkout> tests/data/extracted/<trace> <preset> 5
```

| machine | trace (preset) | base time s | new time s | base peak GB | new peak GB |
|---|---|---|---|---|---|
| Mac | dftracer-ai (ai) | 1.511 [1.505, 1.514] | 1.144 [1.138, 1.165] | 0.277 [0.270, 0.289] | 0.215 [0.213, 0.232] |
| Mac | dftracer-dlio (ai) | 1.418 [1.404, 1.423] | 1.005 [1.003, 1.013] | 0.264 [0.263, 0.268] | 0.203 [0.200, 0.204] |
| Mac | dftracer-posix (posix) | 0.679 [0.676, 0.696] | 0.455 [0.454, 0.458] | 0.186 [0.185, 0.187] | 0.119 [0.118, 0.122] |
| Linux | dftracer-ai (ai) | 1.993 [1.977, 2.006] | 1.451 [1.447, 1.500] | 0.380 [0.370, 0.401] | 0.304 [0.294, 0.326] |
| Linux | dftracer-dlio (ai) | 1.882 [1.874, 1.898] | 1.285 [1.279, 1.295] | 0.386 [0.380, 0.427] | 0.287 [0.276, 0.295] |
| Linux | dftracer-posix (posix) | 0.800 [0.797, 0.865] | 0.561 [0.558, 0.585] | 0.244 [0.243, 0.248] | 0.152 [0.149, 0.154] |

Peak is the child's max RSS from `wait4`.

Stages at 5M rows and 200k groups (`bench_main_view.py run 5000000 200000 port`, `bench_views.py run derived 5000000 200000 port`), 5 steps per process.
Base prints the best step only; new gives the median and [range] of the steps.
The base side starts from a pandas HLM and the new side from a library HLM, so the total peak (which includes building the input) is not a like-for-like number; the step peak is.

| machine | stage | base time s | new time s | base step peak GB | new step peak GB |
|---|---|---|---|---|---|
| Mac | main view | 0.309 | 0.180 [0.179, 0.199] | 0.164 | 0.053 [0.050, 0.118] |
| Mac | derived | 0.826 | 0.638 [0.636, 0.655] | 0.647 | 0.089 [0.070, 0.498] |
| Linux | main view | 0.316 | 0.209 [0.206, 0.214] | 0.156 | 0.070 [0.067, 0.138] |
| Linux | derived | 1.034 | 0.962 [0.949, 0.970] | 0.688 | 0.272 [0.260, 0.388] |

The derived step does one filter and group-by per metric instead of one group-by over all metrics.
One wide group-by over 1.4M groups left glibc arenas holding about 0.57 GB on Linux; the eager per-metric form peaks at about a third of the base step.

## Main view

| machine | rows | groups | path | time s | step peak GB (step 1, step 2) | total peak GB |
|---|---|---|---|---|---|---|
| Mac | 1M | 20k | pandas | 0.13 | 0.016, 0.000 | 0.475 |
| Mac | 1M | 20k | port | 0.04 | 0.029, 0.013 | 0.488 |
| Mac | 5M | 200k | pandas | 0.76 | 0.149, 0.107 | 1.915 |
| Mac | 5M | 200k | port | 0.27 | 0.163, 0.065 | 1.839 |
| Mac | 20M | 200k | pandas | 2.71 | 0.127, 0.123 | 6.374 |
| Mac | 20M | 200k | port | 0.75 | 0.174, 0.089 | 6.356 |
| Linux | 1M | 20k | pandas | 0.10 | 0.000, 0.000 | 0.448 |
| Linux | 1M | 20k | port | 0.05 | 0.069, 0.050 | 0.520 |
| Linux | 5M | 200k | pandas | 0.71 | 0.206, 0.243 | 1.202 |
| Linux | 5M | 200k | port | 0.30 | 0.160, 0.067 | 1.117 |
| Linux | 20M | 200k | pandas | 2.54 | 0.960, 0.948 | 4.270 |
| Linux | 20M | 200k | port | 0.94 | 0.179, 0.104 | 3.492 |

The port is 2.0x to 3.6x faster than pandas. Its first-step peak is above pandas on Mac at every
size (by 0.013, 0.014 and 0.047 GB) and on Linux at 1M (by 0.069 GB); this was accepted. On Linux
at 5M and 20M it is below pandas (0.179 against 0.960 GB at 20M). The total peak is below pandas
at 5M and 20M on both machines and above it at 1M (by 0.013 GB on Mac, 0.072 GB on Linux).

The extra step memory is the group-by state (24 bytes per group per aggregation in
dftracer-utils `ac9d9da47`, about 72 MB for 200k groups and 15 columns) and the conversion of the
whole result to pandas. Before `ac9d9da47` the state was about 120 bytes per group per aggregation
and the 20M step peak was 0.445 GB.

## Views, derived metrics and the DaskFrame view

| machine | workload | rows | groups | path | time s | step peak GB (step 1, step 2) | total peak GB |
|---|---|---|---|---|---|---|---|
| Mac | view | 1M | 20k | pandas | 4.07 | 0.864, 0.165 | 1.651 |
| Mac | view | 1M | 20k | port | 1.69 | 0.861, 0.355 | 1.571 |
| Mac | view-sets | 1M | 20k | pandas | 5.37 | 1.132, 0.931 | 2.167 |
| Mac | view-sets | 1M | 20k | port | 2.59 | 1.036, 1.078 | 2.036 |
| Mac | dask | 1M | 20k | port-cluster | 1.82 | 1.256, 0.373 | 2.884 |
| Mac | derived | 1M | 20k | pandas | 1.23 | 2.387, 1.682 | 2.735 |
| Mac | derived | 1M | 20k | port | 0.13 | 0.089, 0.061 | 0.465 |
| Mac | view | 5M | 200k | pandas | 39.74 | 6.947, 2.756 | 10.521 |
| Mac | view | 5M | 200k | port | 11.29 | 4.398, 3.712 | 7.923 |
| Mac | view-sets | 5M | 200k | pandas | 52.36 | 7.899, 5.218 | 12.769 |
| Mac | view-sets | 5M | 200k | port | 15.67 | 5.100, 5.994 | 10.224 |
| Mac | dask | 5M | 200k | port-cluster | 11.34 | 4.963, 3.964 | 12.322 |
| Mac | derived | 5M | 200k | pandas | 6.58 | 11.999, 11.395 | 14.047 |
| Mac | derived | 5M | 200k | port | 0.81 | 0.662, 0.360 | 2.060 |
| Mac | view | 20M | 200k | pandas | 147.44 | 18.921, 20.717 | 29.548 |
| Mac | view | 20M | 200k | port | 53.05 | 16.377, 17.120 | 27.195 |
| Mac | dask | 20M | 200k | port-cluster | 55.24 | 17.797, 15.130 | 29.027 |
| Mac | derived | 20M | 200k | pandas | 35.16 | 29.475, 27.151 | 33.756 |
| Mac | derived | 20M | 200k | port | 1.65 | 0.709, 0.389 | 4.891 |
| Linux | view | 1M | 20k | pandas | 4.80 | 0.871, 0.759 | 1.592 |
| Linux | view | 1M | 20k | port | 2.26 | 0.818, 0.788 | 1.534 |
| Linux | view-sets | 1M | 20k | pandas | 6.11 | 0.963, 0.276 | 1.952 |
| Linux | view-sets | 1M | 20k | port | 3.16 | 1.387, 1.270 | 2.330 |
| Linux | dask | 1M | 20k | port-cluster | 2.46 | 0.835, 0.782 | 2.428 |
| Linux | derived | 1M | 20k | pandas | 1.78 | 2.425, 2.424 | 2.760 |
| Linux | derived | 1M | 20k | port | 0.16 | 0.136, 0.066 | 0.478 |
| Linux | view | 5M | 200k | pandas | 46.27 | 7.762, 6.966 | 9.433 |
| Linux | view | 5M | 200k | port | 15.91 | 5.659, 5.501 | 7.286 |
| Linux | view-sets | 5M | 200k | pandas | 61.26 | 8.159, 7.554 | 10.928 |
| Linux | view-sets | 5M | 200k | port | 20.86 | 6.522, 6.526 | 9.640 |
| Linux | dask | 5M | 200k | port-cluster | 15.94 | 5.656, 5.655 | 8.051 |
| Linux | derived | 5M | 200k | pandas | 8.63 | 12.420, 12.410 | 13.043 |
| Linux | derived | 5M | 200k | port | 1.05 | 0.677, 0.359 | 1.307 |
| Linux | view | 20M | 200k | pandas | 131.20 | 30.077, 26.965 | 35.790 |
| Linux | view | 20M | 200k | port | 58.16 | 22.810, 22.622 | 28.524 |
| Linux | dask | 20M | 200k | port-cluster | 57.14 | 22.801, 22.787 | 29.428 |
| Linux | derived | 20M | 200k | pandas | killed | above 48 GiB | - |
| Linux | derived | 20M | 200k | port | 1.93 | 0.634, 0.438 | 2.616 |

`view-sets` runs at 1M and 5M only: 20M rows of Python set cells do not fit beside the pandas
path. On Linux the container killed the pandas `derived` case at 20M (and its `check`, which runs
both paths) at the 48 GiB limit, with an anonymous RSS of 50.1 GB (`dmesg`, memory cgroup OOM).
The Mac load rose during the view sweep, and both engines ran the 20M `view` case slower than in
the run before (pandas 147 against 115 s, port 53 against 42 s).

The port views are 1.9x to 3.5x faster than pandas and their total peak is lower, with one
exception: `view-sets` at 1M on Linux, 2.330 against 1.952 GB total and 1.387 against 0.963 GB
step peak. The second-step peak is above pandas for `view` at 1M on both machines and at 5M on
Mac, and for `view-sets` at 1M on both machines and at 5M on Mac. The port derived metrics are 8x
to 21x faster and use 18x to 42x less step memory. The DaskFrame view on a 4-worker LocalCluster
with 8 partitions takes about the same time as the one-process port and has a higher total peak
at every size (the workers count). That total peak is above pandas at 1M on both machines and at
5M on Mac (12.322 against 10.521 GB), and below pandas at 20M.

## Main view with key buckets (dftracer-utils `00e9569f8`)

The port before the one-pass group-by. Each 65536-row key bucket gathered its rows from the
unsorted HLM, and that random gather took about half of the time. Same machines and sampler; the
pandas rows of that sweep are within 0.2 s of the table above.

| machine | rows | groups | path | time s | step peak GB (step 1, step 2) | total peak GB |
|---|---|---|---|---|---|---|
| Mac | 1M | 20k | port | 0.23 | 0.019, 0.002 | 0.479 |
| Mac | 5M | 200k | port | 1.81 | 0.106, 0.006 | 1.772 |
| Mac | 20M | 200k | port | 9.32 | 0.116, 0.034 | 6.275 |
| Linux | 1M | 20k | port | 0.32 | 0.013, 0.008 | 0.466 |
| Linux | 5M | 200k | port | 2.01 | 0.017, 0.006 | 0.978 |
| Linux | 20M | 200k | port | 8.68 | 0.017, 0.008 | 3.333 |

## Earlier runs (same sampler)

| rows | groups | port version | dftracer-utils | time s | step peak GB (step 1, step 2) | total peak GB |
|---|---|---|---|---|---|---|
| 1M | 20k | key buckets 262144 + Arrow import | `4f0720a09` | 0.18 | 0.197, 0.024 | 0.678 |
| 5M | 200k | key buckets 262144 + Arrow import | `4f0720a09` | 1.03 | 0.424, 0.042 | 2.154 |
| 20M | 200k | key buckets 262144 + Arrow import | `4f0720a09` | 4.90 | 0.390, 0.062 | 6.717 |
| 1M | 20k | key buckets 262144 + Arrow import | `872e75d4e` | 0.18 | 0.181, 0.023 | 0.67 |
| 5M | 200k | key buckets 262144 + Arrow import | `872e75d4e` | 1.07 | 0.484, 0.048 | 2.24 |
| 20M | 200k | key buckets 262144 + Arrow import | `872e75d4e` | 5.05 | 0.442, 0.012 | 6.74 |
| 1M | 20k | row chunks + fold | old | 2.06 | 0.684 | 1.22 |
| 1M | 20k | row chunks + fold | `872e75d4e` | 1.96 | 0.533 | 1.08 |
| 5M | 200k | row chunks + fold | old | 14.6 | 1.663 | 3.49 |
| 5M | 200k | row chunks + fold + Arrow import | `872e75d4e` | 2.15 | 0.708 | 2.45 |
| 20M | 200k | row chunks + fold | old | 81.7 | 1.561 | 7.85 |

Round 1 code (whole HLM as one native frame, thread sampler): 5M/200k step peak 5.42 GB,
20M/200k total peak 20.72 GB.

## What each cut saved (5M/200k, first step, `4f0720a09`)

| stage of the 262144-row bucket path | MB above the step start |
|---|---|
| bucket scan alone (argsort of 8 buckets per pass) | 119 |
| + `hlm.iloc[rows]` take | 189 |
| + Arrow table | 237 |
| + `from_arrow` | 257 |
| + group-by | 338 |
| + concat of the bucket results | 425 |

Cuts, in order:

1. Scan: 1M rows per pass, int32 positions, no argsort, 256k-row blocks: 119 -> 25 MB.
2. Take only the needed columns at the bucket rows (`_rows_of`), not `hlm.iloc[rows]`.
3. `from_arrow` puts the whole bucket in the pyarrow pool (46 MB per 262144 rows), and mimalloc
   keeps it: with `ARROW_DEFAULT_MEMORY_POOL=system` the RSS growth drops from 278 to 193 MB.
   `DataFrame.from_pandas` uses no Arrow pool: 184 against 238 MB at 262144 rows.
4. Finish each bucket to pandas and write it into output columns allocated once (`_Filled`), so
   the native result and its pandas copy are never whole.
5. Buckets of 65536 rows. At 20M: 32768 rows 11.6 s / 107 MB, 65536 rows 9.3 s / 136 MB,
   131072 rows 8.0 s / 198 MB, 262144 rows 4.9 s / 390 MB (first step).

## Where the remaining first-step memory goes (20M/200k, 65536-row buckets)

Python-traced memory (tracemalloc) peaks at 75 MB: 45 MB is the output, about 10 MB the scan
pass, the rest one bucket. RSS ends 136 to 142 MB above the step start. The other 60 MB is freed
memory the process keeps. It is not the dftracer-utils page heap
(`DFTRACER_UTILS_MEMORY_BACKEND=system` and `DFTRACER_UTILS_MEM_DECAY_MS=0` give the same RSS),
not the native thread count (`DFTRACER_UTILS_THREADS=1` and `4`: 0.136 GB) and not released by
`malloc_zone_pressure_relief`. The second step in the same process reuses it (0.041 GB).

Time at 20M (9.4 s): at 5M, 0.78 of 2.15 s is the random row take from
the unsorted HLM and 0.41 s the group-by. The scan costs about 1.2 s at 20M.
