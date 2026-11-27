"""Time and peak memory of the main-view group-by: the pandas path against the dftracer-utils port.

One process per case.

  python bench_main_view.py run N G pandas|port    time and RSS (JSON line)
  python bench_main_view.py check N G              both paths on one frame, compared

`pandas` is the group-by and the steps after it of the pandas `_compute_main_view`
(`groupby.agg` with the built-in sum, min and max as develop's `_compute_main_view`,
`set_main_metrics`, `replace(0, NA)`, `fix_dtypes`). A sum of only missing values is 0 there,
so a 0/0 rate is NaN where the port has NA; the check counts NaN as missing, as the parity
tests do for develop's saved outputs. `set_main_metrics`
and `fix_dtypes` come from tests/pandas_reference.py.
`port` is `dftracer.analyzer.frames.main_view.group_main_view` of the checkout
under test, on a library HLM made from the same pandas HLM before the step. G is the
number of groups on (proc_name, time_range); the HLM rows also carry the cat, io_cat,
acc_pat and func_name index levels, so a group has about N / G rows.

RSS (this process and its children) is sampled every 10 ms by a child process while the step runs, and only then. `step_peak_gb`
is the highest sample minus the RSS just before the step, so the HLM build does not hide
the step. `rss_peak_gb` is the highest sample. The step runs REPS times after one warm-up
call on a small HLM; time and step peak are reported as the median and the range of the runs.
"""

import gc
import json
import os
import statistics
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

warnings.filterwarnings("ignore")
VIEW_TYPES = ["proc_name", "time_range"]
TIME_RANGES = 20
RTOL = 1e-12
REPS = 5
SAMPLE_S = 0.01
GB = 1e9


_SAMPLER = """
import os, select, sys, psutil
proc = psutil.Process(int(sys.argv[1]))
me = os.getpid()
def rss():
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        if child.pid != me:
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
    return total
peak = rss()
while not select.select([sys.stdin], [], [], float(sys.argv[2]))[0]:
    peak = max(peak, rss())
sys.stdin.readline()
print(max(peak, rss()), flush=True)
"""


def tree_rss() -> int:
    """RSS of this process and its child processes, such as Dask workers."""
    proc = psutil.Process()
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass
    return total


class RssSampler:
    """Highest RSS of this process and its children, sampled every SAMPLE_S seconds between enter and exit.

    The sampler is a child process, so a long call that holds the GIL does not stop it.
    It leaves itself out of the sum.
    """

    def __init__(self):
        self.before = self.peak = tree_rss()

    def __enter__(self):
        self._child = subprocess.Popen(
            [sys.executable, "-c", _SAMPLER, str(os.getpid()), str(SAMPLE_S)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.5)
        self.before = tree_rss() - psutil.Process(self._child.pid).memory_info().rss
        return self

    def __exit__(self, *exc):
        out, _ = self._child.communicate("stop\n")
        self.peak = int(out)


FUNC_NAMES = ("read", "write", "open", "close", "stat")


def make_hlm_index(rng, n: int, groups: int, io_cats: int = 4, func_names=FUNC_NAMES) -> pd.MultiIndex:
    """The six HLM index levels; about `groups` (proc_name, time_range) keys."""
    n_proc = max(1, groups // TIME_RANGES)
    names = pd.array([f"app#host#{i}#{i}" for i in range(n_proc)], dtype="string")
    proc = pd.array(names.take(rng.integers(0, n_proc, n)), dtype="string")
    return pd.MultiIndex.from_arrays(
        [
            proc,
            rng.integers(0, TIME_RANGES, n).astype("int64"),
            pd.array(rng.choice(["posix", "stdio"], n), dtype="string"),
            rng.integers(0, io_cats, n).astype("int64"),
            rng.integers(0, 3, n).astype("int64"),
            np.array(func_names, dtype=object)[rng.integers(0, len(func_names), n)],
        ],
        names=["proc_name", "time_range", "cat", "io_cat", "acc_pat", "func_name"],
    )


def make_hlm(n: int, groups: int, seed: int = 0) -> pd.DataFrame:
    """An HLM as the analyzer holds it: six index levels, nullable metric columns, a zero is a missing value."""
    rng = np.random.default_rng(seed)
    index = make_hlm_index(rng, n, groups)
    cols = {}
    floats = ["time", "time_call_min", "time_call_max", "read_time", "write_time"]
    ints = ["count", "size", "read_count", "read_size", "write_count", "write_size"]
    ints += [f"size_bin_{i}" for i in range(4)]
    for name in floats:
        cols[name] = pd.arrays.FloatingArray(rng.random(n) * 100, rng.random(n) < 0.05)
    for name in ints:
        cols[name] = pd.arrays.IntegerArray(rng.integers(1, 1000, n), rng.random(n) < 0.05)
    return pd.DataFrame(cols, index=index)


def pandas_main_view(hlm: pd.DataFrame, view_types: list, time_sliced: bool = False) -> pd.DataFrame:
    from tests.pandas_reference import fix_dtypes, set_main_metrics

    agg = {}
    for name in hlm.columns:
        if name.endswith("_call_min"):
            agg[name] = "min"
        elif name.endswith("_call_max"):
            agg[name] = "max"
        else:
            agg[name] = "sum"
    view = hlm.groupby(view_types).agg(agg)
    view = set_main_metrics(view).replace(0, pd.NA)
    return fix_dtypes(view, time_sliced=time_sliced)


def port_main_view(hlm, view_types: list, time_sliced: bool = False):
    from dftracer.analyzer.frames.main_view import group_main_view

    return group_main_view(hlm, view_types, time_sliced)


def library(frame: pd.DataFrame):
    """`frame` as a dftracer-utils frame with its index."""
    from dftracer.utils import DataFrame

    return DataFrame.from_pandas(frame)


ENGINES = {"pandas": pandas_main_view, "port": port_main_view}
INPUTS = {"pandas": lambda frame: frame, "port": library}


def _missing(values: pd.Series) -> np.ndarray:
    """NA, and NaN in a float column: develop's 0/0 rates are NaN where the port has NA."""
    missing = values.isna().to_numpy()
    if pd.api.types.is_float_dtype(values.dtype):
        missing |= np.isnan(values.to_numpy(dtype="float64", na_value=np.nan))
    return missing


def assert_match(got, expected: pd.DataFrame) -> None:
    """Same columns, index values, dtypes and missing values (NaN counts as missing); numbers within RTOL relative.

    A library frame `got` is compared as its nullable pandas frame. The index is compared by value.
    """
    if not isinstance(got, pd.DataFrame):
        got = got.to_pandas(nullable=True)
    assert list(got.columns) == list(expected.columns)
    assert list(got.index) == list(expected.index)
    assert list(got.index.names) == list(expected.index.names)
    for name in expected.columns:
        g, e = got[name], expected[name]
        assert str(g.dtype) == str(e.dtype), f"{name}: dtype {g.dtype} != {e.dtype}"
        assert (_missing(g) == _missing(e)).all(), f"{name}: missing values differ"
        present = ~_missing(e)
        gv = g.to_numpy(dtype="float64", na_value=np.nan)[present]
        ev = e.to_numpy(dtype="float64", na_value=np.nan)[present]
        np.testing.assert_allclose(gv, ev, rtol=RTOL, atol=0, err_msg=name)


def measure(step) -> dict:
    """Time and RSS of REPS calls of `step` (no arguments), as the JSON fields of one case."""
    gc.collect()
    times, steps, peaks, befores = [], [], [], []
    groups_out = 0
    for _ in range(REPS):
        with RssSampler() as rss:
            t = time.perf_counter()
            out = step()
            times.append(time.perf_counter() - t)
        groups_out = len(out)
        del out
        gc.collect()
        steps.append(rss.peak - rss.before)
        peaks.append(rss.peak)
        befores.append(rss.before)
    return {
        "groups": groups_out,
        "agg_s": round(statistics.median(times), 3),
        "agg_s_range": [round(min(times), 3), round(max(times), 3)],
        "agg_s_runs": [round(x, 3) for x in times],
        "rss_before_gb": [round(x / GB, 3) for x in befores],
        "step_peak_gb": round(statistics.median(steps) / GB, 3),
        "step_peak_gb_range": [round(min(steps) / GB, 3), round(max(steps) / GB, 3)],
        "step_peak_gb_runs": [round(x / GB, 3) for x in steps],
        "rss_peak_gb": round(max(peaks) / GB, 3),
    }


def main() -> None:
    mode, n, groups = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    if mode == "check":
        hlm = make_hlm(n, groups)
        got = port_main_view(library(hlm), VIEW_TYPES)
        expected = pandas_main_view(hlm, VIEW_TYPES)
        assert_match(got, expected)
        print(json.dumps({"check": "ok", "rows": n, "groups": len(expected), "rtol": RTOL}))
        return
    engine = sys.argv[4]
    fn, prepare = ENGINES[engine], INPUTS[engine]
    fn(prepare(make_hlm(1000, 100)), VIEW_TYPES)
    hlm = prepare(make_hlm(n, groups))
    result = measure(lambda: fn(hlm, VIEW_TYPES))
    print(json.dumps({"engine": engine, "rows": n, "groups_target": groups, **result}), flush=True)


if __name__ == "__main__":
    main()
