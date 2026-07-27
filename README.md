# Data Flow Analyzer

![Build and Test](https://github.com/LLNL/dfanalyzer/actions/workflows/ci.yml/badge.svg)
![PyPI - Version](https://img.shields.io/pypi/v/dftracer-analyzer?label=PyPI)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/dftracer-analyzer?label=Wheel)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dftracer-analyzer?label=Python)

## Overview

DFAnalyzer is an open-source tool for analyzing performance data from large-scale workflows on distributed systems. It presents a hierarchical, layer-by-layer summary of an application's execution, from high-level application events down to low-level POSIX calls. For each layer, DFAnalyzer quantifies time, operation counts, and data volume, and calculates key performance metrics like bandwidth and operations per second. It also visualizes the overlap between different layers, helping to characterize and understand complex I/O and compute patterns.

## Installation

To install DFAnalyzer through `pip` (recommended for most users):

```bash
pip install dftracer-analyzer
```

To install DFAnalyzer from source (for developers or custom builds):

```bash
# 1. Install Python build dependencies:
python -m pip install --upgrade pip setuptools wheel

# 2. Install DFAnalyzer from the root of this repository:
pip install -e .

# (Optional) Install dependencies for running tests if you plan to contribute or run local tests:
# pip install -r tests/requirements.txt
```

## Usage

Here's an example of how to run DFAnalyzer using sample data included in the repository:

```bash
# Before running, ensure the sample data is extracted.
# For example, to extract the 'dftracer-ai' sample used below:
# mkdir -p tests/data/extracted
# tar -xzf tests/data/dftracer-ai.tar.gz -C tests/data/extracted
dfanalyzer analyzer/preset=ai trace_path=tests/data/extracted/dftracer-ai view_types=[time_range]
```

This command analyzes the traces and prints a high-level summary of the application's execution. Below is a sample of the "Time Period Summary" output:

```bash
                                                     Time Period Summary
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                                                                         ┃ Unit              ┃                Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Job Time                                                                       │ seconds           │               86.998 │
│ Trace Count                                                                    │ count             │              125,669 │
│ Profile Count                                                                  │ count             │                    0 │
│ Total Count                                                                    │ count             │              125,669 │
│ Total Files                                                                    │ count             │                   47 │
│ Total Nodes                                                                    │ count             │                    1 │
│ Total Processes                                                                │ count             │                    1 │
│ App Count                                                                      │ count             │                    1 │
│ Training Count                                                                 │ count             │                    5 │
│ Compute Count                                                                  │ count             │                   50 │
│ Fetch Data Count                                                               │ count             │                   50 │
│ Checkpoint Count                                                               │ count             │                    3 │
│ DLIO Data Loader Count                                                         │ count             │                  302 │
│ DLIO Data Loader Fork Count                                                    │ count             │                   10 │
│ Reader Count                                                                   │ count             │                  800 │
│ POSIX - All Count                                                              │ count             │              124,025 │
│ POSIX - All Size                                                               │ MB                │            28757.953 │
│ POSIX - All Bandwidth                                                          │ MB/s              │             1228.160 │
│ POSIX - All Avg Transfer Size                                                  │ MB                │                0.232 │
│ POSIX - Reader Count                                                           │ count             │              124,004 │
│ POSIX - Reader Size                                                            │ MB                │            28757.942 │
│ POSIX - Reader Bandwidth                                                       │ MB/s              │             1234.529 │
│ POSIX - Reader Avg Transfer Size                                               │ MB                │                0.232 │
│ POSIX - Checkpoint Count                                                       │ count             │                   10 │
│ POSIX - Checkpoint Size                                                        │ MB                │                0.011 │
│ POSIX - Checkpoint Bandwidth                                                   │ MB/s              │                1.043 │
│ POSIX - Checkpoint Avg Transfer Size                                           │ MB                │                0.001 │
└────────────────────────────────────────────────────────────────────────────────┴───────────────────┴──────────────────────┘
```

DFAnalyzer also provides a detailed breakdown of performance metrics for each layer of the application. Here is a snippet of the "Layer Breakdown" section from the same run, which includes the percentage of time each layer overlaps with its parent layer:

```bash
                                               Layer Breakdown (w/ overlap %)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Layer                     ┃          Time (s) ┃               Ops ┃    Ops/sec ┃           Size (MB) ┃   Bandwidth (MB/s) ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ App                       │     85.534 (----) │          1 (----) │      0.012 │                   - │                  - │
│ Training                  │     85.384 (----) │          5 (----) │      0.059 │                   - │                  - │
│ Compute                   │     68.079 (----) │         50 (----) │      0.734 │                   - │                  - │
│ Fetch Data                │     15.428 (----) │         50 (----) │      3.241 │                   - │                  - │
│ Checkpoint                │      0.078 (----) │          3 (----) │     38.240 │                   - │                  - │
│ DLIO Data Loader          │    102.992 ( 66%) │        302 (  0%) │      2.932 │                   - │                  - │
│ DLIO Data Loader Fork     │      0.109 (  0%) │         10 (  0%) │     91.467 │                   - │                  - │
│ Reader                    │     56.889 ( 57%) │        800 ( 59%) │     14.063 │                   - │                  - │
│ POSIX - All               │     23.415 ( 45%) │    124,025 ( 63%) │   5296.712 │    28757.953 ( 63%) │           1228.160 │
│ POSIX - Reader            │     23.295 ( 45%) │    124,004 ( 63%) │   5323.278 │    28757.942 ( 63%) │           1234.529 │
│ POSIX - Checkpoint        │      0.010 (----) │         10 (----) │    957.671 │        0.011 (----) │              1.043 │
└───────────────────────────┴───────────────────┴───────────────────┴────────────┴─────────────────────┴────────────────────┘
```

### Web app

A Streamlit front end is available for interactive analysis. It uploads traces,
runs the same analysis as the CLI, and renders the per-layer breakdown.

```bash
pip install ".[web]"
streamlit run streamlit_app.py
```

Upload one or more `.pfw.gz` traces, pick a preset, and press **Analyze**.
Uncompressed `.pfw` files are not accepted: the indexer only reads `.pfw.gz`,
and an uncompressed trace parses to zero events.

The web app is for small traces. Peak memory is roughly 290 MB of fixed
overhead plus 11-17 MB per 1,000 trace events, so uploads are capped at 2 MB
total and a warning appears past 450 KB. For reference, a 3.8 MB trace
(284,041 events) peaks at about 3.5 GB — well beyond what a hosted instance
provides. Use the CLI for anything real; it has no such cap.

## Analysis facts (DFDiagnoser integration)

Beyond the human-readable summary, DFAnalyzer can emit **analysis facts** — compact,
machine-readable bottleneck signals (`analyzer.fact-envelope.v1`) that
[DFDiagnoser](https://github.com/LLNL/dfdiagnoser) turns into longitudinal findings
and [DFOptimizer](https://github.com/LLNL/dfoptimizer) turns into tuning actions. Facts
are **opt-in** and additive: with `facts.enabled=false` (the default) the analysis
output is unchanged.

A fact is produced per view per analysis window by either builder:

- **rule** (`facts.eval_mode=rule`) — YAML conditions over view metrics
  (`facts.eval_rule_file=<rules.yaml>`), e.g. *fetch time dominates compute*.
- **metric** (`facts.eval_mode=metric`) — WISIO-style slope detection: an entity whose
  share of time is disproportionate to its share of operations.

Each fact carries a continuous `severity` in [0,1], a two-level `scope`
(`layer:view` aggregate or `layer:view:entity` detail), and `opportunity_tags`.

### Producing facts to a bundle (offline)

`output=file` writes the deliverable bundle — `facts.jsonl` (one envelope per window),
`detail_view_*.parquet`, and `raw_stats.json` — that `dfdiagnoser input=file` consumes:

```bash
dfanalyzer analyzer/preset=ai trace_path=tests/data/extracted/dftracer-ai \
    view_types=[time_range] \
    facts.enabled=true facts.eval_mode=rule \
    facts.eval_rule_file=python/dftracer/analyzer/configs/fact_rules/dlio.yaml \
    output=file output.path=/tmp/bundle
```

```text
[info ] file_output.facts   path=/tmp/bundle/facts.jsonl
$ ls /tmp/bundle
facts.jsonl  detail_view_proc_name.parquet  detail_view_time_range.parquet  raw_stats.json
```

### Full offline chain (analyzer → diagnoser → optimizer)

```bash
# 0. a minimal time_range rule (the shipped dlio.yaml rules target the streaming
#    epoch axis; offline rules are workload-specific). Save as /tmp/tr.yaml:
#
#   schema_version: analysisfact-rules.v1
#   defaults: {rule_version: "1.0.0", emit_mode: aggregate, confidence: "0.80"}
#   rules:
#     - id: tr.reader_pressure.v1
#       priority: 100
#       source_view: time_range
#       fact_type: reader_pressure
#       required_metrics: [reader_posix_time_proc_max, app_time_proc_max]
#       derived_metrics:
#         reader_frac: "fillna0(reader_posix_time_proc_max) / max(fillna0(app_time_proc_max), 1e-9)"
#       when: "reader_frac >= 0.10"
#       severity_score: "clip01(reader_frac)"
#       opportunity_tags: [dataloader_prefetch, reader_parallelism]

# 1. analyze -> fact bundle (facts on the time_range temporal axis)
dfanalyzer analyzer/preset=ai trace_path=tests/data/extracted/dftracer-ai \
    view_types=[time_range] facts.enabled=true \
    facts.eval_rule_file=/tmp/tr.yaml output=file output.path=/tmp/bundle

# 2. diagnose -> longitudinal findings
dfdiagnoser input=file input.path=/tmp/bundle output=console

# 3. optimize -> ActionPlans (offline replay of the diagnoser's findings.jsonl)
#    (from the dfoptimizer repo root; DFOPTIMIZER_BOOTSTRAP_DLIO=1 loads the DLIO knobs)
DFOPTIMIZER_BOOTSTRAP_DLIO=1 python main.py --transport file --findings-file findings.jsonl
```

Verified end-to-end on `dftracer-ai`: a `reader_pressure` rule on `time_range` ->
76 facts -> diagnoser finding (persistence 39) -> 2 ActionPlans (`dlio.prefetch_size`
2->3, `dlio.read_threads` 1->2).

The temporal axis for longitudinal facts is **`time_range`** offline; **`epoch`/`window`**
are produced on the **streaming** path (ZMQ/Mofka), where each event is window-tagged.
Spatial views (`file_name`/`proc_name`) yield one-shot facts.

### Facts configuration

| key | default | meaning |
|---|---|---|
| `facts.enabled` | `false` | master switch; off = analysis output unchanged |
| `facts.eval_mode` | `rule` | `rule` (YAML conditions) or `metric` (slope) |
| `facts.eval_rule_file` | `""` | rule YAML (when `eval_mode=rule`) |
| `facts.emit_mode` | `aggregate` | `aggregate` (per-view rollup) or `detail` (per-entity) |
| `facts.emit_flat_views` | `true` | also write the detail views into the bundle |

## Further Information

For more details, to report issues, or to contribute to DFAnalyzer, please refer to the following resources:

- **[Official DFAnalyzer Documentation](https://dfanalyzer.readthedocs.io/)**: For detailed usage, configuration options, and information about analyzers.
- **[Issue Tracker](https://github.com/LLNL/dfanalyzer/issues)**: To report bugs or suggest new features.
- **[Contributing Guidelines](./CONTRIBUTING.md)**: For information on how to contribute to the project, including setting up a development environment and coding standards.
- **[Citation File](./CITATION.cff)**: If you use DFAnalyzer in your research, please cite it using the information in this file.

## Acknowledgments

This work was performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344. This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research under the DOE Early Career Research Program (LLNL-CONF-862440). Also, this research is supported in part by the National Science Foundation (NSF) under Grants OAC-2104013, OAC-2313154, and OAC-2411318.
