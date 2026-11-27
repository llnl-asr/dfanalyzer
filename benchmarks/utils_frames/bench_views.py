"""Time and peak memory of the view aggregation and of the main view with derived metrics.

One process per case.

  python bench_views.py run WORKLOAD N G ENGINE    time and RSS (JSON line)
  python bench_views.py check WORKLOAD N G         every port engine against pandas on one frame

Workloads and engines:

  view     pandas | port                  `aggregate_view` of a view input grouped on proc_name
  view-sets pandas | port                 the same with a `file_name` column of sets
  dask     port-cluster                   `aggregate_view` as a DaskFrame of NPARTITIONS partitions
                                          on a LocalCluster of WORKERS processes
  derived  pandas | port                  the main view with the POSIX preset's derived metrics

The view input has the column shape of the reader layer: for 5 prefixes a count, an ops
and a time column (full statistics), the time `_sq` (sum) and the time call min and max.
G is the number of proc_name keys of `view` and `dask` and of (proc_name, time_range) keys of
`derived`, whose input has the 9 columns of the posix parity HLM. `pandas` of `view` and `dask` is the partial, merge and
finalize path of the parity tests (`tests/test_utils_frame_parity.py::_pandas_view`); the
pandas Dask path is gone, so the `dask` engines compare with it too. `pandas` of `derived` is
`set_layer_metrics` of tests/pandas_reference.py and then the pandas main view of bench_main_view.py.
The port engines start from a library frame made from the same pandas input before the step
(a set cell becomes a LIST of its sorted values).

Memory and the REPS runs are measured as in bench_main_view.py, over this process and its children, so the
cluster workers count. The cluster starts before the step.
"""

import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from bench_main_view import RTOL, VIEW_TYPES, assert_match, make_hlm_index, measure, pandas_main_view

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

PREFIXES = ["", "read_", "write_", "open_", "close_"]
TIME_RANGES = 20
LOGICAL_VIEWS = {
    "file_name": ["file_dir", "file_pattern"],
    "proc_name": ["host_name", "proc_id", "thread_id"],
}
NPARTITIONS = 8
FILES = 1000
WORKERS = 4
DERIVED_FUNC_NAMES = ("read", "write", "open", "close", "lseek", "stat", "fstat", "fsync")


def make_view_input(n: int, groups: int, seed: int = 0, sets: bool = False) -> pd.DataFrame:
    """A view input as the analyzer passes it: (proc_name, time_range) index, nullable metric columns.

    With `sets`, a `file_name` column holds a set of 1 to 3 of FILES file names per row
    (missing in 5 percent), so the view takes its set union and unique count.
    """
    rng = np.random.default_rng(seed)
    names = pd.array([f"app#host#{i}#{i}" for i in range(groups)], dtype="string")
    index = pd.MultiIndex.from_arrays(
        [
            pd.array(names.take(rng.integers(0, groups, n)), dtype="string"),
            rng.integers(0, TIME_RANGES, n).astype("int64"),
        ],
        names=VIEW_TYPES,
    )
    cols = {}
    for p in PREFIXES:
        time = rng.random(n) * 100
        missing = rng.random(n) < 0.05
        cols[f"{p}count"] = pd.arrays.IntegerArray(rng.integers(1, 1000, n), missing)
        cols[f"{p}ops"] = pd.arrays.FloatingArray(rng.random(n) * 10, missing)
        cols[f"{p}time"] = pd.arrays.FloatingArray(time, missing)
        cols[f"{p}time_sq"] = pd.arrays.FloatingArray(time * time, missing)
        cols[f"{p}time_call_min"] = pd.arrays.FloatingArray(time * 0.5, missing)
        cols[f"{p}time_call_max"] = pd.arrays.FloatingArray(time * 2.0, missing)
    if sets:
        files = np.array([f"/data/file_{i}.bin" for i in range(FILES)], dtype=object)
        sizes = rng.integers(1, 4, n)
        picks = rng.integers(0, FILES, (n, 3))
        absent = rng.random(n) < 0.05
        cols["file_name"] = pd.Series(
            [None if a else set(files[row[:k]]) for a, k, row in zip(absent, sizes, picks)], dtype=object
        ).to_numpy()
    return pd.DataFrame(cols, index=index)


def pandas_view(records: pd.DataFrame) -> pd.DataFrame:
    from tests.test_utils_frame_parity import _pandas_view

    return _pandas_view(records, "proc_name", VIEW_TYPES, LOGICAL_VIEWS, "reader", False)


def library(frame: pd.DataFrame):
    """`frame` as a dftracer-utils frame with its index; a column of set cells becomes a LIST column."""
    from tests.test_utils_frame_parity import as_frame

    return as_frame(frame)


def port_view(records, client=None, npartitions: int = 0):
    from dftracer.analyzer.frames.view import aggregate_view

    return aggregate_view(
        records, "proc_name", VIEW_TYPES, LOGICAL_VIEWS, "reader", False, client=client, npartitions=npartitions
    )


def make_derived_hlm(n: int, groups: int, seed: int = 0) -> pd.DataFrame:
    """A POSIX HLM before the derive step, with the columns of the posix parity trace."""
    rng = np.random.default_rng(seed)
    index = make_hlm_index(rng, n, groups, io_cats=8, func_names=DERIVED_FUNC_NAMES)
    time = rng.random(n) * 100
    size = rng.integers(1, 1 << 20, n)
    missing = rng.random(n) < 0.05
    no_size = missing | (rng.random(n) < 0.3)
    cols = {
        "time": pd.arrays.FloatingArray(time, missing),
        "count": pd.arrays.IntegerArray(rng.integers(1, 1000, n), missing),
        "size": pd.arrays.IntegerArray(size, no_size),
        "time_sq": pd.arrays.FloatingArray(time * time, missing),
        "size_sq": pd.arrays.FloatingArray(size.astype("float64") ** 2, no_size),
        "time_call_min": pd.arrays.FloatingArray(time * 0.5, missing),
        "time_call_max": pd.arrays.FloatingArray(time * 2.0, missing),
        "size_call_min": pd.arrays.IntegerArray(size // 2, no_size),
        "size_call_max": pd.arrays.IntegerArray(size * 2, no_size),
    }
    return pd.DataFrame(cols, index=index)


def pandas_derived(hlm: pd.DataFrame) -> pd.DataFrame:
    from dftracer.analyzer.config import DERIVED_POSIX_METRICS, DERIVED_POSIX_SIZE_METRICS

    from tests.pandas_reference import set_layer_metrics

    derived = set_layer_metrics(hlm, DERIVED_POSIX_METRICS, list(DERIVED_POSIX_SIZE_METRICS))
    return pandas_main_view(derived, VIEW_TYPES)


def port_derived(hlm):
    from dftracer.analyzer.config import DERIVED_POSIX_METRICS, DERIVED_POSIX_SIZE_METRICS
    from dftracer.analyzer.frames.main_view import group_main_view

    return group_main_view(hlm, VIEW_TYPES, False, DERIVED_POSIX_METRICS, DERIVED_POSIX_SIZE_METRICS)


def _cluster():
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(n_workers=WORKERS, threads_per_worker=1, processes=True, silence_logs=40)
    return cluster, Client(cluster)


def _engine(workload: str, engine: str, client):
    if workload in ("view", "view-sets"):
        return {"pandas": pandas_view, "port": port_view}[engine]
    if workload == "dask":
        if engine == "port-cluster":
            return lambda records: port_view(records, client, NPARTITIONS)
        if engine == "pandas":
            return pandas_view
    if workload == "derived":
        return {"pandas": pandas_derived, "port": port_derived}[engine]
    raise SystemExit(f"unknown workload or engine: {workload} {engine}")


def _input(workload: str, n: int, groups: int) -> pd.DataFrame:
    if workload == "derived":
        return make_derived_hlm(n, groups)
    return make_view_input(n, groups, sets=workload == "view-sets")


def main() -> None:
    mode, workload, n, groups = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    if mode == "check":
        data = _input(workload, n, groups)
        if workload == "derived":
            assert_match(port_derived(library(data)), pandas_derived(data.copy()))
        else:
            expected = pandas_view(data.copy())
            assert_match(port_view(library(data)), expected)
            cluster, client = _cluster()
            try:
                assert_match(port_view(library(data), client, NPARTITIONS), expected)
            finally:
                client.close()
                cluster.close()
        print(json.dumps({"check": "ok", "workload": workload, "rows": n, "groups_target": groups, "rtol": RTOL}))
        return
    engine = sys.argv[5]
    cluster = client = None
    if engine == "port-cluster":
        cluster, client = _cluster()
    try:
        fn = _engine(workload, engine, client)
        prepare = (lambda frame: frame) if engine == "pandas" else library
        fn(prepare(_input(workload, 1000, 100)))
        data = prepare(_input(workload, n, groups))
        result = measure(lambda: fn(data))
    finally:
        if client is not None:
            client.close()
            cluster.close()
    print(
        json.dumps({"workload": workload, "engine": engine, "rows": n, "groups_target": groups, **result}),
        flush=True,
    )


if __name__ == "__main__":
    main()
