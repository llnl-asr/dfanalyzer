"""The distributed and in-process read paths must agree.

`read_trace` scans on Dask workers and decodes per-worker Arrow IPC; when there
is no client, `read_trace` falls back to `read_trace_local`, which runs the same
C++ aggregation in this process. Both are supposed to hand the analyzer the same
frame, but nothing downstream re-derives the schema, so a divergence here is
silent until arithmetic fails much later.

Two divergences this pins down:

* the C++ aggregator dictionary-encodes string columns, and only the
  distributed path decoded them -- the local path produced `category`, which
  `normalize_arrow_dtypes` demotes to `object`, so `count / time` followed
  Python scalar rules and raised ZeroDivisionError instead of yielding inf.
* `_time_origin` was only ever set by the distributed path, so the local path
  bucketed `time_range` against a different origin.
"""

import pathlib
import pytest
from dask.distributed import LocalCluster
from dftracer.analyzer import init_with_hydra


# 8 files, so the distributed path spreads the scan across workers and each may
# dictionary-encode independently -- which is what the decoding exists for.
TRACE_PATH = "tests/data/extracted/dftracer-dlio"
PRESET = "ai"


@pytest.fixture(scope="module")
def dask_cluster():
    cluster = LocalCluster(processes=False, protocol="tcp", worker_class="distributed.nanny.Nanny")
    yield cluster
    cluster.close()


@pytest.fixture(scope="module")
def analyzer(tmp_path_factory, dask_cluster):
    tmp_path = tmp_path_factory.mktemp("read_paths")
    overrides = [
        "analyzer=dftracer",
        f"analyzer/preset={PRESET}",
        "analyzer.checkpoint=False",
        f"analyzer.checkpoint_dir={tmp_path}/checkpoints",
        "cluster=external",
        "cluster.restart_on_connect=False",
        f"cluster.scheduler_address={dask_cluster.scheduler_address}",
        f"hydra.run.dir={tmp_path}",
        f"hydra.runtime.output_dir={tmp_path}",
        f"trace_path={TRACE_PATH}",
        "view_types=[time_range]",
    ]
    dfa = init_with_hydra(hydra_overrides=overrides)
    yield dfa.analyzer
    dfa.shutdown()


@pytest.fixture(scope="module")
def read_paths(analyzer):
    """Both read paths, plus the `_time_origin` each one left behind."""
    trace_path = str(pathlib.Path(TRACE_PATH).resolve())

    distributed = analyzer.read_trace(trace_path).traces.compute()
    distributed_origin = getattr(analyzer, "_time_origin", None)

    # Clear it first: the distributed read above already set `_time_origin`, and
    # a local path that never sets one would otherwise inherit that value and
    # look correct.
    analyzer._time_origin = None
    local = analyzer.read_trace_local(trace_path).traces.compute()
    local_origin = getattr(analyzer, "_time_origin", None)

    return distributed, distributed_origin, local, local_origin


@pytest.mark.smoke
def test_read_paths_agree_on_schema(read_paths) -> None:
    distributed, _, local, _ = read_paths

    assert len(distributed) > 0, "fixture produced no events -- the comparison would be vacuous"
    assert len(distributed) == len(local), f"row counts differ: distributed={len(distributed)} local={len(local)}"
    assert list(distributed.columns) == list(local.columns), "column sets differ"

    mismatched = {
        col: (str(distributed[col].dtype), str(local[col].dtype))
        for col in distributed.columns
        if str(distributed[col].dtype) != str(local[col].dtype)
    }
    assert not mismatched, f"dtype mismatches between read paths: {mismatched}"


@pytest.mark.smoke
def test_read_paths_agree_on_string_columns(read_paths) -> None:
    """Guards the decoding specifically: `category` here is the regression."""
    distributed, _, local, _ = read_paths

    string_cols = [col for col in distributed.columns if str(distributed[col].dtype) == "string"]
    assert string_cols, "no string columns -- the decoding regression could not be observed"

    categorical = [col for col in string_cols if str(local[col].dtype) == "category"]
    assert not categorical, f"local path left columns dictionary-encoded: {categorical}"


@pytest.mark.smoke
def test_local_read_sets_time_origin(read_paths) -> None:
    distributed, distributed_origin, _, local_origin = read_paths

    assert distributed_origin is not None, "distributed path did not set a time origin"
    assert local_origin is not None, "local path left _time_origin unset"
    assert local_origin == distributed_origin, (
        f"time origins differ: distributed={distributed_origin} local={local_origin}"
    )


@pytest.mark.smoke
def test_read_trace_rejects_dir_with_only_nested_traces(analyzer, tmp_path) -> None:
    nested = tmp_path / "sub"
    nested.mkdir()
    src = next(pathlib.Path(TRACE_PATH).glob("*.pfw*"))
    (nested / src.name).write_bytes(src.read_bytes())

    with pytest.raises(FileNotFoundError, match="subdirectories are not searched"):
        analyzer.read_trace(trace_path=str(tmp_path))
