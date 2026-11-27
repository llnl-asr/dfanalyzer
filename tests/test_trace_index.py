"""`TraceIndex.resolve` reads only the file system; `ensure` builds the index."""

import os

import pytest
from dftracer.analyzer.trace_index import TraceIndex

from .test_file_grain import _write_trace

pytestmark = [pytest.mark.smoke, pytest.mark.full]

TRACE_NAME = "dftracer-dlio"


@pytest.fixture
def dask_cluster():
    from dask.distributed import LocalCluster

    cluster = LocalCluster(processes=False, protocol="tcp", worker_class="distributed.nanny.Nanny")
    yield cluster
    cluster.close()


def test_resolve_writes_no_index(trace_copy):
    trace_path = trace_copy(TRACE_NAME)
    index = TraceIndex.resolve(trace_path, None)
    assert index.files == trace_path
    assert index.index_dir == os.path.join(trace_path, ".dftindex")
    assert not os.path.exists(index.index_dir)


def test_resolve_glob_lists_files_beside_the_index(trace_copy):
    trace_path = trace_copy(TRACE_NAME)
    index = TraceIndex.resolve(os.path.join(trace_path, "*.pfw.gz"), None)
    assert index.files and all(f.endswith(".pfw.gz") for f in index.files)
    assert index.index_dir == os.path.join(trace_path, ".dftindex")
    assert not os.path.exists(index.index_dir)


def test_ensure_builds_the_index(trace_copy):
    trace_path = trace_copy(TRACE_NAME)
    index = TraceIndex.resolve(trace_path, None)
    index.ensure(time_interval_ms=5000.0)
    assert os.path.isdir(index.index_dir)


def test_trace_files_lists_only_what_the_indexer_reads(tmp_path):
    (tmp_path / "a.pfw.gz").write_bytes(b"")
    (tmp_path / "b.pfw").write_text("")
    index = TraceIndex.resolve(str(tmp_path), None)
    assert index.trace_files() == [str(tmp_path / "a.pfw.gz")]


def test_detect_preset_ignores_plain_pfw_files(tmp_path):
    from dftracer.analyzer.utils.preset_utils import detect_preset

    (tmp_path / "b.pfw").write_text('{"ph":"X","name":"read","cat":"POSIX","pid":1,"tid":1,"ts":1,"dur":1}\n')
    with pytest.raises(FileNotFoundError, match="No matching"):
        detect_preset(str(tmp_path))


def test_a_second_run_with_checkpoints_in_the_trace_dir_reads_only_traces(tmp_path, dask_cluster):
    from dftracer.analyzer import init_with_hydra

    trace_path = _write_trace(tmp_path / "traces")

    def run():
        dfa = init_with_hydra(
            hydra_overrides=[
                "analyzer=dftracer",
                "analyzer/preset=auto",
                "analyzer.checkpoint=True",
                f"analyzer.checkpoint_dir={trace_path}",
                "cluster=external",
                "cluster.restart_on_connect=True",
                f"cluster.scheduler_address={dask_cluster.scheduler_address}",
                f"hydra.run.dir={tmp_path}",
                f"hydra.runtime.output_dir={tmp_path}",
                f"trace_path={trace_path}",
                "view_types=[time_range]",
            ]
        )
        try:
            result = dfa.analyze_trace()
            return int(result.raw_stats.unique_file_count)
        finally:
            dfa.shutdown()

    first = run()
    assert any(f.endswith(".pfw.gz") and not f.startswith("trace-") for f in os.listdir(trace_path))
    assert run() == first
