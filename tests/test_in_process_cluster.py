"""`cluster=none` must analyse a trace without a cluster, and agree with one.

The analyzer talks to a Dask client in a handful of places -- submitting scans,
gathering partials, scheduling checkpoint writes. `NullClient` satisfies those
calls inline so one code path serves both modes.

The subtle failure this guards is not a crash but a silent divergence: if the
View HLM is bypassed, the analyzer falls back to
`Analyzer._compute_high_level_metrics`, a second implementation whose dtypes
differ (its time columns come back as `object`, so `count / time` raises
ZeroDivisionError rather than yielding inf). Reading through the client rather
than around it is what keeps a single HLM; a test that only checked "did it
run" would not notice that regressing.
"""

import pathlib
import pytest
from dask.distributed import LocalCluster
from dftracer.analyzer import init_with_hydra
from dftracer.analyzer.analyzer import Analyzer
from dftracer.analyzer.cluster import InProcessCluster, NullClient
from pandas.testing import assert_frame_equal


# 8 files, so the clustered comparison is a genuine multi-worker scan rather
# than a single worker dressed up as one.
TRACE_PATH = "tests/data/extracted/dftracer-dlio"
PRESET = "ai"


@pytest.fixture(scope="module")
def dask_cluster():
    cluster = LocalCluster(processes=False, protocol="tcp", worker_class="distributed.nanny.Nanny")
    yield cluster
    cluster.close()


def _overrides(tmp_path: pathlib.Path, cluster_overrides, checkpoint: bool):
    return [
        "analyzer=dftracer",
        f"analyzer/preset={PRESET}",
        f"analyzer.checkpoint={checkpoint}",
        f"analyzer.checkpoint_dir={tmp_path}/checkpoints",
        *cluster_overrides,
        f"hydra.run.dir={tmp_path}",
        f"hydra.runtime.output_dir={tmp_path}",
        f"trace_path={TRACE_PATH}",
        "view_types=[time_range,proc_name]",
    ]


def _in_process(tmp_path: pathlib.Path):
    return _overrides(tmp_path, ["cluster=none"], checkpoint=False)


def _distributed(tmp_path: pathlib.Path, cluster):
    return _overrides(
        tmp_path,
        [
            "cluster=external",
            "cluster.restart_on_connect=False",
            f"cluster.scheduler_address={cluster.scheduler_address}",
        ],
        checkpoint=False,
    )


def _flat_views(overrides):
    dfa = init_with_hydra(hydra_overrides=overrides)
    try:
        result = dfa.analyze_trace()
        return {key: view.copy() for key, view in result.flat_views.items()}
    finally:
        dfa.shutdown()


@pytest.mark.smoke
def test_in_process_matches_distributed(tmp_path: pathlib.Path, dask_cluster) -> None:
    """Dropping the cluster must not change a single value or dtype."""
    distributed = _flat_views(_distributed(tmp_path / "dist", dask_cluster))
    in_process = _flat_views(_in_process(tmp_path / "inproc"))

    assert distributed, "expected at least one flat view"
    assert set(distributed) == set(in_process), (
        f"different view keys: distributed={sorted(map(str, distributed))} in_process={sorted(map(str, in_process))}"
    )

    for view_key in distributed:
        assert_frame_equal(
            distributed[view_key],
            in_process[view_key],
            check_dtype=True,
            check_exact=False,
            rtol=1e-9,
            obj=f"flat view {view_key}",
        )


@pytest.mark.smoke
def test_in_process_builds_no_cluster(tmp_path: pathlib.Path) -> None:
    dfa = init_with_hydra(hydra_overrides=_in_process(tmp_path))
    try:
        assert isinstance(dfa.cluster, InProcessCluster), f"expected no cluster, got {type(dfa.cluster)}"
        assert isinstance(dfa.client, NullClient), f"expected NullClient, got {type(dfa.client)}"
        assert dfa.client.is_distributed is False
        assert dfa.analyzer.dask_client.is_distributed is False, "analyzer resolved a distributed client"
    finally:
        dfa.shutdown()


@pytest.mark.smoke
def test_in_process_uses_the_view_hlm(tmp_path: pathlib.Path, monkeypatch) -> None:
    """The base-class HLM is the stale path -- reaching it is the regression."""
    called = []
    original = Analyzer._compute_high_level_metrics

    def spy(self, *args, **kwargs):
        called.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Analyzer, "_compute_high_level_metrics", spy)
    _flat_views(_in_process(tmp_path))

    assert not called, "in-process run fell back to Analyzer._compute_high_level_metrics"


@pytest.mark.smoke
def test_in_process_checkpoints_round_trip(tmp_path: pathlib.Path) -> None:
    """Checkpointing goes through client.compute/cancel, which NullClient fakes."""
    checkpoint_dir = tmp_path / "checkpoints"

    cold = _flat_views(_overrides(tmp_path, ["cluster=none"], checkpoint=True))
    written = sorted(p.name for p in checkpoint_dir.iterdir()) if checkpoint_dir.is_dir() else []
    assert written, "no checkpoints written -- the test would pass without exercising the write path"

    warm = _flat_views(_overrides(tmp_path, ["cluster=none"], checkpoint=True))
    assert set(cold) == set(warm), "restoring from checkpoint changed which views were produced"

    for view_key in cold:
        # Index types are not compared: restoring from Parquet widens string
        # indexes to `large_string[pyarrow]`, which predates this mode and
        # happens on the clustered path too.
        assert_frame_equal(
            cold[view_key],
            warm[view_key],
            check_dtype=True,
            check_index_type=False,
            check_exact=False,
            rtol=1e-9,
            obj=f"flat view {view_key} (warm checkpoint)",
        )
