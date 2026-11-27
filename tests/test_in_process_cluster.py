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
TRACE_NAME = "dftracer-dlio"
PRESET = "ai"


@pytest.fixture(scope="module")
def dask_cluster():
    cluster = LocalCluster(processes=False, protocol="tcp", worker_class="distributed.nanny.Nanny")
    yield cluster
    cluster.close()


def _overrides(tmp_path: pathlib.Path, trace_path: str, cluster_overrides, checkpoint: bool):
    return [
        "analyzer=dftracer",
        f"analyzer/preset={PRESET}",
        f"analyzer.checkpoint={checkpoint}",
        f"analyzer.checkpoint_dir={tmp_path}/checkpoints",
        *cluster_overrides,
        f"hydra.run.dir={tmp_path}",
        f"hydra.runtime.output_dir={tmp_path}",
        f"trace_path={trace_path}",
        "view_types=[time_range,proc_name]",
    ]


def _in_process(tmp_path: pathlib.Path, trace_path: str):
    return _overrides(tmp_path, trace_path, ["cluster=none"], checkpoint=False)


def _distributed(tmp_path: pathlib.Path, trace_path: str, cluster):
    return _overrides(
        tmp_path,
        trace_path,
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
        return {key: view.to_pandas(nullable=True) for key, view in result.flat_views.items()}
    finally:
        dfa.shutdown()


@pytest.mark.smoke
def test_in_process_matches_distributed(tmp_path: pathlib.Path, dask_cluster, trace_copy) -> None:
    """Dropping the cluster must not change a single value or dtype."""
    distributed = _flat_views(_distributed(tmp_path / "dist", trace_copy(TRACE_NAME), dask_cluster))
    in_process = _flat_views(_in_process(tmp_path / "inproc", trace_copy(TRACE_NAME)))

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
def test_in_process_builds_no_cluster(tmp_path: pathlib.Path, trace_copy) -> None:
    dfa = init_with_hydra(hydra_overrides=_in_process(tmp_path, trace_copy(TRACE_NAME)))
    try:
        assert isinstance(dfa.cluster, InProcessCluster), f"expected no cluster, got {type(dfa.cluster)}"
        assert isinstance(dfa.client, NullClient), f"expected NullClient, got {type(dfa.client)}"
        assert dfa.client.is_distributed is False
        assert dfa.analyzer.dask_client.is_distributed is False, "analyzer resolved a distributed client"
    finally:
        dfa.shutdown()


@pytest.mark.smoke
def test_in_process_uses_the_view_hlm(tmp_path: pathlib.Path, monkeypatch, trace_copy) -> None:
    """The base-class HLM is the stale path -- reaching it is the regression."""
    called = []
    original = Analyzer._compute_high_level_metrics

    def spy(self, *args, **kwargs):
        called.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Analyzer, "_compute_high_level_metrics", spy)
    _flat_views(_in_process(tmp_path, trace_copy(TRACE_NAME)))

    assert not called, "in-process run fell back to Analyzer._compute_high_level_metrics"


@pytest.mark.smoke
def test_in_process_checkpoints_round_trip(tmp_path: pathlib.Path, trace_copy) -> None:
    """Checkpointing goes through client.compute/cancel, which NullClient fakes."""
    checkpoint_dir = tmp_path / "checkpoints"
    trace_path = trace_copy(TRACE_NAME)

    cold = _flat_views(_overrides(tmp_path, trace_path, ["cluster=none"], checkpoint=True))
    written = sorted(p.name for p in checkpoint_dir.iterdir()) if checkpoint_dir.is_dir() else []
    assert written, "no checkpoints written -- the test would pass without exercising the write path"

    warm = _flat_views(_overrides(tmp_path, trace_path, ["cluster=none"], checkpoint=True))
    assert set(cold) == set(warm), "restoring from checkpoint changed which views were produced"

    for view_key in cold:
        assert_frame_equal(
            cold[view_key],
            warm[view_key],
            check_dtype=True,
            check_index_type=True,
            check_exact=False,
            rtol=1e-9,
            obj=f"flat view {view_key} (warm checkpoint)",
        )


@pytest.mark.smoke
def test_warm_run_restores_the_flat_views(tmp_path: pathlib.Path, monkeypatch, trace_copy) -> None:
    """A warm run reads the flat views from their checkpoint instead of computing them again."""
    trace_path = trace_copy(TRACE_NAME)
    _flat_views(_overrides(tmp_path, trace_path, ["cluster=none"], checkpoint=True))

    computed = []
    original = Analyzer._process_flat_view

    def spy(self, *args, **kwargs):
        computed.append(kwargs.get("view_key"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Analyzer, "_process_flat_view", spy)
    _flat_views(_overrides(tmp_path, trace_path, ["cluster=none"], checkpoint=True))

    assert not computed, f"warm run computed flat views again: {computed}"


@pytest.mark.smoke
def test_restored_flat_views_equal_fresh_ones(tmp_path: pathlib.Path, trace_copy) -> None:
    """A flat view read from its checkpoint has the values, dtypes and index of the one computed."""
    trace_path = trace_copy(TRACE_NAME)

    def library_flat_views():
        dfa = init_with_hydra(hydra_overrides=_overrides(tmp_path, trace_path, ["cluster=none"], checkpoint=True))
        try:
            return dfa.analyze_trace().flat_views
        finally:
            dfa.shutdown()

    fresh = library_flat_views()
    restored = library_flat_views()
    assert fresh and set(fresh) == set(restored)
    for view_key in fresh:
        assert restored[view_key].index_names() == fresh[view_key].index_names(), view_key
        assert restored[view_key].columns == fresh[view_key].columns, view_key
        assert restored[view_key].dtypes == fresh[view_key].dtypes, view_key
        assert_frame_equal(
            fresh[view_key].to_pandas(nullable=True),
            restored[view_key].to_pandas(nullable=True),
            check_dtype=True,
            check_exact=False,
            rtol=1e-12,
            obj=f"flat view {view_key} (restored)",
        )
