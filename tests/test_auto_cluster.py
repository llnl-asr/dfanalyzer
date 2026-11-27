"""`cluster=auto` picks in-process or a cluster, and agrees either way.

It scans in this process first and compares the aggregated Arrow payloads
against ``cluster.max_bytes``. Under budget, the scan already in hand is used
and no cluster is ever built. Over it, the payloads are dropped, a
``LocalCluster`` is started, and the read is redone across its workers.

Two failure modes are easy to miss and are covered here:

* the analyzer resolving its own client instead of adopting the one the
  entrypoint built. That still *works* -- it gets a `NullClient` -- but it
  cannot promote, so `cluster=auto` silently degrades to `cluster=none` and
  every trace is analysed in process no matter how large.
* the escalation path never being exercised, since no bundled fixture comes
  close to the default budget. `cluster.max_bytes` is forced down to trigger it.
"""

import gc
import pathlib
import weakref

import pytest
from dask.distributed import Client, LocalCluster
from dftracer.analyzer import cluster as cluster_module
from dftracer.analyzer import init_with_hydra, trace_index
from dftracer.analyzer.cluster import AutoClient, AutoCluster
from pandas.testing import assert_frame_equal

# 8 files across several pids, so the escalated run is a real multi-worker scan.
TRACE_NAME = "dftracer-dlio"
PRESET = "ai"


def _overrides(tmp_path: pathlib.Path, trace_path: str, extra):
    return [
        "analyzer=dftracer",
        f"analyzer/preset={PRESET}",
        "analyzer.checkpoint=False",
        f"analyzer.checkpoint_dir={tmp_path}/checkpoints",
        *extra,
        f"hydra.run.dir={tmp_path}",
        f"hydra.runtime.output_dir={tmp_path}",
        f"trace_path={trace_path}",
        "view_types=[time_range,proc_name]",
    ]


def _analyze(overrides):
    """Run once, returning the flat views and whether a cluster was started."""
    dfa = init_with_hydra(hydra_overrides=overrides)
    try:
        result = dfa.analyze_trace()
        return (
            {key: view.to_pandas(nullable=True) for key, view in result.flat_views.items()},
            bool(getattr(dfa.client, "is_distributed", True)),
        )
    finally:
        dfa.shutdown()


@pytest.fixture(scope="module")
def runs(tmp_path_factory, module_trace_copy):
    """Every analysis this module needs, run once, in an order that works.

    Closing a Dask client shuts the dftracer C++ runtime down in this process,
    so an in-process scan attempted afterwards fails with "Runtime is shut
    down". Measured directly: `none` then `none` is fine, `none` then `local` is
    fine, `local` then `none` is not. Both auto runs scan in process, so they
    have to precede the clustered baseline.
    """
    tmp_path = tmp_path_factory.mktemp("auto")
    fits_views, fits_promoted = _analyze(
        _overrides(tmp_path / "fits", module_trace_copy(TRACE_NAME), ["cluster=auto"])
    )
    big_views, big_promoted = _analyze(
        _overrides(
            tmp_path / "big",
            module_trace_copy(TRACE_NAME),
            ["cluster=auto", "cluster.max_bytes=1", "cluster.n_workers=2"],
        )
    )
    clustered_views, _ = _analyze(
        _overrides(
            tmp_path / "local",
            module_trace_copy(TRACE_NAME),
            ["cluster=local", "cluster.n_workers=2", "cluster.processes=False"],
        )
    )
    return {
        "fits": (fits_views, fits_promoted),
        "escalated": (big_views, big_promoted),
        "clustered": clustered_views,
    }


@pytest.mark.smoke
def test_auto_is_the_default(trace_copy) -> None:
    dfa = init_with_hydra(hydra_overrides=["analyzer=dftracer", f"trace_path={trace_copy(TRACE_NAME)}"])
    try:
        assert isinstance(dfa.cluster, AutoCluster), f"default cluster is {type(dfa.cluster).__name__}"
    finally:
        dfa.shutdown()


@pytest.mark.smoke
def test_auto_adopts_the_entrypoint_client(tmp_path: pathlib.Path, trace_copy) -> None:
    """A self-resolved client cannot promote, so auto would degrade to none."""
    dfa = init_with_hydra(hydra_overrides=_overrides(tmp_path, trace_copy(TRACE_NAME), ["cluster=auto"]))
    try:
        assert isinstance(dfa.client, AutoClient)
        assert dfa.analyzer.dask_client is dfa.client, "analyzer did not adopt the entrypoint's client"
        assert getattr(dfa.analyzer.dask_client, "can_promote", False), "analyzer's client cannot promote"
    finally:
        dfa.shutdown()


@pytest.mark.smoke
def test_auto_stays_in_process_when_it_fits(runs) -> None:
    _, promoted = runs["fits"]
    assert promoted is False, "started a cluster for a trace that fits in process"


@pytest.mark.smoke
def test_auto_escalates_when_over_budget(runs) -> None:
    """No bundled fixture reaches the real budget, so it is forced down to 1 byte."""
    _, promoted = runs["escalated"]
    assert promoted is True, "stayed in process despite exceeding the budget"


@pytest.mark.smoke
def test_auto_agrees_with_a_cluster(runs) -> None:
    """Both auto paths must match a plain clustered run, value for value."""
    clustered = runs["clustered"]
    assert clustered, "expected at least one flat view"

    for label in ("fits", "escalated"):
        actual, _ = runs[label]
        assert set(clustered) == set(actual), f"{label} produced different view keys"
        for view_key in clustered:
            assert_frame_equal(
                clustered[view_key],
                actual[view_key],
                check_dtype=True,
                check_exact=False,
                rtol=1e-9,
                obj=f"flat view {view_key} ({label})",
            )


class _Payload:
    """A weak-referenceable stand-in for a gathered result."""


class _FakeClient:
    def __init__(self, cluster):
        self.cluster = cluster

    def gather(self, futures):
        return futures


class _FakeCluster:
    def close(self):
        pass


def _fake_auto_client(monkeypatch, calls):
    monkeypatch.setattr(cluster_module, "Client", _FakeClient)
    monkeypatch.setattr(AutoCluster, "build_cluster", lambda self: _FakeCluster())
    monkeypatch.setattr(cluster_module, "register_auto_thread_plugin", lambda: calls.append("registered"))
    client = AutoClient(AutoCluster())
    client._impl.gather = lambda futures: futures
    return client


@pytest.mark.smoke
def test_over_budget_result_is_released_before_the_rerun(monkeypatch) -> None:
    calls = []
    client = _fake_auto_client(monkeypatch, calls)
    monkeypatch.setattr(AutoClient, "_fits", lambda self, nbytes: False)
    refs = []

    def submit():
        if refs:
            gc.collect()
            assert refs[0]() is None, "the rejected result is still alive during the rerun"
            return "second"
        payload = _Payload()
        refs.append(weakref.ref(payload))
        return payload

    assert client.gather_within_budget(submit, lambda: 0, lambda results: 1) == "second"
    assert len(refs) == 1


@pytest.mark.smoke
def test_promote_registers_the_thread_plugin(monkeypatch) -> None:
    calls = []
    client = _fake_auto_client(monkeypatch, calls)
    client.promote()
    client.promote()
    assert calls == ["registered"]


@pytest.mark.smoke
def test_none_cluster_ignores_the_ambient_client_for_the_index(
    tmp_path: pathlib.Path, trace_copy, monkeypatch
) -> None:
    seen = []
    real = trace_index.build_index_distributed

    def spy(*args, **kwargs):
        seen.append(kwargs.get("client"))
        return real(*args, **kwargs)

    monkeypatch.setattr(trace_index, "build_index_distributed", spy)
    ambient = Client(LocalCluster(processes=False, n_workers=1, dashboard_address=None))
    try:
        dfa = init_with_hydra(hydra_overrides=_overrides(tmp_path, trace_copy(TRACE_NAME), ["cluster=none"]))
        try:
            dfa.analyze_trace()
        finally:
            dfa.shutdown()
    finally:
        ambient.close()
    assert seen and all(client is None for client in seen), f"index built on {seen}"
