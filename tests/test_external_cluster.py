"""`cluster=external` must fail at connect when the cluster has no worker.

A restart that loses the only worker (a nanny kill that fails under load) leaves
the cluster empty, and a later restart of an empty cluster succeeds at once. The
index build then queues its tasks and waits for a worker forever.
"""

import pathlib

import dftracer.analyzer.cluster as cluster_module
import pytest
from dask.distributed import LocalCluster
from dftracer.analyzer import init_with_hydra

pytestmark = [pytest.mark.smoke, pytest.mark.full]


@pytest.fixture
def empty_cluster():
    cluster = LocalCluster(n_workers=0, processes=False, protocol="tcp")
    yield cluster
    cluster.close()


@pytest.mark.parametrize("restart", [True, False])
def test_external_cluster_without_workers_fails_at_connect(
    empty_cluster, tmp_path: pathlib.Path, trace_copy, monkeypatch, restart
):
    monkeypatch.setattr(cluster_module, "CLUSTER_RESTART_TIMEOUT_SECONDS", 2)
    with pytest.raises(TimeoutError, match="no worker"):
        init_with_hydra(
            hydra_overrides=[
                "analyzer=dftracer",
                "analyzer/preset=posix",
                "analyzer.checkpoint=False",
                "cluster=external",
                f"cluster.restart_on_connect={restart}",
                f"cluster.scheduler_address={empty_cluster.scheduler_address}",
                f"hydra.run.dir={tmp_path}",
                f"hydra.runtime.output_dir={tmp_path}",
                f"trace_path={trace_copy('dftracer-dlio')}",
                "view_types=[time_range]",
            ]
        )
