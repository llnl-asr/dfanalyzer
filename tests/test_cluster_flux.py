import socket
from hydra.utils import instantiate

from dftracer.analyzer.config import FluxClusterConfig, _default_cluster_scheduler_host


def test_default_cluster_scheduler_host_uses_resolved_ip(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "cluster-login")
    monkeypatch.setattr(socket, "gethostbyname", lambda host: "192.168.192.100")

    assert _default_cluster_scheduler_host() == "192.168.192.100"


def test_flux_cluster_job_script_uses_flux_queue_and_run_wrapper(tmp_path):
    cluster = instantiate(
        FluxClusterConfig(
            cores=4,
            local_directory=str(tmp_path),
            memory="8GB",
            processes=2,
            queue="pdebug",
            job_nodes=2,
        )
    )

    try:
        job_script = cluster.job_script()
    finally:
        cluster.close()

    assert "#flux: -q pdebug" in job_script
    assert "#flux: -N 2" in job_script
    assert "#flux: -n 2" in job_script
    assert "flux run -N 2 -n 2" in job_script
    assert "--nworkers" not in job_script
