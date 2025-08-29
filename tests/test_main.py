import pathlib
import pytest
import random
from dask.distributed import LocalCluster
from dfanalyzer import init_with_hydra
from glob import glob


# Full test matrix for comprehensive testing
full_analyzer_trace_params = [
    # ("darshan", "posix", "tests/data/extracted/darshan-posix"),
    ("darshan", "posix", "tests/data/extracted/darshan-posix-dxt"),
    ("dftracer", "dlio", "tests/data/extracted/dftracer-dlio"),
    ("dftracer", "posix", "tests/data/extracted/dftracer-posix"),
    ("recorder", "posix", "tests/data/extracted/recorder-posix-parquet"),
]
full_checkpoint_params = [True, False]

# Reduced matrix for smoke testing (fast runs)
smoke_analyzer_trace_params = [random.choice(full_analyzer_trace_params)]
smoke_checkpoint_params = [False]  # Skip checkpoint to make tests faster


@pytest.fixture(scope="session")
def dask_cluster():
    cluster = LocalCluster(processes=False, protocol="tcp", worker_class="distributed.nanny.Nanny")
    yield cluster
    # This teardown code runs after all tests are done
    cluster.close()


@pytest.mark.full
@pytest.mark.parametrize("analyzer, preset, trace_path", full_analyzer_trace_params)
@pytest.mark.parametrize("checkpoint", full_checkpoint_params)
def test_e2e_full(
    analyzer: str,
    preset: str,
    trace_path: str,
    checkpoint: bool,
    tmp_path: pathlib.Path,
    dask_cluster: LocalCluster,
) -> None:
    """Full test suite with all parameter combinations."""
    _test_e2e(analyzer, preset, trace_path, checkpoint, tmp_path, dask_cluster)


@pytest.mark.smoke
@pytest.mark.parametrize("analyzer, preset, trace_path", smoke_analyzer_trace_params)
@pytest.mark.parametrize("checkpoint", smoke_checkpoint_params)
def test_e2e_smoke(
    analyzer: str,
    preset: str,
    trace_path: str,
    checkpoint: bool,
    tmp_path: pathlib.Path,
    dask_cluster: LocalCluster,
) -> None:
    """Smoke test with minimal parameter combinations for quick validation."""
    _test_e2e(analyzer, preset, trace_path, checkpoint, tmp_path, dask_cluster)


def _test_e2e(
    analyzer: str,
    preset: str,
    trace_path: str,
    checkpoint: bool,
    tmp_path: pathlib.Path,
    dask_cluster: LocalCluster,
) -> None:
    """Common test logic extracted to avoid duplication."""
    checkpoint_dir = f"{tmp_path}/checkpoints"
    scheduler_address = dask_cluster.scheduler_address

    view_types = ["proc_name", "time_range"]
    if trace_path.endswith("darshan-posix"):
        view_types = ["file_name", "proc_name"]

    dfa = init_with_hydra(
        hydra_overrides=[
            f"analyzer={analyzer}",
            f"analyzer/preset={preset}",
            f"analyzer.checkpoint={checkpoint}",
            f"analyzer.checkpoint_dir={checkpoint_dir}",
            "cluster=external",
            f"cluster.restart_on_connect={True}",
            f"cluster.scheduler_address={scheduler_address}",
            f"hydra.run.dir={tmp_path}",
            f"hydra.runtime.output_dir={tmp_path}",
            f"trace_path={trace_path}",
            f"view_types=[{','.join(view_types)}]",
        ]
    )

    assert dfa.hydra_config.analyzer.checkpoint == checkpoint
    assert dfa.hydra_config.analyzer.checkpoint_dir == checkpoint_dir
    assert dfa.hydra_config.analyzer.preset.name == preset
    assert dfa.hydra_config.trace_path == trace_path

    # Run the main function
    result = dfa.analyze_trace()

    assert len(result.flat_views) == len(dfa.hydra_config.view_types), (
        f"Expected {len(dfa.hydra_config.view_types)} views, got {len(result.flat_views)}"
    )
    assert len(result.layers) == len(dfa.hydra_config.analyzer.preset.layer_defs), (
        f"Expected {len(dfa.hydra_config.analyzer.preset.layer_defs)} layers, got {len(result.layers)}"
    )
    if checkpoint:
        assert any(glob(f"{result.checkpoint_dir}/*.json")), "No checkpoint found"

    # Shutdown the Dask client and cluster
    dfa.shutdown()

    # Verify that the Dask client is closed
    assert dfa.client.status == "closed", "Dask client should be closed after shutdown"
