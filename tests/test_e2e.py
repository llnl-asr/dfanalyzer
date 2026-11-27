import dask
import json
import os
import pathlib
import pytest
import random
from dask.distributed import LocalCluster
from dftracer.analyzer import init_with_hydra
from glob import glob


# Full test matrix for comprehensive testing
full_analyzer_trace_params = [
    ("dftracer", "ai", "dftracer-ai"),
    ("dftracer", "dlio", "dftracer-dlio"),
    ("dftracer", "posix", "dftracer-posix"),
]
full_checkpoint_params = [True, False]

# Ground-truth event/process counts per fixture, measured with a single worker.
#
# Without these, the e2e assertions below (view counts, layer counts, a
# checkpoint glob) all hold on an empty result, so a fixture that silently
# loads zero events keeps CI green while testing nothing -- exactly what
# happened before #63, when the dftracer fixtures shipped uncompressed .pfw
# traces the indexer could not read.
EXPECTED_TRACE_STATS = {
    "dftracer-ai": (125669, 1),
    "dftracer-dlio": (18039, 8),
    "dftracer-posix": (2056, 1),
}

# Reduced matrix for smoke testing (fast runs)
smoke_analyzer_trace_params = [random.choice(full_analyzer_trace_params)]
smoke_checkpoint_params = [False]  # Skip checkpoint to make tests faster


@pytest.fixture
def dask_cluster():
    cluster = LocalCluster(processes=False, protocol="tcp", worker_class="distributed.nanny.Nanny")
    yield cluster
    # This teardown code runs after all tests are done
    cluster.close()


@pytest.mark.smoke
def test_expected_trace_stats_keys_are_live() -> None:
    """Every entry in EXPECTED_TRACE_STATS must name a fixture that is tested.

    A key that no longer matches any parameter is not an error anyone sees --
    the lookup just misses and the assertion silently stops running. That is
    how renaming a fixture quietly removes its coverage.
    """
    tested = {trace_path for _, _, trace_path in full_analyzer_trace_params}
    dead = sorted(set(EXPECTED_TRACE_STATS) - tested)
    assert not dead, (
        f"EXPECTED_TRACE_STATS keys match no tested fixture: {dead}. "
        "Update them alongside any fixture rename, or their assertions stop running."
    )


@pytest.mark.full
@pytest.mark.parametrize("analyzer, preset, trace_name", full_analyzer_trace_params)
@pytest.mark.parametrize("checkpoint", full_checkpoint_params)
def test_e2e_full(
    analyzer: str,
    preset: str,
    trace_name: str,
    checkpoint: bool,
    tmp_path: pathlib.Path,
    dask_cluster: LocalCluster,
    trace_copy,
) -> None:
    """Full test suite with all parameter combinations."""
    _test_e2e(analyzer, preset, trace_copy(trace_name), trace_name, checkpoint, tmp_path, dask_cluster)


@pytest.mark.smoke
@pytest.mark.parametrize("analyzer, preset, trace_name", smoke_analyzer_trace_params)
@pytest.mark.parametrize("checkpoint", smoke_checkpoint_params)
def test_e2e_smoke(
    analyzer: str,
    preset: str,
    trace_name: str,
    checkpoint: bool,
    tmp_path: pathlib.Path,
    dask_cluster: LocalCluster,
    trace_copy,
) -> None:
    """Smoke test with minimal parameter combinations for quick validation."""
    _test_e2e(analyzer, preset, trace_copy(trace_name), trace_name, checkpoint, tmp_path, dask_cluster)


@pytest.mark.smoke
def test_json_output_file(tmp_path: pathlib.Path, dask_cluster: LocalCluster, trace_copy) -> None:
    """Verify JSON output file is created with the expected schema and views."""
    checkpoint_dir = f"{tmp_path}/checkpoints"
    scheduler_address = dask_cluster.scheduler_address
    output_path = tmp_path / "analysis.json"
    hydra_overrides = [
        "analyzer=dftracer",
        "analyzer/preset=posix",
        "analyzer.checkpoint=False",
        f"analyzer.checkpoint_dir={checkpoint_dir}",
        "cluster=external",
        "cluster.restart_on_connect=True",
        f"cluster.scheduler_address={scheduler_address}",
        "output=json",
        f"output.file_path={output_path}",
        f"hydra.run.dir={tmp_path}",
        f"hydra.runtime.output_dir={tmp_path}",
        f"trace_path={trace_copy('dftracer-posix')}",
        "view_types=[time_range,proc_name]",
    ]

    dfa = init_with_hydra(hydra_overrides=hydra_overrides)
    result = dfa.analyze_trace()
    dfa.output.handle_result(result)

    assert output_path.exists(), f"Expected JSON output at {output_path}"
    with output_path.open() as f:
        payload = json.load(f)

    assert payload["schema_version"] == "1"
    assert "raw_stats" in payload
    assert "views" in payload
    assert "time_range" in payload["views"]
    assert "proc_name" in payload["views"]
    assert "summary" in payload["views"]["time_range"]
    assert "additional_metrics" in payload["views"]["time_range"]
    assert "flat_views" not in payload

    dfa.shutdown()
    assert dfa.client.status == "closed", "Dask client should be closed after shutdown"


def _test_e2e(
    analyzer: str,
    preset: str,
    trace_path: str,
    trace_name: str,
    checkpoint: bool,
    tmp_path: pathlib.Path,
    dask_cluster: LocalCluster,
) -> None:
    """Common test logic extracted to avoid duplication."""
    checkpoint_dir = f"{tmp_path}/checkpoints"
    scheduler_address = dask_cluster.scheduler_address

    view_types = ["proc_name", "time_range"]

    hydra_overrides = [
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

    # Allow enabling debug logs for investigation via env var
    if os.getenv("DFANALYZER_DEBUG", "").lower() in {"1", "true", "yes"}:
        hydra_overrides.append("debug=True")

    # Both the DLIOBenchmark ("dlio") and AI-logging ("ai") presets define an
    # `epoch` layer, so keep the membership test explicit rather than matching a
    # shared name prefix.
    assign_epochs = analyzer == "dftracer" and preset in ("dlio", "ai")
    if assign_epochs:
        hydra_overrides.append("analyzer.assign_epochs=True")

    dfa = init_with_hydra(hydra_overrides=hydra_overrides)

    assert dfa.hydra_config.analyzer.checkpoint == checkpoint
    assert dfa.hydra_config.analyzer.checkpoint_dir == checkpoint_dir
    assert dfa.hydra_config.analyzer.preset.name == preset
    assert dfa.hydra_config.trace_path == trace_path
    if assign_epochs:
        assert dfa.hydra_config.analyzer.assign_epochs

    # Run the main function
    result = dfa.analyze_trace()

    assert len(result.flat_views) == len(dfa.hydra_config.view_types), (
        f"Expected {len(dfa.hydra_config.view_types)} views, got {len(result.flat_views)}"
    )
    assert len(result.layers) == len(dfa.hydra_config.analyzer.preset.layer_defs), (
        f"Expected {len(dfa.hydra_config.analyzer.preset.layer_defs)} layers, got {len(result.layers)}"
    )
    if checkpoint:
        assert any(glob(f"{result.checkpoint_dir}/*.pfw.gz")), "No checkpoint found"

    expected = EXPECTED_TRACE_STATS.get(trace_name)
    if expected is not None:
        expected_events, expected_procs = expected
        raw_stats = dask.compute(result.raw_stats)[0]
        stats = raw_stats if isinstance(raw_stats, dict) else raw_stats.__dict__
        actual_events = int(stats["total_event_count"])
        actual_procs = int(stats["unique_process_count"])
        assert actual_events == expected_events, (
            f"{trace_name}: expected {expected_events} events, got {actual_events}. "
            "A zero or short count means the trace was not read, not that the "
            "analysis is empty."
        )
        assert actual_procs == expected_procs, (
            f"{trace_name}: expected {expected_procs} processes, got {actual_procs}"
        )

    # Shutdown the Dask client and cluster
    dfa.shutdown()

    # Verify that the Dask client is closed
    assert dfa.client.status == "closed", "Dask client should be closed after shutdown"
