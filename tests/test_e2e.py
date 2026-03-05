import os
import pandas as pd
import pathlib
import pytest
import random
from dask.distributed import LocalCluster
from dftracer.analyzer import init_with_hydra
from glob import glob


# Full test matrix for comprehensive testing
full_analyzer_trace_params = [
    ("darshan", "posix", "tests/data/extracted/darshan-posix"),
    ("darshan", "posix", "tests/data/extracted/darshan-posix-dxt"),
    ("datacrumbs", "stack", "tests/data/extracted/datacrumbs-hdf5"),
    ("datacrumbs", "stack", "tests/data/extracted/datacrumbs-ior"),
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
    if analyzer == "datacrumbs":
        view_types = ["proc_name", "func_name"]
    elif trace_path.endswith("darshan-posix"):
        view_types = ["file_name", "proc_name"]

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

    assign_epochs = analyzer == "dftracer" and preset == "dlio"
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
        assert any(glob(f"{result.checkpoint_dir}/*.parquet")), "No checkpoint found"

    if preset == "stack":
        stack_layer = next(iter(result.layers), None)
        assert stack_layer is not None, "Expected a stack layer to be present"

        # Schema sanity + hierarchy invariants for layer views
        for view_key, view_df in result.views[stack_layer].items():
            view = view_df.reset_index()
            for col in ("parent_id", "root_id", "depth"):
                assert col in view.columns, f"Missing {col} in stack view {view_key}"
            assert view["root_id"].ne("").all(), f"Empty root_id values in stack view {view_key}"
            depth_zero = view["depth"] == 0
            if depth_zero.any():
                assert view.loc[depth_zero, "parent_id"].isin(["", None]).all(), (
                    f"Root rows should not have parent_id in stack view {view_key}"
                )
            if (~depth_zero).any():
                assert (view.loc[~depth_zero, "parent_id"] != "").all(), (
                    f"Non-root rows missing parent_id in stack view {view_key}"
                )

            # Basic invariants for time metrics when present
            time_sum = view["time_sum"]
            self_sum = view["self_time_sum"]
            child_sum = view["child_time_sum"]
            assert (time_sum.dropna() >= 0).all(), f"Negative time_sum in {view_key}"
            assert (self_sum.dropna() >= 0).all(), f"Negative self_time_sum in {view_key}"
            assert (child_sum.dropna() >= 0).all(), f"Negative child_time_sum in {view_key}"
            assert (time_sum.dropna() >= self_sum.dropna()).all(), (
                f"time_sum < self_time_sum in {view_key}"
            )
            delta = (self_sum + child_sum) - time_sum
            assert (delta.abs().dropna() < 1e-6).all(), (
                f"self_time_sum + child_time_sum != time_sum in {view_key}"
            )

            # Deterministic top child selection for roots with a strict max
            if "func_name" in view.columns:
                candidate = view[view["depth"] > 0].copy()
                if not candidate.empty:
                    candidate = candidate.dropna(subset=["time_frac_root", "root_id"])
                    if not candidate.empty:
                        time_sort = "time_sum" if "time_sum" in candidate.columns else None
                        sort_cols = ["root_id", "time_frac_root"]
                        sort_asc = [True, False]
                        if time_sort:
                            sort_cols.append(time_sort)
                            sort_asc.append(False)
                        sort_cols.append("func_name")
                        sort_asc.append(True)
                        top = candidate.sort_values(sort_cols, ascending=sort_asc).groupby("root_id").head(2)
                        for root_id, group in top.groupby("root_id"):
                            if len(group) < 2:
                                continue
                            first, second = group.iloc[0], group.iloc[1]
                            if pd.isna(first["time_frac_root"]) or pd.isna(second["time_frac_root"]):
                                continue
                            if first["time_frac_root"] > second["time_frac_root"]:
                                max_row = candidate[candidate["root_id"] == root_id].loc[
                                    candidate[candidate["root_id"] == root_id]["time_frac_root"].idxmax()
                                ]
                                assert first["func_name"] == max_row["func_name"], (
                                    f"Non-deterministic top child in root {root_id} for {view_key}"
                                )

    # Fraction bounds for flat views
    for view_key, flat_view in result.flat_views.items():
        frac_cols = [col for col in flat_view.columns if "_frac_" in col]
        assert frac_cols, f"Expected fraction columns in flat view {view_key}"
        for col in frac_cols:
            series = flat_view[col]
            in_range = series.between(0, 1) | series.isna()
            assert in_range.all(), f"Out-of-range fraction in {view_key}.{col}"

    # Shutdown the Dask client and cluster
    dfa.shutdown()

    # Verify that the Dask client is closed
    assert dfa.client.status == "closed", "Dask client should be closed after shutdown"
