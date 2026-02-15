import numpy as np
import pandas as pd
import dask.dataframe as dd
import pytest
from dask.distributed import Client, LocalCluster
from dftracer.analyzer.analyzer import Analyzer
from dftracer.analyzer.config import AnalyzerPresetConfigPOSIX, AnalyzerPresetConfigDLIO
from dftracer.analyzer.constants import VIEW_TYPES
from dftracer.analyzer.metrics import set_view_metrics
from typing import List, Optional, Dict

# Ensure this module runs in both smoke and full CI modes
pytestmark = [pytest.mark.smoke, pytest.mark.full]


class DummyAnalyzer(Analyzer):
    """Minimal concrete Analyzer for testing internal helpers.

    We only need to satisfy the abstract API; tests will call
    _compute_high_level_metrics directly with synthetic frames.
    """

    def read_trace(
        self,
        trace_path: str,
        extra_columns: Optional[Dict[str, str]],
        extra_columns_fn: Optional[callable],
    ) -> dd.DataFrame:
        raise NotImplementedError


def make_analyzer() -> DummyAnalyzer:
    preset = AnalyzerPresetConfigPOSIX()
    # Disable checkpointing to avoid requiring a real checkpoint dir
    return DummyAnalyzer(
        preset=preset,
        checkpoint=False,
        checkpoint_dir="",
        time_granularity=1.0,
        time_resolution=1.0,
    )


def build_base_df() -> pd.DataFrame:
    # Two files, two procs, two funcs; include bins and core metrics
    return pd.DataFrame(
        {
            "file_name": ["f1", "f1", "f2", "f2"],
            "proc_name": ["p#h#1#t", "p#h#1#t", "q#h#2#t", "q#h#2#t"],
            "time_range": [0, 0, 1, 1],
            "cat": ["posix", "posix", "posix", "posix"],
            "io_cat": [1, 2, 1, 3],
            "acc_pat": ["seq", "rand", "seq", "seq"],
            "func_name": ["read", "write", "read", "stat"],
            "time": [1.0, 2.0, 0.0, 3.0],
            "count": [10, 0, 5, 5],
            "size": [100, 200, 0, 300],
            # bin/histogram columns
            "size_bin_0": [1, 0, 0, 2],
            "size_bin_1": [0, 1, 0, 0],
        }
    )


# ---------- Helpers ----------


def _assert_groupby_and_aggs_basic(out: pd.DataFrame, view_types: List[str]) -> None:
    for col in set(view_types).union({"cat", "io_cat", "acc_pat", "func_name"}):
        assert col in out.index.names
    grp = out.reset_index()
    read_rows = grp[(grp["proc_name"] == "p#h#1#t") & (grp["func_name"] == "read")]
    assert pytest.approx(read_rows["time"].iloc[0]) == 1.0
    assert pytest.approx(read_rows["count"].iloc[0]) == 10.0
    assert pytest.approx(read_rows["size"].iloc[0]) == 100.0
    write_rows = grp[(grp["proc_name"] == "p#h#1#t") & (grp["func_name"] == "write")]
    assert pd.isna(write_rows["count"].iloc[0])
    assert out["size_bin_0"].dtype.name == "Int32"
    assert out["size_bin_1"].dtype.name == "Int32"


def _assert_unique_sets_non_selected(out: pd.DataFrame, selected: List[str]) -> None:
    assert "proc_name" in set(VIEW_TYPES) - set(selected)
    assert "time_range" in set(VIEW_TYPES) - set(selected)
    vals = out.reset_index().loc[0]
    assert isinstance(vals["proc_name"], (set, frozenset)) or hasattr(vals["proc_name"], "flatten")
    assert isinstance(vals["time_range"], (set, frozenset)) or hasattr(vals["time_range"], "flatten")


def _assert_multiple_view_types_selection(out: pd.DataFrame, selected: List[str]) -> None:
    assert "time_range" in set(VIEW_TYPES) - set(selected)
    tr_vals = out.reset_index()["time_range"].iloc[0]
    assert isinstance(tr_vals, (set, frozenset)) or hasattr(tr_vals, "flatten")


def _assert_empty_input(out: pd.DataFrame, view_types: List[str]) -> None:
    assert out.shape[0] == 0
    for col in set(view_types).union({"cat", "io_cat", "acc_pat", "func_name"}):
        assert col in out.index.names


# Helpers for main view tests
def _build_hlm(analyzer: "DummyAnalyzer", pdf: pd.DataFrame, view_types: List[str], as_dask: bool):
    traces = dd.from_pandas(pdf, npartitions=2) if as_dask else pdf
    hlm = analyzer._compute_high_level_metrics(traces=traces, view_types=view_types, partition_size="64MB")
    if as_dask:
        return hlm
    return hlm.compute() if isinstance(hlm, dd.DataFrame) else hlm


def _assert_no_infinities(df: pd.DataFrame):
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            coerced = pd.to_numeric(series, errors="coerce")
            assert not np.isinf(coerced.to_numpy()).any(), f"Infinite values found in column {col}"


# ---------- Fixtures ----------


@pytest.fixture(scope="session", autouse=True)
def _dask_client_session():
    cluster = LocalCluster(processes=False)
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


@pytest.fixture
def dummy_analyzer() -> DummyAnalyzer:
    return make_analyzer()


@pytest.fixture
def analyzer_dlio() -> DummyAnalyzer:
    return DummyAnalyzer(
        preset=AnalyzerPresetConfigDLIO(),
        checkpoint=False,
        checkpoint_dir="",
        time_granularity=1.0,
        time_resolution=1.0,
    )


@pytest.fixture
def dummy_analyzer_quantiles() -> DummyAnalyzer:
    preset = AnalyzerPresetConfigPOSIX()
    return DummyAnalyzer(
        preset=preset,
        checkpoint=False,
        checkpoint_dir="",
        time_granularity=1.0,
        time_resolution=1.0,
        quantile_stats=True,
    )


# ---------- Tests (pandas) ----------


def test_hlm_groupby_and_aggregations_basic(dummy_analyzer: DummyAnalyzer):
    view_types = ["proc_name"]
    pdf = build_base_df()
    hlm = dummy_analyzer._compute_high_level_metrics(traces=pdf, view_types=view_types, partition_size="64MB")
    out = hlm  # already pandas
    _assert_groupby_and_aggs_basic(out, view_types)


def test_hlm_unique_sets_for_non_selected_view_types(dummy_analyzer: DummyAnalyzer):
    view_types = ["file_name"]
    pdf = build_base_df()
    hlm = dummy_analyzer._compute_high_level_metrics(traces=pdf, view_types=view_types, partition_size="64MB")
    out = hlm
    _assert_unique_sets_non_selected(out, view_types)


def test_hlm_multiple_view_types_selection(dummy_analyzer: DummyAnalyzer):
    view_types = ["file_name", "proc_name"]
    pdf = build_base_df()
    hlm = dummy_analyzer._compute_high_level_metrics(traces=pdf, view_types=view_types, partition_size="64MB")
    out = hlm
    _assert_multiple_view_types_selection(out, view_types)


def test_hlm_empty_input(dummy_analyzer: DummyAnalyzer):
    view_types = ["proc_name"]
    base = build_base_df().iloc[0:0]
    hlm = dummy_analyzer._compute_high_level_metrics(traces=base, view_types=view_types, partition_size="64MB")
    out = hlm
    _assert_empty_input(out, view_types)


def test_hlm_minimal_required_columns(dummy_analyzer: DummyAnalyzer):
    pdf = pd.DataFrame(
        {
            "file_name": ["f1"],
            "proc_name": ["p#h#1#t"],
            "time_range": [0],
            "cat": ["posix"],
            "io_cat": [1],
            "acc_pat": ["seq"],
            "func_name": ["read"],
            "time": [0.0],
            "count": [1],
            "size": [10],
        }
    )
    hlm = dummy_analyzer._compute_high_level_metrics(traces=pdf, view_types=["file_name"], partition_size="64MB")
    out = hlm
    row = out.reset_index().iloc[0]
    assert pd.isna(row["time"])  # zero normalized to NaN
    assert row["count"] == 1
    assert row["size"] == 10


@pytest.mark.parametrize("npartitions", [1, 3])
def test_hlm_dask_partitioning_behavior(dummy_analyzer: DummyAnalyzer, npartitions: int):
    pdf = build_base_df().sample(frac=1.0, random_state=42).reset_index(drop=True)
    ddf = dd.from_pandas(pdf, npartitions=npartitions)

    hlm = dummy_analyzer._compute_high_level_metrics(traces=ddf, view_types=["proc_name"], partition_size="64MB")
    out = hlm.compute()

    # Re-run with a different partitioning to ensure stable results
    ddf2 = dd.from_pandas(pdf, npartitions=max(1, npartitions - 1))
    hlm2 = dummy_analyzer._compute_high_level_metrics(traces=ddf2, view_types=["proc_name"], partition_size="32MB")
    out2 = hlm2.compute()

    # Sort for deterministic comparison
    idx_cols = list(set(["proc_name"]).union({"cat", "io_cat", "acc_pat", "func_name"}))
    out_s = out.reset_index().sort_values(idx_cols).reset_index(drop=True)
    out2_s = out2.reset_index().sort_values(idx_cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(out_s, out2_s, check_like=True)


# ---------- Tests (dask) ----------


def test_hlm_groupby_and_aggregations_basic_dask(dummy_analyzer: DummyAnalyzer):
    view_types = ["proc_name"]
    pdf = build_base_df()
    ddf = dd.from_pandas(pdf, npartitions=2)
    hlm = dummy_analyzer._compute_high_level_metrics(traces=ddf, view_types=view_types, partition_size="64MB")
    out = hlm.compute()
    _assert_groupby_and_aggs_basic(out, view_types)


def test_hlm_unique_sets_for_non_selected_view_types_dask(dummy_analyzer: DummyAnalyzer):
    view_types = ["file_name"]
    pdf = build_base_df()
    ddf = dd.from_pandas(pdf, npartitions=1)
    hlm = dummy_analyzer._compute_high_level_metrics(traces=ddf, view_types=view_types, partition_size="64MB")
    out = hlm.compute()
    _assert_unique_sets_non_selected(out, view_types)


def test_hlm_multiple_view_types_selection_dask(dummy_analyzer: DummyAnalyzer):
    view_types = ["file_name", "proc_name"]
    pdf = build_base_df()
    ddf = dd.from_pandas(pdf, npartitions=2)
    hlm = dummy_analyzer._compute_high_level_metrics(traces=ddf, view_types=view_types, partition_size="64MB")
    out = hlm.compute()
    _assert_multiple_view_types_selection(out, view_types)


def test_hlm_empty_input_dask(dummy_analyzer: DummyAnalyzer):
    view_types = ["proc_name"]
    base = build_base_df().iloc[0:0]
    ddf = dd.from_pandas(base, npartitions=1)
    hlm = dummy_analyzer._compute_high_level_metrics(traces=ddf, view_types=view_types, partition_size="64MB")
    out = hlm.compute()
    _assert_empty_input(out, view_types)


def test_hlm_minimal_required_columns_dask(dummy_analyzer: DummyAnalyzer):
    pdf = pd.DataFrame(
        {
            "file_name": ["f1"],
            "proc_name": ["p#h#1#t"],
            "time_range": [0],
            "cat": ["posix"],
            "io_cat": [1],
            "acc_pat": ["seq"],
            "func_name": ["read"],
            "time": [0.0],
            "count": [1],
            "size": [10],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=1)
    hlm = dummy_analyzer._compute_high_level_metrics(traces=ddf, view_types=["file_name"], partition_size="64MB")
    out = hlm.compute()
    row = out.reset_index().iloc[0]
    assert pd.isna(row["time"])  # zero normalized to NaN
    assert row["count"] == 1
    assert row["size"] == 10


# ---------- Main view tests (moved here) ----------


def test_main_view_posix_pandas_basic(dummy_analyzer: DummyAnalyzer):
    pdf = build_base_df()
    view_types = ["proc_name"]
    hlm = _build_hlm(dummy_analyzer, pdf, view_types, as_dask=False)

    main = dummy_analyzer._compute_main_view(layer="posix", hlm=hlm, view_types=view_types, partition_size="64MB")

    # Aggregation index
    assert list(main.index.names) == view_types
    _assert_no_infinities(main)

    # Derived layer metrics present and main metrics computed
    cols = main.columns.tolist()
    assert any(c.endswith("read_size") for c in cols)
    assert any(c.endswith("read_time") for c in cols)
    assert any(c.endswith("read_bw") for c in cols)
    assert any(c.endswith("read_intensity") for c in cols)
    assert any(c.endswith("read_ops") for c in cols)

    # Zero-to-NaN normalization: write_count was 0 for p#h#1#t
    grp = main.reset_index()
    row = grp[grp["proc_name"] == "p#h#1#t"].iloc[0]
    assert pd.isna(row.filter(like="write_count").iloc[0])

    # Numeric assertions for proc p#h#1#t
    assert pytest.approx(row["time"], rel=1e-6) == 3.0
    assert pytest.approx(row["count"], rel=1e-6) == 10.0
    assert pytest.approx(row["size"], rel=1e-6) == 300.0
    assert pytest.approx(row["bw"], rel=1e-6) == 100.0  # 300/3
    assert pytest.approx(row["intensity"], rel=1e-6) == (10.0 / 300.0)
    assert pytest.approx(row["ops"], rel=1e-6) == (10.0 / 3.0)
    # Read-derived
    assert pytest.approx(row["read_time"], rel=1e-6) == 1.0
    assert pytest.approx(row["read_count"], rel=1e-6) == 10.0
    assert pytest.approx(row["read_size"], rel=1e-6) == 100.0
    assert pytest.approx(row["read_bw"], rel=1e-6) == 100.0  # 100/1
    assert pytest.approx(row["read_intensity"], rel=1e-6) == 0.1  # 10/100
    assert pytest.approx(row["read_ops"], rel=1e-6) == 10.0  # 10/1

    # Numeric assertions for proc q#h#2#t
    row_q = grp[grp["proc_name"] == "q#h#2#t"].iloc[0]
    assert pytest.approx(row_q["time"], rel=1e-6) == 3.0
    assert pytest.approx(row_q["count"], rel=1e-6) == 10.0
    assert pytest.approx(row_q["size"], rel=1e-6) == 300.0
    assert pytest.approx(row_q["bw"], rel=1e-6) == 100.0
    assert pytest.approx(row_q["intensity"], rel=1e-6) == (10.0 / 300.0)
    assert pytest.approx(row_q["ops"], rel=1e-6) == (10.0 / 3.0)
    # Read-derived should be NA for bw/intensity/ops due to missing size/time in read group
    assert pd.isna(row_q.filter(like="read_time").iloc[0])
    assert pd.isna(row_q.filter(like="read_size").iloc[0])
    assert pd.isna(row_q.filter(like="read_bw").iloc[0])
    assert pd.isna(row_q.filter(like="read_intensity").iloc[0])
    assert pd.isna(row_q.filter(like="read_ops").iloc[0])
    # Metadata-derived ops
    assert pytest.approx(row_q["metadata_time"], rel=1e-6) == 3.0
    assert pytest.approx(row_q["metadata_count"], rel=1e-6) == 5.0
    assert pytest.approx(row_q["metadata_ops"], rel=1e-6) == (5.0 / 3.0)


def test_main_view_posix_dask_basic(dummy_analyzer: DummyAnalyzer):
    pdf = build_base_df()
    view_types = ["proc_name"]
    hlm = _build_hlm(dummy_analyzer, pdf, view_types, as_dask=True)

    main_dd = dummy_analyzer._compute_main_view(layer="posix", hlm=hlm, view_types=view_types, partition_size="64MB")
    main = main_dd.compute() if isinstance(main_dd, dd.DataFrame) else main_dd

    assert list(main.index.names) == view_types
    _assert_no_infinities(main)

    # Spot-check core numbers for p#h#1#t
    grp = main.reset_index()
    row = grp[grp["proc_name"] == "p#h#1#t"].iloc[0]
    assert pytest.approx(row["time"], rel=1e-6) == 3.0
    assert pytest.approx(row["count"], rel=1e-6) == 10.0
    assert pytest.approx(row["size"], rel=1e-6) == 300.0
    assert pytest.approx(row["bw"], rel=1e-6) == 100.0
    assert pytest.approx(row["read_time"], rel=1e-6) == 1.0
    assert pytest.approx(row["read_count"], rel=1e-6) == 10.0
    assert pytest.approx(row["read_size"], rel=1e-6) == 100.0


def test_main_view_posix_unique_set_columns(dummy_analyzer: DummyAnalyzer):
    pdf = build_base_df()
    view_types = ["proc_name"]
    hlm = _build_hlm(dummy_analyzer, pdf, view_types, as_dask=False)

    main = dummy_analyzer._compute_main_view(layer="posix", hlm=hlm, view_types=view_types, partition_size="64MB")
    vals = main.reset_index().iloc[0]
    # Non-selected view types -> unique sets
    assert isinstance(vals.get("file_name"), (set, frozenset)) or hasattr(vals.get("file_name"), "flatten")
    assert isinstance(vals.get("time_range"), (set, frozenset)) or hasattr(vals.get("time_range"), "flatten")


def test_main_view_posix_empty_hlm(dummy_analyzer: DummyAnalyzer):
    pdf = build_base_df().iloc[0:0]
    view_types = ["proc_name"]
    hlm = _build_hlm(dummy_analyzer, pdf, view_types, as_dask=False)

    main = dummy_analyzer._compute_main_view(layer="posix", hlm=hlm, view_types=view_types, partition_size="64MB")
    assert main.shape[0] == 0


def test_main_view_nonposix_drops_size_and_file(analyzer_dlio: DummyAnalyzer):
    pdf = build_base_df()
    view_types = ["proc_name"]
    hlm = _build_hlm(analyzer_dlio, pdf, view_types, as_dask=False)

    # Use a non-posix layer e.g. 'compute'
    main = analyzer_dlio._compute_main_view(layer="compute", hlm=hlm, view_types=view_types, partition_size="64MB")

    # size* and file_name dropped before aggregation
    assert not any(col.startswith("size") for col in main.columns)
    assert "file_name" not in main.columns
    # Consequently bw/intensity shouldn't be present
    assert not any(col.endswith("_bw") for col in main.columns)
    assert not any(col.endswith("_intensity") for col in main.columns)
    _assert_no_infinities(main)


def test_main_view_nonposix_dask(analyzer_dlio: DummyAnalyzer):
    pdf = build_base_df()
    view_types = ["proc_name"]
    hlm = _build_hlm(analyzer_dlio, pdf, view_types, as_dask=True)

    main_dd = analyzer_dlio._compute_main_view(layer="compute", hlm=hlm, view_types=view_types, partition_size="64MB")
    main = main_dd.compute() if isinstance(main_dd, dd.DataFrame) else main_dd

    assert list(main.index.names) == view_types
    assert not any(col.startswith("size") for col in main.columns)
    assert "file_name" not in main.columns
    _assert_no_infinities(main)


# ---------- Compute view tests ----------


def test_compute_view_proc_pandas_basic(dummy_analyzer: DummyAnalyzer):
    pdf = build_base_df()
    # Build pipeline: HLM -> main_view, then compute view
    view_types = ["proc_name", "file_name"]
    hlm = _build_hlm(dummy_analyzer, pdf, view_types, as_dask=False)
    records = dummy_analyzer._compute_main_view(layer="posix", hlm=hlm, view_types=view_types, partition_size="64MB")
    view_key = ("proc_name",)
    view_type = "proc_name"

    view = dummy_analyzer._compute_view(
        layer="posix",
        records=records,
        view_key=view_key,
        view_type=view_type,
        view_types=view_types,
    )

    out = view  # pandas
    assert list(out.index.names) == ["proc_name"]
    _assert_no_infinities(out)

    # Columns include expected aggregates and unique count for file_name
    cols = out.columns.tolist()
    assert "time_sum" in cols and "count_sum" in cols and "size_sum" in cols
    assert "file_name_nunique" in cols
    # Enrich to compute frac totals and ops metrics
    enriched = set_view_metrics(out.copy(), metric_boundaries={}, is_view_process_based=True)
    ecols = enriched.columns.tolist()
    assert "time_frac_total" in ecols and "count_frac_total" in ecols and "size_frac_total" in ecols
    assert "ops_slope" in ecols  # from set_view_metrics

    grp = enriched.reset_index()
    row_p = grp[grp["proc_name"] == "p#h#1#t"].iloc[0]
    # Aggregates
    assert pytest.approx(row_p["time_sum"], rel=1e-6) == 3.0
    assert pytest.approx(row_p["count_sum"], rel=1e-6) == 10.0
    assert pytest.approx(row_p["size_sum"], rel=1e-6) == 300.0
    # Min across grouped (proc,file) rows after pre-grouping: zero got summed into 10
    assert pytest.approx(row_p["count_min"], rel=1e-6) == 10.0
    # Frac totals should be 0.5 for both procs; slope 1.0
    assert pytest.approx(row_p["time_frac_total"], rel=1e-6) == 0.5
    assert pytest.approx(row_p["count_frac_total"], rel=1e-6) == 0.5
    assert pytest.approx(row_p["size_frac_total"], rel=1e-6) == 0.5
    assert pytest.approx(row_p["ops_slope"], rel=1e-6) == 1.0
    # Unique file_name count
    assert row_p["file_name_nunique"] == 1


def test_compute_view_file_pandas_basic(dummy_analyzer: DummyAnalyzer):
    pdf = build_base_df()
    # Build pipeline: HLM -> main_view, then compute view
    view_types = ["file_name", "proc_name"]
    hlm = _build_hlm(dummy_analyzer, pdf, view_types, as_dask=False)
    records = dummy_analyzer._compute_main_view(layer="posix", hlm=hlm, view_types=view_types, partition_size="64MB")
    view_key = ("file_name",)
    view_type = "file_name"

    view = dummy_analyzer._compute_view(
        layer="posix",
        records=records,
        view_key=view_key,
        view_type=view_type,
        view_types=view_types,
    )

    out = view
    assert list(out.index.names) == ["file_name"]
    _assert_no_infinities(out)

    enriched = set_view_metrics(out.copy(), metric_boundaries={}, is_view_process_based=False)
    grp = enriched.reset_index()
    row_f1 = grp[grp["file_name"] == "f1"].iloc[0]
    row_f2 = grp[grp["file_name"] == "f2"].iloc[0]
    # Process-unaware time metrics use max over per-proc-summed time
    assert pytest.approx(row_f1["time_max"], rel=1e-6) == 3.0
    assert pytest.approx(row_f2["time_max"], rel=1e-6) == 3.0
    # Frac totals 0.5 each
    assert pytest.approx(row_f1["time_frac_total"], rel=1e-6) == 0.5
    assert pytest.approx(row_f2["time_frac_total"], rel=1e-6) == 0.5
    assert pytest.approx(row_f1["count_frac_total"], rel=1e-6) == 0.5
    assert pytest.approx(row_f2["count_frac_total"], rel=1e-6) == 0.5
    assert pytest.approx(row_f1["size_frac_total"], rel=1e-6) == 0.5
    assert pytest.approx(row_f2["size_frac_total"], rel=1e-6) == 0.5
    # Unique proc_name count per file
    assert row_f1["proc_name_nunique"] == 1
    assert row_f2["proc_name_nunique"] == 1


def test_compute_view_proc_dask_basic(dummy_analyzer: DummyAnalyzer):
    pdf = build_base_df()
    view_types = ["proc_name", "file_name"]
    traces = dd.from_pandas(pdf, npartitions=2)
    hlm = dummy_analyzer._compute_high_level_metrics(traces=traces, view_types=view_types, partition_size="64MB")
    main_view = dummy_analyzer._compute_main_view(layer="posix", hlm=hlm, view_types=view_types, partition_size="64MB")
    ddf = main_view  # already dask
    view_key = ("proc_name",)
    view_type = "proc_name"

    view_dd = dummy_analyzer._compute_view(
        layer="posix",
        records=ddf,
        view_key=view_key,
        view_type=view_type,
        view_types=view_types,
    )
    out = view_dd.compute() if isinstance(view_dd, dd.DataFrame) else view_dd
    assert list(out.index.names) == ["proc_name"]
    _assert_no_infinities(out)
    enriched = set_view_metrics(out.copy(), metric_boundaries={}, is_view_process_based=True)
    grp = enriched.reset_index()
    row_p = grp[grp["proc_name"] == "p#h#1#t"].iloc[0]
    assert pytest.approx(row_p["time_frac_total"], rel=1e-6) == 0.5
    assert pytest.approx(row_p["count_frac_total"], rel=1e-6) == 0.5
    assert pytest.approx(row_p["size_frac_total"], rel=1e-6) == 0.5


def test_compute_view_file_dask_basic(dummy_analyzer: DummyAnalyzer):
    pdf = build_base_df()
    view_types = ["file_name", "proc_name"]
    traces = dd.from_pandas(pdf, npartitions=2)
    hlm = dummy_analyzer._compute_high_level_metrics(traces=traces, view_types=view_types, partition_size="64MB")
    main_view = dummy_analyzer._compute_main_view(layer="posix", hlm=hlm, view_types=view_types, partition_size="64MB")
    ddf = main_view  # already dask
    view_key = ("file_name",)
    view_type = "file_name"

    view_dd = dummy_analyzer._compute_view(
        layer="posix",
        records=ddf,
        view_key=view_key,
        view_type=view_type,
        view_types=view_types,
    )
    out = view_dd.compute() if isinstance(view_dd, dd.DataFrame) else view_dd
    assert list(out.index.names) == ["file_name"]
    _assert_no_infinities(out)
    enriched = set_view_metrics(out.copy(), metric_boundaries={}, is_view_process_based=False)
    grp = enriched.reset_index()
    row_f1 = grp[grp["file_name"] == "f1"].iloc[0]
    row_f2 = grp[grp["file_name"] == "f2"].iloc[0]
    assert pytest.approx(row_f1["time_frac_total"], rel=1e-6) == 0.5
    assert pytest.approx(row_f2["time_frac_total"], rel=1e-6) == 0.5


# ---------- Dask compute_view quantile stats ----------


def _assert_quantile_columns_present(df: pd.DataFrame, metric_prefix: str):
    cols = df.columns.tolist()
    for rng in ("q1_q99", "q5_q95", "q10_q90", "q25_q75"):
        assert f"{metric_prefix}_{rng}_mean" in cols
        assert f"{metric_prefix}_{rng}_std" in cols
        assert f"{metric_prefix}_{rng}_count" in cols
        # Ensure legacy _stats column absent
        assert f"{metric_prefix}_{rng}_stats" not in cols


def test_compute_view_proc_dask_quantiles(dummy_analyzer_quantiles: DummyAnalyzer):
    pdf = build_base_df()
    traces = dd.from_pandas(pdf, npartitions=2)
    view_types = ["proc_name", "file_name"]
    hlm = dummy_analyzer_quantiles._compute_high_level_metrics(
        traces=traces, view_types=view_types, partition_size="64MB"
    )
    main_view = dummy_analyzer_quantiles._compute_main_view(
        layer="posix", hlm=hlm, view_types=view_types, partition_size="64MB"
    )

    view_dd = dummy_analyzer_quantiles._compute_view(
        layer="posix",
        records=main_view,
        view_key=("proc_name",),
        view_type="proc_name",
        view_types=view_types,
    )
    out = view_dd.compute()
    assert list(out.index.names) == ["proc_name"]
    _assert_no_infinities(out)
    _assert_quantile_columns_present(out, "time")
    # Spot check numeric presence
    row = out.reset_index().iloc[0]
    assert pd.api.types.is_number(row["time_q1_q99_mean"]) or pd.isna(row["time_q1_q99_mean"])


def test_compute_view_file_dask_quantiles(dummy_analyzer_quantiles: DummyAnalyzer):
    pdf = build_base_df()
    traces = dd.from_pandas(pdf, npartitions=2)
    view_types = ["file_name", "proc_name"]
    hlm = dummy_analyzer_quantiles._compute_high_level_metrics(
        traces=traces, view_types=view_types, partition_size="64MB"
    )
    main_view = dummy_analyzer_quantiles._compute_main_view(
        layer="posix", hlm=hlm, view_types=view_types, partition_size="64MB"
    )

    view_dd = dummy_analyzer_quantiles._compute_view(
        layer="posix",
        records=main_view,
        view_key=("file_name",),
        view_type="file_name",
        view_types=view_types,
    )
    out = view_dd.compute()
    assert list(out.index.names) == ["file_name"]
    _assert_no_infinities(out)
    _assert_quantile_columns_present(out, "time")
