import numpy as np
import pandas as pd
import pytest

from .pandas_reference import (
    set_cross_layer_metrics,
    set_main_metrics,
    set_quantile_metrics,
    set_view_metrics,
)


# Ensure this module runs in both smoke and full CI modes
pytestmark = [pytest.mark.smoke, pytest.mark.full]


def _assert_no_infinities(df: pd.DataFrame):
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            coerced = pd.to_numeric(s, errors="coerce")
            assert not np.isinf(coerced.to_numpy()).any(), f"Infinite values found in column {col}"


def test_set_main_metrics_basic_single_metric():
    # Single set of size/time/count
    df = pd.DataFrame(
        {
            "time": [2.0, 4.0],
            "count": [10.0, 20.0],
            "size": [100.0, 80.0],
        }
    )
    out = set_main_metrics(df.copy())

    # bw = size / time; intensity = count / size; ops = count / time
    assert pytest.approx(out.loc[0, "bw"], rel=1e-6) == 50.0
    assert pytest.approx(out.loc[1, "bw"], rel=1e-6) == 20.0
    assert pytest.approx(out.loc[0, "intensity"], rel=1e-6) == 0.1
    assert pytest.approx(out.loc[1, "intensity"], rel=1e-6) == 0.25
    assert pytest.approx(out.loc[0, "ops"], rel=1e-6) == 5.0
    assert pytest.approx(out.loc[1, "ops"], rel=1e-6) == 5.0
    _assert_no_infinities(out)


def test_set_main_metrics_masks_and_infs_to_na_zero_size():
    # size <= 0 should be masked to NA for size, bw, intensity
    df = pd.DataFrame(
        {
            "time": [2.0, 3.0, 1.0],
            "count": [10.0, 5.0, 0.0],
            "size": [0.0, -5.0, 0.0],
        }
    )
    out = set_main_metrics(df.copy())

    # size masked to NA
    assert pd.isna(out.loc[0, "size"]) and pd.isna(out.loc[1, "size"]) and pd.isna(out.loc[2, "size"])
    # bw/intensity masked to NA
    assert pd.isna(out.loc[0, "bw"]) and pd.isna(out.loc[1, "bw"]) and pd.isna(out.loc[2, "bw"])
    assert pd.isna(out.loc[0, "intensity"]) and pd.isna(out.loc[1, "intensity"]) and pd.isna(out.loc[2, "intensity"])
    # ops still computed as count/time
    assert pytest.approx(out.loc[0, "ops"], rel=1e-6) == 5.0
    assert pytest.approx(out.loc[1, "ops"], rel=1e-6) == (5.0 / 3.0)
    assert pytest.approx(out.loc[2, "ops"], rel=1e-6) == 0.0
    _assert_no_infinities(out)


def test_set_main_metrics_infinite_handling_zero_time():
    # time == 0 with positive size/count should produce inf which must become NA
    df = pd.DataFrame(
        {
            "time": [0.0, 0.0],
            "count": [10.0, 0.0],
            "size": [100.0, 50.0],
        }
    )
    out = set_main_metrics(df.copy())

    # bw and ops would be inf; function should convert to NA
    assert pd.isna(out.loc[0, "bw"]) and pd.isna(out.loc[1, "bw"])  # 100/0, 50/0
    assert pd.isna(out.loc[0, "ops"]) and pd.isna(out.loc[1, "ops"])  # 10/0, 0/0
    # intensity = count/size remains finite
    assert pytest.approx(out.loc[0, "intensity"], rel=1e-6) == 0.1
    assert pytest.approx(out.loc[1, "intensity"], rel=1e-6) == 0.0
    _assert_no_infinities(out)


def test_set_main_metrics_multiple_prefixes_read_write():
    # Two metric families (read_*, write_*) are handled independently by endswith
    df = pd.DataFrame(
        {
            "read_time": [1.0, 2.0],
            "read_count": [10.0, 4.0],
            "read_size": [100.0, 20.0],
            "write_time": [2.0, 3.0],
            "write_count": [0.0, 6.0],
            "write_size": [0.0, 120.0],
        }
    )
    out = set_main_metrics(df.copy())

    # read_* derived
    assert pytest.approx(out.loc[0, "read_bw"], rel=1e-6) == 100.0
    assert pytest.approx(out.loc[0, "read_intensity"], rel=1e-6) == 0.1
    assert pytest.approx(out.loc[0, "read_ops"], rel=1e-6) == 10.0
    assert pytest.approx(out.loc[1, "read_bw"], rel=1e-6) == 10.0
    assert pytest.approx(out.loc[1, "read_intensity"], rel=1e-6) == 0.2
    assert pytest.approx(out.loc[1, "read_ops"], rel=1e-6) == 2.0

    # write_* derived: write_size is 0 => masked to NA for row 0; row 1 valid
    assert (
        pd.isna(out.loc[0, "write_size"])
        and pd.isna(out.loc[0, "write_bw"])
        and pd.isna(out.loc[0, "write_intensity"])
    )
    assert pytest.approx(out.loc[1, "write_bw"], rel=1e-6) == 40.0
    assert pytest.approx(out.loc[1, "write_intensity"], rel=1e-6) == (6.0 / 120.0)
    assert pytest.approx(out.loc[1, "write_ops"], rel=1e-6) == 2.0
    _assert_no_infinities(out)


def test_set_view_metrics_process_based_basic():
    # Uses time_sum when process-based
    df = pd.DataFrame(
        {
            "count_sum": [6.0, 2.0],
            "size_sum": [300.0, 100.0],
            "time_sum": [3.0, 1.0],
        }
    )
    metric_boundaries = {"time_sum": 10.0}

    out = set_view_metrics(df.copy(), metric_boundaries=metric_boundaries, is_view_process_based=True)

    # frac_total values
    assert pytest.approx(out.loc[0, "count_frac_total"], rel=1e-6) == 6.0 / 8.0
    assert pytest.approx(out.loc[1, "count_frac_total"], rel=1e-6) == 2.0 / 8.0
    assert pytest.approx(out.loc[0, "size_proc_frac_total"], rel=1e-6) == 300.0 / 400.0
    assert pytest.approx(out.loc[1, "size_proc_frac_total"], rel=1e-6) == 100.0 / 400.0
    # size_call_frac_total (same as proc when process-based, both from size_sum)
    assert pytest.approx(out.loc[0, "size_call_frac_total"], rel=1e-6) == 300.0 / 400.0
    assert pytest.approx(out.loc[1, "size_call_frac_total"], rel=1e-6) == 100.0 / 400.0
    assert pytest.approx(out.loc[0, "time_proc_frac_total"], rel=1e-6) == 3.0 / 4.0
    assert pytest.approx(out.loc[1, "time_proc_frac_total"], rel=1e-6) == 1.0 / 4.0
    # time_call_frac_total (same as proc when process-based, both from time_sum)
    assert pytest.approx(out.loc[0, "time_call_frac_total"], rel=1e-6) == 3.0 / 4.0
    assert pytest.approx(out.loc[1, "time_call_frac_total"], rel=1e-6) == 1.0 / 4.0
    # ops slope = time_proc_frac_total / count_frac_total
    assert pytest.approx(out.loc[0, "ops_slope"], rel=1e-6) == 1.0
    assert pytest.approx(out.loc[1, "ops_slope"], rel=1e-6) == 1.0
    # ops percentile ranks; equal slopes -> equal percentile
    assert pytest.approx(out.loc[0, "ops_percentile"], rel=1e-6) == pytest.approx(
        out.loc[1, "ops_percentile"], rel=1e-6
    )
    _assert_no_infinities(out)


def test_set_view_metrics_non_process_based_basic():
    # Uses time_proc_max / size_proc_max when not process-based
    df = pd.DataFrame(
        {
            "count_sum": [6.0, 2.0],
            "size_sum": [300.0, 100.0],
            "size_proc_max": [200.0, 80.0],
            "time_sum": [5.0, 3.0],
            "time_proc_max": [3.0, 1.0],
        }
    )
    metric_boundaries = {"time_proc_max": 10.0}

    out = set_view_metrics(df.copy(), metric_boundaries=metric_boundaries, is_view_process_based=False)

    assert pytest.approx(out.loc[0, "time_proc_frac_total"], rel=1e-6) == 3.0 / 4.0
    assert pytest.approx(out.loc[1, "time_proc_frac_total"], rel=1e-6) == 1.0 / 4.0
    # time_call_frac_total (from time_sum, not time_proc_max)
    assert pytest.approx(out.loc[0, "time_call_frac_total"], rel=1e-6) == 5.0 / 8.0
    assert pytest.approx(out.loc[1, "time_call_frac_total"], rel=1e-6) == 3.0 / 8.0
    # count fractions based on sums
    assert pytest.approx(out.loc[0, "count_frac_total"], rel=1e-6) == 6.0 / 8.0
    assert pytest.approx(out.loc[1, "count_frac_total"], rel=1e-6) == 2.0 / 8.0
    # size_proc_frac_total (from size_proc_max, not size_sum)
    assert pytest.approx(out.loc[0, "size_proc_frac_total"], rel=1e-6) == 200.0 / 280.0
    assert pytest.approx(out.loc[1, "size_proc_frac_total"], rel=1e-6) == 80.0 / 280.0
    # size_call_frac_total (from size_sum)
    assert pytest.approx(out.loc[0, "size_call_frac_total"], rel=1e-6) == 300.0 / 400.0
    assert pytest.approx(out.loc[1, "size_call_frac_total"], rel=1e-6) == 100.0 / 400.0
    # ops slope defined and finite
    assert pytest.approx(out.loc[0, "ops_slope"], rel=1e-6) == (3 / 4) / (6 / 8)
    assert pytest.approx(out.loc[1, "ops_slope"], rel=1e-6) == (1 / 4) / (2 / 8)
    _assert_no_infinities(out)


def test_set_view_metrics_infinite_handling_zero_count():
    # count_sum == 0 leads to infinite slope (time_per / 0) -> must be NA
    df = pd.DataFrame(
        {
            "count_sum": [0.0, 2.0],
            "size_sum": [100.0, 100.0],
            "time_sum": [1.0, 1.0],
        }
    )
    metric_boundaries = {"time_sum": 10.0}

    out = set_view_metrics(df.copy(), metric_boundaries=metric_boundaries, is_view_process_based=True)

    # If slope is NA, percentile should also be NA
    assert pd.isna(out.loc[0, "ops_slope"]) and pd.isna(out.loc[0, "ops_percentile"])
    assert not pd.isna(out.loc[1, "ops_slope"]) and not pd.isna(out.loc[1, "ops_percentile"])  # finite
    _assert_no_infinities(out)


def test_set_view_metrics_multiple_prefixes():
    # Two prefixes (read_, write_) should generate their own frac and ops metrics
    df = pd.DataFrame(
        {
            "read_count_sum": [4.0, 6.0],
            "read_size_sum": [40.0, 60.0],
            "read_time_sum": [1.0, 3.0],
            "write_count_sum": [2.0, 8.0],
            "write_size_sum": [20.0, 80.0],
            "write_time_sum": [2.0, 2.0],
        }
    )
    metric_boundaries = {"read_time_sum": 10.0, "write_time_sum": 10.0}

    out = set_view_metrics(df.copy(), metric_boundaries=metric_boundaries, is_view_process_based=True)

    # read_* fractions
    assert pytest.approx(out.loc[0, "read_count_frac_total"], rel=1e-6) == 4.0 / 10.0
    assert pytest.approx(out.loc[0, "read_time_proc_frac_total"], rel=1e-6) == 1.0 / 4.0
    # write_* fractions
    assert pytest.approx(out.loc[1, "write_count_frac_total"], rel=1e-6) == 8.0 / 10.0
    assert pytest.approx(out.loc[1, "write_time_proc_frac_total"], rel=1e-6) == 2.0 / 4.0
    # size_call_frac_total (same as proc when process-based)
    assert pytest.approx(out.loc[0, "read_size_call_frac_total"], rel=1e-6) == 40.0 / 100.0
    assert pytest.approx(out.loc[1, "write_size_call_frac_total"], rel=1e-6) == 80.0 / 100.0
    # time_call_frac_total (same as proc when process-based)
    assert pytest.approx(out.loc[0, "read_time_call_frac_total"], rel=1e-6) == 1.0 / 4.0
    assert pytest.approx(out.loc[1, "write_time_call_frac_total"], rel=1e-6) == 2.0 / 4.0
    # ops_slope & ops_percentile exist per family
    assert "read_ops_slope" in out.columns and "write_ops_slope" in out.columns
    assert "read_ops_percentile" in out.columns and "write_ops_percentile" in out.columns
    _assert_no_infinities(out)


def test_set_view_metrics_zero_size_sum_denominator_safe():
    # Sum of size_sum is zero -> size_proc_frac_total and size_call_frac_total should be NA
    df = pd.DataFrame(
        {
            "count_sum": [1.0, 2.0],
            "size_sum": [0.0, 0.0],
            "time_sum": [1.0, 1.0],
        }
    )
    metric_boundaries = {"time_sum": 10.0}
    out = set_view_metrics(df.copy(), metric_boundaries=metric_boundaries, is_view_process_based=True)
    assert pd.isna(out.loc[0, "size_proc_frac_total"]) and pd.isna(out.loc[1, "size_proc_frac_total"])
    assert pd.isna(out.loc[0, "size_call_frac_total"]) and pd.isna(out.loc[1, "size_call_frac_total"])
    _assert_no_infinities(out)


def test_set_view_metrics_zero_count_sum_denominator_safe():
    # Sum of count_sum is zero -> count_frac_total NA; slope & percentile NA
    df = pd.DataFrame(
        {
            "count_sum": [0.0, 0.0],
            "size_sum": [10.0, 20.0],
            "time_sum": [1.0, 3.0],
        }
    )
    metric_boundaries = {"time_sum": 10.0}
    out = set_view_metrics(df.copy(), metric_boundaries=metric_boundaries, is_view_process_based=True)
    assert pd.isna(out.loc[0, "count_frac_total"]) and pd.isna(out.loc[1, "count_frac_total"])
    assert pd.isna(out.loc[0, "ops_slope"]) and pd.isna(out.loc[0, "ops_percentile"])
    assert pd.isna(out.loc[1, "ops_slope"]) and pd.isna(out.loc[1, "ops_percentile"])
    _assert_no_infinities(out)


def test_set_view_metrics_zero_time_denominator_safe_process_and_non():
    # All time metrics zero -> time_proc_frac_total should be NA
    df_proc = pd.DataFrame(
        {
            "count_sum": [1.0, 1.0],
            "size_sum": [10.0, 20.0],
            "time_sum": [0.0, 0.0],
        }
    )
    out_proc = set_view_metrics(df_proc.copy(), metric_boundaries={}, is_view_process_based=True)
    assert pd.isna(out_proc.loc[0, "time_proc_frac_total"]) and pd.isna(out_proc.loc[1, "time_proc_frac_total"])
    assert pd.isna(out_proc.loc[0, "time_call_frac_total"]) and pd.isna(out_proc.loc[1, "time_call_frac_total"])

    df_non = pd.DataFrame(
        {
            "count_sum": [1.0, 1.0],
            "size_sum": [0.0, 0.0],
            "size_proc_max": [0.0, 0.0],
            "time_sum": [0.0, 0.0],
            "time_proc_max": [0.0, 0.0],
        }
    )
    out_non = set_view_metrics(df_non.copy(), metric_boundaries={}, is_view_process_based=False)
    assert pd.isna(out_non.loc[0, "time_proc_frac_total"]) and pd.isna(out_non.loc[1, "time_proc_frac_total"])
    assert pd.isna(out_non.loc[0, "time_call_frac_total"]) and pd.isna(out_non.loc[1, "time_call_frac_total"])
    assert pd.isna(out_non.loc[0, "size_proc_frac_total"]) and pd.isna(out_non.loc[1, "size_proc_frac_total"])
    assert pd.isna(out_non.loc[0, "size_call_frac_total"]) and pd.isna(out_non.loc[1, "size_call_frac_total"])
    _assert_no_infinities(out_proc)
    _assert_no_infinities(out_non)


def test_set_cross_layer_metrics_process_based_basic():
    # Layers: A (root), B and C children of A; B is async
    layer_deps = {"A": None, "B": "A", "C": "A"}
    async_layers = ["B"]
    # Row0: overhead present; Row1: no overhead; Row2: all zeros to exercise NA paths
    df = pd.DataFrame(
        {
            "A_time_sum": [10.0, 8.0, 0.0],
            "B_time_sum": [3.0, 8.0, 0.0],
            "C_time_sum": [2.0, 0.0, 0.0],
            "compute_time_sum": [4.0, 1.0, 0.0],
        }
    )

    out = set_cross_layer_metrics(
        df.copy(),
        layers=["A", "B", "C"],
        layer_deps=layer_deps,
        async_layers=async_layers,
        derived_metrics={},
        is_view_process_based=True,
        time_boundary_layer="A",
    )

    # Root boundary fractions present and correct (A_time / A_time)
    assert pytest.approx(out.loc[0, "A_time_proc_frac_A"], rel=1e-6) == 1.0
    assert pytest.approx(out.loc[1, "A_time_proc_frac_A"], rel=1e-6) == 1.0
    assert pd.isna(out.loc[2, "A_time_proc_frac_A"])  # 0/0 -> NA

    # Child boundary fractions: layer_time / A_time
    assert pytest.approx(out.loc[0, "B_time_proc_frac_A"], rel=1e-6) == 3.0 / 10.0
    assert pytest.approx(out.loc[1, "B_time_proc_frac_A"], rel=1e-6) == 8.0 / 8.0
    assert pytest.approx(out.loc[0, "C_time_proc_frac_A"], rel=1e-6) == 2.0 / 10.0

    # Overhead for A: max(A - B - C, 0)
    assert pytest.approx(out.loc[0, "o_A_time_sum"], rel=1e-6) == 5.0
    assert pytest.approx(out.loc[1, "o_A_time_sum"], rel=1e-6) == 0.0
    assert pytest.approx(out.loc[0, "o_A_time_proc_frac_self"], rel=1e-6) == 5.0 / 10.0
    assert pytest.approx(out.loc[0, "o_A_time_proc_frac_A"], rel=1e-6) == 5.0 / 10.0
    # Total overhead fraction across rows: 5 / (5 + 0 + 0)
    assert pytest.approx(out.loc[0, "o_A_time_proc_frac_total"], rel=1e-6) == 1.0
    assert pytest.approx(out.loc[1, "o_A_time_proc_frac_total"], rel=1e-6) == 0.0

    # Child to parent fractions
    assert pytest.approx(out.loc[0, "B_time_proc_frac_parent"], rel=1e-6) == 3.0 / 10.0
    assert pytest.approx(out.loc[1, "B_time_proc_frac_parent"], rel=1e-6) == 8.0 / 8.0

    # Unoverlapped for async layer B: max(B - compute, 0)
    assert pytest.approx(out.loc[0, "u_B_time_sum"], rel=1e-6) == 0.0
    assert pytest.approx(out.loc[1, "u_B_time_sum"], rel=1e-6) == 7.0
    # Fractions
    assert pytest.approx(out.loc[1, "u_B_time_proc_frac_self"], rel=1e-6) == 7.0 / 8.0
    assert pytest.approx(out.loc[1, "u_B_time_proc_frac_A"], rel=1e-6) == 7.0 / 8.0
    assert pytest.approx(out.loc[1, "u_B_time_proc_frac_total"], rel=1e-6) == 1.0
    assert pytest.approx(out.loc[1, "u_B_time_proc_frac_parent"], rel=1e-6) == 7.0 / 8.0

    # Call variants: for process-based views, call == proc (both use time_sum)
    assert pytest.approx(out.loc[0, "A_time_call_frac_A"], rel=1e-6) == 1.0
    assert pytest.approx(out.loc[0, "B_time_call_frac_A"], rel=1e-6) == 3.0 / 10.0
    assert pytest.approx(out.loc[0, "o_A_time_call_frac_self"], rel=1e-6) == 5.0 / 10.0
    assert pytest.approx(out.loc[0, "o_A_time_call_frac_A"], rel=1e-6) == 5.0 / 10.0
    assert pytest.approx(out.loc[0, "o_A_time_call_frac_total"], rel=1e-6) == 1.0
    assert pytest.approx(out.loc[0, "B_time_call_frac_parent"], rel=1e-6) == 3.0 / 10.0
    assert pytest.approx(out.loc[1, "u_B_time_call_frac_self"], rel=1e-6) == 7.0 / 8.0
    assert pytest.approx(out.loc[1, "u_B_time_call_frac_parent"], rel=1e-6) == 7.0 / 8.0

    # NA cleanup (no infinities)
    _assert_no_infinities(out)


def test_set_cross_layer_metrics_non_process_based_basic():
    layer_deps = {"A": None, "B": "A"}
    async_layers = ["B"]
    df = pd.DataFrame(
        {
            "A_time_proc_max": [5.0, 0.0],
            "B_time_proc_max": [2.0, 0.0],
            "compute_time_proc_max": [1.0, 0.0],
            # Call track always uses time_sum (aggregate across processes)
            "A_time_sum": [12.0, 0.0],
            "B_time_sum": [7.0, 0.0],
            "compute_time_sum": [3.0, 0.0],
        }
    )

    out = set_cross_layer_metrics(
        df.copy(),
        layers=["A", "B"],
        layer_deps=layer_deps,
        async_layers=async_layers,
        derived_metrics={},
        is_view_process_based=False,
        time_boundary_layer="A",
    )

    # Proc: root boundary fraction
    assert pytest.approx(out.loc[0, "A_time_proc_frac_A"], rel=1e-6) == 1.0
    assert pd.isna(out.loc[1, "A_time_proc_frac_A"])  # 0/0 -> NA

    # Proc: unoverlapped for B with time_proc_max
    assert pytest.approx(out.loc[0, "u_B_time_proc_max"], rel=1e-6) == 1.0
    assert pytest.approx(out.loc[0, "u_B_time_proc_frac_self"], rel=1e-6) == 1.0 / 2.0
    assert pytest.approx(out.loc[0, "u_B_time_proc_frac_A"], rel=1e-6) == 1.0 / 5.0

    # Proc: overhead for A: max(5 - 2, 0) = 3
    assert pytest.approx(out.loc[0, "o_A_time_proc_max"], rel=1e-6) == 3.0
    assert pytest.approx(out.loc[0, "o_A_time_proc_frac_A"], rel=1e-6) == 3.0 / 5.0

    # Proc: child to parent: B/A = 2/5
    assert pytest.approx(out.loc[0, "B_time_proc_frac_parent"], rel=1e-6) == 2.0 / 5.0

    # Call: boundary fractions (uses time_sum)
    assert pytest.approx(out.loc[0, "A_time_call_frac_A"], rel=1e-6) == 1.0
    assert pd.isna(out.loc[1, "A_time_call_frac_A"])  # 0/0 -> NA
    assert pytest.approx(out.loc[0, "B_time_call_frac_A"], rel=1e-6) == 7.0 / 12.0

    # Call: overhead for A: max(12 - 7, 0) = 5
    assert pytest.approx(out.loc[0, "o_A_time_sum"], rel=1e-6) == 5.0
    assert pytest.approx(out.loc[0, "o_A_time_call_frac_A"], rel=1e-6) == 5.0 / 12.0
    assert pytest.approx(out.loc[0, "o_A_time_call_frac_self"], rel=1e-6) == 5.0 / 12.0

    # Call: child to parent: B/A = 7/12
    assert pytest.approx(out.loc[0, "B_time_call_frac_parent"], rel=1e-6) == 7.0 / 12.0

    # Call: unoverlapped for B: max(7 - 3, 0) = 4
    assert pytest.approx(out.loc[0, "u_B_time_sum"], rel=1e-6) == 4.0
    assert pytest.approx(out.loc[0, "u_B_time_call_frac_self"], rel=1e-6) == 4.0 / 7.0
    assert pytest.approx(out.loc[0, "u_B_time_call_frac_A"], rel=1e-6) == 4.0 / 12.0
    assert pytest.approx(out.loc[0, "u_B_time_call_frac_parent"], rel=1e-6) == 4.0 / 12.0

    # Verify proc and call differ (non-process-based view)
    assert out.loc[0, "B_time_proc_frac_A"] != out.loc[0, "B_time_call_frac_A"]
    assert out.loc[0, "o_A_time_proc_frac_A"] != out.loc[0, "o_A_time_call_frac_A"]

    _assert_no_infinities(out)


def test_set_cross_layer_metrics_zero_denominators_produce_na():
    # Boundary time zero triggers NA in fractions; parent with zero time as well
    layer_deps = {"A": None, "B": "A"}
    async_layers = ["B"]
    df = pd.DataFrame(
        {
            "A_time_sum": [0.0],
            "B_time_sum": [1.0],
            "compute_time_sum": [2.0],
        }
    )

    out = set_cross_layer_metrics(
        df.copy(),
        layers=["A", "B"],
        layer_deps=layer_deps,
        async_layers=async_layers,
        derived_metrics={},
        is_view_process_based=True,
        time_boundary_layer="A",
    )

    # Divisions by zero -> NA
    assert pd.isna(out.loc[0, "A_time_proc_frac_A"])  # 0/0
    assert pd.isna(out.loc[0, "B_time_proc_frac_A"])  # 1/0 -> NA after cleanup
    assert pd.isna(out.loc[0, "B_time_proc_frac_parent"])  # 1/0
    # Unoverlapped 0 vs compute
    assert pytest.approx(out.loc[0, "u_B_time_sum"], rel=1e-6) == 0.0
    assert pd.isna(out.loc[0, "u_B_time_proc_frac_A"])  # 0/0

    # Call variants: same zero-denominator behavior
    assert pd.isna(out.loc[0, "A_time_call_frac_A"])  # 0/0
    assert pd.isna(out.loc[0, "B_time_call_frac_A"])  # 1/0 -> NA
    assert pd.isna(out.loc[0, "B_time_call_frac_parent"])  # 1/0
    assert pd.isna(out.loc[0, "u_B_time_call_frac_A"])  # 0/0
    _assert_no_infinities(out)


def test_set_cross_layer_metrics_with_derived_metrics():
    # Derived metric under root layer A
    layer_deps = {"A": None, "B": "A"}
    async_layers = []
    derived_metrics = {"A": {"foo": ""}}

    df = pd.DataFrame(
        {
            "A_time_sum": [10.0, 10.0],
            "B_time_sum": [3.0, 7.0],
            "A_foo_time_sum": [4.0, 6.0],
        }
    )

    out = set_cross_layer_metrics(
        df.copy(),
        layers=["A", "B"],
        layer_deps=layer_deps,
        async_layers=async_layers,
        derived_metrics=derived_metrics,
        is_view_process_based=True,
        time_boundary_layer="A",
    )

    # Derived metric fractions (proc)
    assert pytest.approx(out.loc[0, "A_foo_time_proc_frac_A"], rel=1e-6) == 4.0 / 10.0
    assert pytest.approx(out.loc[1, "A_foo_time_proc_frac_A"], rel=1e-6) == 6.0 / 10.0
    assert pytest.approx(out.loc[0, "A_foo_time_proc_frac_parent"], rel=1e-6) == 4.0 / 10.0
    assert pytest.approx(out.loc[1, "A_foo_time_proc_frac_parent"], rel=1e-6) == 6.0 / 10.0
    assert pytest.approx(out.loc[0, "A_foo_time_proc_frac_total"], rel=1e-6) == 0.4
    assert pytest.approx(out.loc[1, "A_foo_time_proc_frac_total"], rel=1e-6) == 0.6

    # Derived metric fractions (call): same as proc for process-based views
    assert pytest.approx(out.loc[0, "A_foo_time_call_frac_A"], rel=1e-6) == 4.0 / 10.0
    assert pytest.approx(out.loc[1, "A_foo_time_call_frac_A"], rel=1e-6) == 6.0 / 10.0
    assert pytest.approx(out.loc[0, "A_foo_time_call_frac_parent"], rel=1e-6) == 4.0 / 10.0
    assert pytest.approx(out.loc[0, "A_foo_time_call_frac_total"], rel=1e-6) == 0.4
    _assert_no_infinities(out)


def test_set_cross_layer_metrics_derived_metrics_zero_denoms():
    layer_deps = {"A": None}
    async_layers = []
    derived_metrics = {"A": {"foo": ""}}

    # Boundary zero and derived zero across rows -> NA where appropriate
    df = pd.DataFrame(
        {
            "A_time_sum": [0.0, 0.0],
            "A_foo_time_sum": [0.0, 0.0],
        }
    )

    out = set_cross_layer_metrics(
        df.copy(),
        layers=["A"],
        layer_deps=layer_deps,
        async_layers=async_layers,
        derived_metrics=derived_metrics,
        is_view_process_based=True,
        time_boundary_layer="A",
    )

    assert pd.isna(out.loc[0, "A_foo_time_proc_frac_A"])  # 0/0
    assert pd.isna(out.loc[0, "A_foo_time_proc_frac_parent"])  # 0/0
    # total fraction is 0/0 as well -> NA for both rows
    assert pd.isna(out.loc[0, "A_foo_time_proc_frac_total"]) and pd.isna(out.loc[1, "A_foo_time_proc_frac_total"])

    # Call variants: same zero-denominator behavior
    assert pd.isna(out.loc[0, "A_foo_time_call_frac_A"])  # 0/0
    assert pd.isna(out.loc[0, "A_foo_time_call_frac_parent"])  # 0/0
    assert pd.isna(out.loc[0, "A_foo_time_call_frac_total"]) and pd.isna(out.loc[1, "A_foo_time_call_frac_total"])
    _assert_no_infinities(out)


# ---------- set_quantile_metrics tests ----------


def test_set_quantile_metrics_basic_extraction():
    df = pd.DataFrame(
        {
            "time_q1_q99_stats": [[1.5, 0.5, 10], [3.0, 1.0, 20]],
            "size_q5_q95_stats": [[100.0, 10.0, 8], [200.0, 20.0, 12]],
        }
    )

    out = set_quantile_metrics(df.copy())

    # Original columns dropped
    assert "time_q1_q99_stats" not in out.columns
    assert "size_q5_q95_stats" not in out.columns

    # New columns created with correct values and dtypes
    assert pytest.approx(out.loc[0, "time_q1_q99_mean"], rel=1e-6) == 1.5
    assert pytest.approx(out.loc[1, "time_q1_q99_std"], rel=1e-6) == 1.0
    assert out["time_q1_q99_count"].dtype.name in ("Int64", "Int32")
    assert out.loc[0, "time_q1_q99_count"] == 10

    assert pytest.approx(out.loc[0, "size_q5_q95_mean"], rel=1e-6) == 100.0
    assert pytest.approx(out.loc[1, "size_q5_q95_std"], rel=1e-6) == 20.0
    assert out.loc[1, "size_q5_q95_count"] == 12


def test_set_quantile_metrics_handles_nan_triplets():
    df = pd.DataFrame(
        {
            "time_q10_q90_stats": [[np.nan, np.nan, np.nan], [1.0, 0.0, 0]],
        }
    )

    out = set_quantile_metrics(df.copy())

    # First row NA triplet -> all NA
    assert pd.isna(out.loc[0, "time_q10_q90_mean"]) and pd.isna(out.loc[0, "time_q10_q90_std"]) and pd.isna(
        out.loc[0, "time_q10_q90_count"]
    )
    # Second row has values
    assert pytest.approx(out.loc[1, "time_q10_q90_mean"], rel=1e-6) == 1.0
    assert pytest.approx(out.loc[1, "time_q10_q90_std"], rel=1e-6) == 0.0
    assert out.loc[1, "time_q10_q90_count"] == 0


def test_set_quantile_metrics_mixed_list_tuple_and_noop():
    df = pd.DataFrame(
        {
            "time_q25_q75_stats": [(1.0, 0.2, 5), [2.0, 0.3, 6]],  # tuple and list
            "plain_col": [42, 43],
        }
    )

    out = set_quantile_metrics(df.copy())
    # plain_col preserved
    assert list(out["plain_col"]) == [42, 43]
    # New columns present
    assert "time_q25_q75_mean" in out.columns and "time_q25_q75_std" in out.columns and "time_q25_q75_count" in out.columns
    # Stats dropped
    assert "time_q25_q75_stats" not in out.columns
