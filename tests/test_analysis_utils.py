import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd

from dfanalyzer.analysis_utils import split_duration_records_vectorized
from dfanalyzer.constants import (
    COL_TIME,
    COL_TIME_START,
    COL_TIME_RANGE,
    COL_COUNT,
    COL_SIZE,
)

# Ensure this module runs in both smoke and full CI modes
pytestmark = [pytest.mark.smoke, pytest.mark.full]


def build_df(time_values, time_starts, counts=None, sizes=None):
    if counts is None:
        counts = [1] * len(time_values)
    if sizes is None:
        sizes = [100] * len(time_values)
    return pd.DataFrame(
        {
            COL_TIME: time_values,
            COL_TIME_START: time_starts,
            COL_COUNT: counts,
            COL_SIZE: sizes,
        }
    )


@pytest.mark.parametrize(
    "time_granularity,time_resolution",
    [
        (1.0, 1.0),
        (0.5, 1.0),
        (1.0, 1_000_000.0),
    ],
)
def test_exact_multiple_splitting(time_granularity, time_resolution):
    df = build_df([2.0], [0], [6], [300])

    out = split_duration_records_vectorized(df.copy(), time_granularity, time_resolution)

    n_chunks = int(np.ceil(2.0 / time_granularity))
    assert len(out) == n_chunks
    # All chunks should be exactly granularity long
    assert np.allclose(out[COL_TIME].to_numpy(), np.full(n_chunks, time_granularity))
    # Starts should increase by granularity * resolution
    expected_starts = 0 + np.arange(n_chunks) * time_granularity * time_resolution
    assert np.allclose(out[COL_TIME_START].to_numpy(), expected_starts)
    # Buckets should be floor(start / (gran*res))
    expected_ranges = (expected_starts // (time_granularity * time_resolution)).astype("int64")
    assert np.array_equal(out[COL_TIME_RANGE].astype("int64").to_numpy(), expected_ranges)
    # Even split counts/sizes
    assert np.allclose(out[COL_COUNT].sum(), 6)
    assert np.allclose(out[COL_SIZE].sum(), 300)


def test_non_multiple_splitting_with_remainder():
    # 2.5 split by 1.0 → [1.0, 1.0, 0.5]
    df = build_df([2.5], [10.0], [9], [90])
    out = split_duration_records_vectorized(df.copy(), 1.0, 1.0)

    assert len(out) == 3
    assert np.allclose(out[COL_TIME].to_numpy(), [1.0, 1.0, 0.5])
    assert np.allclose(out[COL_TIME_START].to_numpy(), [10.0, 11.0, 12.0])
    assert np.array_equal(out[COL_TIME_RANGE].astype("int64").to_numpy(), [10, 11, 12])
    assert np.isclose(out[COL_COUNT].sum(), 9)
    assert np.isclose(out[COL_SIZE].sum(), 90)


def test_zero_duration_rows_short_circuit():
    df = build_df([0.0, 0.0], [0.0, 3.0], [2, 4], [20, 40])
    out = split_duration_records_vectorized(df.copy(), 1.0, 1.0)

    # No expansion
    assert len(out) == 2
    # time_range computed from start
    assert np.array_equal(out[COL_TIME_RANGE].astype("int64").to_numpy(), [0, 3])
    # Original values kept
    assert np.array_equal(out[COL_TIME].to_numpy(), [0.0, 0.0])
    assert np.array_equal(out[COL_COUNT].to_numpy(), [2, 4])
    assert np.array_equal(out[COL_SIZE].to_numpy(), [20, 40])


def test_mixed_rows_vectorization():
    # Mix of exact, remainder, and zero
    df = build_df(
        [0.0, 2.0, 2.5],
        [0.0, 100.0, 200.0],
        [3, 6, 9],
        [30, 60, 90],
    )

    out = split_duration_records_vectorized(df.copy(), 1.0, 1.0)

    # Expected lengths: 0→0-chunks (shortcut makes whole df returned, but in vector case
    # we handle using the implemented short-circuit only when all max_chunks == 0).
    # Here max_chunks > 0, so 0-duration row will be repeated 0 times (no rows for it).
    # 2.0 → 2, 2.5 → 3, so total 5
    assert len(out) == 5
    # First 2 rows correspond to 2.0 duration
    assert np.allclose(out.iloc[0:2][COL_TIME].to_numpy(), [1.0, 1.0])
    assert np.allclose(out.iloc[0:2][COL_TIME_START].to_numpy(), [100.0, 101.0])
    # Next 3 rows correspond to 2.5 duration
    assert np.allclose(out.iloc[2:][COL_TIME].to_numpy(), [1.0, 1.0, 0.5])
    assert np.allclose(out.iloc[2:][COL_TIME_START].to_numpy(), [200.0, 201.0, 202.0])
    # Totals preserved for non-zero rows only (zero-row disappears in expansion)
    assert np.isclose(out[COL_COUNT].sum(), 6 + 9)
    assert np.isclose(out[COL_SIZE].sum(), 60 + 90)


def test_large_granularity_larger_than_duration():
    # If granularity > duration, still one chunk with the remainder (=duration)
    df = build_df([0.4], [5.0], [5], [50])
    out = split_duration_records_vectorized(df.copy(), 1.0, 1.0)

    assert len(out) == 1
    assert np.allclose(out[COL_TIME].to_numpy(), [0.4])
    assert np.allclose(out[COL_TIME_START].to_numpy(), [5.0])
    assert np.array_equal(out[COL_TIME_RANGE].astype("int64").to_numpy(), [5])
    assert np.isclose(out[COL_COUNT].sum(), 5)
    assert np.isclose(out[COL_SIZE].sum(), 50)


def test_time_resolution_scaling():
    # Verify that time_resolution scales start and bucket boundaries
    df = build_df([2.0], [10_000_000.0], [4], [40])  # start at 10s when res=1e6
    out = split_duration_records_vectorized(df.copy(), 1.0, 1_000_000.0)

    assert len(out) == 2
    assert np.allclose(out[COL_TIME_START].to_numpy(), [10_000_000.0, 11_000_000.0])
    assert np.array_equal(out[COL_TIME_RANGE].astype("int64").to_numpy(), [10, 11])


def test_dtype_of_time_range_is_integer_extension():
    df = build_df([2.0], [0.0])
    out = split_duration_records_vectorized(df.copy(), 1.0, 1.0)
    # Should be concrete int64 dtype (non-nullable)
    assert out[COL_TIME_RANGE].dtype == np.dtype("int64")


def test_counts_and_sizes_even_split_per_row():
    df = build_df([3.0], [0.0], [9], [120])  # 3 chunks at gran=1.0
    out = split_duration_records_vectorized(df.copy(), 1.0, 1.0)

    assert len(out) == 3
    # Each chunk gets 1/3
    assert np.allclose(out[COL_COUNT].to_numpy(), [3.0, 3.0, 3.0])
    assert np.allclose(out[COL_SIZE].to_numpy(), [40.0, 40.0, 40.0])


def test_multiple_rows_preserve_per_row_splitting_and_totals():
    df = build_df([1.5, 2.0], [0.0, 10.0], [3, 4], [30, 40])
    out = split_duration_records_vectorized(df.copy(), 1.0, 1.0)

    # 1.5 -> 2 chunks; 2.0 -> 2 chunks
    assert len(out) == 4
    # Totals preserved
    assert np.isclose(out.groupby((np.arange(4) < 2))[COL_COUNT].sum().sum(), 7)
    assert np.isclose(out.groupby((np.arange(4) < 2))[COL_SIZE].sum().sum(), 70)


def test_dftracer_microsecond_events_single_chunk():
    # Mimic dftracer: ts in microseconds, dur in microseconds but stored as seconds in COL_TIME
    time_resolution = 1_000_000.0
    time_granularity = 1.0  # 1 second buckets
    ts_values = [1744611950820800.0, 1744611950820822.0, 1744611950820833.0, 1744611950820854.0]
    dur_us = [7, 2, 2, 2]
    durations_sec = [d / time_resolution for d in dur_us]

    df = build_df(durations_sec, ts_values, [1, 2, 3, 4], [10, 20, 30, 40])
    out = split_duration_records_vectorized(df.copy(), time_granularity, time_resolution)

    # Each is < 1 second → single chunk with exact duration
    assert len(out) == 4
    assert np.allclose(out[COL_TIME].to_numpy(), durations_sec)
    # Starts unchanged
    assert np.allclose(out[COL_TIME_START].to_numpy(), ts_values)
    # Buckets are floor(ts / 1e6)
    expected_ranges = (np.array(ts_values) // (time_granularity * time_resolution)).astype("int64")
    assert np.array_equal(out[COL_TIME_RANGE].astype("int64").to_numpy(), expected_ranges)
    # Totals preserved
    assert np.isclose(out[COL_COUNT].sum(), sum([1, 2, 3, 4]))
    assert np.isclose(out[COL_SIZE].sum(), sum([10, 20, 30, 40]))


def test_dftracer_cross_second_boundary_no_split_when_duration_lt_granularity():
    # Start just before the second boundary, small duration crossing into next second
    time_resolution = 1_000_000.0
    df = build_df([10 / time_resolution], [999_999.0], [1], [10])
    out = split_duration_records_vectorized(df.copy(), 1.0, time_resolution)

    assert len(out) == 1
    assert np.allclose(out[COL_TIME].to_numpy(), [10 / time_resolution])
    # Bucket remains the starting second (0)
    assert np.array_equal(out[COL_TIME_RANGE].astype("int64").to_numpy(), [0])


def test_dftracer_large_duration_multiple_seconds_with_us_remainder():
    time_resolution = 1_000_000.0
    time_granularity = 1.0
    ts = 1744611950820800.0
    duration_sec = 2.000007  # 2 seconds + 7 microseconds
    df = build_df([duration_sec], [ts], [6], [300])

    out = split_duration_records_vectorized(df.copy(), time_granularity, time_resolution)

    assert len(out) == 3
    assert np.allclose(out[COL_TIME].to_numpy(), [1.0, 1.0, 7 / time_resolution])
    # Starts at ts, then +1e6, +2e6
    assert np.allclose(out[COL_TIME_START].to_numpy(), [ts, ts + 1_000_000.0, ts + 2_000_000.0])
    assert np.array_equal(
        out[COL_TIME_RANGE].astype("int64").to_numpy(),
        [int(ts // 1_000_000.0) + 0, int(ts // 1_000_000.0) + 1, int(ts // 1_000_000.0) + 2],
    )
    # Totals preserved
    assert np.isclose(out[COL_COUNT].sum(), 6)
    assert np.isclose(out[COL_SIZE].sum(), 300)


@pytest.mark.parametrize("npartitions", [1, 2])
def test_dask_map_partitions_basic(npartitions: int):
    # Basic case without zeros: should be partition-invariant
    base_df = build_df([2.0, 2.5], [0.0, 10.0], [6, 9], [60, 90])
    expected = split_duration_records_vectorized(base_df.copy(), 1.0, 1.0)

    ddf = dd.from_pandas(base_df, npartitions=npartitions)
    out_ddf = ddf.map_partitions(
        split_duration_records_vectorized,
        time_granularity=1.0,
        time_resolution=1.0,
    )
    out = out_ddf.compute()

    # Sort for stable comparison
    cols = [COL_TIME_START, COL_TIME]
    out_sorted = out.sort_values(cols).reset_index(drop=True)
    expected_sorted = expected.sort_values(cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(out_sorted, expected_sorted, check_like=True)


@pytest.mark.parametrize("npartitions", [1, 2])
def test_dask_map_partitions_with_zeros_no_zero_only_partition(npartitions: int):
    # Mix zeros but ensure each partition contains at least one non-zero duration
    base_df = build_df([0.0, 2.0, 2.5, 0.0], [0.0, 100.0, 200.0, 300.0], [3, 6, 9, 12], [30, 60, 90, 120])
    expected = split_duration_records_vectorized(base_df.copy(), 1.0, 1.0)

    # For 2 partitions, we will manually repartition to avoid any zero-only partition
    if npartitions == 2:
        # Partition 1: rows 0,1 (contains zero and non-zero)
        # Partition 2: rows 2,3 (contains non-zero and zero)
        ddf_p1 = dd.from_pandas(base_df.iloc[[0, 1]], npartitions=1)
        ddf_p2 = dd.from_pandas(base_df.iloc[[2, 3]], npartitions=1)
        ddf = dd.concat([ddf_p1, ddf_p2])
    else:
        ddf = dd.from_pandas(base_df, npartitions=npartitions)

    out_ddf = ddf.map_partitions(
        split_duration_records_vectorized,
        time_granularity=1.0,
        time_resolution=1.0,
    )
    out = out_ddf.compute()

    cols = [COL_TIME_START, COL_TIME]
    out_sorted = out.sort_values(cols).reset_index(drop=True)
    expected_sorted = expected.sort_values(cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(out_sorted, expected_sorted, check_like=True)


@pytest.mark.parametrize("npartitions", [1, 2])
def test_dask_map_partitions_dftracer_microseconds(npartitions: int):
    # Microsecond timestamps with short durations
    time_resolution = 1_000_000.0
    base_df = build_df([2.0, 2.000007], [1744611950820800.0, 1744611951820800.0], [6, 9], [60, 90])
    expected = split_duration_records_vectorized(base_df.copy(), 1.0, time_resolution)

    ddf = dd.from_pandas(base_df, npartitions=npartitions)
    out_ddf = ddf.map_partitions(
        split_duration_records_vectorized,
        time_granularity=1.0,
        time_resolution=time_resolution,
    )
    out = out_ddf.compute()

    cols = [COL_TIME_START, COL_TIME]
    out_sorted = out.sort_values(cols).reset_index(drop=True)
    expected_sorted = expected.sort_values(cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(out_sorted, expected_sorted, check_like=True)
