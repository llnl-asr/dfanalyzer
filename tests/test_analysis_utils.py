import math

import numpy as np
import pytest
from dftracer.analyzer.analysis_utils import set_size_bins, split_duration_records_vectorized
from dftracer.analyzer.constants import (
    COL_COUNT,
    COL_SIZE,
    COL_TIME,
    COL_TIME_RANGE,
    COL_TIME_START,
    SIZE_BINS,
)
from dftracer.utils import DataFrame, Series

from . import pandas_reference

# Ensure this module runs in both smoke and full CI modes
pytestmark = [pytest.mark.smoke, pytest.mark.full]


def build_df(time_values, time_starts, counts=None, sizes=None):
    if counts is None:
        counts = [1] * len(time_values)
    if sizes is None:
        sizes = [100] * len(time_values)
    return DataFrame.from_dict(
        {
            COL_TIME: Series(time_values, dtype="float64"),
            COL_TIME_START: time_starts,
            COL_COUNT: counts,
            COL_SIZE: sizes,
        }
    )


def split(df, time_granularity, time_resolution):
    """The split of the library frame `df`, as pandas, checked against the pandas path."""
    out = split_duration_records_vectorized(df.copy(), time_granularity, time_resolution).to_pandas()
    expected = pandas_reference.split_duration_records_vectorized(df.to_pandas(), time_granularity, time_resolution)
    assert list(out.columns) == list(expected.columns)
    for col in expected.columns:
        np.testing.assert_allclose(
            out[col].to_numpy(dtype=float), expected[col].to_numpy(dtype=float), rtol=1e-12, atol=0
        )
    return out


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

    out = split(df, time_granularity, time_resolution)

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
    out = split(df, 1.0, 1.0)

    assert len(out) == 3
    assert np.allclose(out[COL_TIME].to_numpy(), [1.0, 1.0, 0.5])
    assert np.allclose(out[COL_TIME_START].to_numpy(), [10.0, 11.0, 12.0])
    assert np.array_equal(out[COL_TIME_RANGE].astype("int64").to_numpy(), [10, 11, 12])
    assert np.isclose(out[COL_COUNT].sum(), 9)
    assert np.isclose(out[COL_SIZE].sum(), 90)


def test_zero_duration_rows_short_circuit():
    df = build_df([0.0, 0.0], [0.0, 3.0], [2, 4], [20, 40])
    out = split(df, 1.0, 1.0)

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

    out = split(df, 1.0, 1.0)

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
    out = split(df, 1.0, 1.0)

    assert len(out) == 1
    assert np.allclose(out[COL_TIME].to_numpy(), [0.4])
    assert np.allclose(out[COL_TIME_START].to_numpy(), [5.0])
    assert np.array_equal(out[COL_TIME_RANGE].astype("int64").to_numpy(), [5])
    assert np.isclose(out[COL_COUNT].sum(), 5)
    assert np.isclose(out[COL_SIZE].sum(), 50)


def test_time_resolution_scaling():
    # Verify that time_resolution scales start and bucket boundaries
    df = build_df([2.0], [10_000_000.0], [4], [40])  # start at 10s when res=1e6
    out = split(df, 1.0, 1_000_000.0)

    assert len(out) == 2
    assert np.allclose(out[COL_TIME_START].to_numpy(), [10_000_000.0, 11_000_000.0])
    assert np.array_equal(out[COL_TIME_RANGE].astype("int64").to_numpy(), [10, 11])


def test_dtype_of_time_range_is_integer_extension():
    df = build_df([2.0], [0.0])
    out = split(df, 1.0, 1.0)
    # Should be concrete int64 dtype (non-nullable)
    assert out[COL_TIME_RANGE].dtype == np.dtype("int64")


def test_counts_and_sizes_even_split_per_row():
    df = build_df([3.0], [0.0], [9], [120])  # 3 chunks at gran=1.0
    out = split(df, 1.0, 1.0)

    assert len(out) == 3
    # Each chunk gets 1/3
    assert np.allclose(out[COL_COUNT].to_numpy(), [3.0, 3.0, 3.0])
    assert np.allclose(out[COL_SIZE].to_numpy(), [40.0, 40.0, 40.0])


def test_multiple_rows_preserve_per_row_splitting_and_totals():
    df = build_df([1.5, 2.0], [0.0, 10.0], [3, 4], [30, 40])
    out = split(df, 1.0, 1.0)

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
    out = split(df, time_granularity, time_resolution)

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
    out = split(df, 1.0, time_resolution)

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

    out = split(df, time_granularity, time_resolution)

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


SIZE_EDGES = [float(e) for e in SIZE_BINS]


@pytest.mark.parametrize(
    "sizes",
    [
        SIZE_EDGES,
        [e + 1 for e in SIZE_EDGES[1:-1]] + [e - 1 for e in SIZE_EDGES[1:-1]],
        [0.0, 1.0, -1.0, -math.inf, math.inf, math.nan, None],
        [None, None],
    ],
    ids=["edges", "around-edges", "special", "all-null"],
)
def test_size_bins_match_pandas_cut(sizes):
    """Each edge, the infinities, NaN and null land in the bin `pd.cut(right=True, include_lowest=True)` picks."""
    df = DataFrame.from_dict({COL_SIZE: Series(sizes, dtype="float64"), "row": list(range(len(sizes)))})
    got = set_size_bins(df).to_pandas()
    expected = pandas_reference.set_size_bins(df.to_pandas())
    assert list(got.columns) == list(expected.columns)
    bins = [c for c in expected.columns if c.startswith("size_bin_")]
    assert got[bins].to_numpy().tolist() == expected[bins].astype("int64").to_numpy().tolist()
    assert all(got[c].dtype == np.dtype("int64") for c in bins)


def test_size_bins_keep_the_input_frame():
    df = DataFrame.from_dict({COL_SIZE: [1, None]})
    set_size_bins(df)
    assert df.columns == [COL_SIZE]


@pytest.mark.parametrize(
    "time_values,time_starts,time_granularity",
    [
        ([0.0, 0.0], [1.0, 0.5], 0.1),
        ([0.3, 0.25], [1.0, 0.5], 0.1),
        ([0.7], [2.9], 0.3),
    ],
    ids=["short-circuit", "split", "split-0.3"],
)
def test_time_range_uses_numpy_floor_division(time_values, time_starts, time_granularity):
    """`time_start // bucket` follows numpy floor division, not floor(a / b): 1.0 // 0.1 is 9."""
    split(build_df(time_values, time_starts), time_granularity, 1.0)


def test_negative_duration_raises_as_pandas():
    df = build_df([-2.0, 3.0], [0.0, 10.0])
    with pytest.raises(ValueError):
        pandas_reference.split_duration_records_vectorized(df.to_pandas(), 1.0, 1.0)
    with pytest.raises(ValueError):
        split_duration_records_vectorized(df.copy(), 1.0, 1.0)


@pytest.mark.parametrize("divisor", [0.1, 0.3, -0.7, 2.5, 1e6, 3.0, -3.0, 7])
def test_floor_divide_matches_numpy(divisor):
    import random

    from dftracer.analyzer.analysis_utils import floor_divide

    rng = random.Random(7)
    values = [rng.uniform(-1e6, 1e6) * rng.choice([1, 1e-9, 1e9, 1e-3]) for _ in range(5000)]
    values += [0.0, -0.0, 1.0, 0.5, -1.0, 1e300, -1e300, math.inf, -math.inf, math.nan, 2.0**53 + 2]
    got = floor_divide(Series(values, dtype="float64"), divisor).to_list()
    expected = np.floor_divide(np.array(values), divisor).tolist()
    assert len(got) == len(expected)
    for g, e, v in zip(got, expected, values):
        if math.isnan(e):
            assert g is None or math.isnan(g), v
        else:
            assert g == e, (v, g, e)


def test_floor_divide_keeps_integers_and_nulls():
    from dftracer.analyzer.analysis_utils import floor_divide

    ints = Series([7, -7, 0, None], dtype="int64")
    assert floor_divide(ints, 2).to_list() == [3, -4, 0, None]
    assert floor_divide(ints, 2).dtype == ints.dtype
    assert floor_divide(ints, 0.1).to_list()[:3] == np.floor_divide(np.array([7.0, -7.0, 0.0]), 0.1).tolist()
    assert floor_divide(ints, 0.1).to_list()[3] is None


@pytest.mark.smoke
def test_split_duration_records_leaves_the_input_unchanged_when_no_row_is_split():
    df = DataFrame.from_dict(
        {
            "time": Series([0.0, None]),
            "time_start": Series([1_500_000, 2_500_000], dtype="int64"),
            "count": Series([1, 1], dtype="int64"),
            "size": Series([4, 8], dtype="int64"),
        }
    )
    out = split_duration_records_vectorized(df, 1.0, 1e6)
    assert out["time_range"].to_list() == [1, 2]
    assert "time_range" not in df.columns
