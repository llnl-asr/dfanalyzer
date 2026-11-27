import math
import os
import re

from dftracer.utils import DataFrame, DType, Series

from .constants import (
    COL_COUNT,
    COL_FILE_NAME,
    COL_SIZE,
    COL_TIME,
    COL_TIME_RANGE,
    COL_TIME_START,
    FILE_PATTERN_PLACEHOLDER,
    SIZE_BINS,
    SIZE_BIN_SUFFIXES,
)


_INT_DTYPES = frozenset(
    {DType.INT8, DType.INT16, DType.INT32, DType.INT64, DType.UINT8, DType.UINT16, DType.UINT32, DType.UINT64}
)


def is_missing(value) -> bool:
    """Whether a scalar is a missing value: None or a float NaN."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def nan_as_null(values: Series, as_float: bool = False) -> Series:
    """`values` with a float NaN as a missing value, as it is when the pandas path builds an Arrow table.

    With `as_float` the result is a float64 column.
    """
    if as_float and values.dtype != DType.FLOAT64:
        values = values.cast("float64")
    if not values.dtype.is_float or not values.is_nan().any():
        return values
    return values.fill_nan(None)


def frame_nbytes(frame: DataFrame) -> int:
    """Bytes held by the columns and the index levels of `frame`."""
    return int(frame.memory_usage().sum())


def floor_divide(values: Series, divisor: float) -> Series:
    """``values // divisor`` with numpy ``floor_divide`` results; null stays null.

    Integers divided by an integer stay integers. Anything else is float.
    """
    if values.dtype in _INT_DTYPES and isinstance(divisor, int):
        return values // divisor
    return values.astype("float64") // float(divisor)


def build_view_rename_map(columns):
    """Build a column rename dict for view-level statistics.

    Renames per-process stats (e.g., time_min → time_proc_min) and
    fixes double-suffixed per-call stats (e.g., time_call_min_min → time_call_min).
    """
    rename_map = {}
    for col in columns:
        if col.endswith("_call_min_min"):
            rename_map[col] = col[: -len("_min")]
        elif col.endswith("_call_max_max"):
            rename_map[col] = col[: -len("_max")]
        elif col.endswith("_min") and not col.endswith("_call_min"):
            rename_map[col] = col[: -len("_min")] + "_proc_min"
        elif col.endswith("_max") and not col.endswith("_call_max"):
            rename_map[col] = col[: -len("_max")] + "_proc_max"
        elif col.endswith("_mean"):
            rename_map[col] = col[: -len("_mean")] + "_proc_mean"
        elif col.endswith("_std"):
            rename_map[col] = col[: -len("_std")] + "_proc_std"
    return rename_map


def _with_file_name_column(df: DataFrame, name: str, fn) -> DataFrame:
    """A copy of `df` with column `name` set to `fn` of each file name index value; a missing name stays missing."""
    if COL_FILE_NAME not in df.index_names():
        return df
    names = df.get_level_values(COL_FILE_NAME).to_list()
    out = df.copy()
    out[name] = Series([None if v is None else fn(v) for v in names], dtype="string")
    return out


def set_file_dir(df: DataFrame) -> DataFrame:
    return _with_file_name_column(df, "file_dir", os.path.dirname)


def set_file_pattern(df: DataFrame) -> DataFrame:
    return _with_file_name_column(df, "file_pattern", lambda v: re.sub("[0-9]+", FILE_PATTERN_PLACEHOLDER, v))


def set_size_bins(df: DataFrame) -> DataFrame:
    """A copy of `df` with one 0/1 ``size_bin_<suffix>`` column per `SIZE_BINS` interval.

    Intervals are closed on the right and the first one also holds its lower
    edge, as ``pandas.cut(right=True, include_lowest=True)``. A missing size is in no bin.
    """
    size = df[COL_SIZE].astype("float64")
    bins = size.cut(Series([float(edge) for edge in SIZE_BINS]), right=True, outer=False, include_lowest=True)
    out = df.copy()
    for i, suffix in enumerate(SIZE_BIN_SUFFIXES):
        out[f"size_bin_{suffix}"] = (bins == i).fillna(False).astype("int64")
    return out


def split_duration_records_vectorized(
    df: DataFrame,
    time_granularity: float,
    time_resolution: float,
) -> DataFrame:
    """Split each row into `time_granularity` long chunks of its duration.

    Counts and sizes divide evenly over the chunks, and the last chunk holds
    the remainder. With no row longer than zero, only `time_range` is set.
    """
    if df.num_rows == 0:
        return df
    durations = df[COL_TIME].fillna(0)
    n_chunks = (durations / time_granularity).ceil().astype("int64")
    bucket = time_granularity * time_resolution

    if n_chunks.max() == 0:
        df = df.copy()
        df[COL_TIME_RANGE] = floor_divide(df[COL_TIME_START], bucket).astype("int64")
        return df
    if n_chunks.min() < 0:
        raise ValueError("negative dimensions are not allowed: a duration is negative")

    # No native per-row repeat, so the expanded row positions are built in Python; measured faster
    # than the numpy repeat of the pandas path at 1M rows (dftracer-utils branch feat/analyzer-port-gaps: repeat).
    counts = n_chunks.to_list()
    positions = Series([row for row, n in enumerate(counts) for _ in range(n)])
    chunk = Series([c for n in counts for c in range(n)])
    is_last = Series([c == n - 1 for n in counts for c in range(n)])
    repeats = n_chunks.gather(positions)

    remainders = durations % time_granularity
    remainders = remainders.mask(remainders == 0, time_granularity)

    result = df.gather(positions)
    full = Series([float(time_granularity)] * len(positions))
    result[COL_TIME] = full.mask(is_last, remainders.gather(positions))
    result[COL_TIME_START] = df[COL_TIME_START].gather(positions) + chunk * time_granularity * time_resolution
    result[COL_TIME_RANGE] = floor_divide(result[COL_TIME_START], bucket).astype("int64")
    result[COL_COUNT] = df[COL_COUNT].gather(positions) / repeats
    result[COL_SIZE] = df[COL_SIZE].gather(positions) / repeats
    return result
