import hashlib
import numpy as np
import os
import pandas as pd
import re
import structlog
from typing import List, Union

from .constants import (
    COL_COUNT,
    COL_FILE_NAME,
    COL_PROC_NAME,
    COL_SIZE,
    COL_TIME,
    COL_TIME_RANGE,
    COL_TIME_START,
    FILE_PATTERN_PLACEHOLDER,
    PROC_NAME_SEPARATOR,
    SIZE_BINS,
    SIZE_BIN_SUFFIXES,
)

logger = structlog.get_logger()



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


def derive_call_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Derive per-call mean and std from sum-of-squares support columns.

    For each column ending in '_sq_sum', computes:
      - {prefix}_call_mean = sum / count
      - {prefix}_call_std  = sqrt(sq/n - mean^2)
    Then drops the support column.
    """
    sq_cols = [c for c in df.columns if c.endswith("_sq_sum")]
    if not sq_cols:
        return df
    df = df.copy()
    for sq_col in sq_cols:
        prefix = sq_col[: -len("_sq_sum")]
        sum_col = f"{prefix}_sum"

        # Determine the count column for this metric family
        if prefix in ("time", "size"):
            count_col = "count_sum"
        elif prefix.endswith("_time"):
            count_col = prefix[: -len("time")] + "count_sum"
        elif prefix.endswith("_size"):
            count_col = prefix[: -len("size")] + "count_sum"
        else:
            continue

        if sum_col not in df.columns or count_col not in df.columns:
            continue

        n = df[count_col].astype("Float64")
        s = df[sum_col].astype("Float64")
        sq = df[sq_col].astype("Float64")
        mean = s / n
        var = (sq / n - mean**2).clip(lower=0)
        df[f"{prefix}_call_mean"] = mean
        df[f"{prefix}_call_std"] = var ** 0.5

    df = df.drop(columns=sq_cols)
    return df


def fix_dtypes(df: pd.DataFrame, time_sliced: bool = False):
    if df.empty:
        return df
    int_cols = []
    int_cols.extend([col for col in df.columns if col.endswith('_nunique')])
    double_cols = []
    double_cols.extend([col for col in df.columns if col.endswith('_bw')])
    double_cols.extend([col for col in df.columns if col.endswith('_intensity')])
    double_cols.extend([col for col in df.columns if col.endswith('_ops')])
    double_cols.extend([col for col in df.columns if col.endswith('_time')])
    double_cols.extend([col for col in df.columns if col.endswith('_slope')])
    double_cols.extend([col for col in df.columns if col.endswith('_pct')])
    size_cols = [col for col in df.columns if col.endswith('_size')]
    if time_sliced:
        double_cols.extend([col for col in df.columns if '_bin_' in col])
        double_cols.extend([col for col in df.columns if col.endswith('_count')])
    else:
        int_cols.extend([col for col in df.columns if '_bin_' in col])
        int_cols.extend([col for col in df.columns if col.endswith('_count')])
    df[int_cols] = df[int_cols].astype('Int64')
    df[double_cols] = df[double_cols].astype('Float64')
    df[size_cols] = df[size_cols].astype('Int64')
    return df


def fix_size_values(df: pd.DataFrame):
    size_cols = [col for col in df.columns if 'size' in col]
    df[size_cols] = df[size_cols].replace(0, pd.NA)
    return df


def fix_std_cols(df: pd.DataFrame, std_cols: List[str]):
    """
    Convert specified standard deviation columns to float64 dtype.

    This function is needed to ensure that columns representing standard deviations
    are stored as float64, which avoids issues with object arrays in Dask and ensures
    consistent numeric operations.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    std_cols : List[str]
        List of column names to convert to float64.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns converted to float64 dtype.
    """
    return df.assign(**{col: pd.to_numeric(df[col], errors="coerce").astype("float64") for col in std_cols})


def set_app_name(df: pd.DataFrame):
    return df.assign(
        app_name=lambda df: df.index.get_level_values(COL_PROC_NAME)
        .str.split(PROC_NAME_SEPARATOR)
        .str[0]
        .astype("string"),
    )


def set_host_name(df: pd.DataFrame):
    return df.assign(
        app_name=lambda df: df.index.get_level_values(COL_PROC_NAME)
        .str.split(PROC_NAME_SEPARATOR)
        .str[1]
        .astype("string"),
    )


def set_file_dir(df: pd.DataFrame):
    if COL_FILE_NAME not in df.index.names:
        return df
    return df.assign(
        file_dir=df.index.get_level_values(COL_FILE_NAME).map(os.path.dirname).astype("string"),
    )


def set_file_pattern(df: pd.DataFrame):
    if COL_FILE_NAME not in df.index.names:
        return df

    def _apply_regex(file_name: str):
        return re.sub('[0-9]+', FILE_PATTERN_PLACEHOLDER, file_name)

    return df.assign(
        file_pattern=df.index.get_level_values(COL_FILE_NAME).map(_apply_regex).astype("string"),
    )


def set_id(ix: Union[tuple, str, int]):
    ix_str = '_'.join(map(str, ix)) if isinstance(ix, tuple) else str(ix)
    return int(hashlib.md5(ix_str.encode()).hexdigest(), 16)


def set_proc_name_parts(df: pd.DataFrame):
    if COL_PROC_NAME not in df.index.names:
        return df

    proc_names = df.index.get_level_values(COL_PROC_NAME)

    first_proc_name_parts = proc_names[0].split(PROC_NAME_SEPARATOR)

    if first_proc_name_parts[0] == 'app':
        return df.assign(
            proc_name_parts=proc_names.str.split(PROC_NAME_SEPARATOR),
            app_name=lambda df: df.proc_name_parts.str[0].astype(str),
            host_name=lambda df: df.proc_name_parts.str[1].astype(str),
            # node_name=lambda df: pd.NA,
            proc_id=lambda df: df.proc_name_parts.str[2].astype(str),
            rank=lambda df: pd.NA,
            thread_id=lambda df: df.proc_name_parts.str[3].astype(str),
        ).drop(columns=['proc_name_parts'])

    return df.assign(
        proc_name_parts=proc_names.str.split(PROC_NAME_SEPARATOR),
        app_name=lambda df: df.proc_name_parts.str[0].astype(str),
        host_name=lambda df: df.proc_name_parts.str[1].astype(str),
        # node_name=lambda df: df.proc_name_parts.str[1].astype(str),
        proc_id=lambda df: pd.NA,
        rank=lambda df: df.proc_name_parts.str[2].astype(str),
        thread_id=lambda df: pd.NA,
    ).drop(columns=['proc_name_parts'])


def set_size_bins(df: pd.DataFrame):
    # `_sanitize_size_offset` replaces 0 with pd.NA, upcasting size to an
    # object/nullable column; pd.cut raises on pd.NA. Coerce to float NaN first.
    size_bins = pd.cut(
        pd.to_numeric(df['size'], errors='coerce'),
        bins=SIZE_BINS,
        labels=SIZE_BIN_SUFFIXES,
        right=True,
        include_lowest=True,
    )
    size_bin_dummies = pd.get_dummies(size_bins, prefix='size_bin', dtype=int)
    # Ensure all expected size_bin columns exist (needed for Dask partition consistency)
    for suffix in SIZE_BIN_SUFFIXES:
        col = f"size_bin_{suffix}"
        if col not in size_bin_dummies.columns:
            size_bin_dummies[col] = 0
    return pd.concat([df, size_bin_dummies], axis=1)


def set_unique_counts(df: pd.DataFrame, layer: str):
    # Defragment once before deriving many unique-count columns.
    df = df.copy()
    unique_cols = [col for col in df.columns if col.endswith('_unique')]
    nunique_cols = {}
    for unique_col in unique_cols:
        if COL_FILE_NAME in unique_col and 'posix' not in layer:
            continue
        nunique_col = unique_col.replace('_unique', '_nunique')
        if df[unique_col].isnull().all():
            if df[unique_col].dtype != 'object':
                logger.warning(
                    "Column '%s' is not of object dtype (actual: %s) and all values are null during 'set_unique_counts'",
                    unique_col,
                    df[unique_col].dtype,
                )
            nunique_cols[nunique_col] = pd.Series(0, index=df.index, dtype='Int32')
        else:
            nunique_cols[nunique_col] = df[unique_col].map(len).astype('Int32')

    if nunique_cols:
        nunique_df = pd.DataFrame(nunique_cols, index=df.index).astype('Int32')
        overlapping_cols = [col for col in nunique_df.columns if col in df.columns]
        if overlapping_cols:
            df = df.drop(columns=overlapping_cols)
        df = pd.concat([df, nunique_df], axis=1)

    return df.drop(columns=unique_cols)


def split_duration_records_vectorized(
    df: pd.DataFrame,
    time_granularity: float,
    time_resolution: float,
) -> pd.DataFrame:
    # Convert duration column to numpy array
    durations = df[COL_TIME].fillna(0).to_numpy()

    if durations.size == 0:
        return df

    # Calculate number of chunks needed for each row
    n_chunks = np.ceil(durations / time_granularity).astype(int)
    max_chunks = n_chunks.max()

    if max_chunks == 0:
        df[COL_TIME_RANGE] = df[COL_TIME_START] // (time_granularity * time_resolution)
        df[COL_TIME_RANGE] = df[COL_TIME_RANGE].astype('int64')
        return df.copy()

    # Create expansion indices
    row_idx = np.arange(len(df))
    repeated_idx = np.repeat(row_idx, n_chunks)

    # Create the expanded dataframe
    result_df = df.iloc[repeated_idx].copy()

    # Calculate chunk numbers for each expanded row
    chunk_numbers = np.concatenate([np.arange(n) for n in n_chunks])

    # Calculate time values
    is_last_chunk = chunk_numbers == (n_chunks.repeat(n_chunks) - 1)
    time_values = np.full(len(chunk_numbers), time_granularity)

    # Handle remainders for last chunks
    remainders = durations % time_granularity
    remainders = np.where(remainders == 0, time_granularity, remainders)
    time_values[is_last_chunk] = remainders.repeat(n_chunks)[is_last_chunk]

    # Update time column
    result_df[COL_TIME] = time_values

    # Calculate and update timestamps
    ts_base = df[COL_TIME_START].to_numpy()
    ts_offsets = chunk_numbers * time_granularity * time_resolution  # Convert to microseconds
    result_df[COL_TIME_START] = ts_base.repeat(n_chunks) + ts_offsets

    result_df[COL_TIME_RANGE] = result_df[COL_TIME_START] // (time_granularity * time_resolution)
    result_df[COL_TIME_RANGE] = result_df[COL_TIME_RANGE].astype('int64')

    counts = df[COL_COUNT].to_numpy()
    expanded_counts = counts.repeat(n_chunks)
    # Divide counts by number of chunks for each original row
    result_df[COL_COUNT] = expanded_counts / n_chunks.repeat(n_chunks)

    sizes = df[COL_SIZE].to_numpy()
    expanded_sizes = sizes.repeat(n_chunks)
    # Divide sizes by number of chunks for each original row
    result_df[COL_SIZE] = expanded_sizes / n_chunks.repeat(n_chunks)

    return result_df.reset_index(drop=True)
