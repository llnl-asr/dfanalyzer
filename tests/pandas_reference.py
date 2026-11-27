"""The pandas path the frame implementations are checked against.

The package no longer runs this code. The parity tests and the benchmarks keep it
as the reference: the partial view aggregation (`partial_arrow_view_groupby`,
`merge_view_partials`, `finalize_view_partials`), the pandas view and main-view
metrics, the derived layer metrics and the read-path transforms (`set_size_bins`,
`split_duration_records_vectorized`, `expand_profile_buckets`, `standardize_system`).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import structlog
from betterset import BetterSet
from dftracer.analyzer.constants import (
    COL_COUNT,
    COL_FILE_NAME,
    COL_SIZE,
    COL_TIME,
    COL_TIME_RANGE,
    COL_TIME_START,
    SIZE_BIN_SUFFIXES,
    SIZE_BINS,
)
from dftracer.analyzer.dftracer import SYSTEM_OUTPUT_COLUMNS
from dftracer.analyzer.metrics import _find_metric_pairs
from dftracer.analyzer.types import MetricBoundaries

if TYPE_CHECKING:
    from pandas.core.groupby import SeriesGroupBy

logger = structlog.get_logger()

_Dtype = Union[pd.api.extensions.ExtensionDtype, "np.dtype[Any]", str]


def _partial_agg_columns(
    full_cols: Sequence[str],
    sum_cols: Sequence[str],
    min_cols: Sequence[str],
    max_cols: Sequence[str],
    set_cols_items: Sequence[tuple[str, object]],
    dtype_of: Callable[[str], _Dtype | None],
) -> dict[str, _Dtype | None]:
    """Metric column -> dtype map for a partial view aggregation, in the one
    order every partial has: per full column its sum, count, min, max, m2,
    mean_hi and mean_lo, then the sum, min, max and set columns. The data
    (`partial_arrow_view_groupby`), the empty partition and the dask meta all
    follow it. `dtype_of(col)` resolves a metric column's dtype from the
    caller's source (a live frame or the dask meta)."""
    cols: dict[str, _Dtype | None] = {}
    for c in full_cols:
        cols[f"{c}_sum"] = dtype_of(c)
        cols[f"{c}_count"] = pd.ArrowDtype(pa.int64())
        cols[f"{c}_min"] = dtype_of(c)
        cols[f"{c}_max"] = dtype_of(c)
        cols[f"{c}_m2"] = pd.ArrowDtype(pa.float64())
        cols[f"{c}_mean_hi"] = pd.ArrowDtype(pa.float64())
        cols[f"{c}_mean_lo"] = pd.ArrowDtype(pa.float64())
    for c in sum_cols:
        cols[f"{c}_sum"] = dtype_of(c)
    for c in min_cols:
        cols[f"{c}_min"] = dtype_of(c)
    for c in max_cols:
        cols[f"{c}_max"] = dtype_of(c)
    for c, _ in set_cols_items:
        cols[f"{c}_unique"] = "object"
    return cols


def _central_moments(work: pd.DataFrame, view_type: str, cols: Sequence[str]) -> pd.DataFrame:
    """Per view row and metric column: `{c}_m2`, the sum of squared deviations
    from the mean, and the mean as the pair `{c}_mean_hi` + `{c}_mean_lo` (a
    value near the mean and the small remainder), so a variance merged across
    partitions keeps its digits when the mean is large next to the spread.

    The deviations are taken from a first estimate of the mean and corrected
    (`m2 = sum(d^2) - sum(d)^2 / n`), the corrected two-pass form; nothing here
    subtracts two sums of squares."""
    out = {}
    key = work[view_type]
    for c in cols:
        x = pd.to_numeric(work[c], errors="coerce").astype("float64")
        grouped = x.groupby(key)
        n = grouped.count().astype("float64")
        hi = grouped.mean()
        d = x - key.map(hi)
        sd = d.groupby(key).sum()
        sd2 = (d * d).groupby(key).sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            lo = (sd / n).where(n > 0, 0.0)
            m2 = (sd2 - sd * sd / n).where(n > 0, 0.0).clip(lower=0.0)
        out[f"{c}_m2"] = m2
        out[f"{c}_mean_hi"] = hi.fillna(0.0)
        out[f"{c}_mean_lo"] = lo
    return pd.DataFrame(out)


def merge_view_partials(
    df: pd.DataFrame,
    full_cols: Sequence[str],
    sum_cols: Sequence[str] = (),
    min_cols: Sequence[str] = (),
    max_cols: Sequence[str] = (),
    set_cols: Sequence[str] = (),
    flatten_fn: Callable[[Iterable[AbstractSet[Hashable]]], AbstractSet[Hashable]] | None = None,
) -> pd.DataFrame:
    """Merge partial view aggregations (the output of `partial_arrow_view_groupby`,
    one row per view row per partition) into one row per view row, ready for
    `finalize_view_partials`. The columns are named as the partials name them:

    - full columns (`full_cols`): `_sum` and `_count` add, `_min` and `_max` take
      their extremes, and `_m2` with the mean pair (`_mean_hi`, `_mean_lo`)
      combine with the pairwise (Chan) formula about each row's first mean, so a
      variance is never rebuilt from a sum of squares;
    - `sum_cols`, `min_cols`, `max_cols`: `_sum` adds, `_min` and `_max` take
      their extreme;
    - `set_cols` (the names of the set columns): `_unique` cells are joined with
      `flatten_fn`, the function the partials were flattened with.

    A column that no partial holds is skipped. The result has the partials' column
    order and each column keeps the dtype of the partials' column of that name (an
    all-null column stays its type). The index is the view row (a MultiIndex is
    grouped by every level)."""
    if set_cols and flatten_fn is None:
        raise ValueError("merge_view_partials: set_cols need a flatten_fn")
    levels = list(range(df.index.nlevels))

    def g(s: pd.Series) -> SeriesGroupBy:
        return s.groupby(level=levels)

    out = {}
    for c in full_cols:
        need = [f"{c}_{k}" for k in ("sum", "count", "min", "max", "m2", "mean_hi", "mean_lo")]
        if any(name not in df.columns for name in need):
            continue
        n = df[f"{c}_count"].astype("float64").fillna(0.0)
        hi = df[f"{c}_mean_hi"].astype("float64").fillna(0.0)
        lo = df[f"{c}_mean_lo"].astype("float64").fillna(0.0)
        m2 = df[f"{c}_m2"].astype("float64").fillna(0.0)
        ref_hi, ref_lo = g(hi).transform("first"), g(lo).transform("first")
        delta = (hi - ref_hi) + (lo - ref_lo)  # each partition's mean from the first one's
        total = g(n).transform("sum")
        with np.errstate(invalid="ignore", divide="ignore"):
            centre = (g(n * delta).transform("sum") / total).where(total > 0, 0.0)
        spread = n * (delta - centre) ** 2
        out[f"{c}_sum"] = g(df[f"{c}_sum"]).sum()
        out[f"{c}_count"] = g(df[f"{c}_count"]).sum()
        out[f"{c}_min"] = g(df[f"{c}_min"]).min()
        out[f"{c}_max"] = g(df[f"{c}_max"]).max()
        out[f"{c}_m2"] = g(m2 + spread).sum().astype(df[f"{c}_m2"].dtype)
        out[f"{c}_mean_hi"] = g(ref_hi).first().astype(df[f"{c}_mean_hi"].dtype)
        out[f"{c}_mean_lo"] = (g(ref_lo).first() + g(pd.Series(centre, index=df.index)).first()).astype(
            df[f"{c}_mean_lo"].dtype
        )
    for c in sum_cols:
        if f"{c}_sum" in df.columns:
            out[f"{c}_sum"] = g(df[f"{c}_sum"]).sum()
    for c in min_cols:
        if f"{c}_min" in df.columns:
            out[f"{c}_min"] = g(df[f"{c}_min"]).min()
    for c in max_cols:
        if f"{c}_max" in df.columns:
            out[f"{c}_max"] = g(df[f"{c}_max"]).max()
    for c in set_cols:
        if f"{c}_unique" in df.columns:
            out[f"{c}_unique"] = g(df[f"{c}_unique"]).agg(flatten_fn)
    merged = pd.DataFrame(out)
    order = _partial_agg_columns(
        full_cols, sum_cols, min_cols, max_cols, list(zip(set_cols, set_cols)), lambda c: None
    )
    return merged[[name for name in order if name in merged.columns]]


def partial_arrow_view_groupby(
    df: pd.DataFrame,
    view_type: str,
    full_cols: Sequence[str],
    sum_cols: Sequence[str],
    min_cols: Sequence[str],
    max_cols: Sequence[str],
    set_cols_items: Sequence[tuple[str, object]],
    flatten_fn: Callable[..., object] | None,
) -> pd.DataFrame:
    """Per-partition Arrow groupby emitting mergeable partial aggregates.

    `flatten_fn` flattens a grouped set-column series when its aggregation
    object exposes no `chunk` method.
    """
    view_type_in_index = (isinstance(df.index, pd.MultiIndex) and view_type in df.index.names) or (
        df.index.name == view_type
    )
    work = df.reset_index() if view_type_in_index else df
    if work.empty:
        # Derive dtypes from the input columns so an empty partition matches
        # the meta declared by the caller (which uses the same rule).
        def _col_dtype(col: str, default: _Dtype | None = None) -> _Dtype:
            if col in work.columns:
                return work[col].dtype
            return pd.ArrowDtype(pa.float64()) if default is None else default

        cols = _partial_agg_columns(full_cols, sum_cols, min_cols, max_cols, set_cols_items, _col_dtype)
        out = pd.DataFrame({name: pd.Series(dtype=dt) for name, dt in cols.items()})
        out.index = pd.Index(
            [],
            name=view_type,
            dtype=_col_dtype(view_type, default=pd.ArrowDtype(pa.int64())),
        )
        return out

    arrow_keep = [view_type]
    for lst in (full_cols, sum_cols, min_cols, max_cols):
        for c in lst:
            if c in work.columns and c not in arrow_keep:
                arrow_keep.append(c)
    tbl = pa.Table.from_pandas(work[arrow_keep], preserve_index=False)

    agg_specs = []
    for c in full_cols:
        if c not in tbl.schema.names:
            continue
        agg_specs += [
            (c, "sum"),
            (c, "count"),
            (c, "min"),
            (c, "max"),
        ]
    for c in sum_cols:
        if c in tbl.schema.names:
            agg_specs.append((c, "sum"))
    for c in min_cols:
        if c in tbl.schema.names:
            agg_specs.append((c, "min"))
    for c in max_cols:
        if c in tbl.schema.names:
            agg_specs.append((c, "max"))

    if agg_specs:
        result = tbl.group_by([view_type]).aggregate(agg_specs)
        out = result.to_pandas(types_mapper=pd.ArrowDtype)
        out = out.set_index(view_type)
        moments = _central_moments(work, view_type, [c for c in full_cols if c in work.columns])
        for name in moments.columns:
            out[name] = moments[name].reindex(out.index).astype(pd.ArrowDtype(pa.float64()))
    else:
        uniq = work[view_type].drop_duplicates().reset_index(drop=True)
        out = pd.DataFrame(index=pd.Index(uniq, name=view_type))

    for col, agg in set_cols_items:
        if col not in work.columns:
            continue
        sgb = work.groupby(view_type)[col]
        chunk_fn = getattr(agg, "chunk", None)
        partial = chunk_fn(sgb) if chunk_fn is not None else sgb.apply(flatten_fn)
        partial.name = f"{col}_unique"
        out = out.join(partial, how="left")
    order = _partial_agg_columns(full_cols, sum_cols, min_cols, max_cols, set_cols_items, lambda c: None)
    return out[[name for name in order if name in out.columns]]


def finalize_view_partials(df: pd.DataFrame, full_cols: Sequence[str]) -> pd.DataFrame:
    """Compute mean/std per view_type row from merged partials; drop helper cols."""
    if df.empty:
        return df
    out = df.copy()
    drop = []
    for c in full_cols:
        sum_c = f"{c}_sum"
        count_c = f"{c}_count"
        m2_c = f"{c}_m2"
        if sum_c not in out.columns or count_c not in out.columns:
            continue
        s = out[sum_c].astype("float64")
        n = out[count_c].astype("float64")
        mean_v = s / n
        out[f"{c}_mean"] = mean_v.astype(pd.ArrowDtype(pa.float64()))
        if m2_c in out.columns:
            m2 = out[m2_c].astype("float64")
            # sample variance is undefined for n <= 1 -> std is NaN, matching
            # pandas .std(ddof=1); avoids a divide-by-zero on (n - 1).
            with np.errstate(invalid="ignore", divide="ignore"):
                var_v = (m2 / (n - 1)).where(n > 1, np.nan)
            out[f"{c}_std"] = np.sqrt(var_v).astype(pd.ArrowDtype(pa.float64()))
            drop += [m2_c, f"{c}_mean_hi", f"{c}_mean_lo"]
        drop.append(count_c)
    if drop:
        out = out.drop(columns=drop)
    return out


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
        df[f"{prefix}_call_std"] = var**0.5

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


def fix_std_cols(df: pd.DataFrame, std_cols: list[str]):
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


def set_main_metrics(df: pd.DataFrame):
    df = df.copy()

    count_cols = [col for col in df.columns if col.endswith('count')]
    size_cols = [col for col in df.columns if col.endswith('size')]

    new_metrics: list[str] = []

    for size_col in size_cols:
        bw_col = size_col.replace('size', 'bw')
        count_col = size_col.replace('size', 'count')
        intensity_col = size_col.replace('size', 'intensity')
        time_col = size_col.replace('size', 'time')
        df[size_col] = df[size_col].where(df[size_col] > 0, pd.NA)
        df[bw_col] = (df[size_col] / df[time_col]).where(df[size_col] > 0, pd.NA)
        df[intensity_col] = (df[count_col] / df[size_col]).where(df[size_col] > 0, pd.NA)
        new_metrics.extend([bw_col, intensity_col, time_col])

    for count_col in count_cols:
        ops_col = count_col.replace('count', 'ops')
        time_col = count_col.replace('count', 'time')
        df[ops_col] = df[count_col] / df[time_col]
        new_metrics.append(ops_col)

    df[new_metrics] = df[new_metrics].replace([np.inf, -np.inf], pd.NA).astype('Float64')

    return df.sort_index(axis=1)


def set_view_metrics(
    df: pd.DataFrame,
    metric_boundaries: MetricBoundaries,
    is_view_process_based: bool,
) -> pd.DataFrame:
    df = df.copy()

    count_metric = 'count_sum'
    size_proc_metric = 'size_sum' if is_view_process_based else 'size_proc_max'
    size_call_metric = 'size_sum'
    time_proc_metric = 'time_sum' if is_view_process_based else 'time_proc_max'
    time_call_metric = 'time_sum'

    view_metrics = list(set(df.columns.tolist()))
    new_metrics: dict[str, pd.Series] = {}

    for metric in view_metrics:
        if metric.endswith(count_metric):
            count_col = metric
            count_frac_total_col = metric.replace(count_metric, 'count_frac_total')
            count_sum = df[count_col].sum()
            if count_sum > 0:
                new_metrics[count_frac_total_col] = df[count_col] / count_sum
            else:
                new_metrics[count_frac_total_col] = pd.Series(pd.NA, index=df.index, dtype='Float64')
        elif metric.endswith(size_proc_metric):
            size_col = metric
            size_proc_frac_total_col = metric.replace(size_proc_metric, 'size_proc_frac_total')
            size_total = df[size_col].sum()
            if size_total > 0:
                new_metrics[size_proc_frac_total_col] = df[size_col] / size_total
            else:
                new_metrics[size_proc_frac_total_col] = pd.Series(pd.NA, index=df.index, dtype='Float64')
        elif metric.endswith(time_proc_metric):
            time_col = metric
            time_proc_frac_total_col = metric.replace(time_proc_metric, 'time_proc_frac_total')
            time_total = df[time_col].sum()
            if time_total > 0:
                new_metrics[time_proc_frac_total_col] = df[time_col] / time_total
            else:
                new_metrics[time_proc_frac_total_col] = pd.Series(pd.NA, index=df.index, dtype='Float64')

    # Compute call_frac_total metrics (always from *_sum, regardless of view type)
    for metric in view_metrics:
        if metric.endswith(size_call_metric):
            size_call_col = metric
            size_call_frac_total_col = metric.replace(size_call_metric, 'size_call_frac_total')
            size_call_total = df[size_call_col].sum()
            if size_call_total > 0:
                new_metrics[size_call_frac_total_col] = df[size_call_col] / size_call_total
            else:
                new_metrics[size_call_frac_total_col] = pd.Series(pd.NA, index=df.index, dtype='Float64')
        elif metric.endswith(time_call_metric):
            time_call_col = metric
            time_call_frac_total_col = metric.replace(time_call_metric, 'time_call_frac_total')
            time_call_total = df[time_call_col].sum()
            if time_call_total > 0:
                new_metrics[time_call_frac_total_col] = df[time_call_col] / time_call_total
            else:
                new_metrics[time_call_frac_total_col] = pd.Series(pd.NA, index=df.index, dtype='Float64')

    count_time_proc_frac_metric_pairs = _find_metric_pairs(
        list(new_metrics.keys()), 'count_frac_total', 'time_proc_frac_total'
    )
    for count_frac_total_col, time_proc_frac_total_col in count_time_proc_frac_metric_pairs:
        ops_percentile_col = count_frac_total_col.replace('count_frac_total', 'ops_percentile')
        ops_slope_col = count_frac_total_col.replace('count_frac_total', 'ops_slope')
        ops_slope = new_metrics[time_proc_frac_total_col] / new_metrics[count_frac_total_col]
        ops_slope = ops_slope.replace([np.inf, -np.inf], pd.NA)
        new_metrics[ops_percentile_col] = ops_slope.rank(pct=True)
        new_metrics[ops_slope_col] = ops_slope

    if new_metrics:
        new_metrics_df = pd.DataFrame(new_metrics, index=df.index)
        new_metrics_df = new_metrics_df.replace([np.inf, -np.inf], pd.NA).astype('Float64')
        overlapping_cols = [col for col in new_metrics_df.columns if col in df.columns]
        if overlapping_cols:
            df = df.drop(columns=overlapping_cols)
        df = pd.concat([df, new_metrics_df], axis=1)

    return df.sort_index(axis=1)


def set_cross_layer_metrics(
    df: pd.DataFrame,
    layers: list[str],
    layer_deps: dict[str, str | None],
    async_layers: list[str],
    derived_metrics: dict[str, dict[str, str]],
    is_view_process_based: bool,
    time_boundary_layer: str,
) -> pd.DataFrame:
    time_proc_metric = 'time_sum' if is_view_process_based else 'time_proc_max'
    time_call_metric = 'time_sum'
    compute_time_proc_metric = f"compute_{time_proc_metric}"
    compute_time_call_metric = f"compute_{time_call_metric}"
    time_proc_boundary_metric = f"{time_boundary_layer}_{time_proc_metric}"
    time_call_boundary_metric = f"{time_boundary_layer}_{time_call_metric}"

    # Collect new columns and assign them in batch to avoid fragmentation warnings
    x_layer_metrics: dict[str, pd.Series] = {}

    # Set relational time metrics for layers
    for layer in layers:
        layer_time_proc = df[f"{layer}_{time_proc_metric}"]
        layer_time_call = df[f"{layer}_{time_call_metric}"]

        # Proc: frac_boundary
        time_proc_frac_boundary_col = f"{layer}_time_proc_frac_{time_boundary_layer}"
        x_layer_metrics[time_proc_frac_boundary_col] = layer_time_proc / df[time_proc_boundary_metric]
        # Call: frac_boundary
        time_call_frac_boundary_col = f"{layer}_time_call_frac_{time_boundary_layer}"
        x_layer_metrics[time_call_frac_boundary_col] = layer_time_call / df[time_call_boundary_metric]

        child_layers = [child for child, parent in layer_deps.items() if parent == layer]
        if not child_layers:
            continue

        # Proc: overhead
        o_time_proc_col = f"o_{layer}_{time_proc_metric}"
        o_time_proc_frac_boundary_col = f"o_{layer}_time_proc_frac_{time_boundary_layer}"
        o_time_proc_frac_self_col = f"o_{layer}_time_proc_frac_self"
        o_time_proc_frac_total_col = o_time_proc_col.replace(time_proc_metric, 'time_proc_frac_total')

        child_time_proc_sum = sum(df[f"{child}_{time_proc_metric}"].fillna(0) for child in child_layers)
        o_time_proc = np.maximum(layer_time_proc - child_time_proc_sum, 0)
        o_time_proc_total = o_time_proc.sum()

        o_time_proc_series = pd.array(o_time_proc, dtype='Float64')
        x_layer_metrics[o_time_proc_col] = o_time_proc_series
        x_layer_metrics[o_time_proc_frac_boundary_col] = o_time_proc_series / df[time_proc_boundary_metric]
        x_layer_metrics[o_time_proc_frac_self_col] = o_time_proc_series / layer_time_proc
        x_layer_metrics[o_time_proc_frac_total_col] = pd.NA
        if o_time_proc_total > 0:
            x_layer_metrics[o_time_proc_frac_total_col] = o_time_proc_series / o_time_proc_total

        # Call: overhead
        o_time_call_col = f"o_{layer}_{time_call_metric}"
        o_time_call_frac_boundary_col = f"o_{layer}_time_call_frac_{time_boundary_layer}"
        o_time_call_frac_self_col = f"o_{layer}_time_call_frac_self"
        o_time_call_frac_total_col = f"o_{layer}_time_call_frac_total"

        child_time_call_sum = sum(df[f"{child}_{time_call_metric}"].fillna(0) for child in child_layers)
        o_time_call = np.maximum(layer_time_call - child_time_call_sum, 0)
        o_time_call_total = o_time_call.sum()

        o_time_call_series = pd.array(o_time_call, dtype='Float64')
        x_layer_metrics[o_time_call_col] = o_time_call_series
        x_layer_metrics[o_time_call_frac_boundary_col] = o_time_call_series / df[time_call_boundary_metric]
        x_layer_metrics[o_time_call_frac_self_col] = o_time_call_series / layer_time_call
        x_layer_metrics[o_time_call_frac_total_col] = pd.NA
        if o_time_call_total > 0:
            x_layer_metrics[o_time_call_frac_total_col] = o_time_call_series / o_time_call_total

        # Proc + Call: frac_parent
        layer_has_time_proc = layer_time_proc.sum() > 0
        layer_has_time_call = layer_time_call.sum() > 0
        for child_layer in child_layers:
            time_proc_frac_parent_col = f"{child_layer}_time_proc_frac_parent"
            x_layer_metrics[time_proc_frac_parent_col] = pd.NA
            if layer_has_time_proc:
                x_layer_metrics[time_proc_frac_parent_col] = df[f"{child_layer}_{time_proc_metric}"] / layer_time_proc

            time_call_frac_parent_col = f"{child_layer}_time_call_frac_parent"
            x_layer_metrics[time_call_frac_parent_col] = pd.NA
            if layer_has_time_call:
                x_layer_metrics[time_call_frac_parent_col] = df[f"{child_layer}_{time_call_metric}"] / layer_time_call

    # Set relational time metrics for derived metrics
    for layer, layer_derived in derived_metrics.items():
        for dm in layer_derived:
            dm_col = f"{layer}_{dm}"
            dm_time_proc_col = f"{dm_col}_{time_proc_metric}"
            dm_time_call_col = f"{dm_col}_{time_call_metric}"

            # Proc
            if dm_time_proc_col in df.columns:
                dm_time_proc = df[dm_time_proc_col]
                dm_time_proc_total = dm_time_proc.sum()

                x_layer_metrics[f"{dm_col}_time_proc_frac_{time_boundary_layer}"] = (
                    dm_time_proc / df[time_proc_boundary_metric]
                )
                x_layer_metrics[f"{dm_col}_time_proc_frac_parent"] = dm_time_proc / df[f"{layer}_{time_proc_metric}"]
                x_layer_metrics[f"{dm_col}_time_proc_frac_total"] = pd.NA
                if dm_time_proc_total > 0:
                    x_layer_metrics[f"{dm_col}_time_proc_frac_total"] = dm_time_proc / dm_time_proc_total

            # Call
            if dm_time_call_col in df.columns:
                dm_time_call = df[dm_time_call_col]
                dm_time_call_total = dm_time_call.sum()

                x_layer_metrics[f"{dm_col}_time_call_frac_{time_boundary_layer}"] = (
                    dm_time_call / df[time_call_boundary_metric]
                )
                x_layer_metrics[f"{dm_col}_time_call_frac_parent"] = dm_time_call / df[f"{layer}_{time_call_metric}"]
                x_layer_metrics[f"{dm_col}_time_call_frac_total"] = pd.NA
                if dm_time_call_total > 0:
                    x_layer_metrics[f"{dm_col}_time_call_frac_total"] = dm_time_call / dm_time_call_total

    # Set unoverlapped times if there is compute time
    if compute_time_proc_metric in df.columns:
        compute_time_proc = df[compute_time_proc_metric].fillna(0).astype('Float64')
        compute_time_call = df[compute_time_call_metric].fillna(0).astype('Float64')
        # Set unoverlapped time metrics
        for async_layer in async_layers:
            # Proc: unoverlapped
            time_proc_col = f"{async_layer}_{time_proc_metric}"
            u_time_proc_col = f"u_{time_proc_col}"

            layer_time_proc = df[time_proc_col]
            u_time_proc = (layer_time_proc - compute_time_proc).clip(lower=0).astype('Float64')
            u_time_proc_total = u_time_proc.sum()

            u_time_proc_series = pd.array(u_time_proc, dtype='Float64')
            x_layer_metrics[u_time_proc_col] = u_time_proc_series
            x_layer_metrics[f"u_{async_layer}_time_proc_frac_self"] = u_time_proc_series / layer_time_proc
            x_layer_metrics[f"u_{async_layer}_time_proc_frac_{time_boundary_layer}"] = (
                u_time_proc_series / df[time_proc_boundary_metric]
            )
            x_layer_metrics[f"u_{async_layer}_time_proc_frac_total"] = pd.NA
            if u_time_proc_total > 0:
                x_layer_metrics[f"u_{async_layer}_time_proc_frac_total"] = u_time_proc_series / u_time_proc_total

            parent_layer = layer_deps.get(async_layer)
            if parent_layer:
                x_layer_metrics[f"u_{async_layer}_time_proc_frac_parent"] = (
                    u_time_proc_series / df[f"{parent_layer}_{time_proc_metric}"]
                )

            # Call: unoverlapped
            time_call_col = f"{async_layer}_{time_call_metric}"
            u_time_call_col = f"u_{time_call_col}"

            layer_time_call = df[time_call_col]
            u_time_call = (layer_time_call - compute_time_call).clip(lower=0).astype('Float64')
            u_time_call_total = u_time_call.sum()

            u_time_call_series = pd.array(u_time_call, dtype='Float64')
            x_layer_metrics[u_time_call_col] = u_time_call_series
            x_layer_metrics[f"u_{async_layer}_time_call_frac_self"] = u_time_call_series / layer_time_call
            x_layer_metrics[f"u_{async_layer}_time_call_frac_{time_boundary_layer}"] = (
                u_time_call_series / df[time_call_boundary_metric]
            )
            x_layer_metrics[f"u_{async_layer}_time_call_frac_total"] = pd.NA
            if u_time_call_total > 0:
                x_layer_metrics[f"u_{async_layer}_time_call_frac_total"] = u_time_call_series / u_time_call_total

            if parent_layer:
                x_layer_metrics[f"u_{async_layer}_time_call_frac_parent"] = (
                    u_time_call_series / df[f"{parent_layer}_{time_call_metric}"]
                )

    if x_layer_metrics:
        x_layer_metrics_df = pd.DataFrame(x_layer_metrics, index=df.index)
        x_layer_metrics_df = x_layer_metrics_df.replace([np.inf, -np.inf], pd.NA).astype('Float64')
        overlapping_cols = [col for col in x_layer_metrics_df.columns if col in df.columns]
        if overlapping_cols:
            df = df.drop(columns=overlapping_cols)
        df = pd.concat([df.copy(), x_layer_metrics_df], axis=1)

    return df.sort_index(axis=1)


def set_quantile_metrics(df: pd.DataFrame):
    quantile_metrics = [col for col in df.columns if col.endswith('_stats') and '_q' in col]

    if not quantile_metrics:
        return df

    new_cols: dict[str, pd.Series] = {}

    for stats_col in quantile_metrics:
        stats = df[stats_col]

        if stats.empty:
            continue

        col_base = stats_col.replace('_stats', '')
        mean_col = f"{col_base}_mean"
        std_col = f"{col_base}_std"
        count_col = f"{col_base}_count"

        mean_series = pd.to_numeric(stats.str[0], errors='coerce').astype('Float64')
        std_series = pd.to_numeric(stats.str[1], errors='coerce').astype('Float64')
        count_series = pd.to_numeric(stats.str[2], errors='coerce').astype('Int64')

        new_cols[mean_col] = mean_series
        new_cols[std_col] = std_series
        new_cols[count_col] = count_series

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        df = df.drop(columns=quantile_metrics)

    return df.sort_index(axis=1)


def set_layer_metrics(
    hlm: pd.DataFrame,
    derived_metrics: dict[str, str],
    size_derived_metrics: list[str] | None = None,
) -> pd.DataFrame:
    # Create an explicit copy to avoid SettingWithCopyWarning
    hlm = hlm.copy()
    hlm_columns = list(hlm.columns)
    size_derived_metric_set = set(size_derived_metrics or [])
    is_size_col = {col: (col == "size" or "size_bin" in col) for col in hlm_columns}
    col_kinds = {}
    numeric_cols = {}

    for col in hlm_columns:
        series = hlm[col]
        if is_size_col[col] or pd.api.types.is_numeric_dtype(series.dtype):
            col_kinds[col] = "numeric"
            numeric_cols[col] = to_nullable_numeric(series)
        elif pd.api.types.is_string_dtype(series.dtype):
            col_kinds[col] = "string"
        elif is_set_like_series(series):
            col_kinds[col] = "set_like"
        else:
            raise TypeError(
                f"Unsupported data type '{series.dtype}' for column '{col}'. "
                "Developer must add explicit handling for this data type in set_layer_metrics."
            )

    # Build derived columns in-memory and append once to avoid repeated fragmentation.
    derived_cols: dict[str, pd.Series] = {}
    for metric, condition in derived_metrics.items():
        metric_mask = hlm.eval(condition)
        is_size_metric = metric in size_derived_metric_set
        for col in hlm_columns:
            if not is_size_metric and is_size_col[col]:
                continue
            metric_col = f"{metric}_{col}"
            if col_kinds[col] in {"string", "set_like"}:
                # The set union of the main view skips None for set-like columns.
                derived_cols[metric_col] = hlm[col].where(metric_mask, None)
            else:
                derived_cols[metric_col] = numeric_cols[col].where(metric_mask)

    if derived_cols:
        hlm = pd.concat([hlm, pd.DataFrame(derived_cols, index=hlm.index)], axis=1)
    return hlm


def is_set_like_series(series: pd.Series) -> bool:
    for value in series.array:
        if value is not None and value is not pd.NA:
            return isinstance(value, BetterSet)
    return False


def to_nullable_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if pd.api.types.is_integer_dtype(numeric.dtype):
        return numeric.astype("Int64")
    if pd.api.types.is_float_dtype(numeric.dtype):
        return numeric.astype("Float64")
    return numeric


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


def expand_profile_buckets(
    df: pd.DataFrame,
    expansion_factor: int,
    time_resolution: float,
    time_granularity: float,
    distribution: str,
) -> pd.DataFrame:
    """Expand each profile row into ``expansion_factor`` sub-bucket rows.

    When the analysis granularity is finer than the profile bucket width
    (e.g. 1s analysis vs 5s profiles, expansion_factor=5), each canonical
    profile row is replicated into *expansion_factor* rows with adjusted
    ``time_range``, ``time_start``, and ``time_end``.

    Measures (``count``, ``time``, ``size``) are distributed across
    sub-buckets using the chosen *distribution* strategy:

    * ``"uniform"``: divide evenly.
    * ``"weighted"``: use ``time_min`` / ``time_max`` to shape the
      distribution so that sub-buckets with higher estimated per-event
      duration receive proportionally more of the aggregate ``time``.
      ``count`` and ``size`` remain uniformly split.
    """
    if df.empty:
        return df

    sub_granularity_us = int(time_granularity * time_resolution)
    output_columns = list(df.columns)

    # Repeat every row expansion_factor times
    expanded = df.loc[df.index.repeat(expansion_factor)].copy()
    sub_idx = np.tile(np.arange(expansion_factor), len(df))

    # Recompute time_start / time_end / time_range for each sub-bucket
    base_start = np.repeat(df["time_start"].values, expansion_factor)
    expanded["time_start"] = pd.array((base_start + sub_idx * sub_granularity_us).astype(np.int64), dtype="Int64")
    expanded["time_end"] = pd.array(
        (expanded["time_start"].to_numpy(dtype=np.int64, na_value=0) + sub_granularity_us),
        dtype="Int64",
    )
    expanded["time_range"] = pd.array(
        expanded["time_start"].to_numpy(dtype=np.int64, na_value=0) // sub_granularity_us,
        dtype="Int64",
    )

    # Distribute measures
    if distribution == "weighted":
        t_min = np.repeat(df["time_min"].fillna(0).values, expansion_factor)
        t_max = np.repeat(df["time_max"].fillna(0).values, expansion_factor)
        if expansion_factor > 1:
            ramp = t_min + (t_max - t_min) * sub_idx / (expansion_factor - 1)
        else:
            ramp = t_min
        ramp_sums = np.repeat(
            ramp.reshape(-1, expansion_factor).sum(axis=1),
            expansion_factor,
        )
        weights = np.where(ramp_sums > 0, ramp / ramp_sums, 1.0 / expansion_factor)
        total_time = np.repeat(df["time"].values, expansion_factor)
        expanded["time"] = (total_time * weights).astype("float64")
    else:
        expanded["time"] = (np.repeat(df["time"].values, expansion_factor) / expansion_factor).astype("float64")

    # count and size always split uniformly (they are discrete totals)
    expanded["count"] = np.repeat(df["count"].fillna(0).values, expansion_factor) / expansion_factor
    base_count = expanded["count"].values
    floor_count = np.floor(base_count).astype(np.int64)
    remainders = np.repeat(
        (df["count"].fillna(0).values % expansion_factor).astype(np.int64),
        expansion_factor,
    )
    floor_count = np.where(sub_idx < remainders, floor_count + 1, floor_count)
    expanded["count"] = pd.array(floor_count, dtype="Int64")

    orig_size = df["size"].values
    has_size = pd.notna(orig_size)
    rep_size = np.repeat(np.where(has_size, orig_size.astype(float), 0.0), expansion_factor)
    rep_has_size = np.repeat(has_size, expansion_factor)
    size_vals = np.where(rep_has_size, (rep_size / expansion_factor).astype(np.int64), 0)
    size_mask = ~rep_has_size
    expanded["size"] = pd.array(size_vals, dtype="Int64")
    expanded.loc[size_mask, "size"] = pd.NA

    # stat columns carry through unchanged (they are per-event bounds)
    expanded.reset_index(drop=True, inplace=True)
    return expanded[output_columns]


def standardize_system(
    df: pd.DataFrame,
    time_origin: int,
    time_granularity: float,
    time_resolution: float,
) -> pd.DataFrame:
    """Aggregate raw system events into per-time_range system metric rows."""
    empty = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in SYSTEM_OUTPUT_COLUMNS.items()})
    if df.empty:
        return empty

    df = df.copy()
    bucket_width_us = int(time_granularity * time_resolution)
    df[COL_TIME_RANGE] = ((df["ts"] - time_origin) // bucket_width_us).astype("Int64")

    group_keys = ["host_hash", COL_TIME_RANGE]

    # Aggregate CPU (name == "cpu"): mean of samples per bucket
    agg_cpu = df[df["name"] == "cpu"]
    cpu_agg = pd.DataFrame()
    if not agg_cpu.empty:
        agg_dict = {}
        for m, out in [
            ("iowait_pct", "sys_cpu_iowait_pct"),
            ("user_pct", "sys_cpu_user_pct"),
            ("system_pct", "sys_cpu_system_pct"),
            ("idle_pct", "sys_cpu_idle_pct"),
        ]:
            if m in agg_cpu.columns:
                agg_dict[out] = (m, "mean")
        if agg_dict:
            cpu_agg = agg_cpu.groupby(group_keys).agg(**agg_dict).reset_index()

    # Per-core cross-core stats (name starts with "cpu-")
    per_core = df[df["name"].str.startswith("cpu-")]
    core_agg = pd.DataFrame()
    if not per_core.empty and "iowait_pct" in per_core.columns:
        core_agg = (
            per_core.groupby(group_keys)
            .agg(
                sys_core_iowait_pct_max=("iowait_pct", "max"),
                sys_core_iowait_pct_p95=("iowait_pct", lambda x: x.quantile(0.95)),
            )
            .reset_index()
        )

    # Memory (name == "memory"): mean of samples per bucket
    mem = df[df["name"] == "memory"]
    mem_agg = pd.DataFrame()
    if not mem.empty:
        mem_dict = {}
        for m, out in [
            ("Dirty", "sys_mem_dirty"),
            ("Cached", "sys_mem_cached"),
            ("MemAvailable", "sys_mem_available"),
        ]:
            if m in mem.columns:
                mem_dict[out] = (m, "mean")
        if mem_dict:
            mem_agg = mem.groupby(group_keys).agg(**mem_dict).reset_index()

    # Merge all on (host_hash, time_range)
    dfs = [d for d in [cpu_agg, core_agg, mem_agg] if not d.empty]
    if not dfs:
        return empty

    result = dfs[0]
    for d in dfs[1:]:
        result = result.merge(d, on=group_keys, how="outer")

    for col, dtype in SYSTEM_OUTPUT_COLUMNS.items():
        if col not in result.columns:
            result[col] = pd.Series(dtype=dtype)
        result[col] = result[col].astype(dtype)
    return result[list(SYSTEM_OUTPUT_COLUMNS)]
