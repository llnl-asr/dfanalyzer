"""View aggregation on a dftracer-utils frame.

One call replaces the partial, merge and finalize steps and the tail of
`DFTracerAnalyzer._compute_view`.
"""

from __future__ import annotations

import itertools as it
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from dftracer.utils import DataFrame, DType, Series, col
from dftracer.utils.dask import DaskFrame

from ..analysis_utils import build_view_rename_map, nan_as_null
from ..constants import COL_FILE_NAME, VIEW_TYPES

_SET_VALUES = "__values"


@dataclass
class ViewPlan:
    """Columns of a view input split by the statistic the view computes for them."""

    full: list[str] = field(default_factory=list)
    sums: list[str] = field(default_factory=list)
    mins: list[str] = field(default_factory=list)
    maxs: list[str] = field(default_factory=list)
    sets: list[str] = field(default_factory=list)


def plan_view(
    records: DataFrame,
    view_types: Sequence[str],
    logical_views: Mapping[str, Iterable[str]],
) -> ViewPlan:
    """Classify the columns of `records`, in column order.

    Text and list columns outside the view types and the logical views are left out.
    Raises TypeError for a column with no aggregation.
    """
    logical = set(it.chain.from_iterable(logical_views.values()))
    keep_object = set(VIEW_TYPES) | set(view_types) | logical
    diff = set(VIEW_TYPES).difference(view_types)
    plan = ViewPlan()
    for name, dtype in zip(records.columns, records.dtypes):
        if (dtype.is_text or dtype == DType.LIST) and not (
            name in keep_object or any(name.endswith(vt) for vt in keep_object)
        ):
            continue
        if "_bin_" in name:
            plan.sums.append(name)
        elif any(name.endswith(vt) for vt in diff) or name in logical:
            plan.sets.append(name)
        elif name.endswith("_sq"):
            plan.sums.append(name)
        elif name.endswith("_call_min"):
            plan.mins.append(name)
        elif name.endswith("_call_max"):
            plan.maxs.append(name)
        elif dtype.is_numeric or dtype == DType.BOOL:
            plan.full.append(name)
        else:
            raise TypeError(f"Unsupported dtype '{dtype.name}' for column '{name}'")
    return plan


_NO_SPLIT = "\x00"


def _as_lists(values: Series) -> Series:
    """A set column as a LIST column: a list cell stays, a value is a one-value list.

    A missing value (or NaN) is an empty list, or a missing list for a text column; a set union skips both.
    """
    if values.dtype == DType.LIST:
        return values
    if values.dtype.is_text and not values.str.contains(_NO_SPLIT, regex=False).any():
        return values.str.split(_NO_SPLIT)
    # dftracer-utils branch feat/analyzer-port-gaps: scalar to list
    # No native op makes a one-value list of a number, so the cells are built in Python.
    return Series([[] if v is None or v != v else [v] for v in values.to_list()])  # noqa: PLR0124


def _column(frame: DataFrame, name: str) -> Series:
    if name in frame.columns:
        return frame[name]
    return frame.get_level_values(name)


def _native_input(records: DataFrame, view_type: str, plan: ViewPlan) -> DataFrame:
    """The view input with no index: the key, the planned columns, set cells as LIST cells."""
    numeric = plan.full + plan.sums + plan.mins + plan.maxs
    full = set(plan.full)
    data = {view_type: nan_as_null(_column(records, view_type))}
    for c in numeric:
        if c == view_type:
            continue
        # fix_std_cols of the pandas path: full columns aggregate as float64
        data[c] = nan_as_null(records[c], as_float=c in full)
    for c in plan.sets:
        data[f"{_SET_VALUES}{c}"] = _as_lists(records[c])
    return DataFrame.from_dict(data)


def _group(frame: DataFrame, view_type: str, plan: ViewPlan) -> DataFrame:
    return _group_unsorted(frame, view_type, plan).sort_values([view_type])


def _scatter(frame: DataFrame, client, npartitions: int) -> DaskFrame:
    """`frame` as `npartitions` row slices on the workers of `client`."""
    rows = len(frame)
    step = -(-rows // npartitions)
    return DaskFrame.from_frames([frame.slice(start, step) for start in range(0, rows, step)], client)


def _group_scattered(parts: DaskFrame, view_type: str, plan: ViewPlan) -> DataFrame:
    """`_group` over the partitions of `parts` on its workers; closes `parts`.

    The shuffle puts every row of a view key in one partition, so each partition groups
    whole keys with the same aggregations as `_group`, set unions included. Each frame
    is closed once the next one exists, so the workers do not hold all three at once.
    """
    frames = [parts]
    try:
        frames.append(parts.shuffle([view_type]))
        parts.close()
        frames.append(frames[-1].map_partitions(_group_unsorted, view_type, plan))
        frames[-2].close()
        return frames[-1].to_frame().sort_values([view_type])
    finally:
        for f in frames:
            f.close()


def _group_unsorted(frame: DataFrame, view_type: str, plan: ViewPlan) -> DataFrame:
    aggs = {}
    for c in plan.full:
        aggs[f"{c}_sum"] = col(c).sum()
        aggs[f"{c}_count"] = col(c).count()
        aggs[f"{c}_min"] = col(c).min()
        aggs[f"{c}_max"] = col(c).max()
        aggs[f"{c}_std"] = col(c).std()
    for names, how in ((plan.sums, "sum"), (plan.mins, "min"), (plan.maxs, "max")):
        for c in names:
            aggs[f"{c}_{how}"] = getattr(col(c), how)()
    for c in plan.sets:
        aggs[f"{c}_unique"] = col(f"{_SET_VALUES}{c}").set_union(typed=True)
    return frame.group_by([view_type]).agg(**aggs)


def _derive_call_stats(frame: DataFrame) -> DataFrame:
    names = list(frame.columns)
    new = {}
    for sq in (c for c in names if c.endswith("_sq_sum")):
        prefix = sq[: -len("_sq_sum")]
        if prefix in ("time", "size"):
            count = "count_sum"
        elif prefix.endswith("_time"):
            count = prefix[: -len("time")] + "count_sum"
        elif prefix.endswith("_size"):
            count = prefix[: -len("size")] + "count_sum"
        else:
            continue
        total = f"{prefix}_sum"
        if total not in names or count not in names:
            continue
        mean = frame[total].cast("float64") / frame[count].cast("float64")
        var = (frame[sq].cast("float64") / frame[count].cast("float64") - mean**2).clip(lower=0)
        new[f"{prefix}_call_mean"] = mean
        new[f"{prefix}_call_std"] = var**0.5
    sq_cols = [c for c in names if c.endswith("_sq_sum")]
    return frame.with_columns(**new).select(*[c for c in frame.columns if c not in sq_cols], *new)


def _fix_dtypes(frame: DataFrame, time_sliced: bool) -> DataFrame:
    names = list(frame.columns)
    ints = [c for c in names if c.endswith("_nunique")]
    doubles = [c for c in names if c.endswith(("_bw", "_intensity", "_ops", "_time", "_slope", "_pct"))]
    sizes = [c for c in names if c.endswith("_size")]
    bins = [c for c in names if "_bin_" in c]
    counts = [c for c in names if c.endswith("_count")]
    (doubles if time_sliced else ints).extend(bins + counts)
    dtype = {}
    for cols, name in ((ints, "int64"), (doubles, "float64"), (sizes, "int64")):
        dtype.update(dict.fromkeys(cols, name))
    return frame.with_columns(**{c: frame[c].cast(t) for c, t in dtype.items()})


def aggregate_view(
    records: DataFrame,
    view_type: str,
    view_types: Sequence[str],
    logical_views: Mapping[str, Iterable[str]],
    layer: str,
    time_sliced: bool,
    client=None,
    npartitions: int = 0,
) -> DataFrame:
    """The view of `records` per `view_type` row, indexed by `view_type`, as `DFTracerAnalyzer._compute_view` returns it.

    `view_type` is a column or an index level of `records`. With a Dask `client` and
    `npartitions` above 0 the group-by runs as a DaskFrame of that many partitions
    on the client's workers. Without a client it groups one frame in this process,
    because a shuffle here only adds copies of the input.
    """
    plan = plan_view(records, view_types, logical_views)
    native = _native_input(records, view_type, plan)
    if client is not None and npartitions > 0 and len(native) > 0:
        parts = _scatter(native, client, npartitions)
        del native
        frame = _group_scattered(parts, view_type, plan)
    else:
        frame = _group(native, view_type, plan)
        del native

    mean_std = {}
    for c in plan.full:
        n = frame[f"{c}_count"].cast("float64")
        mean_std[f"{c}_mean"] = (frame[f"{c}_sum"].cast("float64") / n).where(n > 0)
    frame = frame.with_columns(**mean_std)
    order = (
        [f"{c}_{s}" for c in plan.full for s in ("sum", "min", "max")]
        + [f"{c}_sum" for c in plan.sums]
        + [f"{c}_min" for c in plan.mins]
        + [f"{c}_max" for c in plan.maxs]
        + [f"{c}_{s}" for c in plan.full for s in ("mean", "std")]
    )
    nunique = {
        f"{c}_nunique": frame[f"{c}_unique"].list_len()
        for c in plan.sets
        if not (COL_FILE_NAME in f"{c}_unique" and "posix" not in layer)
    }
    frame = frame.select(view_type, *order, *[f"{c}_unique" for c in plan.sets])

    frame = frame.rename(build_view_rename_map(order))
    metrics = [c for c in frame.columns if c != view_type and not c.endswith("_unique")]
    # pandas gives a missing std, not NaN, for a group with an infinite value.
    frame = frame.with_columns(
        **{
            c: frame[c].where((frame[c] != 0) & frame[c].is_not_nan())
            if c.endswith("_std")
            else frame[c].where(frame[c] != 0)
            for c in metrics
        }
    )
    frame = _derive_call_stats(frame)
    frame = frame.with_columns(**nunique)
    frame = frame.select(*[c for c in frame.columns if not c.endswith("_unique")])
    return _fix_dtypes(frame, time_sliced).set_index([view_type])
