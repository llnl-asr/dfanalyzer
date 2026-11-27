"""Main-view group-by and profile reconcile on dftracer-utils frames.

`group_main_view` replaces the group-by and the steps after it in
`Analyzer._compute_main_view`. `reconcile` replaces `Analyzer._reconcile_hlm`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from dftracer.utils import DataFrame, DType, Series, col

from ..analysis_utils import nan_as_null
from ..constants import VIEW_TYPES
from .expr import evaluate, expression_names
from .view import _as_lists

_SET_VALUES = "__values"
_TRACE_PRESENT = "_trace_present"
_SIGNATURE = "__signature"
HLM_EXTRA_COLS = ("cat", "io_cat", "acc_pat", "func_name")
HLM_SUM_COLS = ("time", "count", "size")


def _agg_for(name: str) -> str:
    if name.endswith("_call_min"):
        return "min"
    if name.endswith("_call_max"):
        return "max"
    return "sum"


def condition_masks(frame: DataFrame, conditions: Iterable[str]) -> dict[str, Series]:
    """One boolean mask per distinct preset condition, a missing value counting as false.

    The conditions are pandas eval strings, evaluated as `frames.expr.evaluate` does. A condition
    that names no column holds for every row or for none.
    """
    masks = {}
    for c in dict.fromkeys(conditions):
        mask = evaluate(frame, c)
        masks[c] = mask.fillna(False) if isinstance(mask, Series) else Series([bool(mask)] * frame.num_rows)
    return masks


def _columns_named(conditions: Iterable[str], columns: Iterable[str]) -> list[str]:
    """The names of `columns` that `conditions` read, in column order."""
    named = set()
    for condition in conditions:
        named.update(expression_names(condition))
    return [c for c in columns if c in named]


def _aggregate(names: Iterable[str], source: Mapping[str, str] | None = None) -> dict:
    """Group-by aggregations: the sum, min or max `_agg_for` picks for each name, read from `source[name]`."""
    source = source or {}
    return {name: getattr(col(source.get(name, name)), _agg_for(name))() for name in names}


def _unions(names: Iterable[str], source: Mapping[str, str] | None = None) -> dict:
    source = source or {}
    return {f"{_SET_VALUES}{n}": col(f"{_SET_VALUES}{source.get(n, n)}").set_union(typed=True) for n in names}


def _signatures(
    frame: DataFrame, needed: Sequence[str], conditions: Sequence[str]
) -> tuple[Series, dict[str, list[int]]]:
    """Per row, the id of the set of `conditions` its `needed` columns meet, and per condition the ids that meet it.

    An id stands for one distinct set of met conditions, so any number of conditions fits one int64.
    Each condition is evaluated once per distinct combination of the `needed` values, not per row.
    With no `needed` column the conditions are constants, evaluated on one row.
    """

    def ids(on: DataFrame) -> tuple[list[int], dict[str, list[int]]]:
        masks = condition_masks(on, conditions)
        seen: dict[tuple, int] = {}
        row_ids = [seen.setdefault(met, len(seen)) for met in zip(*(masks[c].to_list() for c in conditions))]
        return row_ids, {c: [i for met, i in seen.items() if met[b]] for b, c in enumerate(conditions)}

    if not needed:
        row_ids, hits = ids(frame.head(1))
        return Series(row_ids * frame.num_rows, dtype="int64"), hits
    # A group-by finds the distinct combinations faster than drop_duplicates; dropna=False keeps missing values.
    combos = frame.group_by(list(needed), dropna=False).agg(**{_SIGNATURE: col(needed[0]).count()})
    combos = combos.select(*needed)
    row_ids, hits = ids(combos)
    combos = combos.with_columns(**{_SIGNATURE: Series(row_ids, dtype="int64")})
    # A left join keeps the rows of the left frame in order, one row each, since combos has unique keys.
    joined = frame.select(*needed).join(combos, on=list(needed), how="left", nulls_equal=True)
    return joined[_SIGNATURE], hits


def _grouped(
    frame: DataFrame,
    keys: Sequence[str],
    needed: Sequence[str],
    base: Sequence[str],
    base_sets: Sequence[str],
    derived: Mapping[str, tuple[str, Mapping[str, str]]],
) -> DataFrame:
    """`frame` grouped by `keys`: the base columns, then each derived metric's columns.

    A derived metric reads the rows its condition meets: each metric groups, by the keys, the
    rows whose signature holds its condition.

    The filter is eager, not lazy: on glibc the lazy group-by leaves each pool thread's
    partial state in that thread's malloc arena, and the step peak rises by about 0.2 GB on
    5M rows. The eager filter copies the rows of one metric at a time, which costs less.
    """
    keys = list(keys)
    out = frame.group_by(keys).agg(**_aggregate(base), **_unions(base_sets))
    conditions = list(dict.fromkeys(c for c, names in derived.values() if names or base_sets))
    if not conditions:
        return out
    signature, hits_of = _signatures(frame, needed, conditions)
    frame = frame.select(*keys, *base, *[f"{_SET_VALUES}{c}" for c in base_sets])
    frame = frame.with_columns(**{_SIGNATURE: signature})
    del signature
    rows = out.select(*keys)
    for metric, (condition, names) in derived.items():
        sets = {f"{metric}_{c}": c for c in base_sets}
        if not (names or sets):
            continue
        part = frame.filter(frame[_SIGNATURE].is_in(hits_of[condition])).group_by(keys)
        part = part.agg(**_aggregate(names, names), **_unions(sets, sets))
        # The left join keeps the rows of `rows` in order, one each, so its columns line up with `out`.
        part = rows.join(part, on=keys, how="left", nulls_equal=True)
        out = out.with_columns(**{c: part[c] for c in part.columns if c not in keys})
        del part
    return out


def group_main_view(
    hlm: DataFrame,
    view_types: Sequence[str],
    time_sliced: bool,
    derived_metrics: Mapping[str, str] | None = None,
    size_derived_metrics: Iterable[str] = (),
) -> DataFrame:
    """The main view of an HLM: one row per `view_types` key, indexed by `view_types`.

    First the per-metric columns of `derived_metrics` (metric name to a pandas eval
    condition over the HLM keys), as the pandas path's `set_layer_metrics` (tests/pandas_reference.py)
    builds them: the metric's rows keep their value, the others go missing. Size columns are derived
    only for the metrics in `size_derived_metrics`. Then sums, mins and maxs by
    column name and set unions (LIST columns of sorted unique values) for the columns named after a
    view type the view does not group by, the bandwidth, intensity and ops columns, zero as missing
    and the view dtypes. A row with a missing key is left out. Sums can differ from the pandas
    path by float addition order.
    """
    keys = list(view_types)
    derived_metrics = dict(derived_metrics or {})
    size_derived = set(size_derived_metrics)
    diff = set(VIEW_TYPES).difference(keys)
    extra = set(HLM_EXTRA_COLS)
    frame = hlm.reset_index()
    columns = list(hlm.columns)
    base_sets = [c for c in columns if any(c.endswith(v) for v in diff)]
    base = [c for c in columns if c not in base_sets and c not in extra and c not in keys]
    needed = _columns_named(derived_metrics.values(), frame.columns)
    frame = frame.select(*dict.fromkeys([*keys, *needed, *base, *base_sets]))
    # A NaN key is missing, as in pandas, so the group-by leaves its rows out.
    frame = frame.with_columns(
        **{c: nan_as_null(frame[c]) for c in dict.fromkeys([*keys, *base])},
        **{f"{_SET_VALUES}{c}": _as_lists(frame[c]) for c in base_sets},
    )
    if base_sets:
        frame = frame.drop(columns=base_sets)

    derived: dict[str, tuple[str, dict[str, str]]] = {}
    for metric, condition in derived_metrics.items():
        names = {
            f"{metric}_{c}": c for c in base if not ((c == "size" or "size_bin" in c) and metric not in size_derived)
        }
        derived[metric] = (condition, names)

    out = _grouped(frame, keys, needed, base, base_sets, derived)
    del frame
    out = out.sort_values(keys)
    out = _set_main_metrics(out, keys)
    out = _finish(out, keys, time_sliced)
    out = out.rename({c: c[len(_SET_VALUES) :] for c in out.columns if c.startswith(_SET_VALUES)})
    return out.select(*keys, *sorted(c for c in out.columns if c not in keys)).set_index(keys)


def _finite_or_missing(values: Series) -> Series:
    if values.dtype != DType.FLOAT64:
        values = values.cast("float64")
    finite = values.is_finite()
    return values if finite.fillna(True).all() else values.where(finite)


def _map_columns(frame: DataFrame, names: Sequence[str], fn, batch: int = 8) -> DataFrame:
    """`frame` with `fn` applied to each column of `names`, a few columns at a time.

    A batch keeps at most `batch` new columns next to the old ones, so the frame never
    exists twice.
    """
    names = list(names)
    for i in range(0, len(names), batch):
        part = names[i : i + batch]
        frame = frame.with_columns(**{c: fn(frame[c]) for c in part})
    return frame


def _empty_lists(values: Series) -> Series:
    """A missing LIST cell as an empty list of the column's value type, as the pandas path gives an empty set."""
    if not values.null_count:
        return values
    n = len(values)
    item = values.child(0).dtype
    if item == DType.STRING:
        empty = Series([[]] * n)
    elif item == DType.BOOL or item.is_integer or item in (DType.FLOAT32, DType.FLOAT64):
        empty = Series([[]] * n, dtype=f"list<{item.name.lower()}>")
    else:
        # No typed empty list of this item type (dftracer-utils feat/analyzer-port-gaps).
        return Series([[] if v is None else v for v in values.to_list()])
    return values.where(values.is_not_null(), empty)


def _finish(frame: DataFrame, keys: Sequence[str], time_sliced: bool) -> DataFrame:
    """`frame` with zero as missing and the view dtypes, a missing set as an empty one."""
    names = [c for c in frame.columns if c not in keys]
    dtypes = _dtypes(names, time_sliced)

    for i in range(0, len(names), 8):
        done = {}
        for c in names[i : i + 8]:
            values = frame[c]
            if values.dtype == DType.LIST:
                done[c] = _empty_lists(values)
                continue
            # `where` costs far more than the scan, and most columns hold no zero.
            if (values == 0).any():
                values = values.where(values != 0)
            if c in dtypes and values.dtype.name.lower() != dtypes[c]:
                values = values.cast(dtypes[c])
            done[c] = values
        frame = frame.with_columns(**done)
        del done, values
    return frame


def _set_main_metrics(frame: DataFrame, keys: Sequence[str]) -> DataFrame:
    """`set_main_metrics` of the pandas path (tests/pandas_reference.py) on a frame."""
    names = [c for c in frame.columns if c not in keys]
    count_cols = [c for c in names if c.endswith("count")]
    size_cols = [c for c in names if c.endswith("size")]
    new: list[str] = []
    for size in size_cols:
        bw, count, intensity, time = (size.replace("size", s) for s in ("bw", "count", "intensity", "time"))
        frame = frame.with_columns(**{size: frame[size].where(frame[size] > 0)})
        frame = frame.with_columns(
            **{
                bw: (frame[size] / frame[time]).where(frame[size] > 0),
                intensity: (frame[count] / frame[size]).where(frame[size] > 0),
            }
        )
        new.extend([bw, intensity, time])
    for count in count_cols:
        ops, time = count.replace("count", "ops"), count.replace("count", "time")
        frame = frame.with_columns(**{ops: frame[count] / frame[time]})
        new.append(ops)
    new = list(dict.fromkeys(new))
    return _map_columns(frame, new, _finite_or_missing)


def _dtypes(names: Sequence[str], time_sliced: bool) -> dict[str, str]:
    """The dtype the pandas path's `fix_dtypes` (tests/pandas_reference.py) gives each column of `names` it types."""
    ints = [c for c in names if c.endswith("_nunique")]
    doubles = [c for c in names if c.endswith(("_bw", "_intensity", "_ops", "_time", "_slope", "_pct"))]
    sizes = [c for c in names if c.endswith("_size")]
    bins = [c for c in names if "_bin_" in c or c.endswith("_count")]
    (doubles if time_sliced else ints).extend(bins)
    dtype: dict[str, str] = {}
    for cols, name in ((ints, "int64"), (doubles, "float64"), (sizes, "int64")):
        dtype.update(dict.fromkeys(cols, name))
    return dtype


def reconcile(
    trace_hlm: DataFrame,
    profile_hlm: DataFrame,
    view_types: Sequence[str],
) -> tuple[DataFrame, int]:
    """The trace HLM plus the profile rows whose key it lacks, summed per key, indexed by the HLM keys.

    Returns the reconciled HLM and how many profile rows share a key with the
    trace HLM. A column sums to missing when it adds to zero, and bin columns are int32.
    """
    keys = list(dict.fromkeys([*view_types, *HLM_EXTRA_COLS]))
    trace, profile = trace_hlm.reset_index(), profile_hlm.reset_index()
    trace = trace.with_columns(
        **{
            k: trace[k].cast(profile[k].dtype)
            for k in keys
            if k in trace.columns and k in profile.columns and trace[k].dtype != profile[k].dtype
        }
    )
    combined_columns = list(dict.fromkeys([*trace.columns, *profile.columns]))
    bins = [c for c in combined_columns if "_bin_" in c]
    diff = set(VIEW_TYPES).difference(view_types)
    sets = [c for c in combined_columns if c in diff]
    sums = [c for c in (*HLM_SUM_COLS, *bins) if c in combined_columns]
    cells = [*keys, *sums, *[f"{_SET_VALUES}{c}" for c in sets]]

    def native(side: DataFrame) -> DataFrame:
        side = side.with_columns(
            **{c: nan_as_null(side[c]) for c in [*keys, *sums] if c in side.columns},
            **{f"{_SET_VALUES}{c}": _as_lists(side[c]) for c in sets if c in side.columns},
        )
        return side.select(*[c for c in cells if c in side.columns])

    t, p = native(trace), native(profile)
    present = t.select(*keys).drop_duplicates().with_columns(**{_TRACE_PRESENT: 1})
    joined = p.join(present, on=keys, how="left", nulls_equal=True)
    seen = joined[_TRACE_PRESENT].is_not_null()
    overlap = int(seen.sum()) if len(joined) else 0
    # A diagonal concat gives a column one side lacks the type of the other side, as missing values.
    combined = t.concat(joined.filter(~seen).drop(columns=[_TRACE_PRESENT]), how="diagonal").select(*cells)
    aggs = {c: col(c).sum() for c in sums}
    aggs.update(_unions(sets))
    frame = combined.group_by(keys).agg(**aggs).sort_values(keys)
    frame = frame.with_columns(**{c: frame[c].where(frame[c] != 0) for c in sums})
    frame = frame.with_columns(**{c: frame[c].cast("int32") for c in bins})
    frame = frame.rename({f"{_SET_VALUES}{c}": c for c in sets})
    return frame.set_index(keys), overlap
