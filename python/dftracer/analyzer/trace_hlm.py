"""The high-level metrics (HLM) as one aggregation over the trace index."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from dftracer.utils import DataFrame, Series

from .analysis_utils import floor_divide
from .trace_index import TraceRun
from .trace_rules import TraceRules, apply_cat_suffix, drop_matching_files
from .typed_read import compose_proc_name, concat_frames

# file_name is the DFAnalyzer full path, so it maps to file_path.
_HLM_DIM_TO_KEY = {
    "file_name": "file_path",
    "host_name": "host_name",
    "cat": "cat",
    "func_name": "name",
    "io_cat": "io_cat",
    "pid": "pid",
    "tid": "tid",
}
_HLM_METRIC_AGGS = ("count", "sum:dur", "sum:size", "sumsq:dur", "sumsq:size")
_METRIC_COLS = (
    "time",
    "count",
    "size",
    "time_sq",
    "size_sq",
    "time_call_min",
    "time_call_max",
    "size_call_min",
    "size_call_max",
)
_INT_COLS = frozenset({"count", "size", "size_call_min", "size_call_max"})
_INT_INDEX = frozenset({"time_range", "io_cat", "acc_pat", "pid", "tid"})


@dataclass(frozen=True)
class HLMConfig:
    """The HLM domain rules and time settings, supplied by the analyzer.

    `time_origin` is the smallest ``time_bucket`` of the read, so the trace and profile
    HLMs number their time ranges from the same bucket. None uses the frame's own smallest bucket.
    """

    rules: TraceRules = field(default_factory=TraceRules)
    time_granularity: float = 1.0
    time_resolution: float = 1e6
    time_origin: int | None = None


def _ignored_func_predicate(rules: TraceRules) -> str | None:
    clauses: list[str] = []
    if rules.ignored_func_names:
        names = ", ".join(f'"{n.replace(chr(34), "")}"' for n in rules.ignored_func_names)
        clauses.append(f"name not in [{names}]")
    if rules.ignored_func_patterns:
        pat = "|".join(rules.ignored_func_patterns).replace('"', "")
        clauses.append(f'name !~ "{pat}"')
    return " and ".join(clauses) if clauses else None


def hlm(run: TraceRun, view_types: Sequence[str], cfg: HLMConfig, *, profiles: bool = False) -> DataFrame:
    """The HLM of the events, or of the profiles with `profiles`, grouped by `view_types`.

    Returns a frame indexed by `view_types` plus ``cat, io_cat, acc_pat,
    func_name`` with nullable metric columns. The read runs on the run's client
    and is gathered here.
    """
    vts = list(view_types)
    df = _hlm_frame(run, vts, cfg, "aggregated" if profiles else "regular")
    return _to_hlm(df, vts, cfg)


def _hlm_frame(run: TraceRun, vts: list[str], cfg: HLMConfig, family: str) -> DataFrame | None:
    temporal = "time_range" in vts
    bucket_us = int(cfg.time_granularity * cfg.time_resolution)

    keys = [_HLM_DIM_TO_KEY[v] for v in vts if v in _HLM_DIM_TO_KEY]
    if "proc_name" in vts:
        keys += ["pid", "tid", "host_name", "hhash"]
    keys += ["cat", "io_cat", "name"]
    if "file_path" not in keys:
        # Ignore patterns first so an ignored file folds to its pattern (dropped
        # below); cat rules follow so a kept file folds to its suffix substring.
        subs = list(cfg.rules.ignored_file_patterns) + [p for p, _ in cfg.rules.posix_cat_rules]
        if subs:
            args = ", ".join(f"'{s.replace(chr(39), '')}'" for s in subs)
            keys.append(f"bucket(file_path, {args})")

    aggs: list[str] = list(_HLM_METRIC_AGGS)

    viewer = run.viewer()
    v = viewer.phase("events") if family == "regular" else viewer
    pred = _ignored_func_predicate(cfg.rules)
    if pred:
        v = v.filter(pred)
    agg_view = v.group_by(*keys)
    if temporal:
        agg_view = agg_view.time_bucket(bucket_us)
    if family == "regular":
        df = agg_view.agg(*aggs).collect()
    else:
        # Private native route: the public collect_index_records gives pyarrow tables
        # (dftracer-utils branch feat/analyzer-port-gaps: public native index records).
        parts = run.client.gather(agg_view.agg(*aggs)._index_record_futures(None, None, native=True))
        df = concat_frames([getattr(p, family) for p in parts])
        if df is None:
            return None

    # The file-ignore runs after the aggregation because file_path is resolved
    # only then. An ignored file folds to its own pattern, so the drop holds
    # for a full path and for a fold.
    return drop_matching_files(df, cfg.rules.ignored_file_patterns, "file_path")


def _to_hlm(df: DataFrame | None, vts: Sequence[str], cfg: HLMConfig) -> DataFrame:
    idx_names = list(vts) + ["cat", "io_cat", "acc_pat", "func_name"]
    if df is None or "count" not in df.columns or df.num_rows == 0:
        empty = {n: Series([], dtype="int64" if n in _INT_INDEX else "string") for n in idx_names}
        empty.update({c: Series([], dtype="int64" if c in _INT_COLS else "float64") for c in _METRIC_COLS})
        return DataFrame.from_dict(empty).set_index(idx_names)
    tr = float(cfg.time_resolution)
    out: dict[str, Series] = {}

    for v in vts:
        if v == "proc_name":
            out[v] = compose_proc_name(df)
        elif v == "time_range":
            bucket_us = int(cfg.time_granularity * cfg.time_resolution)
            # UINT64 buckets are cast before the subtraction, which would wrap below zero.
            bucket = df["time_bucket"].astype("int64")
            origin = bucket.min() if cfg.time_origin is None else cfg.time_origin
            out[v] = floor_divide(bucket - origin, bucket_us)
        else:
            out[v] = df[_HLM_DIM_TO_KEY[v]]

    n = df.num_rows
    path = df["file_path"].astype("string") if "file_path" in df.columns else Series([""] * n, dtype="string")
    out["cat"] = apply_cat_suffix(df["cat"].astype("string"), path, cfg.rules.posix_cat_rules)
    out["io_cat"] = df["io_cat"].astype("int64")
    out["func_name"] = df["name"]
    out["acc_pat"] = Series([0] * n)

    # 0 -> null so a 0-size open does not skew bandwidth. The _call_min/max
    # columns seed from the group total, not per-event extremes.
    time = (df["sum_dur"].astype("float64") / tr).replace(0, None)
    size = df["sum_size"].astype("int64").replace(0, None)
    out["time"] = time
    out["count"] = df["count"].astype("int64").replace(0, None)
    out["size"] = size
    out["time_sq"] = (df["sumsq_dur"].astype("float64") / (tr * tr)).replace(0, None)
    out["size_sq"] = df["sumsq_size"].astype("float64").replace(0, None)
    out["time_call_min"] = time
    out["time_call_max"] = time
    out["size_call_min"] = size
    out["size_call_max"] = size

    return DataFrame.from_dict(out).set_index(idx_names)
