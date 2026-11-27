"""The typed read of the aggregation index, mapped to the DFAnalyzer frames."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, TypedDict

from dftracer.utils import DataFrame, Series
from dftracer.utils.trace import IndexRecords

from .analysis_utils import floor_divide, frame_nbytes
from .cluster import gather_within_budget
from .trace_rules import drop_matching_files

if TYPE_CHECKING:
    from dask.distributed import Future

    from .trace_index import TraceRun


# View group_by + agg that produces every dfanalyzer metric column below.
_VIEW_METRIC_AGGS = (
    "count",
    "sum:dur",
    "sum:size",
    "sumsq:dur",
    "sumsq:size",
    "min:dur",
    "max:dur",
    "min:size",
    "max:size",
    "min:ts",
    "max:te",  # te = ts + dur, so max:te is the group's end time
)


# Full-grain group_by for the typed read: the dims the HLM groups/derives on.
# Resolved file_name/host_name (not raw hashes) since the HLM uses names for the
# POSIX-category rules and proc_name.
# file_path (full resolved path), not file_name (basename): dfanalyzer's
# file_name column is the full path and its POSIX cat rules match path
# substrings like "/data".
_TYPED_GROUP_KEYS = (
    "cat",
    "name",
    "pid",
    "tid",
    "fhash",
    "hhash",
    "file_path",
    "host_name",
    "io_cat",
)
_TYPED_AGGS = _VIEW_METRIC_AGGS + ("min:offset", "max:offset")


def typed_group_keys(file_buckets: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Group keys for the typed read.

    With no `file_buckets` this is the per-file grain. Given substrings,
    `file_path` folds to whichever one it contains (and the redundant `fhash`
    key drops out), which collapses per-file rows while leaving the caller's
    file-name substring rules matchable on the folded row.
    """
    if not file_buckets:
        return _TYPED_GROUP_KEYS
    args = ", ".join("'{}'".format(b.replace("'", "")) for b in file_buckets)
    return tuple(f"bucket(file_path, {args})" if k == "file_path" else k for k in _TYPED_GROUP_KEYS if k != "fhash")


@dataclass(frozen=True)
class ReadSettings:
    """How `read_frames` groups, buckets and filters the index records.

    `group_keys` is the per-file grain by default; pass `typed_group_keys(...)`
    to fold files. Rows of files matching `drop_file_patterns` are dropped on
    the workers. `profile_columns` maps each profile column to its dtype name.
    """

    time_granularity: float = 1.0
    time_resolution: float = 1e6
    profile_time_granularity: float = 1.0
    profile_columns: Mapping[str, str] = field(default_factory=dict)
    group_keys: tuple[str, ...] = _TYPED_GROUP_KEYS
    drop_file_patterns: tuple[str, ...] = ()


def _nulls(n: int, dtype: str) -> Series:
    return Series([None] * n, dtype=dtype)


def join_proc_name(host: Series, pid: Series, tid: Series) -> Series:
    """``app#<host>#<pid>#<tid>`` per row."""
    prefix = Series(["app"] * len(host), dtype="string")
    return prefix.str.cat(host, sep="#").str.cat(pid.astype("string"), sep="#").str.cat(tid.astype("string"), sep="#")


def proc_names(df: DataFrame, host_hash: str) -> Series:
    """The proc name of each row; the host is `host_name`, else the column `host_hash`, else ``unknown``.

    An empty host name counts as missing, as in the pandas read path.
    """
    n = df.num_rows
    host = df["host_name"].astype("string") if "host_name" in df.columns else _nulls(n, "string")
    if host_hash in df.columns:
        host = host.mask(host.fillna("") == "", df[host_hash].astype("string"))
    return join_proc_name(host.fillna("unknown").replace("", "unknown"), df["pid"], df["tid"])


def compose_proc_name(df: DataFrame) -> Series:
    """The proc name of each event row, from `host_name` or the event host hash `hhash`."""
    return proc_names(df, "hhash")


def _typed_event_frame(
    df: DataFrame,
    time_resolution: float,
    bucket_us: int = 0,
    is_profile: bool = False,
) -> DataFrame | None:
    """Map one collect_index_records table (regular or aggregated) to the dfanalyzer
    event/profile schema, scaling durations by `time_resolution`. None when empty.

    Time columns stay absolute: time_range is the bucket index time_bucket//bucket_us;
    events keep their precise (ts, te) span, while profile rows align to the bucket
    grid (start = bucket, end = start + bucket_us). The client makes them relative."""
    if "count" not in df.columns or df.num_rows == 0:
        return None
    n = df.num_rows
    out = DataFrame.from_dict({"cat": df["cat"].str.lower() if "cat" in df.columns else Series([""] * n)})
    out["func_name"] = df["name"] if "name" in df.columns else _nulls(n, "string")
    for c in ("pid", "tid", "io_cat"):
        if c in df.columns:
            out[c] = df[c].astype("int64")
    # Keep empty strings (not null) for unresolved names, matching the C++ scan;
    # downstream string ops (posix category, proc_name split) are not null-safe.
    # file_name/file_hash are emitted even when folded away: later stages expect them.
    if "file_path" in df.columns:
        out["file_name"] = df["file_path"].astype("string").fillna("")
    else:
        out["file_name"] = Series([""] * n, dtype="string")
    if "host_name" in df.columns:
        out["host_name"] = df["host_name"].astype("string").fillna("")
    if "fhash" in df.columns:
        out["file_hash"] = df["fhash"].astype("string")
    else:
        out["file_hash"] = Series([""] * n, dtype="string")
    if "hhash" in df.columns:
        out["host_hash"] = df["hhash"].astype("string")
    if "pid" in df.columns and "tid" in df.columns:
        out["proc_name"] = compose_proc_name(df)
    # epoch/step (or any aggregated extra key) ride through as-is.
    for c in df.columns:
        if c.startswith("arg_"):
            out[c[len("arg_") :]] = df[c]
    tr = float(time_resolution)
    out["count"] = df["count"].astype("int64")
    out["time"] = df["sum_dur"].astype("float64") / tr
    out["size"] = df["sum_size"].astype("int64")
    out["time_sq"] = df["sumsq_dur"].astype("float64") / (tr * tr)
    # float64: sum-of-squared-sizes overflows int53 for large transfers (and it is a float metric).
    out["size_sq"] = df["sumsq_size"].astype("float64")
    out["time_min"] = df["min_dur"].astype("float64") / tr
    out["time_max"] = df["max_dur"].astype("float64") / tr
    # A group with no transfer size has no minimum or maximum (null from the engine).
    out["size_min"] = df["min_size"].astype("int64")
    out["size_max"] = df["max_size"].astype("int64")
    # Folded-shape per-call columns the distributed HLM combines (min/max across
    # keys) instead of deriving; the derive path seeds them from the group total.
    out["time_call_min"] = out["time"]
    out["time_call_max"] = out["time"]
    out["size_call_min"] = out["size"]
    out["size_call_max"] = out["size"]
    if "min_offset" in df.columns:
        out["offset_min"] = df["min_offset"].astype("int64")
        out["offset_max"] = df["max_offset"].astype("int64")
    out["acc_pat"] = 0
    out["file_nunique"] = 0
    # UINT64 times are cast before any subtraction, which would wrap below zero.
    if "time_bucket" in df.columns:
        bucket = df["time_bucket"].astype("int64")
        out["time_bucket"] = bucket
        out["time_range"] = floor_divide(bucket, bucket_us) if bucket_us else Series([0] * n)
        if is_profile:
            out["time_start"] = bucket
            out["time_end"] = bucket + int(bucket_us)
        else:
            out["time_start"] = df["min_ts"].astype("int64")
            out["time_end"] = df["max_te"].astype("int64")
    else:
        out["time_range"] = 0
        out["time_start"] = df["min_ts"].astype("int64")
        out["time_end"] = df["max_te"].astype("int64")
    return out


def _records_to_frames(
    records: IndexRecords,
    time_resolution: float,
    bucket_us: int,
    drop_file_patterns: tuple[str, ...],
) -> dict[str, DataFrame | None]:
    events = _typed_event_frame(records.regular, time_resolution, bucket_us, False)
    if events is not None:
        events = drop_matching_files(events, drop_file_patterns)
    return {
        "events": events if events is not None and events.num_rows else None,
        "profiles": _typed_event_frame(records.aggregated, time_resolution, bucket_us, True),
        "system": records.counters if records.counters.num_rows else None,
    }


class ReadFrames(TypedDict):
    """Decoded read-path frames: traces (always present), profiles/system (or
    None), and the origins the caller needs to build a ReadTraceResult."""

    traces: DataFrame
    profiles: DataFrame | None
    system: DataFrame | None
    time_origin: int | None
    system_time_origin: int | None


def concat_frames(frames: list[DataFrame]) -> DataFrame | None:
    """The frames stacked in order; columns missing from a part are null."""
    if not frames:
        return None
    first, rest = frames[0], frames[1:]
    if not rest:
        return first
    how = "vertical" if all(f.columns == first.columns for f in rest) else "diagonal"
    return first.concat(*rest, how=how)


def _build_read_frames(
    results: list[dict[str, DataFrame | None]],
    profile_output_columns: Mapping[str, str],
    time_granularity: float,
    time_resolution: float,
    profile_time_granularity: float,
) -> ReadFrames:
    """Join per-shard ``{events, profiles, system}`` frames (from the View typed
    read) into the dfanalyzer frames.

    The View emits absolute time; the returned frames are relativized to the
    global minimum ``time_bucket`` (``time_origin``), keeping values small for
    the checkpoint round-trip. ``traces`` is a typed-empty frame when there are
    no events, so downstream arithmetic works.
    """

    def _parts(key: str) -> list[DataFrame]:
        return [r[key] for r in results if r and r.get(key) is not None and r[key].num_rows]

    events = concat_frames(_parts("events"))
    raw_prof = concat_frames(_parts("profiles"))

    buckets = [int(f["time_bucket"].min()) for f in (events, raw_prof) if f is not None and "time_bucket" in f.columns]
    origin = min(buckets) if buckets else None
    bucket_us = int(time_granularity * time_resolution)

    def _relativize(f: DataFrame) -> None:
        if origin is None:
            return
        f["time_start"] = f["time_start"] - origin
        f["time_end"] = f["time_end"] - origin
        if bucket_us and "time_bucket" in f.columns:
            f["time_range"] = floor_divide(f["time_bucket"] - origin, bucket_us)

    if events is not None:
        _relativize(events)
        traces = events
    else:
        traces = DataFrame.from_dict({c: Series([], dtype=dt) for c, dt in profile_output_columns.items()})

    profiles: DataFrame | None = None
    if raw_prof is not None:
        _relativize(raw_prof)
        window = int(profile_time_granularity * time_resolution)
        raw_prof = raw_prof.drop(columns=["time_end"])
        profiles = _coerce_profile_dtypes(raw_prof, profile_output_columns, profile_window=window)

    system = concat_frames(_parts("system"))
    sys_origin: int | None = None
    if system is not None:
        system["ts"] = system["time_bucket"].astype("int64")
        sys_origin = int(system["ts"].min())

    return {
        "traces": traces,
        "profiles": profiles,
        "system": system,
        "time_origin": origin,
        "system_time_origin": sys_origin,
    }


def _coerce_profile_dtypes(
    df: DataFrame,
    output_columns: Mapping[str, str],
    profile_window: float | None = None,
) -> DataFrame:
    """Normalize C++ aggregator profile output to the `output_columns` schema.

    `output_columns` maps column name -> dtype name. When `profile_window` is
    given, `time_end` is derived as `time_start + profile_window`.
    """
    if df.num_rows == 0:
        return df
    df = df.copy()
    for col, dtype in output_columns.items():
        if col not in df.columns:
            df[col] = _nulls(df.num_rows, dtype)
        elif dtype == "string":
            df[col] = df[col].astype("string").replace("", None)
        else:
            df[col] = df[col].astype(dtype)
    # A profile with no transfer size aggregates to 0; dfanalyzer treats absent
    # size as null (a 0 would skew bandwidth), matching the old scan's null.
    for col in ("size", "size_min", "size_max"):
        if col in df.columns:
            df[col] = df[col].replace(0, None)
    if profile_window is not None:
        df["time_end"] = df["time_start"] + int(profile_window)
    return df


def submit_typed_read(run: TraceRun, settings: ReadSettings) -> list[Future[dict[str, DataFrame | None]]]:
    """The read as one ungathered future per worker of `run`.

    The aggregation tier is index-wide, so each worker reads a disjoint shard
    range of it over all files and returns ``{events, profiles, system}`` as
    frames (or None) with absolute time. Per-shard progress is published under the
    "Reading traces" phase for a `ProgressAggregator`.
    """
    bucket_us = int(settings.time_granularity * settings.time_resolution)
    view = run.viewer().group_by(*settings.group_keys).time_bucket(bucket_us).agg(*_TYPED_AGGS)
    to_frames = partial(
        _records_to_frames,
        time_resolution=settings.time_resolution,
        bucket_us=bucket_us,
        drop_file_patterns=settings.drop_file_patterns,
    )
    # The public index_record_futures converts the records to pyarrow before map_fn
    # (dftracer-utils branch feat/analyzer-port-gaps: public native index records), so take
    # the native records and map them in a task that runs beside them.
    records = view._index_record_futures(None, "Reading traces", native=True)
    return [run.client.submit(to_frames, r, pure=False) for r in records]


def _frames_nbytes(results: list[dict[str, DataFrame | None]]) -> int:
    return sum(frame_nbytes(f) for r in results if r for f in r.values() if f is not None)


def _on_disk_bytes(files: tuple[str, ...]) -> int:
    total = 0
    for path in files:
        try:
            total += os.path.getsize(path)
        except OSError:
            pass
    return total


def read_frames(run: TraceRun, settings: ReadSettings) -> ReadFrames:
    """Read the index of `run` into the DFAnalyzer frames, on the client of `run`.

    On an auto client the read runs in process when its result fits the
    budget, and again on a new cluster when it does not.
    """
    results = gather_within_budget(
        run.client, lambda: submit_typed_read(run, settings), lambda: _on_disk_bytes(run.files), _frames_nbytes
    )
    return _build_read_frames(
        results,
        settings.profile_columns,
        settings.time_granularity,
        settings.time_resolution,
        settings.profile_time_granularity,
    )
