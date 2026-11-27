import math
import os
from typing import List, Optional

import structlog
from dask.distributed import Client, get_client
from dftracer.utils import DataFrame, Series, col
from dftracer.utils.index import Indexer
from dftracer.utils.dask import (
    ProgressAggregator,
    register_auto_thread_plugin,
)

from .analysis_utils import floor_divide
from .analyzer import Analyzer, PartitionedRecords
from .frames.expr import query
from .frames.view import aggregate_view
from .trace_hlm import HLMConfig, hlm
from .trace_index import TraceIndex, TraceRun, aggregation_config
from .trace_rules import DEFAULT_TRACE_RULES, TraceRules
from .typed_read import (
    ReadFrames,
    ReadSettings,
    proc_names,
    read_frames,
    typed_group_keys,
)
from .constants import (
    COL_ACC_PAT,
    COL_COUNT,
    COL_FILE_NAME,
    COL_FUNC_NAME,
    COL_HOST_NAME,
    COL_IO_CAT,
    COL_PROC_NAME,
    COL_SIZE,
    COL_TIME,
    COL_TIME_END,
    COL_TIME_RANGE,
    COL_TIME_START,
)
from .types import ReadTraceResult, ViewType
from .utils.log_utils import current_progress, log_block

logger = structlog.get_logger()

PROFILE_OUTPUT_COLUMNS = {
    "cat": "string",
    COL_FUNC_NAME: "string",
    "pid": "Int64",
    "tid": "Int64",
    "epoch": "Int64",
    "step": "Int64",
    "file_hash": "string",
    "host_hash": "string",
    COL_FILE_NAME: "string",
    COL_HOST_NAME: "string",
    COL_PROC_NAME: "string",
    COL_IO_CAT: "Int8",
    COL_ACC_PAT: "Int8",
    COL_COUNT: "Int64",
    COL_TIME: "float64",
    COL_SIZE: "Int64",
    "time_min": "float64",
    "time_max": "float64",
    "size_min": "Int64",
    "size_max": "Int64",
    "offset_min": "Int64",
    "offset_max": "Int64",
    COL_TIME_RANGE: "Int64",
    COL_TIME_START: "Int64",
    COL_TIME_END: "Int64",
}
SYSTEM_OUTPUT_COLUMNS = {
    "host_hash": "string",
    COL_TIME_RANGE: "Int64",
    "sys_cpu_iowait_pct": "float64",
    "sys_cpu_user_pct": "float64",
    "sys_cpu_system_pct": "float64",
    "sys_cpu_idle_pct": "float64",
    "sys_core_iowait_pct_max": "float64",
    "sys_core_iowait_pct_p95": "float64",
    "sys_mem_dirty": "float64",
    "sys_mem_cached": "float64",
    "sys_mem_available": "float64",
}


_P95 = 0.95


def _numpy_linear_quantile(lower: Series, higher: Series, count: Series, posinf: Series, q: float) -> Series:
    """The linear `q` quantile per group as numpy computes it, from the group's
    ``lower`` and ``higher`` order statistics, value count and +inf count. NaN becomes null.

    The native linear quantile gives inf where numpy gives NaN (inf - inf)
    (dftracer-utils branch feat/analyzer-port-gaps: quantile with infinities).
    """
    virtual = (count - 1) * q
    top = (virtual >= count - 1).fillna(False)
    # numpy interpolates toward the order statistic after floor(virtual) even when
    # virtual is whole, where "higher" stops at floor(virtual); only an inf there matters.
    whole = ((virtual == virtual.floor()) & ~top).fillna(False)
    higher = higher.mask(whole & (posinf >= count - virtual - 1).fillna(False), math.inf)
    gamma = (virtual - virtual.floor()).mask(top, virtual + 1)
    diff = higher - lower
    lerp = (lower + diff * gamma).mask((gamma >= 0.5).fillna(False), higher - diff * (1 - gamma))
    return lerp.fill_nan(None)


class DFTracerAnalyzer(Analyzer):
    rules: TraceRules = DEFAULT_TRACE_RULES

    def __init__(
        self,
        preset,
        assign_epochs: bool = False,
        trace_groups: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(preset, **kwargs)
        self.assign_epochs = assign_epochs
        self.trace_groups = list(trace_groups) if trace_groups else None

    def _index_client(self) -> Client | None:
        """The client to build the index on, or None to build it in this process."""
        client = getattr(self, "dask_client", None)
        client = getattr(client, "_impl", client)
        return client if isinstance(client, Client) else None

    def analyze_trace(self, trace_path, *args, **kwargs):
        """Transparent indexing: ensure the dftracer index exists, then analyze."""
        import time

        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
        )

        from .utils.log_utils import console

        t0 = time.time()
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as _bar:
            _task = _bar.add_task("Indexing", total=None)
            _phase = {"cur": "Indexing"}

            def _on_progress(done: int, total: int, phase: str) -> None:
                # One bar per phase; reset on a phase change so a 0 total pulses
                # instead of showing the previous phase's frozen total.
                if phase != _phase["cur"]:
                    _phase["cur"] = phase
                    _bar.reset(_task, total=(total or None), description=phase)
                _bar.update(_task, completed=done, total=(total or None), description=phase)

            TraceIndex.resolve(trace_path, self.trace_groups).ensure(
                self.time_granularity * 1000.0,
                client=self._index_client(),
                progress=_on_progress,
            )
        console.print(f"[green]✓[/green] Build index [i]({time.time() - t0:.3f}s)[/i]")
        return super().analyze_trace(trace_path, *args, **kwargs)

    def _read_settings(self, group_by_file: bool) -> ReadSettings:
        if group_by_file:
            group_keys, drop = typed_group_keys(), ()
        else:
            # Ignore patterns bucket first: a file matching both must fold to
            # the ignore pattern or the drop cannot see it.
            ignored = self.rules.ignored_file_patterns
            folds = tuple(path for path, _ in self.rules.posix_cat_rules)
            group_keys, drop = typed_group_keys((*ignored, *folds)), ignored
        return ReadSettings(
            time_granularity=self.time_granularity,
            time_resolution=self.time_resolution,
            profile_time_granularity=self.profile_time_granularity,
            profile_columns=PROFILE_OUTPUT_COLUMNS,
            group_keys=group_keys,
            drop_file_patterns=drop,
        )

    def _read_result(self, frames: ReadFrames) -> ReadTraceResult:
        """The read frames as a result, with the system metrics standardized."""
        self._time_origin = frames["time_origin"]
        profiles = frames["profiles"]

        system_metrics = None
        if frames["system"] is not None:
            system_metrics = self._standardize_system(
                frames["system"],
                time_origin=frames["system_time_origin"],
                time_granularity=self.time_granularity,
                time_resolution=self.time_resolution,
            )

        return ReadTraceResult(
            traces=frames["traces"],
            profiles=profiles,
            profile_time_granularity=self.profile_time_granularity if profiles is not None else None,
            system_metrics=system_metrics,
        )

    def read_trace_local(self, trace_path, extra_columns=None, extra_columns_fn=None, group_by_file=True):
        """In-process read: the same View-based path as read_trace, driven by a
        NullClient so the scan runs in this process instead of on cluster workers.
        One read path keeps the schema identical to the distributed read."""
        from .cluster import NullClient

        return self.read_trace(trace_path, extra_columns, extra_columns_fn, group_by_file, client=NullClient())

    def read_trace(
        self,
        trace_path,
        extra_columns=None,
        extra_columns_fn=None,
        group_by_file=True,
        client: Optional[Client] = None,
    ):
        """Read trace using C++ aggregation pipeline.

        Uses the active Dask client to scan the trace across workers. The
        grouped traces, profiles and system metrics come back as library frames.

        Args:
            trace_path: Directory containing trace files or glob pattern.
            extra_columns: Not used (kept for API compatibility).
            extra_columns_fn: Not used (kept for API compatibility).
            client: Dask distributed Client. If None, uses the active client.

        Returns:
            ReadTraceResult with traces, profiles, and system_metrics.
        """
        with log_block("distributed_setup"):
            time_interval_ms = self.time_granularity * 1000.0
            register_auto_thread_plugin()

            trace_index = TraceIndex.resolve(trace_path, self.trace_groups)
            if trace_index.is_empty:
                if os.path.isdir(trace_path):
                    raise FileNotFoundError(
                        f"No .pfw or .pfw.gz files found directly in {trace_path} (subdirectories are not searched)."
                    )
                raise FileNotFoundError("No matching .pfw or .pfw.gz files found.")

        with log_block("cpp_indexer"):
            indexer = Indexer(
                trace_index.files if isinstance(trace_index.files, str) else list(trace_index.files),
                index_dir=trace_index.index_dir,
                require_checkpoint=True,
                require_bloom=True,
                aggregation_config=aggregation_config(time_interval_ms),
            )
            status = indexer.ensure()

            if status.total_files == 0:
                indexer.close()
                raise FileNotFoundError(
                    f"No .pfw or .pfw.gz files found directly in {trace_path} (subdirectories are not searched)."
                )

        with log_block("query_file_info"):
            file_pids = indexer.query_file_info()
            index_path = os.path.abspath(status.index_dir)
            indexer.close()

        with log_block("dask_client_connect"):
            # Prefer the client the analyzer already resolved: a real one, or the
            # `NullClient` that `cluster=none` installs. Everything below is written
            # against the client interface, so it runs unchanged either way -- with
            # NullClient there is exactly one "worker", this process, and `submit`
            # executes inline.
            dask_client = client or getattr(self, "dask_client", None)
            if dask_client is None:
                try:
                    dask_client = get_client()
                except ValueError:
                    dask_client = None

            if dask_client is None:
                return self.read_trace_local(trace_path, group_by_file=group_by_file)

        # Feed per-shard scan progress into the enclosing "Read trace & stats"
        # bar (published by the base analyzer's console_progress_block).
        _scan_bar = current_progress()

        def _feed_scan(done, total, phase):
            if _scan_bar is not None and phase == "Reading traces":
                _scan_bar(done, total)

        self._run = TraceRun(tuple(file_pids), index_path, dask_client)
        with ProgressAggregator(dask_client, _feed_scan), log_block("read_frames"):
            frames = read_frames(self._run, self._read_settings(group_by_file))
        return self._read_result(frames)

    def _compute_high_level_metrics(self, traces, view_types):
        return self._view_hlm(view_types)

    def _view_hlm(self, view_types, profiles=False):
        """Events (or, with profiles=True, profile) HLM."""
        cfg = HLMConfig(
            rules=self.rules,
            time_granularity=self.time_granularity,
            time_resolution=self.time_resolution,
            time_origin=getattr(self, "_time_origin", None),
        )
        return hlm(self._run, view_types, cfg, profiles=profiles)

    def _compute_profile_hlm(self, profiles, view_types):
        if profiles is None:
            return None
        return self._view_hlm(view_types, profiles=True)

    def _compute_view(self, layer, records, view_key, view_type, view_types):
        if isinstance(records, PartitionedRecords):
            return aggregate_view(
                records.frame,
                view_type,
                view_types,
                self.logical_views,
                layer,
                self.time_sliced,
                client=records.client,
                npartitions=records.npartitions,
            )
        return aggregate_view(records, view_type, view_types, self.logical_views, layer, self.time_sliced)

    def postread_trace(
        self,
        traces: DataFrame,
        view_types: List[ViewType],
    ) -> DataFrame:
        with log_block("filter_rows"):
            traces = self.rules.drop_ignored(traces)

        with log_block("assign_epochs"):
            if self.assign_epochs:
                if "epoch" not in self.preset.layer_defs:
                    raise ValueError("Epoch layer definition is missing")
                epochs = query(traces, self.preset.layer_defs["epoch"])
                epochs = epochs.sort_values(["pid", "time_start"]).reset_index(drop=True)
                # cumcount of an empty frame has no column (dftracer-utils branch feat/analyzer-port-gaps).
                rank = epochs.groupby("pid").cumcount()["cumcount"] if epochs.num_rows else Series([], dtype="int64")
                epochs["epoch"] = rank.astype("int64") + 1
                epoch_boundaries = epochs[["pid", "time_start", "time_end", "epoch"]]
                traces = self._set_epochs(traces, epoch_boundaries=epoch_boundaries)

        with log_block("set_basic_columns"):
            # A scalar on an empty frame gives a string or int8 column (dftracer-utils feat/analyzer-port-gaps).
            traces[COL_ACC_PAT] = Series([0] * traces.num_rows, dtype="int64")
            if COL_COUNT not in traces.columns:
                traces[COL_COUNT] = Series([1] * traces.num_rows, dtype="int64")

        traces = self._set_proc_names(traces)
        traces = self.rules.suffix_posix_cat(traces)
        return self._sanitize_size_offset(traces)

    def get_job_time(self, traces):
        return super().get_job_time(traces) / self.time_resolution

    def get_time_boundary_layer(self):
        if self.assign_epochs:
            return "epoch"
        return super().get_time_boundary_layer()

    def get_total_event_count(self, traces: DataFrame) -> int:
        return int(traces[COL_COUNT].sum())

    def get_unique_file_count(self, traces: DataFrame):
        # From the index, so it stays exact no matter what grain the read
        # folded to; summing the frame's per-group counts would instead count a
        # file once per group it appears in. Ignored patterns must be passed or
        # the count includes files the analysis dropped.
        run = getattr(self, "_run", None)
        if run is not None:
            return run.count_files(self.rules.ignored_file_patterns)
        file_hash = traces["file_hash"]
        return file_hash.filter((file_hash != "").fillna(False)).nunique()

    def get_unique_host_count(self, traces: DataFrame):
        return traces["host_hash"].nunique()

    def get_unique_process_count(self, traces: DataFrame):
        return traces["pid"].nunique()

    @staticmethod
    def _standardize_system(
        df: DataFrame,
        time_origin: int,
        time_granularity: float,
        time_resolution: float,
    ) -> DataFrame:
        """Aggregate raw system events into per-time_range system metric rows."""
        empty = DataFrame.from_dict({c: Series([], dtype=dtype) for c, dtype in SYSTEM_OUTPUT_COLUMNS.items()})
        if df.num_rows == 0:
            return empty

        df = df.copy()
        bucket_width_us = int(time_granularity * time_resolution)
        df[COL_TIME_RANGE] = floor_divide(df["ts"] - time_origin, bucket_width_us)
        group_keys = ["host_hash", COL_TIME_RANGE]

        def _agg(rows: DataFrame, aggs: list) -> DataFrame | None:
            present = [(m, op, out) for m, op, out in aggs if m in rows.columns]
            if rows.num_rows == 0 or not present:
                return None
            # A NaN marks a metric the sample does not carry; as null the aggregates skip it.
            rows = rows.select(*group_keys, *dict.fromkeys(m for m, _, _ in present)).fill_nan(None)
            exprs = []
            for m, op, out in present:
                if op == "p95":
                    rows[f"{out}__inf"] = (rows[m] == math.inf).fillna(False).astype("int64")
                    exprs += [
                        col(m).quantile(_P95, "lower").alias(f"{out}__lower"),
                        col(m).quantile(_P95, "higher").alias(f"{out}__higher"),
                        col(m).count().alias(f"{out}__count"),
                        col(f"{out}__inf").sum().alias(f"{out}__posinf"),
                    ]
                else:
                    exprs.append(getattr(col(m), op)().alias(out))
            # The keys go as one list: group_by(*keys) passes the second key in *aggs, where dropna does not apply.
            agg = rows.group_by(group_keys).agg(*exprs)
            for _, op, out in present:
                if op == "p95":
                    parts = [f"{out}__lower", f"{out}__higher", f"{out}__count", f"{out}__posinf"]
                    agg[out] = _numpy_linear_quantile(*(agg[c] for c in parts), _P95)
                    agg = agg.drop(columns=parts)
            return agg

        name = df["name"]
        parts = [
            _agg(
                df.filter((name == "cpu").fillna(False)),
                [
                    ("iowait_pct", "mean", "sys_cpu_iowait_pct"),
                    ("user_pct", "mean", "sys_cpu_user_pct"),
                    ("system_pct", "mean", "sys_cpu_system_pct"),
                    ("idle_pct", "mean", "sys_cpu_idle_pct"),
                ],
            ),
            _agg(
                df.filter(name.str.startswith("cpu-").fillna(False)),
                [("iowait_pct", "max", "sys_core_iowait_pct_max"), ("iowait_pct", "p95", "sys_core_iowait_pct_p95")],
            ),
            _agg(
                df.filter((name == "memory").fillna(False)),
                [
                    ("Dirty", "mean", "sys_mem_dirty"),
                    ("Cached", "mean", "sys_mem_cached"),
                    ("MemAvailable", "mean", "sys_mem_available"),
                ],
            ),
        ]
        parts = [p for p in parts if p is not None and p.num_rows]
        if not parts:
            return empty

        result = parts[0]
        for part in parts[1:]:
            result = result.merge(part, on=group_keys, how="outer")

        for c, dtype in SYSTEM_OUTPUT_COLUMNS.items():
            if c not in result.columns:
                result[c] = Series([None] * result.num_rows, dtype=dtype)
            else:
                result[c] = result[c].astype(dtype)
        # The outer merge appends right-only keys last; key order is the contract.
        return result[list(SYSTEM_OUTPUT_COLUMNS)].sort_values(group_keys).reset_index(drop=True)

    @staticmethod
    def _sanitize_size_offset(df: DataFrame):
        df["size"] = df["size"].replace(0, None)
        if "offset" in df.columns:
            df["offset"] = df["offset"].replace(0, None)
        return df

    @staticmethod
    def _set_epochs(df: DataFrame, epoch_boundaries: DataFrame):
        """Number each row by the epoch whose ``[time_start, time_end)`` of its pid holds its start."""
        epoch = Series([None] * df.num_rows, dtype="int64")
        for boundary in epoch_boundaries.to_dicts():
            mask = (
                (df["pid"] == boundary["pid"])
                & (df["time_start"] >= boundary["time_start"])
                & (df["time_start"] < boundary["time_end"])
            )
            epoch = epoch.mask(mask.fillna(False), boundary["epoch"])
        df["epoch"] = epoch
        return df

    @staticmethod
    def _set_proc_names(df: DataFrame):
        if COL_PROC_NAME in df.columns and df[COL_PROC_NAME].notna().any():
            return df
        df[COL_PROC_NAME] = proc_names(df, "host_hash")
        return df
