import dask
import dask.dataframe as dd
import math
import numpy as np
import os
import pandas as pd
import structlog
import pyarrow as pa
from betterset import BetterSet
from dftracer.utils import AggregationConfig, Indexer
from dftracer.utils.dfanalyzer import (
    build_final_meta,
    build_partial_meta,
    coerce_arrow_numerics_to_pandas_native,
    coerce_profile_dtypes,
    distributed_hlm,
    distributed_time_origin,
    ensure_index,
    finalize_view_partials,
    index_path_for,
    ipc_to_pandas,
    normalize_arrow_dtypes,
    partial_arrow_view_groupby,
    resolve_trace_inputs,
    scan_to_ipc,
    hlm_scan_group_by,
    dfanalyzer_events_meta,
    count_index_hashes,
)
from dftracer.utils.dask import _assign_files_by_pid, register_auto_thread_plugin
from dask.distributed import Client, get_client, wait
from typing import Dict, List, Optional, Tuple

from .analyzer import Analyzer, HLM_AGG, HLM_EXTRA_COLS
from .constants import VIEW_TYPES as _DEFAULT_VIEW_TYPES
from .analysis_utils import (
    build_view_rename_map,
    derive_call_stats,
    fix_dtypes,
    fix_std_cols,
    set_unique_counts,
)
from .utils.dask_agg import unique_set, unique_set_flatten
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
    IOCategory,
)
from .types import ReadTraceResult, ViewType
from .utils.log_utils import log_block

logger = structlog.get_logger()

# HLM groupby columns typed as Int64 (others string); metric columns typed as
# Float64 (others Int64). Passed to the dftracer.utils HLM helpers.
HLM_INT_INDEX_COLS = frozenset({"pid", "tid", "io_cat", "acc_pat", "time_range", "epoch", "step"})
HLM_FLOAT_METRIC_COLS = frozenset(
    {
        "time",
        "time_sq",
        "time_min",
        "time_max",
        "time_call_min",
        "time_call_max",
        "time_start",
        "time_end",
    }
)

IGNORED_FILE_PATTERNS = [
    "/dev/",
    "/etc/",
    "/gapps/python",
    "/lib/python",
    "/proc/",
    "/software/",
    "/sys/",
    "/usr/lib",
    "/usr/tce/backend",
    "/usr/tce/packages",
    "/venv",
    "__pycache__",
]
IGNORED_FUNC_NAMES = [
    "DLIOBenchmark.__init__",
    # 'DLIOBenchmark._train',
    "DLIOBenchmark.initialize",
    # 'DLIOBenchmark.run',
    "FileStorage.__init__",
    "TorchDataset.__init__",
    # "TorchDataset.worker_init",
]
IGNORED_FUNC_PATTERNS = [
    "Checkpointing.__init__",
    "Checkpointing.finalize",
    "Checkpointing.get_tensor",
    "DataLoader.__init__",
    "DataLoader.finalize",
    "DataLoader.get_tensor",
    "DataLoader.next",
    "Framework.get_loader",
    "Framework.init_loader",
    "Framework.is_nativeio_available",
    "Framework.trace_object",
    "Reader.__init__",
    "Reader.load_index",
    "Reader.next",
    "Reader.read_index",
    ".save_state",
    "checkpoint_end_",
    "checkpoint_start_",
]
TRACE_COL_MAPPING = {
    "dur": COL_TIME,
    "name": COL_FUNC_NAME,
    "te": COL_TIME_END,
    "trange": COL_TIME_RANGE,
    "ts": COL_TIME_START,
}
TYPE_EVENT = 0
TYPE_FILE_HASH = 1
TYPE_HOST_HASH = 2
TYPE_STRING_HASH = 3
TYPE_METADATA = 4
TYPE_PROC_METADATA = 5
TYPE_PROFILE = 6
TYPE_SYSTEM = 7
PROFILE_COLUMN_MAPPING = {
    "count": "Int64",
    "count_max": "Int64",
    "count_min": "Int64",
    "count_sum": "Int64",
    "dft_cnt": "Int64",
    "dur": "Int64",
    "dur_max": "Int64",
    "dur_min": "Int64",
    "dur_sum": "Int64",
    "epoch": "Int64",
    "flags": "Int64",
    "offset": "Int64",
    "ret": "Int64",
    "offset_max": "Int64",
    "offset_min": "Int64",
    "offset_sum": "Int64",
    "ret_max": "Int64",
    "ret_min": "Int64",
    "ret_sum": "Int64",
    "whence": "Int64",
    "whence_max": "Int64",
    "whence_min": "Int64",
    "whence_sum": "Int64",
}
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
PROFILE_MEASURE_COLUMNS = [COL_COUNT, COL_TIME, COL_SIZE]
PROFILE_STAT_COLUMNS = ["time_min", "time_max", "size_min", "size_max", "offset_min", "offset_max"]
PROFILE_IDENTITY_COLUMNS = [
    col for col in PROFILE_OUTPUT_COLUMNS if col not in PROFILE_MEASURE_COLUMNS and col not in PROFILE_STAT_COLUMNS
]

# System metric columns extracted from cat="sys" ph="C" events
SYSTEM_CPU_METRICS = ["user_pct", "system_pct", "iowait_pct", "idle_pct", "irq_pct", "softirq_pct"]
SYSTEM_MEMORY_METRICS = ["MemAvailable", "MemFree", "Cached", "Dirty", "Active"]
SYSTEM_COLUMN_MAPPING = {
    **{m: "float64" for m in SYSTEM_CPU_METRICS},
    **{m: "float64" for m in SYSTEM_MEMORY_METRICS},
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


def io_columns():
    columns = {
        "file_hash": "string",
        "host_hash": "string",
        "image_id": "Int64",
        "io_cat": "Int8",
        "size": "Int64",
        "offset": "Int64",
    }
    return columns


class DFTracerAnalyzer(Analyzer):
    POSIX_CAT_RULES = [
        ("/data", "_reader"),
        ("/checkpoint", "_checkpoint"),
        ("/lustre", "_lustre"),
        ("/ssd", "_ssd"),
    ]

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

    def analyze_trace(self, trace_path, *args, **kwargs):
        """Transparent indexing: ensure the dftracer index exists, then analyze."""
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
        )

        from .utils.log_utils import console

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("files"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as _bar:
            _task = _bar.add_task("Indexing", total=None)

            def _on_progress(done: int, total: int) -> None:
                _bar.update(_task, completed=done, total=total)

            ensure_index(
                trace_path,
                self.trace_groups,
                self.time_granularity * 1000.0,
                progress=_on_progress,
            )
        self._view_types = kwargs.get("view_types") or (args[0] if args else None)
        return super().analyze_trace(trace_path, *args, **kwargs)

    def read_trace_local(self, trace_path, extra_columns=None, extra_columns_fn=None):
        """Read trace using C++ aggregation pipeline.

        This is a faster alternative to read_trace() that uses the C++ Indexer
        with fused aggregation to produce pre-aggregated data directly.
        All transformation (hash resolution, time normalization, io_cat, proc_name)
        is done in C++ for performance.

        Args:
            trace_path: Directory containing trace files or glob pattern.
            extra_columns: Not used (kept for API compatibility).
            extra_columns_fn: Not used (kept for API compatibility).

        Returns:
            ReadTraceResult with traces, profiles, and system_metrics.
        """
        with log_block("cpp_indexer_setup"):
            # Configure aggregation to match analyzer time granularity
            time_interval_ms = self.time_granularity * 1000.0  # seconds to ms

            directory, files = resolve_trace_inputs(trace_path, self.trace_groups)

            if not directory and not files:
                raise FileNotFoundError("No matching .pfw or .pfw.gz files found.")

            index_path = index_path_for(trace_path)
            indexer = Indexer(
                directory=directory,
                files=files if files else None,
                index_dir=os.path.dirname(index_path),
                require_checkpoint=True,
                require_bloom=False,
                build_bloom=False,
                require_manifest=False,
                require_aggregation=AggregationConfig(
                    time_interval_ms=time_interval_ms,
                    compute_percentiles=False,
                ),
                force_rebuild=False,
            )

        with log_block("cpp_ensure_indexed"):
            status = indexer.ensure_indexed()
            logger.info(
                "C++ indexing complete",
                total_files=status.total_files,
                ready=len(status.ready),
                needs_work=len(status.needs_work),
            )

        with log_block("cpp_load_hash_tables"):
            # Store hash tables for compatibility with other methods
            file_hashes = indexer.get_hash_table("file")
            host_hashes = indexer.get_hash_table("host")
            self._file_hashes = pd.DataFrame(
                {"name": list(file_hashes.values())},
                index=pd.Index(list(file_hashes.keys()), name="hash", dtype="string"),
            )
            self._host_hashes = pd.DataFrame(
                {"name": list(host_hashes.values())},
                index=pd.Index(list(host_hashes.keys()), name="hash", dtype="string"),
            )
            self._string_hashes = pd.DataFrame(columns=["name"])
            self._metadata = pd.DataFrame(columns=["name", "value"])

        with log_block("cpp_iter_arrow_dfanalyzer"):
            all_batches = indexer.iter_arrow_dfanalyzer_all(
                time_granularity=self.time_granularity,
                time_resolution=self.time_resolution,
            )
            event_batches = [pa.record_batch(b) for b in all_batches.get("events", [])]
            profile_batches = [pa.record_batch(b) for b in all_batches.get("profiles", [])]
            system_batches = [pa.record_batch(b) for b in all_batches.get("system", [])]

        with log_block("cpp_to_dask"):
            # Convert Arrow batches to Dask DataFrames
            if event_batches:
                events_table = pa.Table.from_batches(event_batches)
                events_pd = events_table.to_pandas()
                traces = dd.from_pandas(
                    events_pd,
                    npartitions=max(1, len(event_batches) // 10),
                )
            else:
                traces = dd.from_pandas(
                    pd.DataFrame(columns=list(PROFILE_OUTPUT_COLUMNS.keys())),
                    npartitions=1,
                )

            if profile_batches:
                profiles_table = pa.Table.from_batches(profile_batches)
                profile_window = int(self.profile_time_granularity * self.time_resolution)
                profiles_pd = coerce_profile_dtypes(
                    profiles_table.to_pandas(), PROFILE_OUTPUT_COLUMNS, profile_window=profile_window
                )
                profiles = dd.from_pandas(
                    profiles_pd,
                    npartitions=max(1, len(profile_batches) // 10),
                )
            else:
                profiles = None

            if system_batches:
                system_table = pa.Table.from_batches(system_batches)
                system_pd = system_table.to_pandas()
                # time_bucket is already the bucket-start timestamp in us
                # (compute_time_bucket floors ts to the bucket boundary), so
                # pin ts to it for stable (ts - origin)//bucket_width indexing.
                system_pd["ts"] = system_pd["time_bucket"].astype("int64")
                time_origin = int(system_pd["ts"].min()) if not system_pd.empty else 0
                system_events = dd.from_pandas(
                    system_pd,
                    npartitions=max(1, len(system_batches) // 10),
                )
                system_metrics = self._standardize_system(system_events, time_origin=time_origin)
            else:
                system_metrics = None

        return ReadTraceResult(
            traces=traces,
            profiles=profiles,
            profile_time_granularity=self.profile_time_granularity if profiles is not None else None,
            system_metrics=system_metrics,
        )

    def read_trace(
        self,
        trace_path,
        extra_columns=None,
        extra_columns_fn=None,
        client: Optional[Client] = None,
    ):
        """Read trace using C++ aggregation pipeline.

        Uses the active Dask client for distributed execution across workers.
        Data stays on workers as Dask DataFrame partitions (no coordinator
        materialization).

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

            directory, files = resolve_trace_inputs(trace_path, self.trace_groups)

            if not directory and not files:
                raise FileNotFoundError("No matching .pfw or .pfw.gz files found.")

        with log_block("cpp_indexer"):
            index_path = index_path_for(trace_path)
            indexer = Indexer(
                directory=directory,
                files=files,
                index_dir=os.path.dirname(index_path),
                require_checkpoint=True,
                require_bloom=False,
                build_bloom=False,
                require_manifest=False,
                require_aggregation=AggregationConfig(
                    time_interval_ms=time_interval_ms,
                    compute_percentiles=False,
                ),
                force_rebuild=False,
            )
            status = indexer.ensure_indexed()

            if status.total_files == 0:
                self._file_hashes = pd.DataFrame(columns=["name"])
                self._host_hashes = pd.DataFrame(columns=["name"])
                self._string_hashes = pd.DataFrame(columns=["name"])
                self._metadata = pd.DataFrame(columns=["name", "value"])
                return ReadTraceResult(
                    traces=dd.from_pandas(
                        pd.DataFrame(columns=list(PROFILE_OUTPUT_COLUMNS.keys())),
                        npartitions=1,
                    ),
                    profiles=None,
                    profile_time_granularity=None,
                    system_metrics=None,
                )

        with log_block("query_file_info"):
            file_id_to_path, file_pids = indexer.query_file_info()
            index_path = os.path.abspath(status.index_path)
            indexer.close()

        with log_block("dask_client_connect"):
            try:
                dask_client = client or get_client()
            except ValueError:
                dask_client = None

            if dask_client is None:
                return self.read_trace_local(trace_path)

            worker_nthreads = dask_client.nthreads()
            n_workers = len(worker_nthreads) or 1
            worker_list = list(worker_nthreads.keys())

        event_futures = []
        worker_scan_args = []

        with log_block("submit_workers"):
            all_file_ids = set(file_id_to_path.keys())
            full_file_pids = {fid: file_pids.get(fid, set()) for fid in all_file_ids}
            worker_file_ids = _assign_files_by_pid(full_file_pids, n_workers)
            for worker_id, fids in worker_file_ids.items():
                wfiles = [file_id_to_path[fid] for fid in fids if fid in file_id_to_path]
                if not wfiles:
                    continue
                pids = set()
                for fid in fids:
                    if fid in file_pids:
                        pids.update(file_pids[fid])
                query = None
                if pids:
                    pid_conditions = " or ".join(f"pid == {pid}" for pid in sorted(pids))
                    query = f"({pid_conditions})"
                worker_addr = worker_list[worker_id % len(worker_list)] if worker_list else None
                num_shards = 4096
                n_parts = max(1, len(worker_file_ids))
                span = (num_shards + n_parts - 1) // n_parts
                shard_begin = min(num_shards, worker_id * span)
                shard_end = min(num_shards, shard_begin + span)
                future = dask_client.submit(
                    scan_to_ipc,
                    wfiles,
                    index_path,
                    self.time_granularity,
                    self.time_resolution,
                    None,
                    self._scan_group_by(),
                    shard_begin,
                    shard_end,
                    workers=[worker_addr] if worker_addr else None,
                    pure=False,
                )
                event_futures.append(future)
                worker_scan_args.append((worker_addr, wfiles, query))

            self._worker_ipc_futures = event_futures
            self._worker_scan_args = worker_scan_args
            self._index_path = index_path
            self._dask_client = dask_client
            self._time_origin = distributed_time_origin(event_futures, dask_client)

        with log_block("build_dask_dataframe"):
            events_meta = dfanalyzer_events_meta(self._scan_group_by())

            def _extract_and_decode(ipc_future, key, meta):
                ipc_bytes = ipc_future[key]
                if ipc_bytes is None:
                    return meta.iloc[:0].copy()
                return ipc_to_pandas(ipc_bytes)

            event_delayed = [
                dask.delayed(_extract_and_decode)(dask.delayed(f), "events", events_meta) for f in event_futures
            ]
            traces = (
                dd.from_delayed(event_delayed, meta=events_meta)
                if event_delayed
                else dd.from_pandas(events_meta, npartitions=1)
            )

            def _has_data(result_dict, key):
                return result_dict[key] is not None

            has_profiles_futures = [dask_client.submit(_has_data, f, "profiles", pure=False) for f in event_futures]
            has_profiles = any(dask_client.gather(has_profiles_futures))

            if has_profiles:
                profile_delayed = [
                    dask.delayed(_extract_and_decode)(dask.delayed(f), "profiles", events_meta) for f in event_futures
                ]
                profiles = dd.from_delayed(profile_delayed, meta=events_meta)
                profile_window = int(self.profile_time_granularity * self.time_resolution)
                profiles = profiles.map_partitions(
                    coerce_profile_dtypes, PROFILE_OUTPUT_COLUMNS, profile_window=profile_window
                )
            else:
                profiles = None

            has_system_futures = [dask_client.submit(_has_data, f, "system", pure=False) for f in event_futures]
            has_system = any(dask_client.gather(has_system_futures))
            if has_system:
                system_frames = dask_client.gather(
                    [
                        dask_client.submit(
                            lambda d: ipc_to_pandas(d["system"]) if d.get("system") else None, f, pure=False
                        )
                        for f in event_futures
                    ]
                )
                system_frames = [f for f in system_frames if f is not None and not f.empty]
                if system_frames:
                    system_pd = pd.concat(system_frames, ignore_index=True)
                    system_pd["ts"] = system_pd["time_bucket"].astype("int64")
                    time_origin = int(system_pd["ts"].min())
                    system_events = dd.from_pandas(system_pd, npartitions=1)
                    system_metrics = self._standardize_system(system_events, time_origin=time_origin)
                else:
                    system_metrics = None
            else:
                system_metrics = None

        self._file_hashes = pd.DataFrame(columns=["name"])
        self._host_hashes = pd.DataFrame(columns=["name"])
        self._string_hashes = pd.DataFrame(columns=["name"])
        self._metadata = pd.DataFrame(columns=["name", "value"])

        return ReadTraceResult(
            traces=traces,
            profiles=profiles,
            profile_time_granularity=self.profile_time_granularity if profiles is not None else None,
            system_metrics=system_metrics,
        )

    def _postread_hlm_config(self, data_type):
        """Postread transformations the distributed HLM must replicate."""
        config: Dict[str, object] = {"posix_cat_rules": [list(rule) for rule in self.POSIX_CAT_RULES]}
        if data_type == "events":
            config["ignored_file_patterns"] = list(IGNORED_FILE_PATTERNS)
            config["ignored_func_names"] = list(IGNORED_FUNC_NAMES)
            config["ignored_func_patterns"] = list(IGNORED_FUNC_PATTERNS)
            origin = getattr(self, "_time_origin", None)
            if origin is not None:
                config["time_origin"] = int(origin)
                config["bucket_width_us"] = int(self.time_granularity * self.time_resolution)
            # epoch view: pass the preset's epoch layer query so the distributed scan
            # can assign per-pid epochs (mirrors postread_trace's assign_epochs, which
            # the scan bypasses). Only meaningful when assign_epochs is enabled.
            if self.assign_epochs and "epoch" in self.preset.layer_defs:
                config["epoch_query"] = self.preset.layer_defs["epoch"]
        return config

    def _hlm(self, data_type, view_types, traces):
        return distributed_hlm(
            data_type,
            view_types,
            traces,
            getattr(self, "_worker_ipc_futures", None),
            getattr(self, "_worker_scan_args", []),
            getattr(self, "_dask_client", None),
            HLM_AGG,
            HLM_EXTRA_COLS,
            HLM_INT_INDEX_COLS,
            HLM_FLOAT_METRIC_COLS,
            self._postread_hlm_config(data_type),
            merge_partials=True,
        )

    def _compute_high_level_metrics(self, traces, view_types, partition_size):
        result = self._hlm("events", view_types, traces)
        if result is not None:
            return result
        return super()._compute_high_level_metrics(traces, view_types, partition_size)

    def _compute_profile_hlm(self, profiles, view_types, partition_size):
        result = self._hlm("profiles", view_types, profiles)
        if result is not None:
            return result
        if profiles is None:
            return None
        return super()._compute_profile_hlm(profiles, view_types, partition_size)

    def _compute_view(self, layer, records, view_key, view_type, view_types):
        from .constants import VIEW_TYPES
        import itertools as it

        keep_object_cols = set(VIEW_TYPES) | set(view_types) | set(it.chain.from_iterable(self.logical_views.values()))
        drop_cols = []
        for col in records.columns:
            dtype_str = str(records[col].dtype)
            if dtype_str in ("string", "category"):
                records[col] = records[col].astype("object")
                dtype_str = "object"
            if (
                dtype_str == "object"
                and col not in keep_object_cols
                and not any(col.endswith(vt) for vt in keep_object_cols)
            ):
                drop_cols.append(col)
        if drop_cols:
            records = records.drop(columns=drop_cols, errors="ignore")

        # Arrow-based view computation
        view_types_diff = set(VIEW_TYPES).difference(view_types)
        local_view_types = records.index._meta.names
        local_view_types_diff = set(local_view_types).difference([view_type])

        view_agg = {}
        for col in records.columns:
            if "_bin_" in col:
                view_agg[col] = ["sum"]
            elif any(map(col.endswith, view_types_diff)):
                view_agg[col] = [unique_set_flatten()]
            elif col in it.chain.from_iterable(self.logical_views.values()):
                view_agg[col] = [unique_set_flatten()]
            elif col.endswith("_sq"):
                view_agg[col] = ["sum"]
            elif col.endswith("_call_min"):
                view_agg[col] = ["min"]
            elif col.endswith("_call_max"):
                view_agg[col] = ["max"]
            elif pd.api.types.is_numeric_dtype(records[col].dtype):
                view_agg[col] = ["sum", "min", "max", "mean", "std"]
            else:
                raise TypeError(f"Unsupported dtype '{records[col].dtype}' for column '{col}'")
        view_agg.update({col: [unique_set()] for col in local_view_types_diff})

        # Decompose view_agg into tree-reducible partials. Each input column's
        # aggregation list is split into per-partition chunks that later merge
        # via Dask's tree-reduce (sum/min/max/set-union are all associative).
        full_cols, sum_cols, min_cols, max_cols = [], [], [], []
        set_cols_items = []
        for col, aggs in view_agg.items():
            if col not in records.columns:
                continue
            if all(isinstance(a, str) for a in aggs):
                s = set(aggs)
                if s == {"sum", "min", "max", "mean", "std"}:
                    full_cols.append(col)
                elif s == {"sum"}:
                    sum_cols.append(col)
                elif s == {"min"}:
                    min_cols.append(col)
                elif s == {"max"}:
                    max_cols.append(col)
                else:
                    raise ValueError(f"unsupported agg combo for {col}: {aggs}")
            else:
                set_cols_items.append((col, aggs[0]))

        std_cols = list(full_cols)
        records = records.map_partitions(fix_std_cols, std_cols=std_cols)

        partial_meta = build_partial_meta(records, view_type, full_cols, sum_cols, min_cols, max_cols, set_cols_items)
        partials = records.map_partitions(
            partial_arrow_view_groupby,
            view_type,
            full_cols,
            sum_cols,
            min_cols,
            max_cols,
            set_cols_items,
            BetterSet.flatten,
            meta=partial_meta,
        )

        merge_aggs = {}
        for c in full_cols:
            merge_aggs[f"{c}_sum"] = "sum"
            merge_aggs[f"{c}_count"] = "sum"
            merge_aggs[f"{c}_sumsq"] = "sum"
            merge_aggs[f"{c}_min"] = "min"
            merge_aggs[f"{c}_max"] = "max"
        for c in sum_cols:
            merge_aggs[f"{c}_sum"] = "sum"
        for c in min_cols:
            merge_aggs[f"{c}_min"] = "min"
        for c in max_cols:
            merge_aggs[f"{c}_max"] = "max"
        for c, _ in set_cols_items:
            merge_aggs[f"{c}_unique"] = unique_set_flatten()

        merged = partials.groupby(view_type).agg(merge_aggs)

        final_meta = build_final_meta(merged, full_cols)
        final = merged.map_partitions(finalize_view_partials, full_cols, meta=final_meta)
        final = final.rename(columns=build_view_rename_map(final.columns))
        final = final.replace(0, pd.NA)
        final = (
            final.map_partitions(derive_call_stats)
            .map_partitions(set_unique_counts, layer=layer)
            .map_partitions(fix_dtypes, time_sliced=self.time_sliced)
            .map_partitions(coerce_arrow_numerics_to_pandas_native)
            .persist()
        )
        return final

    def postread_trace(
        self,
        traces: dd.DataFrame,
        view_types: List[ViewType],
    ) -> dd.DataFrame:
        traces = traces.map_partitions(normalize_arrow_dtypes)
        with log_block("filter_rows"):
            traces = traces.map_partitions(
                self._apply_ignore_filters,
                IGNORED_FILE_PATTERNS,
                IGNORED_FUNC_NAMES,
                IGNORED_FUNC_PATTERNS,
            )

        # Set epochs
        with log_block("assign_epochs"):
            if self.assign_epochs:
                if "epoch" not in self.preset.layer_defs:
                    raise ValueError("Epoch layer definition is missing")
                epochs = traces.query(self.preset.layer_defs["epoch"]).compute()
                epochs_with_index = epochs.sort_values(["pid", "time_start"]).reset_index(drop=True)
                epochs_with_index["epoch"] = epochs_with_index.groupby("pid").cumcount() + 1
                epoch_boundaries = epochs_with_index[["pid", "time_start", "time_end", "epoch"]]
                traces = traces.map_partitions(self._set_epochs, epoch_boundaries=epoch_boundaries)

        with log_block("wait"):
            _ = wait(traces)

        with log_block("set_basic_columns"):
            traces[COL_ACC_PAT] = 0
            traces[COL_COUNT] = 1

        # drop columns that are not needed
        # if COL_FILE_NAME not in view_types:
        #     traces = traces.drop(columns=[COL_FILE_NAME], errors='ignore')
        # if COL_HOST_NAME not in view_types:
        #     traces = traces.drop(columns=[COL_HOST_NAME], errors='ignore')

        # Set batches
        # traces['batch'] = traces.groupby(['func_name', 'step']).cumcount() + 1
        # batch_counts = traces['batch'].value_counts()
        # last_valid_batch = batch_counts[batch_counts > 1].index.max()
        # traces['batch'] = traces['batch'].mask(
        #     traces['batch'] > last_valid_batch, pd.NA
        # )

        # pytorch reads images instead of batches
        # e.g. 4 workers = 0..4 images = who starts/finishes first

        # epoch and step make sense in dlio layer

        # to put step back, target variable = previous compute + my io

        # Set steps depending on time ranges
        # step_time_ranges = traces.groupby(['pid', 'epoch', 'step']).agg({'ts': min, 'te': max})
        # traces = traces.map_partitions(
        #     self._set_steps, step_time_ranges=step_time_ranges.reset_index()
        # )

        return (
            traces.map_partitions(self._set_proc_names)
            .map_partitions(self._fix_file_posix_category)
            .map_partitions(self._sanitize_size_offset)
        )

    def get_job_time(self, traces):
        return super().get_job_time(traces) / self.time_resolution

    def get_time_boundary_layer(self):
        if self.assign_epochs:
            return "epoch"
        return super().get_time_boundary_layer()

    def get_total_event_count(self, traces: dd.DataFrame) -> int:
        return traces[COL_COUNT].sum().persist()

    def _scan_group_by(self):
        """Fold the scan to the grain these views need, keeping proc_name so
        the downstream view code still finds it."""
        views = list(getattr(self, "_view_types", None) or _DEFAULT_VIEW_TYPES)
        return hlm_scan_group_by(views + ["proc_name"], HLM_EXTRA_COLS)

    def _index_count(self, hash_type):
        return count_index_hashes(self._index_path, hash_type)

    def get_unique_file_count(self, traces: dd.DataFrame):
        # Folded scans drop file_hash, and this is a property of the index.
        return self._index_count("file")

    def get_unique_host_count(self, traces: dd.DataFrame):
        return self._index_count("host")

    def get_unique_process_count(self, traces: dd.DataFrame):
        return traces["proc_name"].nunique()

    @staticmethod
    def _apply_ignore_filters(df, ignored_file_patterns, ignored_func_names, ignored_func_patterns):
        """Drop ignored files/functions"""
        if ignored_file_patterns and COL_FILE_NAME in df.columns:
            keep = df[COL_FILE_NAME].isna() | ~df[COL_FILE_NAME].str.contains("|".join(ignored_file_patterns))
            df = df[keep]
        if COL_FUNC_NAME in df.columns:
            if ignored_func_names:
                df = df[~df[COL_FUNC_NAME].isin(ignored_func_names)]
            if ignored_func_patterns:
                df = df[~df[COL_FUNC_NAME].str.contains("|".join(ignored_func_patterns))]
        return df

    @classmethod
    def _fix_file_posix_category(cls, df: pd.DataFrame):
        # A folded scan carries no file_name, and the HLM applies this same
        # rule worker-side from postread_config.
        if "file_name" not in df.columns:
            return df
        # base condition is fixed on the original cat; suffixes (purpose then
        # filesystem) are applied cumulatively per POSIX_CAT_RULES order.
        base_condition = df["cat"].str.contains("posix|stdio") & ~df["file_name"].isna()
        for path, suffix in cls.POSIX_CAT_RULES:
            mask = base_condition & df["file_name"].str.contains(path)
            df.loc[mask, "cat"] = df.loc[mask, "cat"] + suffix
        return df

    def _fix_time(self, traces: dd.DataFrame, time_origin: Optional[int] = None) -> dd.DataFrame:
        time_origin = traces["ts"].min() if time_origin is None else time_origin
        traces["ts"] = traces["ts"] - time_origin
        traces["te"] = traces["ts"] + traces["dur"]
        traces["trange"] = traces["ts"] // (self.time_granularity * self.time_resolution)
        traces["ts"] = traces["ts"].astype("Int64")
        traces["te"] = traces["te"].astype("Int64")
        traces["trange"] = traces["trange"].astype("Int16")
        traces["dur"] = traces["dur"] / self.time_resolution
        return traces

    def _standardize_profiles(self, profiles: dd.DataFrame, time_origin: int) -> dd.DataFrame:
        profiles = profiles.map_partitions(self._set_proc_names)
        profiles = profiles.map_partitions(
            self._standardize_profile_partition,
            profile_time_granularity=self.profile_time_granularity,
            time_granularity=self.time_granularity,
            time_origin=time_origin,
            time_resolution=self.time_resolution,
            meta=PROFILE_OUTPUT_COLUMNS,
        )
        profiles = profiles.map_partitions(self._fix_file_posix_category).map_partitions(self._sanitize_size_offset)
        return self._coalesce_profiles(profiles)

    def _coalesce_profiles(self, profiles: dd.DataFrame) -> dd.DataFrame:
        # dft-agg-full can emit multiple counter rows for the same canonical
        # profile bucket. Collapse them here so `read_trace()` returns a stable
        # analyzer-native profile table.
        split_out = max(1, math.ceil(math.sqrt(profiles.npartitions)))
        coalesced = (
            profiles.groupby(PROFILE_IDENTITY_COLUMNS, dropna=False)
            .agg(
                {
                    COL_COUNT: "sum",
                    COL_TIME: "sum",
                    COL_SIZE: "sum",
                    "time_min": "min",
                    "time_max": "max",
                    "size_min": "min",
                    "size_max": "max",
                    "offset_min": "min",
                    "offset_max": "max",
                },
                split_out=split_out,
            )
            .reset_index()
        )
        coalesced[COL_COUNT] = coalesced[COL_COUNT].astype("Int64")
        coalesced[COL_TIME] = coalesced[COL_TIME].astype("float64")
        coalesced[COL_SIZE] = coalesced[COL_SIZE].replace(0, pd.NA).astype("Int64")
        return coalesced[list(PROFILE_OUTPUT_COLUMNS)]

    @staticmethod
    def _standardize_profile_partition(
        df: pd.DataFrame,
        profile_time_granularity: float,
        time_origin: int,
        time_granularity: float,
        time_resolution: float,
    ) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in PROFILE_OUTPUT_COLUMNS.items()})

        df = df.copy()
        duration = df["dur_sum"].where(df["dur_sum"].notna(), df["dur"]).fillna(0)
        size = df["ret_sum"].where(df["ret_sum"].notna(), df["ret"])
        is_sized_io = df[COL_IO_CAT].isin([IOCategory.READ.value, IOCategory.WRITE.value]) & size.notna() & (size > 0)

        profile_df = pd.DataFrame(index=df.index)
        profile_df["cat"] = df["cat"].astype("string")
        profile_df[COL_FUNC_NAME] = df["name"].astype("string")
        profile_df["pid"] = df["pid"].astype("Int64")
        profile_df["tid"] = df["tid"].astype("Int64")
        profile_df["epoch"] = df["epoch"].astype("Int64")
        profile_df["step"] = df["step"].astype("Int64")
        profile_df["file_hash"] = df["file_hash"].astype("string")
        profile_df["host_hash"] = df["host_hash"].astype("string")
        profile_df[COL_FILE_NAME] = df[COL_FILE_NAME].astype("string")
        profile_df[COL_HOST_NAME] = df[COL_HOST_NAME].astype("string")
        profile_df[COL_PROC_NAME] = df[COL_PROC_NAME].astype("string")
        profile_df[COL_IO_CAT] = df[COL_IO_CAT].fillna(IOCategory.OTHER.value).astype("Int8")
        profile_df[COL_ACC_PAT] = pd.Series(0, index=df.index, dtype="Int8")
        profile_df[COL_COUNT] = df["dft_cnt"].fillna(0).astype("Int64")
        profile_df[COL_TIME] = duration.astype("float64") / time_resolution
        profile_df[COL_SIZE] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        profile_df.loc[is_sized_io, COL_SIZE] = size.loc[is_sized_io].astype("Int64")
        dur_min = df["dur_min"].where(df["dur_min"].notna(), df["dur"])
        profile_df["time_min"] = dur_min.astype("float64") / time_resolution
        dur_max = df["dur_max"].where(df["dur_max"].notna(), df["dur"])
        profile_df["time_max"] = dur_max.astype("float64") / time_resolution
        profile_df["size_min"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        profile_df["size_max"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        profile_df.loc[is_sized_io, "size_min"] = (
            df["ret_min"].where(df["ret_min"].notna(), df["ret"]).loc[is_sized_io].astype("Int64")
        )
        profile_df.loc[is_sized_io, "size_max"] = (
            df["ret_max"].where(df["ret_max"].notna(), df["ret"]).loc[is_sized_io].astype("Int64")
        )
        profile_df["offset_min"] = df["offset_min"].where(df["offset_min"].notna(), df["offset"]).astype("Int64")
        profile_df["offset_max"] = df["offset_max"].where(df["offset_max"].notna(), df["offset"]).astype("Int64")
        profile_df[COL_TIME_START] = (df["ts"] - time_origin).astype("Int64")
        profile_df[COL_TIME_END] = profile_df[COL_TIME_START] + int(profile_time_granularity * time_resolution)
        profile_df[COL_TIME_RANGE] = (profile_df[COL_TIME_START] // int(time_granularity * time_resolution)).astype(
            "Int64"
        )
        return profile_df[list(PROFILE_OUTPUT_COLUMNS)]

    @staticmethod
    def _standardize_system_partition(
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

    def _standardize_system(self, system_events: dd.DataFrame, time_origin: int) -> dd.DataFrame:
        """Standardize raw system events into per-time_range metrics."""
        meta = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in SYSTEM_OUTPUT_COLUMNS.items()})
        return system_events.map_partitions(
            self._standardize_system_partition,
            time_origin=time_origin,
            time_granularity=self.time_granularity,
            time_resolution=self.time_resolution,
            meta=meta,
        )

    def _get_columns(self, extra_columns: Optional[Dict[str, str]]):
        columns = {
            "name": "string",
            "cat": "string",
            "type": "Int8",
            "pid": "Int64",
            "tid": "Int64",
            "ts": "Int64",
            "te": "Int64",
            "dur": "Int64",
            "epoch": "Int64",
            "step": "Int64",
            "tinterval": "Int64" if self.time_approximate else "string",
            "trange": "Int64",
            "level": "Int8",
        }
        metadata_columns = {
            "hash": "string",
            "host_hash": "string",
            "value": "string",
        }
        columns.update(io_columns())
        columns.update(PROFILE_COLUMN_MAPPING)
        columns.update(SYSTEM_COLUMN_MAPPING)
        columns.update(metadata_columns)
        columns.update(extra_columns or {})
        logger.debug("get_columns", columns=columns)
        return columns

    def _handle_metadata(self, raw_traces: dd.DataFrame) -> Tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame]:
        is_dask = isinstance(raw_traces, dd.DataFrame)
        traces = raw_traces.query(f"type == {TYPE_EVENT}")
        profiles = raw_traces.query(f"type == {TYPE_PROFILE}")
        system_events = raw_traces.query(f"type == {TYPE_SYSTEM}")
        file_hashes = raw_traces.query(f"type == {TYPE_FILE_HASH}")[["name", "hash"]].groupby("hash").first()
        host_hashes = raw_traces.query(f"type == {TYPE_HOST_HASH}")[["name", "hash"]].groupby("hash").first()
        string_hashes = raw_traces.query(f"type == {TYPE_STRING_HASH}")[["name", "hash"]].groupby("hash").first()
        metadata = raw_traces.query(f"type == {TYPE_METADATA}")[["name", "value"]]
        file_hashes.index = file_hashes.index.astype(str)
        host_hashes.index = host_hashes.index.astype(str)
        string_hashes.index = string_hashes.index.astype(str)
        if is_dask:
            file_hashes = file_hashes.persist()
            host_hashes = host_hashes.persist()
            string_hashes = string_hashes.persist()
            metadata = metadata.persist()
        traces = self._attach_metadata(traces, file_hashes=file_hashes, host_hashes=host_hashes)
        profiles = self._attach_metadata(profiles, file_hashes=file_hashes, host_hashes=host_hashes)
        self._file_hashes = file_hashes
        self._host_hashes = host_hashes
        self._string_hashes = string_hashes
        self._metadata = metadata
        return traces, profiles, system_events

    @staticmethod
    def _attach_metadata(records: dd.DataFrame, file_hashes: dd.DataFrame, host_hashes: dd.DataFrame):
        records = records.merge(
            file_hashes.rename(columns={"name": COL_FILE_NAME}),
            how="left",
            left_on="file_hash",
            right_index=True,
        )
        records = records.merge(
            host_hashes.rename(columns={"name": COL_HOST_NAME}),
            how="left",
            left_on="host_hash",
            right_index=True,
        )
        return records

    @staticmethod
    def _rename_columns(traces: dd.DataFrame) -> dd.DataFrame:
        return traces.rename(columns=TRACE_COL_MAPPING)

    @staticmethod
    def _sanitize_size_offset(df: pd.DataFrame):
        df["size"] = df["size"].replace(0, pd.NA)
        if "offset" in df.columns:
            df["offset"] = df["offset"].replace(0, pd.NA)
        return df

    @staticmethod
    def _set_epochs(df: pd.DataFrame, epoch_boundaries: pd.DataFrame):
        df["epoch"] = pd.NA

        # Iterate over each epoch boundary to find matching events
        for _, epoch_boundary in epoch_boundaries.iterrows():
            pid = epoch_boundary["pid"]
            start = epoch_boundary["time_start"]
            end = epoch_boundary["time_end"]

            # Find rows in the partition that match the pid and fall within the time interval
            mask = (df["pid"] == pid) & (df["time_start"] >= start) & (df["time_start"] < end)

            # Assign the epoch number to the matching rows
            df.loc[mask, "epoch"] = epoch_boundary["epoch"]

        return df

    @staticmethod
    def _set_proc_names(df: pd.DataFrame):
        if COL_PROC_NAME in df.columns and df[COL_PROC_NAME].notna().any():
            return df
        host_component = (
            df[COL_HOST_NAME].astype(str) if COL_HOST_NAME in df.columns else pd.Series(pd.NA, index=df.index)
        )
        if "host_hash" in df.columns:
            host_component = host_component.fillna(df["host_hash"].astype(str))
        df[COL_PROC_NAME] = (
            "app#"
            + host_component.fillna("unknown").astype(str)
            + "#"
            + df["pid"].astype(str)
            + "#"
            + df["tid"].astype(str)
        )
        return df

    @staticmethod
    def _set_steps(df: pd.DataFrame, step_time_ranges: pd.DataFrame):
        mapped_traces = df.copy()

        for pid in df["pid"].unique():
            pid_trace_cond = mapped_traces["pid"] == pid
            pid_traces = mapped_traces[pid_trace_cond]
            pid_step_ranges = step_time_ranges[step_time_ranges["pid"] == pid]

            # Sort step ranges by start timestamp
            pid_step_ranges_sorted = pid_step_ranges.sort_values("ts")

            # Create bins and labels
            bins = pid_step_ranges_sorted["ts"].tolist()
            if len(bins) > 0:
                bins.append(pid_step_ranges_sorted["te"].max())
            # print(pid, bins)
            steps = pid_step_ranges_sorted["step"].tolist()

            # Use np.digitize to find bin indices
            bin_indices = np.digitize(pid_traces["ts"], bins=bins) - 1

            # Map indices to steps, leaving as None for out-of-range timestamps
            mapped_traces.loc[pid_trace_cond, "step"] = [
                steps[idx] if 0 <= idx < len(steps) else pd.NA for idx in bin_indices
            ]

        return mapped_traces
