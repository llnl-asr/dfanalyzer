import abc
import dask
import dask.dataframe as dd
import hashlib
import itertools as it
import json
import math
import numpy as np
import os
import pandas as pd
import structlog
from dask import compute, persist
from dask.distributed import fire_and_forget, get_client, wait
from omegaconf import OmegaConf
from typing import Callable, Dict, List, Optional, Tuple

from .analysis_utils import (
    fix_dtypes,
    set_file_dir,
    set_file_pattern,
    set_size_bins,
    set_unique_counts,
    split_duration_records_vectorized,
)
from .config import CHECKPOINT_VIEWS, HASH_CHECKPOINT_NAMES, AnalyzerPresetConfig
from .constants import (
    COL_FILE_NAME,
    COL_HOST_NAME,
    COL_PROC_NAME,
    COL_TIME_END,
    COL_TIME_START,
    VIEW_TYPES,
    Layer,
)
from .metrics import (
    set_cross_layer_metrics,
    set_main_metrics,
    set_view_metrics,
)
from .types import (
    AnalyzerResultType,
    RawStats,
    ViewKey,
    ViewMetricBoundaries,
    ViewType,
    Views,
)
from .utils.dask_agg import quantile_stats, unique_set, unique_set_flatten
from .utils.dask_utils import flatten_column_names
from .utils.expr_utils import extract_numerator_and_denominators
from .utils.file_utils import ensure_dir
from .utils.json_encoders import NpEncoder
from .utils.log_utils import console_block, log_block


CHECKPOINT_FLAT_VIEW = "_flat_view"
CHECKPOINT_HLM = "_hlm"
CHECKPOINT_MAIN_VIEW = "_main_view"
CHECKPOINT_RAW_STATS = "_raw_stats"
CHECKPOINT_VIEW = "_view"
HLM_AGG = {
    "time": sum,
    "count": sum,
    "size": sum,
}
HLM_EXTRA_COLS = ["cat", "io_cat", "acc_pat", "func_name"]
PARTITION_SIZE = "128MB"
VIEW_PERMUTATIONS = False

logger = structlog.get_logger()


class Analyzer(abc.ABC):
    def __init__(
        self,
        preset: AnalyzerPresetConfig,
        checkpoint: bool = True,
        checkpoint_dir: str = "",
        debug: bool = False,
        quantile_stats: bool = False,
        time_approximate: bool = True,
        time_granularity: float = 1,
        time_resolution: float = 1e6,
        time_sliced: bool = False,
        verbose: bool = False,
    ):
        """Initializes the Analyzer instance.

        Args:
            preset: The configuration preset for the analyzer.
            checkpoint: Whether to enable checkpointing of intermediate results.
            checkpoint_dir: Directory to store checkpoint data.
            debug: Whether to enable debug mode.
            time_approximate: Whether to use approximate time for I/O operations.
            time_granularity: The time granularity for analysis, in seconds.
            time_resolution: The time resolution for analysis, in microseconds.
            time_sliced: Whether to slice time ranges for analysis.
            verbose: Whether to enable verbose logging.
        """
        if checkpoint:
            assert checkpoint_dir != "", "Checkpoint directory must be defined"

        self.checkpoint = checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_tasks = []
        self.dask_client = get_client()
        self.debug = debug
        self.quantile_stats = quantile_stats
        self.layers = list(preset.layer_defs.keys())
        self.logical_views = dict(OmegaConf.to_object(preset.logical_views))  # type: ignore
        self.preset = preset
        self.time_approximate = time_approximate
        self.time_granularity = time_granularity
        self.time_resolution = time_resolution
        self.time_sliced = time_sliced
        self.verbose = verbose
        ensure_dir(self.checkpoint_dir)

    def analyze_trace(
        self,
        trace_path: str,
        view_types: List[ViewType],
        exclude_characteristics: List[str] = [],
        extra_columns: Optional[Dict[str, str]] = None,
        extra_columns_fn: Optional[Callable[[dict], dict]] = None,
        logical_view_types: bool = False,
        metric_boundaries: ViewMetricBoundaries = {},
        unoverlapped_posix_only: Optional[bool] = False,
    ) -> AnalyzerResultType:
        """Analyzes I/O trace data to identify performance bottlenecks.

        This method orchestrates the entire analysis process, including reading
        trace data, computing various metrics and views, evaluating these views
        to detect bottlenecks, and applying rules to characterize them.

        Args:
            trace_path: Path to the I/O trace file or directory.
            accuracy: The analysis accuracy mode ('optimistic' or 'pessimistic').
            exclude_characteristics: A list of I/O characteristics to exclude.
            logical_view_types: Whether to compute views based on logical relationships.
            metrics: A list of metrics to analyze (e.g., 'iops', 'bw', 'time').
            view_types: A list of view types to compute (e.g., 'file_name', 'proc_name').

        Returns:
            An AnalyzerResultType object containing the analysis results.
        """
        # Check if high-level metrics are checkpointed
        proc_view_types = list(sorted(set(view_types).union({COL_PROC_NAME})))
        hlm_checkpoint_name = self.get_hlm_checkpoint_name(view_types=proc_view_types)
        traces = None
        raw_stats = None
        with console_block("Read trace & stats"):
            if not self.checkpoint or not self.has_checkpoint(name=hlm_checkpoint_name):
                # Read trace & stats
                with log_block("read_trace"):
                    traces = self.read_trace(
                        trace_path=trace_path,
                        extra_columns=extra_columns,
                        extra_columns_fn=extra_columns_fn,
                    )
                with log_block("read_stats"):
                    raw_stats = self.read_stats(traces=traces)
                with log_block("postread_trace"):
                    traces = self.postread_trace(traces=traces, view_types=proc_view_types)
                with log_block("set_size_bins"):
                    traces = traces.map_partitions(set_size_bins)
                if self.time_sliced:
                    with log_block("split_duration_records_vectorized"):
                        traces = traces.map_partitions(
                            split_duration_records_vectorized,
                            time_granularity=self.time_granularity,
                            time_resolution=self.time_resolution,
                        )
            else:
                # Restore stats
                with log_block("restore_raw_stats"):
                    raw_stats = self.restore_extra_data(
                        name=self.get_stats_checkpoint_name(),
                        fallback=lambda: None,
                    )

        # Compute high-level metrics
        with console_block("Compute high-level metrics"):
            with log_block("compute_high_level_metrics"):
                hlm = self.compute_high_level_metrics(
                    checkpoint_name=hlm_checkpoint_name,
                    traces=traces,
                    view_types=proc_view_types,
                )
            with log_block("persist"):
                (hlm, raw_stats) = persist(hlm, raw_stats)
            with log_block("wait"):
                wait([hlm, raw_stats])

        # Validate time granularity
        # self.validate_time_granularity(hlm=hlm, view_types=hlm_view_types)

        # Compute layers & views
        with console_block("Compute views"):
            with log_block("create_layers_and_views_tasks"):
                hlms = {}
                main_views = {}
                main_indexes = {}
                views = {}
                view_keys = set()
                for layer, layer_condition in self.preset.layer_defs.items():
                    layer_hlm = hlm.copy()
                    if layer_condition:
                        layer_hlm = hlm.query(layer_condition)
                    layer_main_view = self.compute_main_view(
                        layer=layer,
                        hlm=layer_hlm,
                        view_types=proc_view_types,
                    )
                    layer_main_index = layer_main_view.index.to_frame().reset_index(drop=True)
                    layer_views = self.compute_views(
                        layer=layer,
                        main_view=layer_main_view,
                        view_types=proc_view_types,
                    )
                    if logical_view_types:
                        layer_logical_views = self.compute_logical_views(
                            layer=layer,
                            main_view=layer_main_view,
                            views=layer_views,
                            view_types=proc_view_types,
                        )
                        layer_views.update(layer_logical_views)
                    hlms[layer] = layer_hlm
                    main_views[layer] = layer_main_view
                    main_indexes[layer] = layer_main_index
                    views[layer] = layer_views
                    view_keys.update(layer_views.keys())

            with log_block("compute_views_and_raw_stats"):
                (views, raw_stats) = compute(views, raw_stats)

        # Restore checkpointed flat views if available
        checkpointed_flat_views = {}
        if self.checkpoint:
            with log_block("restore_flat_view_checkpoints"):
                checkpointed_flat_views.update(self.restore_flat_views(view_keys=list(view_keys)))

        # Process views to create flat views
        with console_block("Process views"):
            flat_views = {}
            for layer in views:
                for view_key in views[layer]:
                    if view_key in checkpointed_flat_views:
                        flat_views[view_key] = checkpointed_flat_views[view_key]
                        continue
                    with log_block("merge_flat_view", view_key=view_key):
                        view = views[layer][view_key].copy()
                        view.columns = view.columns.map(lambda col: layer.lower() + "_" + col)
                        if view_key in flat_views:
                            flat_views[view_key] = flat_views[view_key].merge(
                                view,
                                how="outer",
                                left_index=True,
                                right_index=True,
                            )
                        else:
                            flat_views[view_key] = view
                    try:
                        df = flat_views[view_key]
                        mem_bytes = int(df.memory_usage(deep=True).sum()) if hasattr(df, 'memory_usage') else -1
                        logger.debug(
                            "Flat view created",
                            view_key=view_key,
                            shape=getattr(df, 'shape', None),
                            mem_bytes=mem_bytes,
                        )
                    except Exception:
                        pass

            # Compute metric boundaries for flat views
            with log_block("process_flat_views+metric_boundaries"):
                for view_key in flat_views:
                    if view_key in checkpointed_flat_views:
                        continue
                    view_type = view_key[-1]
                    top_layer = list(self.preset.layer_defs)[0]
                    time_suffix = "time_sum" if self.is_view_process_based(view_key) else "time_max"
                    with log_block("calculate_metric_boundary", view_key=view_key):
                        time_boundary = flat_views[view_key][f"{top_layer}_{time_suffix}"].sum()
                        metric_boundaries[view_type] = metric_boundaries.get(view_type, {})
                        for layer in self.preset.layer_defs:
                            metric_boundaries[view_type][f"{layer}_{time_suffix}"] = time_boundary
                    with log_block("process_flat_view", view_key=view_key):
                        # Process flat views to compute metrics and scores
                        flat_views[view_key] = self._process_flat_view(
                            flat_view=flat_views[view_key],
                            view_key=view_key,
                            metric_boundaries=metric_boundaries,
                        )

        # Checkpoint flat views if enabled
        if self.checkpoint:
            with log_block("write_flat_view_checkpoints"):
                self.checkpoint_tasks.extend(self.store_flat_views(flat_views=flat_views))

        # Wait for all checkpoint tasks
        if self.checkpoint:
            with log_block("wait_for_checkpoints"):
                wait(self.checkpoint_tasks)

        return AnalyzerResultType(
            _hlms=hlms,
            _main_views=main_views,
            _metric_boundaries=metric_boundaries,
            _traces=traces,
            checkpoint_dir=self.checkpoint_dir,
            flat_views=flat_views,
            layers=self.layers,
            raw_stats=raw_stats,
            view_types=view_types,
            views=views,
        )

    def read_stats(self, traces: dd.DataFrame) -> RawStats:
        """Computes and restores raw statistics from the trace data.

        Calculates job time and total event count from the traces.
        It attempts to restore these stats from a checkpoint if available,
        otherwise computes them and checkpoints the result.

        Args:
            traces: A Dask DataFrame containing the I/O trace data.

        Returns:
            A RawStats dictionary containing 'job_time', 'time_granularity',
            and 'total_count'.
        """
        job_time = self.get_job_time(traces)
        total_event_count = self.get_total_event_count(traces)
        unique_file_count = self.get_unique_file_count(traces)
        unique_host_count = self.get_unique_host_count(traces)
        unique_process_count = self.get_unique_process_count(traces)
        raw_stats = RawStats(
            **self.restore_extra_data(
                name=self.get_stats_checkpoint_name(),
                fallback=lambda: dict(
                    job_time=job_time,
                    time_granularity=self.time_granularity,
                    time_resolution=self.time_resolution,
                    total_event_count=total_event_count,
                    unique_file_count=unique_file_count,
                    unique_host_count=unique_host_count,
                    unique_process_count=unique_process_count,
                ),
            )
        )
        return raw_stats

    @abc.abstractmethod
    def read_trace(
        self,
        trace_path: str,
        extra_columns: Optional[Dict[str, str]],
        extra_columns_fn: Optional[Callable[[dict], dict]],
    ) -> dd.DataFrame:
        """Reads I/O trace data from the specified path.

        This is an abstract method that must be implemented by subclasses
        to handle specific trace formats.

        Args:
            trace_path: Path to the I/O trace file or directory.

        Returns:
            A Dask DataFrame containing the parsed I/O trace data.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def postread_trace(self, traces: dd.DataFrame, view_types: List[ViewType]) -> dd.DataFrame:
        """Performs any post-processing on the raw trace data.

        This method can be overridden by subclasses to perform additional
        transformations or filtering on the trace data after it has been read.
        By default, it returns the traces unmodified.

        Args:
            traces: A Dask DataFrame containing the I/O trace data.

        Returns:
            A Dask DataFrame with any post-processing applied.
        """
        return traces

    def compute_high_level_metrics(
        self,
        traces: dd.DataFrame,
        view_types: List[ViewType],
        partition_size: str = PARTITION_SIZE,
        checkpoint_name: Optional[str] = None,
    ) -> dd.DataFrame:
        """Computes high-level metrics by aggregating trace data.

        Groups the trace data by the specified view types and extra columns
        (io_cat, acc_pat, func_id) and aggregates metrics like time, count, and size.

        Args:
            traces: A Dask DataFrame containing the I/O trace data.
            view_types: A list of column names to group by for aggregation.
            partition_size: The desired partition size for the resulting Dask DataFrame.

        Returns:
            A Dask DataFrame containing the computed high-level metrics.
        """
        checkpoint_name = checkpoint_name or self.get_hlm_checkpoint_name(view_types)
        return self.restore_view(
            name=checkpoint_name,
            fallback=lambda: self._compute_high_level_metrics(
                partition_size=partition_size,
                traces=traces,
                view_types=view_types,
            ),
        )

    def compute_main_view(
        self,
        layer: Layer,
        hlm: dd.DataFrame,
        view_types: List[ViewType],
        partition_size: str = PARTITION_SIZE,
    ) -> dd.DataFrame:
        """Computes the main aggregated view from high-level metrics.

        This method takes the high-level metrics, sets derived columns,
        and then groups by the specified view_types to create a primary
        aggregated view of the I/O performance data.

        Args:
            hlm: A Dask DataFrame containing high-level metrics.
            view_types: A list of view types to group by for the main view.
            partition_size: The desired partition size for the resulting Dask DataFrame.

        Returns:
            A Dask DataFrame representing the main aggregated view.
        """
        return self.restore_view(
            name=self.get_checkpoint_name(CHECKPOINT_MAIN_VIEW, str(layer), *sorted(view_types)),
            fallback=lambda: self._compute_main_view(
                hlm=hlm,
                layer=layer,
                partition_size=partition_size,
                view_types=view_types,
            ),
        )

    def compute_views(
        self,
        layer: Layer,
        main_view: dd.DataFrame,
        view_types: List[ViewType],
    ) -> Views:
        """Computes multifaceted views for each specified metric.

        Iterates through all permutations of view_types for each metric,
        generating different "perspectives" on the data. Each perspective
        is a ViewResult, containing the filtered data and critical items.

        Args:
            main_view: The main aggregated Dask DataFrame.
            metrics: A list of metrics to compute views for.
            metric_boundaries: A dictionary of precomputed metric boundaries.
            view_types: A list of base view types to permute for creating views.

        Returns:
            A dictionary where keys are metrics and values are dictionaries
            mapping ViewKey to ViewResult.
        """
        views = {}
        for view_key in self.view_permutations(view_types=view_types):
            view_type = view_key[-1]
            parent_view_key = view_key[:-1]
            parent_records = main_view
            for parent_view_type in parent_view_key:
                parent_records = parent_records.query(
                    f"{parent_view_type} in @indices",
                    local_dict={"indices": views[(parent_view_type,)].index},
                )
            views[view_key] = self.compute_view(
                layer=layer,
                records=parent_records,
                view_key=view_key,
                view_type=view_type,
                view_types=view_types,
            )
        return views

    def compute_logical_views(
        self,
        layer: Layer,
        main_view: dd.DataFrame,
        views: Dict[ViewKey, dd.DataFrame],
        view_types: List[ViewType],
    ):
        """Computes views based on predefined logical relationships in the data.

        This method extends the existing view_results by adding new views
        derived from logical columns (e.g., file directory from file name).

        Args:
            main_view: The main aggregated Dask DataFrame.
            metric_boundaries: A dictionary of precomputed metric boundaries.
            metrics: A list of metrics to compute logical views for.
            view_results: The existing dictionary of computed views to be updated.
            view_types: A list of base view types available in the main_view.

        Returns:
            The updated view_results dictionary including the computed logical views.
        """
        logical_views = {}
        for parent_view_type in self.logical_views:
            parent_view_key = (parent_view_type,)
            if parent_view_key not in views:
                continue
            for view_type in self.logical_views[parent_view_type]:
                view_key = (parent_view_type, view_type)
                parent_records = main_view
                for parent_view_type in parent_view_key:
                    parent_records = parent_records.query(
                        f"{parent_view_type} in @indices",
                        local_dict={"indices": views[(parent_view_type,)].index},
                    )
                view_condition = self.logical_views[parent_view_type][view_type]
                if view_condition is None:
                    if view_type == "file_dir":
                        parent_records = parent_records.map_partitions(set_file_dir)
                    elif view_type == "file_pattern":
                        parent_records = parent_records.map_partitions(set_file_pattern)
                    else:
                        raise ValueError("XXX")
                else:
                    parent_records = parent_records.eval(f"{view_type} = {view_condition}")
                logical_views[view_key] = self.compute_view(
                    layer=layer,
                    records=parent_records,
                    view_key=view_key,
                    view_type=view_type,
                    view_types=view_types,
                )
        return logical_views

    def compute_view(
        self,
        layer: Layer,
        view_key: ViewKey,
        view_type: str,
        view_types: List[ViewType],
        records: dd.DataFrame,
    ) -> dd.DataFrame:
        """Computes a single view based on the provided parameters.

        This involves restoring a view from a checkpoint or computing it.

        Args:
            metrics: The list of all metrics being analyzed.
            metric: The specific metric for this view.
            metric_boundary: The precomputed boundary for the current metric.
            records: The Dask DataFrame (parent records) to compute the view from.
            view_key: The key identifying this specific view.
            view_type: The primary dimension/column for this view.

        Returns:
            A ViewResult object containing the computed view, critical items,
            and filtered records.
        """
        return self.restore_view(
            name=self.get_checkpoint_name(CHECKPOINT_VIEW, str(layer), *list(view_key)),
            fallback=lambda: self._compute_view(
                layer=layer,
                records=records,
                view_key=view_key,
                view_type=view_type,
                view_types=view_types,
            ),
            read_from_disk=False,
            write_to_disk=CHECKPOINT_VIEWS,
        )

    def get_checkpoint_name(self, *args) -> str:
        """Generates a standardized name for a checkpoint.

        Joins the provided arguments with underscores. If HASH_CHECKPOINT_NAMES
        is True, it returns an MD5 hash of the name.

        Args:
            *args: String components to form the checkpoint name.

        Returns:
            A string representing the checkpoint name.
        """
        args = list(args) + [str(int(self.time_granularity))]
        checkpoint_name = "_".join(args)
        if HASH_CHECKPOINT_NAMES:
            return hashlib.md5(checkpoint_name.encode("utf-8")).hexdigest()
        return checkpoint_name

    def get_checkpoint_path(self, name: str) -> str:
        """Constructs the full path for a given checkpoint name.

        Args:
            name: The name of the checkpoint.

        Returns:
            The absolute path to the checkpoint directory/file.
        """
        return f"{self.checkpoint_dir}/{name}"

    def get_hlm_checkpoint_name(self, view_types: List[ViewType]) -> str:
        return self.get_checkpoint_name(CHECKPOINT_HLM, *sorted(view_types))

    def get_job_time(self, traces: dd.DataFrame) -> float:
        """Computes the total job execution time from the traces.

        Args:
            traces: A Dask DataFrame containing the I/O trace data,
                    expected to have 'tstart' and 'tend' columns.

        Returns:
            The total job time as a float.
        """
        return traces[COL_TIME_END].max() - traces[COL_TIME_START].min()

    def get_stats_checkpoint_name(self):
        return self.get_checkpoint_name(CHECKPOINT_RAW_STATS)

    def get_total_event_count(self, traces: dd.DataFrame) -> int:
        """Computes the total number of I/O events in the traces.

        Args:
            traces: A Dask DataFrame containing the I/O trace data.

        Returns:
            The total count of I/O events as an integer.
        """
        return traces.index.count().persist()

    def get_unique_host_count(self, traces: dd.DataFrame):
        """Computes the total number of unique hosts accessed in the traces.

        Args:
            traces: A Dask DataFrame containing the I/O trace data.

        Returns:
            The total count of unique hosts accessed as an integer.
        """
        return traces[COL_HOST_NAME].nunique()

    def get_unique_file_count(self, traces: dd.DataFrame):
        """Computes the total number of unique files accessed in the traces.

        Args:
            traces: A Dask DataFrame containing the I/O trace data.

        Returns:
            The total count of unique files accessed as an integer.
        """
        return traces[COL_FILE_NAME].nunique()

    def get_unique_process_count(self, traces: dd.DataFrame):
        """Computes the total number of unique processes accessed in the traces.

        Args:
            traces: A Dask DataFrame containing the I/O trace data.

        Returns:
            The total count of unique processes accessed as an integer.
        """
        return traces[COL_PROC_NAME].nunique()

    def has_checkpoint(self, name: str):
        """Checks if a checkpoint with the given name exists.

        A checkpoint is considered to exist if its `_metadata` file is present.

        Args:
            name: The name of the checkpoint.

        Returns:
            True if the checkpoint exists, False otherwise.
        """
        checkpoint_path = self.get_checkpoint_path(name=name)
        return os.path.exists(f"{checkpoint_path}/_metadata")

    def is_logical_view_of(self, view_key: ViewKey, parent_view_type: ViewType) -> bool:
        if len(view_key) == 2:
            return view_key[1] in self.logical_views[parent_view_type]
        return False

    def is_view_process_based(self, view_key: ViewKey) -> bool:
        view_type = view_key[-1]
        is_proc_view = view_type == COL_PROC_NAME
        is_logical_proc_view = self.is_logical_view_of(view_key, COL_PROC_NAME)
        return is_proc_view or is_logical_proc_view

    def restore_extra_data(self, name: str, fallback: Callable[[], dict], force=False, persist=False) -> dict:
        """Restores extra (non-DataFrame) data from a JSON checkpoint.

        If checkpointing is enabled and the checkpoint file exists (unless 'force'
        is True), it loads the data from the JSON file. Otherwise, it calls the
        'fallback' function to compute the data and then stores it asynchronously.

        Args:
            name: The name of the checkpoint.
            fallback: A callable function that returns the data if not found or forced.
            force: If True, forces recomputation even if a checkpoint exists.
            persist: (Currently unused in the method body, but part of signature)

        Returns:
            A dictionary containing the restored or computed data.
        """
        if self.checkpoint:
            data_path = f"{self.get_checkpoint_path(name=name)}.json"
            if force or not os.path.exists(data_path):
                data = fallback()
                fire_and_forget(
                    self.dask_client.submit(
                        self.store_extra_data,
                        data=self.dask_client.submit(compute, data),
                        data_path=data_path,
                    )
                )
                return data
            with open(data_path, "r") as f:
                return json.load(f)
        return fallback()

    def restore_flat_views(self, view_keys: List[ViewKey]) -> Dict[ViewKey, pd.DataFrame]:
        restored_flat_views = {}
        for view_key in view_keys:
            flat_view_checkpoint_name = self.get_checkpoint_name(CHECKPOINT_FLAT_VIEW, *list(view_key))
            flat_view_checkpoint_path = self.get_checkpoint_path(name=flat_view_checkpoint_name)
            if self.has_checkpoint(name=flat_view_checkpoint_name):
                restored_flat_views[view_key] = pd.read_parquet(f"{flat_view_checkpoint_path}.parquet")
        return restored_flat_views

    def restore_view(
        self,
        name: str,
        fallback: Callable[[], dd.DataFrame],
        force=False,
        write_to_disk=True,
        read_from_disk=False,
    ) -> dd.DataFrame:
        """Restores a Dask DataFrame view from a Parquet checkpoint.

        If checkpointing is enabled and the checkpoint exists (unless 'force' is True),
        it reads the DataFrame from the Parquet store. Otherwise, it calls the
        'fallback' function to compute the DataFrame. If 'write_to_disk' is True,
        the computed DataFrame is then stored as a checkpoint.

        Args:
            name: The name of the checkpoint.
            fallback: A callable function that returns the DataFrame if not found or forced.
            force: If True, forces recomputation even if a checkpoint exists.
            write_to_disk: If True, saves the computed view to disk if it was recomputed.

        Returns:
            A Dask DataFrame representing the restored or computed view.
        """
        if self.checkpoint:
            view_path = self.get_checkpoint_path(name=name)
            if force or not self.has_checkpoint(name=name):
                with log_block("restore_view_fallback_build", name=name):
                    view = fallback()
                if not write_to_disk:
                    return view
                with log_block("restore_view_schedule_store_view", name=name):
                    checkpoint_task = self.dask_client.compute(self.store_view(name=name, view=view), sync=False)
                    self.checkpoint_tasks.append(checkpoint_task)
                if not read_from_disk:
                    return view
                self.dask_client.cancel(checkpoint_task)
            with log_block("restore_view_read_parquet_metadata", name=name):
                return dd.read_parquet(view_path)
        with log_block("restore_view_fallback_build_no_ckpt", name=name):
            return fallback()

    @staticmethod
    def set_layer_metrics(hlm: pd.DataFrame, derived_metrics: Dict[str, str]) -> pd.DataFrame:
        # Create an explicit copy to avoid SettingWithCopyWarning
        hlm = hlm.copy()
        hlm_columns = list(hlm.columns)
        for metric, condition in derived_metrics.items():
            is_data_metric = metric in ["data", "read", "write"]
            for col in hlm_columns:
                is_data_col = col == "size" or "size_bin" in col
                if not is_data_metric and is_data_col:
                    continue
                metric_col = f"{metric}_{col}"
                hlm[metric_col] = pd.NA
                if hlm.dtypes[col].name == "object":
                    hlm[metric_col] = hlm[metric_col].map(lambda x: set())
                hlm[metric_col] = hlm[metric_col].mask(hlm.eval(condition), hlm[col])
                if hlm.dtypes[col].name != "object":
                    hlm[metric_col] = pd.to_numeric(hlm[metric_col], errors="coerce")
        return hlm

    @staticmethod
    def store_extra_data(data: Tuple[Dict], data_path: str):
        """Saves extra (non-DataFrame) data to a JSON file.

        This static method is typically used by Dask workers to persist data.

        Args:
            data: A tuple containing a single dictionary of data to be saved.
            data_path: The full path to the JSON file where data will be stored.
        """
        with open(data_path, "w") as f:
            return json.dump(data[0], f, cls=NpEncoder)

    def store_flat_views(self, flat_views: Dict[ViewKey, pd.DataFrame]):
        store_flat_view_tasks = []
        for view_key in flat_views:
            flat_view_checkpoint_name = self.get_checkpoint_name(CHECKPOINT_FLAT_VIEW, *list(view_key))
            flat_view_checkpoint_path = self.get_checkpoint_path(name=flat_view_checkpoint_name)
            if self.has_checkpoint(name=flat_view_checkpoint_name):
                continue
            store_flat_view_tasks.append(
                self.dask_client.submit(
                    self._save_flat_view,
                    view=flat_views[view_key],
                    view_path=flat_view_checkpoint_path,
                )
            )
        return store_flat_view_tasks

    def store_view(self, name: str, view: dd.DataFrame, partition_size="64MB"):
        """Stores a Dask DataFrame view to a Parquet checkpoint.

        The view DataFrame is repartitioned and then written to a subdirectory
        named `name` within the `checkpoint_dir`.

        Args:
            name: The name of the checkpoint.
            view: The Dask DataFrame to store.
            compute: Whether to compute the DataFrame before writing (Dask default is True).
            partition_size: The desired partition size for the output Parquet files.

        Returns:
            The result of the Dask `to_parquet` operation.
        """
        for col in view.columns:
            if view.dtypes[col].name == "object":
                view[col] = view[col].astype(str)
        if view.npartitions > 1:
            view = view.repartition(partition_size=partition_size)
        return view.to_parquet(
            self.get_checkpoint_path(name=name),
            compute=False,
            write_metadata_file=True,
        )

    def validate_time_granularity(self, hlm: dd.DataFrame, view_types: List[ViewType]):
        if "io_time" in hlm.columns:
            max_io_time = hlm.groupby(view_types)["io_time"].sum().max().compute()
            if max_io_time > self.time_granularity:
                raise ValueError(
                    f"The max 'io_time' exceeds the 'time_granularity' '{self.time_granularity}'. "
                    f"Please adjust the 'time_granularity' to '{int(2 * max_io_time)}' and rerun the analyzer."
                )

    @staticmethod
    def view_permutations(view_types: List[ViewType]):
        """Generates all permutations of view_types for creating multifaceted views.

        For a list of view_types [vt1, vt2, vt3], it will generate permutations
        of length 1, 2, and 3, e.g., (vt1,), (vt2,), (vt1, vt2), (vt2, vt1), ...

        Args:
            view_types: A list of ViewType elements.

        Returns:
            An iterator yielding tuples, where each tuple is a permutation of view_types.
        """

        if not VIEW_PERMUTATIONS:
            return it.permutations(view_types, 1)

        def _iter_permutations(r: int):
            return it.permutations(view_types, r + 1)

        return it.chain.from_iterable(map(_iter_permutations, range(len(view_types))))

    def _compute_high_level_metrics(
        self,
        traces: dd.DataFrame,
        view_types: list,
        partition_size: str,
    ) -> dd.DataFrame:
        # Add layer columns
        hlm_groupby = list(set(view_types).union(HLM_EXTRA_COLS))
        # Build agg_dict
        bin_cols = [col for col in traces.columns if "_bin_" in col]
        view_types_diff = list(set(VIEW_TYPES).difference(view_types))
        hlm_agg = dict(HLM_AGG)
        hlm_agg.update({col: sum for col in bin_cols})
        hlm_agg.update({col: unique_set() for col in view_types_diff})
        hlm = (
            traces.groupby(hlm_groupby)
            .agg(hlm_agg, split_out=math.ceil(math.sqrt(traces.npartitions)))
            .persist()
            .repartition(partition_size=partition_size)
            .replace(0, np.nan)
        )
        hlm[bin_cols] = hlm[bin_cols].astype("uint32[pyarrow]")
        return hlm.persist()

    def _compute_main_view(
        self,
        layer: Layer,
        hlm: dd.DataFrame,
        view_types: List[ViewType],
        partition_size: str,
    ) -> dd.DataFrame:
        with log_block("drop_and_set_metrics", layer=layer):
            if "posix" not in layer.lower():
                size_cols = [col for col in hlm.columns if col.startswith("size")]
                hlm = hlm.drop(columns=size_cols)  # type: ignore
                if "file_name" in hlm.columns:
                    hlm = hlm.drop(columns=["file_name"])  # type: ignore
            hlm = hlm.map_partitions(self.set_layer_metrics, derived_metrics=self.preset.derived_metrics[layer])
        with log_block("build_agg_dict", layer=layer):
            view_types_diff = set(VIEW_TYPES).difference(view_types)
            main_view_agg = {}
            for col in hlm.columns:
                if any(map(col.endswith, view_types_diff)):
                    main_view_agg[col] = unique_set_flatten()
                elif col not in HLM_EXTRA_COLS:
                    main_view_agg[col] = sum
        with log_block("compute_main_view", layer=layer):
            main_view = (
                hlm.groupby(list(view_types))
                .agg(main_view_agg, split_out=hlm.npartitions)
                .map_partitions(set_main_metrics)
                .replace(0, np.nan)
                .map_partitions(fix_dtypes)
                .persist()
            )
        return main_view

    def _compute_view(
        self,
        layer: Layer,
        records: dd.DataFrame,
        view_key: ViewKey,
        view_type: str,
        view_types: List[ViewType],
    ) -> dd.DataFrame:
        is_view_process_based = self.is_view_process_based(view_key)

        view_types_diff = set(VIEW_TYPES).difference(view_types)
        local_view_types = records.index._meta.names
        local_view_types_diff = set(local_view_types).difference([view_type])

        with log_block("build_agg_dict", layer=layer, view_key=view_key):
            view_agg = {}
            for col in records.columns:
                if "_bin_" in col:
                    view_agg[col] = [sum]
                elif any(map(col.endswith, view_types_diff)):
                    view_agg[col] = [unique_set_flatten()]
                elif col in it.chain.from_iterable(self.logical_views.values()):
                    view_agg[col] = [unique_set_flatten()]
                elif pd.api.types.is_numeric_dtype(records[col].dtype):
                    view_agg[col] = [
                        sum,
                        min,
                        max,
                        "mean",
                        "std",
                    ]
                    if self.quantile_stats:
                        view_agg[col].append(quantile_stats(0.01, 0.99))
                        view_agg[col].append(quantile_stats(0.05, 0.95))
                        view_agg[col].append(quantile_stats(0.1, 0.9))
                        view_agg[col].append(quantile_stats(0.25, 0.75))
                else:
                    raise TypeError(
                        f"Unsupported data type '{records[col].dtype}' for column '{col}'. "
                        f"Developer must add explicit handling for this data type in _compute_view method."
                    )
            view_agg.update({col: [unique_set()] for col in local_view_types_diff})

        with log_block("pre_grouping", layer=layer, view_key=view_key):
            pre_view = records.reset_index()
            if view_type != COL_PROC_NAME:
                pre_view = pre_view.groupby([view_type, COL_PROC_NAME]).sum().reset_index()

        with log_block("groupby_agg_pipeline", layer=layer, view_key=view_key):
            view = (
                pre_view.groupby([view_type])
                .agg(view_agg)
                .replace(0, np.nan)
                .map_partitions(
                    set_view_metrics,
                    is_view_process_based=is_view_process_based,
                    time_granularity=self.time_granularity,
                )
            )
        with log_block("finalize", layer=layer, view_key=view_key):
            view = flatten_column_names(view)
            view = view.map_partitions(set_unique_counts, layer=layer).map_partitions(fix_dtypes).persist()

        return view

    def _process_flat_view(
        self,
        flat_view: pd.DataFrame,
        view_key: ViewKey,
        metric_boundaries: ViewMetricBoundaries,
    ):
        view_type = view_key[-1]
        is_view_process_based = self.is_view_process_based(view_key)
        with log_block("set_cross_layer_metrics", view_key=view_key):
            flat_view = set_cross_layer_metrics(
                flat_view,
                layer_defs=self.preset.layer_defs,
                layer_deps=self.preset.layer_deps,
                async_layers=self.preset.async_layers,
                is_view_process_based=is_view_process_based,
            )
        with log_block("set_additional_metrics", view_key=view_key):
            flat_view = self._set_additional_metrics(flat_view, is_view_process_based=is_view_process_based)
        return flat_view.sort_index(axis=1)

    @staticmethod
    def _save_flat_view(view: pd.DataFrame, view_path: str):
        view.to_parquet(f"{view_path}.parquet")

    def _set_additional_metrics(self, view: pd.DataFrame, is_view_process_based: bool, epsilon=1e-9) -> pd.DataFrame:
        time_metric = "time_sum" if is_view_process_based else "time_max"
        for metric, eval_condition in self.preset.additional_metrics.items():
            eval_condition = eval_condition.format(
                epsilon=epsilon,
                time_interval=self.time_granularity,
                time_metric=time_metric,
            )
            view = view.eval(f"{metric} = {eval_condition}")
            numerator_denominators = extract_numerator_and_denominators(eval_condition)
            if numerator_denominators:
                _, denominators = numerator_denominators
                if denominators:
                    denominator_conditions = [f"({denom}.isna() | {denom} == 0)" for denom in denominators]
                    mask_condition = " & ".join(denominator_conditions)
                    view[metric] = view[metric].mask(view.eval(mask_condition), pd.NA)
        return view
