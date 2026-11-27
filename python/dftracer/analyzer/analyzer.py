import abc
import dataclasses as dc
import hashlib
import itertools as it
import json
import math
import os
import re
import structlog
from dask.distributed import get_client
from dftracer.utils import DataFrame, DType, Series
from omegaconf import OmegaConf, open_dict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .analysis_utils import (
    floor_divide,
    frame_nbytes,
    set_file_dir,
    set_file_pattern,
    set_size_bins,
    split_duration_records_vectorized,
)
from .checkpoint_table import FORMAT_TAG, read_table, write_table
from .cluster import AutoClient, NullClient
from .config import CHECKPOINT_VIEWS, HASH_CHECKPOINT_NAMES, AnalyzerPresetConfig, FactsConfig
from .fact_engine import FactPipeline
from .constants import (
    COL_COUNT,
    COL_FILE_NAME,
    COL_HOST_NAME,
    COL_PROC_NAME,
    COL_TIME_END,
    COL_TIME_RANGE,
    COL_TIME_START,
    VIEW_TYPES,
    Layer,
)
from .types import (
    AnalysisResult,
    ReadTraceResult,
    RawStats,
    ViewKey,
    ViewMetricBoundaries,
    ViewType,
    Views,
)
from .frames.expr import evaluate, query
from .frames.flat_metrics import process_flat_metrics
from .frames.main_view import group_main_view, reconcile
from .utils.file_utils import ensure_dir
from .utils.log_utils import console_block, console_progress_block, log_block


CHECKPOINT_FLAT_VIEW = "_flat_view"
CHECKPOINT_HLM = "_hlm"
CHECKPOINT_MAIN_VIEW = "_main_view"
CHECKPOINT_RAW_STATS = "_raw_stats"
CHECKPOINT_VIEW = "_view"
# Categories whose events carry a transfer size, so their layers keep size/bandwidth.
SIZE_CATEGORY_PREFIXES = ("posix", "stdio")
VIEW_PERMUTATIONS = False

logger = structlog.get_logger()


@dc.dataclass(frozen=True)
class PartitionedRecords:
    """A view input whose group-by runs as a DaskFrame of `npartitions` partitions on `client`.

    With no client the group-by runs on one frame in this process.
    """

    frame: DataFrame
    client: Any
    npartitions: int

    def pipe(self, fn: Callable[..., DataFrame], *args) -> "PartitionedRecords":
        """`fn(frame, *args)` on the held frame, kept partitioned."""
        return dc.replace(self, frame=fn(self.frame, *args))


def _rows_in(frame: DataFrame, view_type: str, keys: Series) -> DataFrame:
    """Rows of `frame` whose `view_type` is one of `keys`; a missing value matches a missing key, as in pandas."""
    values = frame[view_type] if view_type in frame.columns else frame.get_level_values(view_type)
    mask = values.is_in(keys)
    if keys.is_null().any():
        mask = mask | values.is_null()
    return frame.filter(mask)


def _view_keys(views: Views, view_type: str) -> Series:
    return views[(view_type,)].get_level_values(view_type)


def _has_non_finite_list_value(frame: DataFrame) -> bool:
    for c, t in zip(frame.columns, frame.dtypes):
        if t in (DType.LIST, DType.LARGE_LIST) and frame[c].child(0).dtype.is_float:
            # A null list cell can keep values under it, which explode shows and the writer skips.
            lists = frame[c]
            values = lists.filter(lists.is_not_null()).explode()
            if (values.is_nan() | values.is_infinite()).fillna(False).any():
                return True
    return False


def _outer_join_on_index(left: DataFrame, right: DataFrame) -> DataFrame:
    """`left` and `right` joined on their index, all keys of both, each column keeping its dtype."""
    keys = left.index_names()
    # A label join widens an int column with a missing row to float; a keyed merge keeps it
    # (dftracer-utils branch feat/analyzer-port-gaps: dtype-keeping index join).
    joined = left.reset_index().merge(right.reset_index(), how="outer", on=keys, nulls_equal=True)
    return joined.set_index(keys)


class Analyzer(abc.ABC):
    def __init__(
        self,
        preset: AnalyzerPresetConfig,
        checkpoint: bool = True,
        checkpoint_dir: str = "",
        debug: bool = False,
        profile_distribution: str = "uniform",
        profile_time_granularity: float = 5,
        quantile_stats: bool = False,
        time_approximate: bool = True,
        time_granularity: float = 1,
        time_resolution: float = 1e6,
        time_sliced: bool = False,
        verbose: bool = False,
        view_materialize_max_bytes: int = 256 * 1024**2,
        facts_config: Optional[Union[FactsConfig, Dict[str, Any]]] = None,
    ):
        """Initializes the Analyzer instance.

        Args:
            preset: The configuration preset for the analyzer.
            checkpoint: Whether to enable checkpointing of intermediate results.
            checkpoint_dir: Directory to store checkpoint data.
            debug: Whether to enable debug mode.
            profile_distribution: Strategy for distributing profile measures
                when expanding buckets ('uniform' or 'weighted').
            profile_time_granularity: The time granularity of profile buckets
                emitted by the tracer, in seconds.
            time_approximate: Whether to use approximate time for I/O operations.
            time_granularity: The time granularity for analysis, in seconds.
            time_resolution: The time resolution for analysis, in microseconds.
            time_sliced: Whether to slice time ranges for analysis.
            verbose: Whether to enable verbose logging.
            view_materialize_max_bytes: Compute a layer's views in this process
                when its main view is smaller than this many bytes, else as a
                DaskFrame on the client's workers. 0 always uses the DaskFrame.
        """
        if checkpoint:
            assert checkpoint_dir != "", "Checkpoint directory must be defined"
        if profile_distribution not in ("uniform", "weighted"):
            raise ValueError(f"profile_distribution must be 'uniform' or 'weighted', got '{profile_distribution}'")

        self.checkpoint = checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.dask_client = self._resolve_client()
        self.debug = debug
        self.profile_distribution = profile_distribution
        self.profile_time_granularity = profile_time_granularity
        self.quantile_stats = quantile_stats
        self.layers = list(preset.layer_defs.keys())
        # Facts pipeline (additive): off unless facts.enabled. When on, builds the
        # rule/metric pipeline that _analyze_hlm runs over the flat views.
        self.facts_config = self._build_facts_config(facts_config)
        self.fact_pipeline = None
        self._build_fact_pipeline()
        self.logical_views = dict(OmegaConf.to_object(preset.logical_views))  # type: ignore
        self.preset = preset
        self.time_approximate = time_approximate
        self.time_granularity = time_granularity
        self.time_resolution = time_resolution
        self.time_sliced = time_sliced
        self.verbose = verbose
        self.view_materialize_max_bytes = view_materialize_max_bytes
        ensure_dir(self.checkpoint_dir)

    @staticmethod
    def _resolve_client():
        """The client to submit work through.

        Normally the one the caller connected, found via `get_client()`. Under
        `cluster=none` there is no ambient client and `get_client()` raises, so
        fall back to a `NullClient` that runs the same calls in this process.
        Constructing a second one is harmless -- it holds no state.
        """
        try:
            return get_client()
        except ValueError:
            return NullClient()

    def use_client(self, client) -> None:
        """Adopt the client the caller set up.

        `_resolve_client` can only find a client that registered itself as the
        ambient one, which the Dask `Client` does and the in-process shims
        cannot. Without this an `AutoClient` would be replaced by a plain
        `NullClient` -- still correct, but silently unable to promote itself to
        a cluster, which is the whole point of `cluster=auto`.
        """
        if client is not None:
            self.dask_client = client

    @staticmethod
    def _build_facts_config(facts_config: Optional[Union[FactsConfig, Dict[str, Any]]]) -> FactsConfig:
        if facts_config is None:
            return FactsConfig()
        if isinstance(facts_config, FactsConfig):
            return facts_config
        if isinstance(facts_config, dict):
            return FactsConfig(**facts_config)
        if OmegaConf.is_config(facts_config):
            config_obj = OmegaConf.to_object(facts_config)
            if isinstance(config_obj, FactsConfig):
                return config_obj
            if isinstance(config_obj, dict):
                return FactsConfig(**config_obj)
            if dc.is_dataclass(config_obj):
                return FactsConfig(**dc.asdict(config_obj))
            raise TypeError(f"Unsupported OmegaConf facts object type: {type(config_obj)}")
        raise TypeError(f"Unsupported facts_config type: {type(facts_config)}")

    def _evaluate_analysis_facts(
        self,
        flat_views: Dict[ViewKey, DataFrame],
        raw_stats: Any,
    ) -> Dict[ViewKey, List[Any]]:
        # facts.enabled is the single switch: if on, the pipeline exists and runs.
        if self.fact_pipeline is None:
            return {}
        return self.fact_pipeline.build(
            flat_views=flat_views,
            raw_stats=raw_stats,
        )

    def _materialize_output_artifacts(
        self,
        flat_views: Dict[ViewKey, DataFrame],
        analysis_facts: Dict[ViewKey, List[Any]],
    ) -> Tuple[Dict[ViewKey, DataFrame], Dict[ViewKey, List[Any]]]:
        # Facts pass through (they're {} unless facts.enabled). Detail (flat) views
        # are still independently gated by emit_flat_views.
        output_flat_views = flat_views if self.facts_config.emit_flat_views else {}
        return output_flat_views, analysis_facts

    @staticmethod
    def _category_layer_key(cat: str, taken: set) -> str:
        key = re.sub(r"[^a-z0-9_]+", "_", cat.lower()).strip("_") or "cat"
        if key == "app":
            key = "app_cat"
        base, i = key, 1
        while key in taken:
            i += 1
            key = f"{base}_{i}"
        taken.add(key)
        return key

    def _build_fact_pipeline(self) -> None:
        """(Re)build the fact pipeline for the current ``self.layers``. Must be
        called again whenever the layer set changes."""
        if not self.facts_config.enabled:
            return
        self.fact_pipeline = FactPipeline.from_facts_config(
            self.facts_config,
            layers=self.layers,
            strict_time_semantics=self.facts_config.strict_time_semantics,
            allow_mixed_time_aggregates=self.facts_config.allow_mixed_time_aggregates,
        )

    @staticmethod
    def _normalize_cats(values) -> List[str]:
        cats = {str(c).lower() for c in values if c is not None and str(c) != ""}
        return sorted(cats - {"dftracer"})

    @classmethod
    def _category_layers_from_cats(cls, cats: List[str]):
        """Build nested layers from ``cat`` values. pydftracer supports dotted
        categories (``ai.data.io``), so each dotted prefix becomes a parent layer
        that also matches its descendants."""
        nodes = set()
        for cat in cats:
            parts = cat.split(".")
            for i in range(1, len(parts) + 1):
                nodes.add(".".join(parts[:i]))
        nodes = sorted(nodes)

        taken: set = set()
        key_of = {node: cls._category_layer_key(node, taken) for node in nodes}

        layer_defs: Dict[str, Optional[str]] = {"app": None}
        layer_deps: Dict[str, Optional[str]] = {"app": None}
        derived_metrics: Dict[str, Dict[str, str]] = {"app": {}}
        size_layers: List[str] = []
        for node in nodes:
            key = key_of[node]
            # json.dumps quotes/escapes the cat so query-significant characters cannot
            # break (or inject into) the expression.
            literal = json.dumps(node)
            if any(other.startswith(f"{node}.") for other in nodes):
                layer_defs[key] = f'cat == {literal} or cat.str.startswith({json.dumps(node + ".")})'
            else:
                layer_defs[key] = f"cat == {literal}"
            parent = node.rsplit(".", 1)[0] if "." in node else None
            layer_deps[key] = key_of[parent] if parent else "app"
            derived_metrics[key] = {}
            if node.startswith(SIZE_CATEGORY_PREFIXES):
                size_layers.append(key)
        return layer_defs, layer_deps, derived_metrics, size_layers

    def _build_category_layers(self, traces: DataFrame) -> None:
        """Rebuild the preset as an ``app`` boundary layer plus nested layers per
        distinct ``cat`` in the trace. No-op if no categories are found."""
        if "cat" not in traces.columns:
            return
        cats = self._normalize_cats(traces["cat"].dropna().unique().to_list())
        if not cats:
            return

        layer_defs, layer_deps, derived_metrics, size_layers = self._category_layers_from_cats(cats)

        with open_dict(self.preset):
            self.preset.layer_defs = layer_defs
            self.preset.layer_deps = layer_deps
            self.preset.derived_metrics = derived_metrics
            self.preset.size_layers = size_layers
            self.preset.size_derived_metrics = {}
        self.layers = list(layer_defs.keys())
        self._build_fact_pipeline()  # was configured for the pre-discovery layer set
        logger.info("Built category layers", layers=self.layers)

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
    ) -> AnalysisResult:
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
            An AnalysisResult object containing the analysis results.

        Raises:
            ValueError: A view type is not one of `VIEW_TYPES`.
        """
        unsupported = sorted(set(view_types).difference(VIEW_TYPES))
        if unsupported:
            raise ValueError(f"view_types {unsupported} are not supported; use some of {VIEW_TYPES}")
        proc_view_types = self.ensure_proc_view_type(view_types=view_types)
        read_result = None
        profiles = None
        traces = None
        raw_stats = None
        with console_progress_block("Read trace & stats"):
            with log_block("read_trace"):
                read_result = self.read_trace(
                    trace_path=trace_path,
                    extra_columns=extra_columns,
                    extra_columns_fn=extra_columns_fn,
                    group_by_file=COL_FILE_NAME in proc_view_types,
                )
                traces = read_result.traces
                profiles = read_result.profiles
            if profiles is not None:
                with log_block("validate_and_expand_profiles"):
                    ptg = read_result.profile_time_granularity or self.profile_time_granularity
                    profiles = self._validate_and_expand_profiles(
                        profiles=profiles,
                        profile_time_granularity=ptg,
                    )
                    read_result.profiles = profiles
            with log_block("read_stats"):
                raw_stats = self.read_stats(traces=traces, profiles=profiles)
            with log_block("postread_trace"):
                traces = self.postread_trace(traces=traces, view_types=proc_view_types)
            if getattr(self.preset, "auto_layers_by_category", False):
                with log_block("build_category_layers"):
                    self._build_category_layers(traces=traces)
            with log_block("set_size_bins"):
                traces = set_size_bins(traces)
            if self.time_sliced:
                with log_block("split_duration_records_vectorized"):
                    traces = split_duration_records_vectorized(
                        traces,
                        time_granularity=self.time_granularity,
                        time_resolution=self.time_resolution,
                    )
            read_result.traces = traces

        # Compute high-level metrics
        with console_block("Compute high-level metrics"):
            if profiles is None:
                with log_block("compute_high_level_metrics"):
                    hlm_checkpoint_name = self.get_hlm_checkpoint_name(view_types=proc_view_types)
                    hlm = self.compute_high_level_metrics(
                        checkpoint_name=hlm_checkpoint_name,
                        traces=traces,
                        view_types=proc_view_types,
                    )
                trace_hlm = None
                profile_hlm = None
            else:
                with log_block("compute_trace_hlm"):
                    trace_hlm = self.compute_high_level_metrics(
                        checkpoint_name=self.get_trace_hlm_checkpoint_name(view_types=proc_view_types),
                        traces=traces,
                        view_types=proc_view_types,
                    )
                with log_block("compute_profile_hlm"):
                    profile_hlm = self.compute_profile_hlm(
                        checkpoint_name=self.get_profile_hlm_checkpoint_name(view_types=proc_view_types),
                        profiles=profiles,
                        view_types=proc_view_types,
                    )
                hlm = None

        # Analyze HLM
        result = self._analyze_hlm(
            hlm=hlm,
            logical_view_types=logical_view_types,
            metric_boundaries=metric_boundaries,
            profile_hlm=profile_hlm,
            proc_view_types=proc_view_types,
            raw_stats=raw_stats,
            system_metrics=read_result.system_metrics,
            trace_hlm=trace_hlm,
        )

        # Attach correct traces & view types
        result._read_result = read_result
        result._trace_run = getattr(self, "_run", None)
        result.view_types = view_types

        # Return result
        return result

    def read_stats(self, traces: DataFrame, profiles: Optional[DataFrame] = None) -> RawStats:
        """Computes and restores raw statistics from the trace data.

        Calculates job time and total event count from the traces.
        It attempts to restore these stats from a checkpoint if available,
        otherwise computes them and checkpoints the result.

        Args:
            traces: A DataFrame containing the I/O trace data.

        Returns:
            A RawStats dictionary containing 'job_time', 'time_granularity',
            and 'total_count'.
        """
        job_time = self.get_job_time(traces)
        trace_event_count = self.get_total_event_count(traces)
        profile_event_count = self.get_profile_event_count(profiles)
        total_event_count = trace_event_count + profile_event_count
        unique_file_count = self.get_unique_file_count(traces)
        unique_host_count = self.get_unique_host_count(traces)
        unique_process_count = self.get_unique_process_count(traces)
        raw_stats_data = self.restore_extra_data(
            name=self.get_stats_checkpoint_name(),
            fallback=lambda: dict(
                job_time=job_time,
                time_granularity=self.time_granularity,
                time_resolution=self.time_resolution,
                trace_event_count=trace_event_count,
                profile_event_count=profile_event_count,
                total_event_count=total_event_count,
                unique_file_count=unique_file_count,
                unique_host_count=unique_host_count,
                unique_process_count=unique_process_count,
            ),
        )
        if "trace_event_count" not in raw_stats_data:
            raw_stats_data["trace_event_count"] = raw_stats_data.get("total_event_count", 0)
        if "profile_event_count" not in raw_stats_data:
            raw_stats_data["profile_event_count"] = 0
        if "total_event_count" not in raw_stats_data:
            raw_stats_data["total_event_count"] = (
                raw_stats_data["trace_event_count"] + raw_stats_data["profile_event_count"]
            )
        raw_stats = RawStats(**raw_stats_data)
        return raw_stats

    @abc.abstractmethod
    def read_trace(
        self,
        trace_path: str,
        extra_columns: Optional[Dict[str, str]],
        extra_columns_fn: Optional[Callable[[dict], dict]],
        group_by_file: bool = True,
    ) -> ReadTraceResult:
        """Reads I/O trace data from the specified path.

        This is an abstract method that must be implemented by subclasses
        to handle specific trace formats.

        Args:
            trace_path: Path to the I/O trace file or directory.
        Returns:
            A ReadTraceResult containing the parsed I/O trace data and any
            additional native profile streams for the analyzer.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def postread_trace(self, traces: DataFrame, view_types: List[ViewType]) -> DataFrame:
        """Performs any post-processing on the raw trace data.

        This method can be overridden by subclasses to perform additional
        transformations or filtering on the trace data after it has been read.
        By default, it returns the traces unmodified.

        Args:
            traces: A DataFrame containing the I/O trace data.

        Returns:
            A DataFrame with any post-processing applied.
        """
        return traces

    def compute_high_level_metrics(
        self,
        traces: DataFrame,
        view_types: List[ViewType],
        checkpoint_name: Optional[str] = None,
    ) -> DataFrame:
        """Computes high-level metrics by aggregating trace data.

        Groups the trace data by the specified view types and extra columns
        (io_cat, acc_pat, func_id) and aggregates metrics like time, count, and size.

        Args:
            traces: A DataFrame containing the I/O trace data.
            view_types: A list of column names to group by for aggregation.

        Returns:
            A DataFrame containing the computed high-level metrics.
        """
        checkpoint_name = checkpoint_name or self.get_hlm_checkpoint_name(view_types)
        return self.restore_view(
            name=checkpoint_name,
            fallback=lambda: self._compute_high_level_metrics(
                traces=traces,
                view_types=view_types,
            ),
        )

    def compute_profile_hlm(
        self,
        profiles: DataFrame,
        view_types: List[ViewType],
        checkpoint_name: Optional[str] = None,
    ) -> DataFrame:
        checkpoint_name = checkpoint_name or self.get_profile_hlm_checkpoint_name(view_types)
        return self.restore_view(
            name=checkpoint_name,
            fallback=lambda: self._compute_profile_hlm(
                profiles=profiles,
                view_types=view_types,
            ),
        )

    def compute_hybrid_hlm(
        self,
        traces: DataFrame,
        profiles: DataFrame,
        view_types: List[ViewType],
        checkpoint_name: Optional[str] = None,
    ) -> DataFrame:
        checkpoint_name = checkpoint_name or self.get_hlm_checkpoint_name(view_types)
        return self.restore_view(
            name=checkpoint_name,
            fallback=lambda: self._compute_hybrid_hlm(
                profiles=profiles,
                traces=traces,
                view_types=view_types,
            ),
        )

    def reconcile_hlm(
        self,
        trace_hlm: DataFrame,
        profile_hlm: DataFrame,
        view_types: List[ViewType],
        layer: Optional[Layer] = None,
    ) -> DataFrame:
        return self._reconcile_hlm(
            layer=layer,
            profile_hlm=profile_hlm,
            trace_hlm=trace_hlm,
            view_types=view_types,
        )

    def compute_main_view(
        self,
        layer: Layer,
        hlm: DataFrame,
        view_types: List[ViewType],
    ) -> DataFrame:
        """Computes the main aggregated view from high-level metrics.

        This method takes the high-level metrics, sets derived columns,
        and then groups by the specified view_types to create a primary
        aggregated view of the I/O performance data.

        Args:
            hlm: A DataFrame containing high-level metrics.
            view_types: A list of view types to group by for the main view.

        Returns:
            A DataFrame representing the main aggregated view.
        """
        return self.restore_view(
            name=self.get_checkpoint_name(CHECKPOINT_MAIN_VIEW, str(layer), *sorted(view_types)),
            fallback=lambda: self._compute_main_view(
                hlm=hlm,
                layer=layer,
                view_types=view_types,
            ),
        )

    def compute_views(
        self,
        layer: Layer,
        main_view: DataFrame,
        view_types: List[ViewType],
    ) -> Views:
        """Computes multifaceted views for each specified metric.

        Iterates through all permutations of view_types for each metric,
        generating different "perspectives" on the data.

        Args:
            main_view: The main aggregated DataFrame.
            metrics: A list of metrics to compute views for.
            metric_boundaries: A dictionary of precomputed metric boundaries.
            view_types: A list of base view types to permute for creating views.

        Returns:
            A dictionary mapping each ViewKey to its view, indexed by the view type.
        """
        records = self._materialize_if_small(self._partitioned(main_view))
        views = {}
        for view_key in self.view_permutations(view_types=view_types):
            view_type = view_key[-1]
            parent_view_key = view_key[:-1]
            parent_records = records
            for parent_view_type in parent_view_key:
                parent_records = parent_records.pipe(_rows_in, parent_view_type, _view_keys(views, parent_view_type))
            views[view_key] = self.compute_view(
                layer=layer,
                records=parent_records,
                view_key=view_key,
                view_type=view_type,
                view_types=view_types,
            )
        return views

    def _partitioned(self, main_view: DataFrame) -> PartitionedRecords:
        """`main_view` as a view input that groups on the client's workers, one partition per worker."""
        client = self.dask_client
        if isinstance(client, AutoClient):
            client = client._impl
        if not getattr(client, "is_distributed", True):
            return PartitionedRecords(main_view, None, 1)
        return PartitionedRecords(main_view, client, max(1, len(client.nthreads())))

    def _materialize_if_small(self, records: PartitionedRecords):
        """The plain frame of `records` when it is under `view_materialize_max_bytes`, else `records`.

        A view of a small main view groups faster in this process than as a
        DaskFrame, and these frames are routinely a few hundred rows.
        """
        if not self.view_materialize_max_bytes:
            return records
        if frame_nbytes(records.frame) < self.view_materialize_max_bytes:
            return records.frame
        return records

    def compute_logical_views(
        self,
        layer: Layer,
        main_view: DataFrame,
        views: Dict[ViewKey, DataFrame],
        view_types: List[ViewType],
    ):
        """Computes views based on predefined logical relationships in the data.

        This method extends the existing view_results by adding new views
        derived from logical columns (e.g., file directory from file name).

        Args:
            main_view: The main aggregated DataFrame.
            metric_boundaries: A dictionary of precomputed metric boundaries.
            metrics: A list of metrics to compute logical views for.
            view_results: The existing dictionary of computed views to be updated.
            view_types: A list of base view types available in the main_view.

        Returns:
            The updated view_results dictionary including the computed logical views.
        """
        records = self._materialize_if_small(self._partitioned(main_view))
        logical_views = {}
        for parent_view_type in self.logical_views:
            parent_view_key = (parent_view_type,)
            if parent_view_key not in views:
                continue
            for view_type in self.logical_views[parent_view_type]:
                view_key = (parent_view_type, view_type)
                parent_records = records
                for parent_view_type in parent_view_key:
                    parent_records = parent_records.pipe(
                        _rows_in, parent_view_type, _view_keys(views, parent_view_type)
                    )
                view_condition = self.logical_views[parent_view_type][view_type]
                if view_condition is None:
                    if view_type == "file_dir":
                        parent_records = parent_records.pipe(set_file_dir)
                    elif view_type == "file_pattern":
                        parent_records = parent_records.pipe(set_file_pattern)
                    else:
                        raise ValueError("XXX")
                else:
                    parent_records = parent_records.pipe(evaluate, f"{view_type} = {view_condition}")
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
        records: Union[DataFrame, PartitionedRecords],
    ) -> DataFrame:
        """Computes a single view based on the provided parameters.

        This involves restoring a view from a checkpoint or computing it.

        Args:
            metrics: The list of all metrics being analyzed.
            metric: The specific metric for this view.
            metric_boundary: The precomputed boundary for the current metric.
            records: The parent records to compute the view from, in memory or partitioned.
            view_key: The key identifying this specific view.
            view_type: The primary dimension/column for this view.

        Returns:
            The view, indexed by `view_type`.
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
        checkpoint_name = "_".join((*args, FORMAT_TAG))
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

    def get_profile_hlm_checkpoint_name(self, view_types: List[ViewType]) -> str:
        return self.get_checkpoint_name(CHECKPOINT_HLM, "profile", *sorted(view_types))

    def get_trace_hlm_checkpoint_name(self, view_types: List[ViewType]) -> str:
        return self.get_checkpoint_name(CHECKPOINT_HLM, "trace", *sorted(view_types))

    def get_job_time(self, traces: DataFrame) -> float:
        """Computes the total job execution time from the traces.

        Args:
            traces: A DataFrame containing the I/O trace data,
                    expected to have 'time_start' and 'time_end' columns.

        Returns:
            The total job time as a float, NaN without traces.
        """
        end, start = traces[COL_TIME_END].max(), traces[COL_TIME_START].min()
        if end is None or start is None:
            return math.nan
        return end - start

    def ensure_proc_view_type(self, view_types: List[ViewType]) -> List[ViewType]:
        """Ensures that COL_PROC_NAME is always included in the list of view types.

        Args:
            view_types: A list of view types to be used for analysis.

        Returns:
            A sorted list of view types that always includes COL_PROC_NAME.
        """
        return list(sorted(set(view_types).union({COL_PROC_NAME})))

    def get_stats_checkpoint_name(self):
        return self.get_checkpoint_name(CHECKPOINT_RAW_STATS)

    def get_time_boundary_layer(self):
        return list(self.preset.layer_defs)[0]

    def get_total_event_count(self, traces: DataFrame) -> int:
        """Computes the total number of I/O events in the traces.

        Args:
            traces: A DataFrame containing the I/O trace data.

        Returns:
            The total count of I/O events as an integer.
        """
        return traces.num_rows

    def get_profile_event_count(self, profiles: Optional[DataFrame]):
        if profiles is None:
            return 0
        if COL_COUNT in profiles.columns:
            return int(profiles[COL_COUNT].fillna(0).sum())
        return profiles.num_rows

    def get_unique_host_count(self, traces: DataFrame):
        """Computes the total number of unique hosts accessed in the traces.

        Args:
            traces: A DataFrame containing the I/O trace data.

        Returns:
            The total count of unique hosts accessed as an integer.
        """
        return traces[COL_HOST_NAME].nunique()

    def get_unique_file_count(self, traces: DataFrame):
        """Computes the total number of unique files accessed in the traces.

        Args:
            traces: A DataFrame containing the I/O trace data.

        Returns:
            The total count of unique files accessed as an integer.
        """
        return traces[COL_FILE_NAME].nunique()

    def get_unique_process_count(self, traces: DataFrame):
        """Computes the total number of unique processes accessed in the traces.

        Args:
            traces: A DataFrame containing the I/O trace data.

        Returns:
            The total count of unique processes accessed as an integer.
        """
        return traces[COL_PROC_NAME].nunique()

    def checkpoint_file(self, name: str) -> str:
        """The table file of checkpoint `name`."""
        return f"{self.get_checkpoint_path(name=name)}.pfw.gz"

    def has_checkpoint(self, name: str) -> bool:
        return os.path.exists(self.checkpoint_file(name))

    def write_checkpoint(self, view: DataFrame, name: str) -> None:
        """Write the frame `view` as checkpoint `name`.

        A table list cell cannot hold NaN or inf (a caller time range can). Such a view gets no
        checkpoint, and an older one is removed, so the next run computes it again.
        """
        path = self.checkpoint_file(name)
        if _has_non_finite_list_value(view):
            logger.warning("Checkpoint skipped: a list cell holds NaN or inf", name=name)
            if os.path.exists(path):
                os.remove(path)
            return
        write_table(view, path, name)

    def read_checkpoint(self, name: str) -> DataFrame:
        """Read checkpoint `name` with its dtypes and index restored."""
        return read_table(self.checkpoint_file(name))

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
        'fallback' function to compute the data and then stores it.

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
                self.store_extra_data(data=data, data_path=data_path)
                return data
            with open(data_path, "r") as f:
                return json.load(f)
        return fallback()

    def restore_flat_views(self, view_keys: List[ViewKey]) -> Dict[ViewKey, DataFrame]:
        restored_flat_views = {}
        for view_key in view_keys:
            flat_view_checkpoint_name = self.get_checkpoint_name(CHECKPOINT_FLAT_VIEW, *list(view_key))
            if self.has_checkpoint(name=flat_view_checkpoint_name):
                restored_flat_views[view_key] = self.read_checkpoint(flat_view_checkpoint_name)
        return restored_flat_views

    def restore_view(
        self,
        name: str,
        fallback: Callable[[], DataFrame],
        force=False,
        write_to_disk=True,
    ) -> DataFrame:
        """Restores a DataFrame view from its checkpoint.

        If checkpointing is enabled and the checkpoint exists (unless 'force' is True),
        it reads the DataFrame from the checkpoint table. Otherwise, it calls the
        'fallback' function to compute the DataFrame. If 'write_to_disk' is True,
        the computed DataFrame is then written as a checkpoint in this process,
        so the frame is not copied into a Dask graph.

        Args:
            name: The name of the checkpoint.
            fallback: A callable function that returns the DataFrame if not found or forced.
            force: If True, forces recomputation even if a checkpoint exists.
            write_to_disk: If True, saves the computed view to disk if it was recomputed.

        Returns:
            A DataFrame representing the restored or computed view.
        """
        if self.checkpoint:
            if force or not self.has_checkpoint(name=name):
                with log_block("restore_view_fallback_build", name=name):
                    view = fallback()
                if not write_to_disk:
                    return view
                with log_block("restore_view_write_checkpoint", name=name):
                    self.write_checkpoint(view, name)
                return view
            with log_block("restore_view_read_checkpoint", name=name):
                return self.read_checkpoint(name)
        with log_block("restore_view_fallback_build_no_ckpt", name=name):
            return fallback()

    @staticmethod
    def store_extra_data(data: Dict, data_path: str):
        """Saves extra (non-DataFrame) data to a JSON file.

        Args:
            data: The dictionary of data to be saved.
            data_path: The full path to the JSON file where data will be stored.
        """
        with open(data_path, "w") as f:
            return json.dump(data, f)

    def store_flat_views(self, flat_views: Dict[ViewKey, DataFrame]) -> None:
        for view_key in flat_views:
            flat_view_checkpoint_name = self.get_checkpoint_name(CHECKPOINT_FLAT_VIEW, *list(view_key))
            if self.has_checkpoint(name=flat_view_checkpoint_name):
                continue
            self.write_checkpoint(flat_views[view_key], flat_view_checkpoint_name)

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

    def _analyze_hlm(
        self,
        hlm: Optional[DataFrame],
        proc_view_types: List[ViewType],
        metric_boundaries: ViewMetricBoundaries,
        trace_hlm: Optional[DataFrame],
        profile_hlm: Optional[DataFrame],
        raw_stats: RawStats,
        logical_view_types: bool,
        layer_main_views: Optional[Dict[Layer, DataFrame]] = None,
        system_metrics: Optional[DataFrame] = None,
    ) -> AnalysisResult:
        """
        Analyze the high-level metrics (HLM) and compute views for each layer.

        This method computes the main views and additional views for each layer, either from the provided
        high-level metrics DataFrame (`hlm`) or from precomputed main views (`layer_main_views`). At least
        one of `hlm` or `layer_main_views` must be provided. If `layer_main_views` is given and contains
        a main view for a layer, it will be used; otherwise, the main view will be computed from `hlm`.

        Args:
            hlm (DataFrame): The high-level metrics DataFrame. Required unless layer-specific HLM data or
                all main views are provided in `layer_main_views`.
            proc_view_types (List[ViewType]): List of view types to process for each layer.
            metric_boundaries (ViewMetricBoundaries): Boundaries for metrics used in view computation.
            trace_hlm (Optional[DataFrame]): Trace-derived HLM used for per-layer reconciliation.
            profile_hlm (Optional[DataFrame]): Profile-derived HLM used for per-layer reconciliation.
            raw_stats (RawStats): Raw statistics to be computed alongside the views.
            logical_view_types (bool): Whether to compute logical views in addition to main views.
            layer_main_views (Optional[Dict[Layer, DataFrame]]): Optional dictionary mapping each layer to its
                precomputed main view. If not provided, main views will be computed from `hlm`.

        Returns:
            AnalysisResult: The result of the analysis, including computed views and statistics.

        Raises:
            ValueError: If neither `hlm` nor `layer_main_views` is provided for a required layer.
        """
        # Compute layers & views
        with console_block("Compute views"):
            with log_block("create_layers_and_views_tasks"):
                hlms = {}
                main_views = {}
                views = {}
                view_keys = set()
                for layer, layer_condition in self.preset.layer_defs.items():
                    layer_hlm = None
                    if layer_main_views is not None and layer in layer_main_views:
                        layer_main_view = layer_main_views[layer]
                    else:
                        if hlm is not None:
                            layer_hlm = query(hlm, layer_condition) if layer_condition else hlm
                        else:
                            if trace_hlm is None and profile_hlm is None:
                                raise ValueError(
                                    "Either hlm or at least one of trace_hlm/profile_hlm must be provided "
                                    "when layer_main_views is not supplied"
                                )
                            layer_trace_hlm = trace_hlm
                            layer_profile_hlm = profile_hlm
                            if layer_condition:
                                if layer_trace_hlm is not None:
                                    layer_trace_hlm = query(layer_trace_hlm, layer_condition)
                                if layer_profile_hlm is not None:
                                    layer_profile_hlm = query(layer_profile_hlm, layer_condition)
                            if layer_trace_hlm is None:
                                layer_hlm = layer_profile_hlm
                            elif layer_profile_hlm is None:
                                layer_hlm = layer_trace_hlm
                            else:
                                layer_hlm = self.reconcile_hlm(
                                    layer=layer,
                                    profile_hlm=layer_profile_hlm,
                                    trace_hlm=layer_trace_hlm,
                                    view_types=proc_view_types,
                                )
                        layer_main_view = self.compute_main_view(
                            layer=layer,
                            hlm=layer_hlm,
                            view_types=proc_view_types,
                        )
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
                    views[layer] = layer_views
                    view_keys.update(layer_views.keys())

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
                        view = views[layer][view_key]
                        view = view.rename(columns={c: f"{layer.lower()}_{c}" for c in view.columns})
                        if view_key in flat_views:
                            flat_views[view_key] = _outer_join_on_index(flat_views[view_key], view)
                        else:
                            flat_views[view_key] = view
                    df = flat_views[view_key]
                    logger.debug("Flat view created", view_key=view_key, shape=df.shape, mem_bytes=frame_nbytes(df))

            # Compute metric boundaries for flat views
            with log_block("process_flat_views+metric_boundaries"):
                for view_key in flat_views:
                    if view_key in checkpointed_flat_views:
                        continue
                    view_type = view_key[-1]
                    top_layer = list(self.preset.layer_defs)[0]
                    time_proc_suffix = "time_sum" if self.is_view_process_based(view_key) else "time_proc_max"
                    # Sort before any order-sensitive metric is computed. Row
                    # order otherwise comes from whichever engine produced the
                    # view -- a DaskFrame shuffle or one frame's group-by -- and the
                    # `_frac_total` sums are floating point, so their last bits
                    # follow summation order. That is invisible until
                    # `rank(pct=True)` turns a near-tie into a visible jump.
                    # Sorting makes the output reproducible across engines and
                    # partition counts.
                    flat_views[view_key] = flat_views[view_key].sort_index()
                    with log_block("calculate_metric_boundary", view_key=view_key):
                        time_boundary = flat_views[view_key][f"{top_layer}_{time_proc_suffix}"].sum()
                        metric_boundaries.setdefault(view_type, {})
                        for layer in self.preset.layer_defs:
                            metric_boundaries[view_type][f"{layer}_{time_proc_suffix}"] = time_boundary
                    with log_block("process_flat_view", view_key=view_key):
                        # Process flat views to compute metrics and scores
                        flat_views[view_key] = self._process_flat_view(
                            flat_view=flat_views[view_key],
                            view_key=view_key,
                            system_metrics=system_metrics,
                        )

        # Checkpoint flat views if enabled
        if self.checkpoint:
            with log_block("write_flat_view_checkpoints"):
                self.store_flat_views(flat_views=flat_views)

        # Evaluate analysis facts over the flat views (no-op {} unless facts.enabled),
        # then gate the flat views by emit_flat_views. Facts-off leaves both untouched.
        with log_block("evaluate_analysis_facts"):
            analysis_facts = self._evaluate_analysis_facts(flat_views=flat_views, raw_stats=raw_stats)
        output_flat_views, output_analysis_facts = self._materialize_output_artifacts(
            flat_views=flat_views,
            analysis_facts=analysis_facts,
        )

        return AnalysisResult(
            _hlms=hlms,
            _main_views=main_views,
            _metric_boundaries=metric_boundaries,
            additional_metrics={
                view_type: list(metrics.keys())
                for view_type, metrics in (self.preset.additional_metrics or {}).items()
            },  # type: ignore
            analysis_facts=output_analysis_facts,
            checkpoint_dir=self.checkpoint_dir,
            flat_views=output_flat_views,
            layers=self.layers,
            raw_stats=raw_stats,
            view_types=proc_view_types,
            views=views,
        )

    @abc.abstractmethod
    def _compute_high_level_metrics(
        self,
        traces: DataFrame,
        view_types: list,
    ) -> DataFrame:
        """The HLM of `traces`: one row per `view_types` and HLM extra column key."""
        raise NotImplementedError

    def _compute_profile_hlm(
        self,
        profiles: DataFrame,
        view_types: List[ViewType],
    ) -> DataFrame:
        profiles = set_size_bins(profiles)
        return self._compute_high_level_metrics(
            traces=profiles,
            view_types=view_types,
        )

    def _validate_and_expand_profiles(
        self,
        profiles: DataFrame,
        profile_time_granularity: float,
    ) -> DataFrame:
        profile_ratio = self.time_granularity / profile_time_granularity
        inverse_ratio = profile_time_granularity / self.time_granularity
        is_coarser_aligned = math.isclose(profile_ratio, round(profile_ratio), rel_tol=0, abs_tol=1e-9)
        is_finer_aligned = math.isclose(inverse_ratio, round(inverse_ratio), rel_tol=0, abs_tol=1e-9)
        if not is_coarser_aligned and not is_finer_aligned:
            raise ValueError(
                f"Analysis granularity ({self.time_granularity}s) must evenly divide into "
                f"or be an integer multiple of the profile bucket width "
                f"({profile_time_granularity}s)"
            )
        if self.time_granularity < profile_time_granularity:
            expansion_factor = int(round(inverse_ratio))
            profiles = self._expand_profile_buckets(
                profiles,
                expansion_factor=expansion_factor,
                time_resolution=self.time_resolution,
                time_granularity=self.time_granularity,
                distribution=self.profile_distribution,
            )
        return profiles

    @staticmethod
    def _expand_profile_buckets(
        df: DataFrame,
        expansion_factor: int,
        time_resolution: float,
        time_granularity: float,
        distribution: str,
    ) -> DataFrame:
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
        if df.num_rows == 0:
            return df

        factor = expansion_factor
        sub_granularity_us = int(time_granularity * time_resolution)
        # No native per-row repeat (dftracer-utils branch feat/analyzer-port-gaps: repeat).
        positions = Series([row for row in range(df.num_rows) for _ in range(factor)])
        sub_idx = Series(list(range(factor)) * df.num_rows)
        expanded = df.gather(positions)

        expanded["time_start"] = expanded["time_start"] + sub_idx * sub_granularity_us
        expanded["time_end"] = expanded["time_start"] + sub_granularity_us
        expanded["time_range"] = floor_divide(expanded["time_start"], sub_granularity_us)

        if distribution == "weighted":
            t_min, t_max = df["time_min"].fillna(0), df["time_max"].fillna(0)
            span = t_max - t_min

            def ramp_at(k):
                return t_min + span * k / (factor - 1) if factor > 1 else t_min

            # Summed in sub-bucket order, as numpy sums a row of fewer than 8 values.
            ramp_sums = ramp_at(0)
            for k in range(1, factor):
                ramp_sums = ramp_sums + ramp_at(k)
            ramp = ramp_at(0).gather(positions)
            if factor > 1:
                ramp = t_min.gather(positions) + span.gather(positions) * sub_idx / (factor - 1)
            ramp_sums = ramp_sums.gather(positions)
            weights = (ramp / ramp_sums).mask(~(ramp_sums > 0), 1.0 / factor)
            expanded["time"] = (expanded["time"] * weights).astype("float64")
        else:
            expanded["time"] = (expanded["time"] / factor).astype("float64")

        # count and size always split uniformly (they are discrete totals); the
        # first `count % factor` sub-buckets take one more count.
        count = expanded["count"].fillna(0)
        expanded["count"] = count // factor + (sub_idx < count % factor).astype("int64")
        expanded["size"] = (expanded["size"].astype("float64") / factor).trunc().astype("int64")
        return expanded

    def _compute_hybrid_hlm(
        self,
        traces: DataFrame,
        profiles: DataFrame,
        view_types: List[ViewType],
    ) -> DataFrame:
        trace_hlm = self._compute_high_level_metrics(
            traces=traces,
            view_types=view_types,
        )
        profile_hlm = self.compute_profile_hlm(
            profiles=profiles,
            view_types=view_types,
        )
        return self._reconcile_hlm(
            profile_hlm=profile_hlm,
            trace_hlm=trace_hlm,
            view_types=view_types,
        )

    def _reconcile_hlm(
        self,
        trace_hlm: DataFrame,
        profile_hlm: DataFrame,
        view_types: List[ViewType],
        layer: Optional[Layer] = None,
    ) -> DataFrame:
        hlm, overlap_count = reconcile(trace_hlm, profile_hlm, view_types)
        if overlap_count > 0:
            logger.warning("Hybrid reconcile found overlapping HLM keys", layer=layer, overlap_count=overlap_count)
        else:
            logger.info("Hybrid reconcile found no overlapping HLM keys", layer=layer, overlap_count=overlap_count)
        return hlm

    def _compute_main_view(
        self,
        layer: Layer,
        hlm: DataFrame,
        view_types: List[ViewType],
    ) -> DataFrame:
        with log_block("drop_and_set_metrics", layer=layer):
            size_layers = {configured_layer.lower() for configured_layer in (self.preset.size_layers or [])}
            keep_size_for_layer = layer.lower() in size_layers
            if not keep_size_for_layer:
                size_cols = [col for col in hlm.columns if col.startswith("size")]
                hlm = hlm.drop(columns=size_cols)
                if "file_name" in hlm.columns:
                    hlm = hlm.drop(columns=["file_name"])
        with log_block("compute_main_view", layer=layer):
            main_view = group_main_view(
                hlm,
                view_types,
                self.time_sliced,
                derived_metrics=self.preset.derived_metrics[layer],
                size_derived_metrics=(self.preset.size_derived_metrics or {}).get(layer.lower(), []),
            )
        return main_view

    @abc.abstractmethod
    def _compute_view(
        self,
        layer: Layer,
        records: Union[DataFrame, PartitionedRecords],
        view_key: ViewKey,
        view_type: str,
        view_types: List[ViewType],
    ) -> DataFrame:
        """The view of `records` per `view_type` row."""
        raise NotImplementedError

    def _process_flat_view(
        self,
        flat_view: DataFrame,
        view_key: ViewKey,
        system_metrics: Optional[DataFrame] = None,
    ):
        view_type = view_key[-1]
        is_view_process_based = self.is_view_process_based(view_key)
        with log_block("set_flat_view_metrics", view_key=view_key):
            flat_view = process_flat_metrics(
                flat_view,
                is_view_process_based=is_view_process_based,
                layers=self.layers,
                layer_deps=self.preset.layer_deps,
                async_layers=self.preset.async_layers,
                derived_metrics=self.preset.derived_metrics,
                time_boundary_layer=self.get_time_boundary_layer(),
                additional=(self.preset.additional_metrics or {}).get(view_type, {}),
                time_interval=self.time_granularity,
            )
        if system_metrics is not None and system_metrics.num_rows and COL_TIME_RANGE in view_key:
            with log_block("join_system_metrics", view_key=view_key):
                sys_cols = [c for c in system_metrics.columns if c.startswith("sys_")]
                if sys_cols:
                    keys = flat_view.index_names()
                    sys_lookup = system_metrics.select(COL_TIME_RANGE, *sys_cols)
                    flat_view = (
                        flat_view.reset_index()
                        .merge(sys_lookup, how="left", on=COL_TIME_RANGE, nulls_equal=True)
                        .set_index(keys)
                    )
        return flat_view[sorted(flat_view.columns)]
