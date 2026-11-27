from __future__ import annotations

import dataclasses as dc
import hashlib
import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, Tuple

from dftracer.utils import DataFrame

from .constants import HUMANIZED_METRICS, HUMANIZED_VIEW_TYPES, Layer


class Score(Enum):
    TRIVIAL = 'trivial'
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'


Metric = str
ViewType = Literal['file_name', 'host_name', 'proc_name', 'step', 'time_range']
ViewKey = Union[
    Tuple[ViewType],
    Tuple[ViewType, ViewType],
    Tuple[ViewType, ViewType, ViewType],
    Tuple[ViewType, ViewType, ViewType, ViewType],
]


@dc.dataclass
class AnalysisRuntimeConfig:
    checkpoint: bool
    cluster_type: str
    debug: bool
    memory: int
    num_threads_per_worker: int
    num_workers: int
    processes: bool
    verbose: bool
    working_dir: str


@dc.dataclass
class RawStats:
    job_time: float
    time_granularity: int
    time_resolution: int
    trace_event_count: int
    profile_event_count: int
    total_event_count: int
    unique_file_count: int
    unique_host_count: int
    unique_process_count: int


@dc.dataclass
class ReadTraceResult:
    traces: DataFrame
    profiles: Optional[DataFrame] = None
    profile_time_granularity: Optional[float] = None
    system_metrics: Optional[DataFrame] = None


@dc.dataclass
class RuleReason:
    condition: str
    message: str


@dc.dataclass
class Rule:
    name: str
    condition: str
    layers: Optional[List[Layer]] = None
    reasons: Optional[List[RuleReason]] = None


@dc.dataclass
class RuleResultReason:
    description: str
    # value: Optional[float]


@dc.dataclass
class RuleResult:
    description: str
    compact_desc: Optional[str] = None
    detail_list: Optional[List[str]] = None
    extra_data: Optional[dict] = None
    object_hash: Optional[int] = None
    reasons: Optional[List[RuleResultReason]] = None
    value: Optional[Union[float, int, tuple]] = None
    value_fmt: Optional[str] = None


View = DataFrame

Characteristics = Dict[str, RuleResult]
MetricBoundary = Union[int, float]
MetricBoundaries = Dict[str, Any]
ViewMetricBoundaries = Dict[str, MetricBoundaries]
Views = Dict[ViewKey, View]


# --- Analysis facts (analyzer.fact-envelope.v1) -------------------------------
# The analyzer emits AnalysisFacts (one per analysis window per view); DFDiagnoser
# consumes them. See docs/window-as-longitudinal-axis.md.


@dc.dataclass
class FactWindow:
    run_id: Optional[str] = None
    view_type: Optional[str] = None
    # Longitudinal coordinate fields, one per temporal view type. For a given
    # temporal view exactly one is the AXIS (window->window_index, epoch->epoch,
    # step->step, time_range->time_bucket); the diagnoser selects it by view_type.
    # The others, when set, are workload metadata for context. t0_ns/t1_ns are the
    # time bounds (time_range). See docs/window-as-longitudinal-axis.md.
    window_index: Optional[int] = None
    epoch: Optional[int] = None
    step: Optional[int] = None
    time_bucket: Optional[int] = None
    t0_ns: Optional[int] = None
    t1_ns: Optional[int] = None
    trigger: Optional[str] = None


@dc.dataclass
class FactScope:
    workload: Optional[str] = None
    layer: Optional[str] = None
    entity: Optional[str] = None
    rank_set: Optional[str] = None
    node: Optional[str] = None


@dc.dataclass
class FactSeverity:
    score: float
    label: str
    method: str = "rule_expr"


@dc.dataclass
class FactProvenance:
    rule_id: str
    rule_version: str
    metric_source: str = "flat_view"
    view_key: Optional[List[str]] = None


@dc.dataclass
class AnalysisFact:
    fact_type: str
    window: FactWindow
    scope: FactScope
    evidence: Dict[str, Any]
    severity: Optional[FactSeverity] = None
    confidence: Optional[float] = None
    opportunity_tags: List[str] = dc.field(default_factory=list)
    suppresses_tags: List[str] = dc.field(default_factory=list)
    provenance: Optional[FactProvenance] = None
    schema_version: str = "analysisfact.v1"
    fact_id: Optional[str] = None

    def finalize_id(self):
        """Set deterministic fact_id when it is not explicitly provided."""
        if self.fact_id:
            return self.fact_id
        scope_key = f"{self.scope.layer}:{self.scope.entity}:{self.scope.rank_set}"
        window_key = (
            f"{self.window.run_id}:{self.window.view_type}:{self.window.epoch}:{self.window.step}:"
            f"{self.window.t0_ns}:{self.window.t1_ns}"
        )
        fact_key = f"{self.schema_version}:{self.fact_type}:{window_key}:{scope_key}"
        self.fact_id = f"af_{hashlib.md5(fact_key.encode('utf-8')).hexdigest()[:12]}"
        return self.fact_id


@dc.dataclass
class FactEnvelopeContext:
    run_id: Optional[str] = None
    layers: List[str] = dc.field(default_factory=list)
    view_types: List[str] = dc.field(default_factory=list)
    time_granularity: Optional[float] = None
    time_resolution: Optional[float] = None
    total_event_count: Optional[int] = None
    window_type_counts: Dict[str, int] = dc.field(default_factory=dict)


@dc.dataclass
class FactEnvelope:
    schema_version: str = "analyzer.fact-envelope.v1"
    context: FactEnvelopeContext = dc.field(default_factory=FactEnvelopeContext)
    facts: List[AnalysisFact] = dc.field(default_factory=list)
    fact_count_by_view: Dict[str, int] = dc.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dc.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dc.dataclass
class OutputCharacteristicsType:
    complexity: float
    io_time: float
    job_time: float
    num_apps: int
    num_files: int
    num_nodes: int
    num_ops: int
    num_procs: int
    num_time_periods: int
    per_io_time: float


@dc.dataclass
class OutputCountsType:
    raw_count: int
    hlm_count: int
    main_view_count: int
    avg_perspective_count: Dict[str, int]
    avg_perspective_count_std: Dict[str, float]
    avg_perspective_critical_count: Dict[str, int]
    avg_perspective_critical_count_std: Dict[str, float]
    perspective_skewness: Dict[str, float]
    root_perspective_skewness: Dict[str, float]
    per_records_discarded: Dict[str, float]
    per_records_retained: Dict[str, float]
    num_metrics: int
    num_perspectives: int
    num_rules: int
    evaluated_records: Dict[str, int]
    perspective_count_tree: Dict[str, Dict[str, int]]
    perspective_critical_count_tree: Dict[str, Dict[str, int]]
    perspective_record_count_tree: Dict[str, Dict[str, int]]
    reasoned_records: Dict[str, int]
    slope_filtered_records: Dict[str, int]


@dc.dataclass
class OutputSeveritiesType:
    critical_count: Dict[str, int]
    critical_tree: Dict[str, Dict[str, int]]
    very_high_count: Dict[str, int]
    very_high_tree: Dict[str, Dict[str, int]]
    high_count: Dict[str, int]
    high_tree: Dict[str, Dict[str, int]]
    medium_count: Dict[str, int]
    medium_tree: Dict[str, Dict[str, int]]
    low_count: Dict[str, int]
    very_low_count: Dict[str, int]
    trivial_count: Dict[str, int]
    none_count: Dict[str, int]
    root_critical_count: Dict[str, int]
    root_very_high_count: Dict[str, int]
    root_high_count: Dict[str, int]
    root_medium_count: Dict[str, int]
    root_low_count: Dict[str, int]
    root_very_low_count: Dict[str, int]
    root_trivial_count: Dict[str, int]
    root_none_count: Dict[str, int]


@dc.dataclass
class OutputThroughputsType:
    evaluated_records: Dict[str, float]
    perspectives: Dict[str, float]
    reasoned_records: Dict[str, float]
    rules: Dict[str, float]
    slope_filtered_records: Dict[str, float]


@dc.dataclass
class OutputTimingsType:
    read_traces: Dict[str, float]
    compute_hlm: Dict[str, float]
    compute_main_view: Dict[str, float]
    compute_perspectives: Dict[str, float]
    attach_reasons: Dict[str, float]


@dc.dataclass
class OutputType:
    _characteristics: Characteristics
    _raw_stats: RawStats
    characteristics: OutputCharacteristicsType
    counts: OutputCountsType
    severities: OutputSeveritiesType
    throughputs: OutputThroughputsType
    timings: OutputTimingsType


@dc.dataclass
class AnalysisResult:
    additional_metrics: Dict[ViewType, List[str]]
    checkpoint_dir: str
    flat_views: Dict[ViewKey, DataFrame]
    layers: List[Layer]
    raw_stats: RawStats
    view_types: List[ViewType]
    views: Dict[Layer, Views]
    _hlms: Dict[Layer, DataFrame]
    _main_views: Dict[Layer, DataFrame]
    _metric_boundaries: ViewMetricBoundaries
    analysis_facts: Dict[ViewKey, List[AnalysisFact]] = dc.field(default_factory=dict)
    _read_result: Optional[ReadTraceResult] = None
    _trace_run: Optional[Any] = None

    def get_hlm(self, layer: Layer) -> DataFrame:
        return self._hlms[layer]

    def get_main_view(self, layer: Layer) -> DataFrame:
        return self._main_views[layer]

    @property
    def traces(self) -> Optional[Any]:
        """The raw events of the analyzed traces, one row per event.

        A lazy, distributed ``DaskTraceViewer`` over the analyzed files, on the
        analyzer's Dask client. Nothing is read until a terminal call, such as
        ``result.traces.page(10).table`` or
        ``for table in result.traces.filter('cat == "POSIX"').pages(): ...``.
        """
        if self._trace_run is None or not self._trace_run.files:
            return None
        return self._trace_run.viewer()

    @property
    def aggregated_traces(self) -> Optional[DataFrame]:
        """The events grouped by category, function, process, file and time bucket."""
        if self._read_result is None:
            return None
        return self._read_result.traces

    @property
    def profiles(self) -> Optional[DataFrame]:
        if self._read_result is None:
            return None
        return self._read_result.profiles

    @property
    def system_metrics(self) -> Optional[DataFrame]:
        if self._read_result is None:
            return None
        return self._read_result.system_metrics

    def get_flat_view(self, view_key_type: Union[ViewKey, ViewType]) -> DataFrame:
        if not isinstance(view_key_type, tuple):
            view_key_type = (view_key_type,)
        return self.flat_views[view_key_type]

    def get_layer_view(self, layer: Layer, view_key_type: Union[ViewKey, ViewType]) -> DataFrame:
        if not isinstance(view_key_type, tuple):
            view_key_type = (view_key_type,)
        return self.views[layer][view_key_type]

    def get_analysis_facts(self, view_key_type: Union[ViewKey, ViewType]) -> List[AnalysisFact]:
        if not isinstance(view_key_type, tuple):
            view_key_type = (view_key_type,)
        return self.analysis_facts.get(view_key_type, [])

    def iter_analysis_facts(self):
        for view_key, facts in self.analysis_facts.items():
            for fact in facts:
                yield view_key, fact

    def to_fact_envelope(self) -> FactEnvelope:
        raw_stats_dict: Dict[str, Any] = {}
        if isinstance(self.raw_stats, dict):
            raw_stats_dict = dict(self.raw_stats)
        elif dc.is_dataclass(self.raw_stats):
            raw_stats_dict = dc.asdict(self.raw_stats)

        run_id = raw_stats_dict.get("run_id")
        fact_count_by_view: Dict[str, int] = {}
        window_type_counts: Dict[str, int] = {}
        facts_flat: List[AnalysisFact] = []

        for view_key, fact in self.iter_analysis_facts():
            view_key_name = "_".join(view_key)
            fact_count_by_view[view_key_name] = fact_count_by_view.get(view_key_name, 0) + 1
            if fact.window.view_type:
                window_type_counts[fact.window.view_type] = window_type_counts.get(fact.window.view_type, 0) + 1
            if run_id is None and fact.window.run_id:
                run_id = fact.window.run_id
            facts_flat.append(fact)

        context = FactEnvelopeContext(
            run_id=None if run_id is None else str(run_id),
            layers=[str(layer) for layer in self.layers],
            view_types=[str(view_type) for view_type in self.view_types],
            time_granularity=_to_opt_float(raw_stats_dict.get("time_granularity")),
            time_resolution=_to_opt_float(raw_stats_dict.get("time_resolution")),
            total_event_count=_to_opt_int(raw_stats_dict.get("total_event_count")),
            window_type_counts=window_type_counts,
        )
        return FactEnvelope(
            context=context,
            facts=facts_flat,
            fact_count_by_view=fact_count_by_view,
        )


def _to_opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_opt_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def humanized_metric_name(metric: Metric):
    return HUMANIZED_METRICS[metric]


def humanized_view_name(view_key_type: Union[ViewKey, ViewType], separator='_'):
    if isinstance(view_key_type, tuple):
        return separator.join([HUMANIZED_VIEW_TYPES[view_type] for view_type in view_key_type])
    return HUMANIZED_VIEW_TYPES[view_key_type]


def view_name(view_key_type: Union[ViewKey, ViewType], separator='_'):
    return separator.join(view_key_type) if isinstance(view_key_type, tuple) else view_key_type
