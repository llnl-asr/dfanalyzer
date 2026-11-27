import dataclasses as dc
import math
import os
from pathlib import Path
import structlog
from typing import Any, Dict, List, Mapping, Optional
from dftracer.utils import DataFrame, Series
from omegaconf import OmegaConf

from .analysis_utils import is_missing, nan_as_null
from .frames.expr import evaluate
from .fact_rules import FactRule, FactRuleValidationError, build_fact_rule
from .scoring import normalize_slope
from .types import AnalysisFact, FactProvenance, FactScope, FactSeverity, FactWindow, ViewKey


logger = structlog.get_logger()
_VALUE = "__fact_value"
_SEVERITY = "__fact_severity"
_CONFIDENCE = "__fact_confidence"
_POSITION = "__fact_position"
_RANK = "__fact_rank"

RESERVED_IDENTIFIERS = {
    "True",
    "False",
    "None",
    "and",
    "or",
    "not",
    "max",
    "min",
    "abs",
    "clip01",
    "fillna0",
    "math",
}


def _as_dict(raw_stats: Any) -> Dict[str, Any]:
    if raw_stats is None:
        return {}
    if isinstance(raw_stats, Mapping):
        return dict(raw_stats)
    if dc.is_dataclass(raw_stats):
        return dc.asdict(raw_stats)
    return {}


def _to_scalar(value: Any) -> Any:
    return None if is_missing(value) else value


def _to_series(df: DataFrame, value: Any) -> Series:
    """`value` as a Series with the row labels of `df`; a scalar repeats on every row."""
    if isinstance(value, Series):
        return value
    out = df[[]]
    out[_VALUE] = Series([value] * df.num_rows)
    return out[_VALUE]


def _first_row(df: DataFrame, names) -> dict[str, Any]:
    """The values of the columns `names` on the first row of `df`, a missing column as None."""
    row = df.head(1).to_dicts()
    row = row[0] if row else {}
    return {m: _to_scalar(row.get(m)) for m in names}


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except Exception:
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _to_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _severity_label(score: float) -> str:
    if score >= 0.85:
        return "critical"
    if score >= 0.60:
        return "high"
    if score >= 0.30:
        return "medium"
    if score > 0:
        return "low"
    return "none"


def _clip01(value: Any) -> Any:
    if isinstance(value, Series):
        return value.clip(lower=0.0, upper=1.0)
    try:
        return min(1.0, max(0.0, float(value)))
    except Exception:
        return value


def _fillna0(value: Any) -> Any:
    if isinstance(value, Series):
        return value.fillna(0.0)
    return 0.0 if is_missing(value) else value


def _na_safe_pick(a: Any, b: Any, prefer_a) -> Any:
    """`a` or `b` per row by `prefer_a(a, b)`; a missing value loses to a number."""
    if not isinstance(a, Series):
        a, b = b, a
    a = nan_as_null(a, as_float=True)
    if isinstance(b, Series):
        b = nan_as_null(b, as_float=True)
        return a.where(prefer_a(a, b).fillna(False) | b.is_null(), b)
    if is_missing(b):
        return a
    return a.where(prefer_a(a, b).fillna(False), float(b))


def _na_safe_max(*args):
    """`max` of numbers or Series; on a Series a missing value loses to a number, as `np.fmax`."""
    result = args[0]
    for arg in args[1:]:
        if isinstance(result, Series) or isinstance(arg, Series):
            result = _na_safe_pick(result, arg, lambda a, b: a >= b)
        else:
            result = max(result, arg)
    return result


def _na_safe_min(*args):
    """`min` of numbers or Series; on a Series a missing value loses to a number, as `np.fmin`."""
    result = args[0]
    for arg in args[1:]:
        if isinstance(result, Series) or isinstance(arg, Series):
            result = _na_safe_pick(result, arg, lambda a, b: a <= b)
        else:
            result = min(result, arg)
    return result


def _abs(value: Any) -> Any:
    return value.abs() if isinstance(value, Series) else abs(value)


FACT_HELPERS: dict[str, Any] = {
    "max": _na_safe_max,
    "min": _na_safe_min,
    "abs": _abs,
    "clip01": _clip01,
    "fillna0": _fillna0,
    "math": math,
}


def _eval_expr(df: DataFrame, expression: str) -> Series:
    """A fact rule expression on `df`, one value per row."""
    return _to_series(df, evaluate(df, expression, names=FACT_HELPERS))


DEFAULT_AGG_TOP_K = 5
# The TEMPORAL view: its rows are analysis WINDOWS (the longitudinal axis the
# diagnoser tracks persistence/trend on), not entities. Such facts emit per row
# with an entity-free scope and the window number as FactWindow.window. Everything
# else is spatial (rolled up / per-entity). See docs/window-as-longitudinal-axis.md.
# Temporal views are longitudinal axes the diagnoser tracks persistence/trend on.
# Online streaming uses `window`; offline batch uses the trace's natural temporal
# dimensions (epoch/step/time_range). Spatial views (file/proc/host) are one-shot.
TEMPORAL_VIEW_TYPES = {"window", "epoch", "step", "time_range"}


class FactEngine:
    def __init__(self, rules: List[FactRule], default_emit_mode: str = "aggregate"):
        self.rules = sorted(rules, key=lambda rule: rule.priority, reverse=True)
        self.default_emit_mode = default_emit_mode
        self.emitter = FactEmitter()

    @classmethod
    def from_rule_file(
        cls,
        rule_file: str,
        *,
        strict_time_semantics: bool,
        allow_mixed_time_aggregates: bool,
        default_emit_mode: str = "aggregate",
    ) -> "FactEngine":
        if not rule_file:
            logger.info("facts.rules.none")
            return cls([], default_emit_mode=default_emit_mode)
        resolved_rule_file = cls._resolve_rule_file(rule_file)
        if not os.path.exists(resolved_rule_file):
            raise FactRuleValidationError(f"Fact rule file does not exist: {resolved_rule_file}")

        config = OmegaConf.load(resolved_rule_file)
        config_obj = OmegaConf.to_object(config)
        defaults = dict(config_obj.get("defaults", {}))
        raw_rules = list(config_obj.get("rules", []))

        compiled_rules: List[FactRule] = []
        for raw_rule in raw_rules:
            try:
                compiled_rules.append(
                    build_fact_rule(
                        raw_rule=raw_rule,
                        defaults=defaults,
                        strict_time_semantics=strict_time_semantics,
                        allow_mixed_time_aggregates=allow_mixed_time_aggregates,
                    )
                )
            except FactRuleValidationError as error:
                logger.error("facts.rule.invalid", rule_id=raw_rule.get("id"), error=str(error))
                raise

        logger.info("facts.rules.loaded", rule_file=resolved_rule_file, count=len(compiled_rules))
        return cls(compiled_rules, default_emit_mode=default_emit_mode)

    @staticmethod
    def _resolve_rule_file(rule_file: str) -> str:
        path = Path(rule_file)
        candidates: List[Path] = []
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(Path.cwd() / path)
            if path.suffix == "":
                candidates.append(Path.cwd() / f"{rule_file}.yaml")
                candidates.append(Path.cwd() / f"{rule_file}.yml")
            package_rules = Path(__file__).resolve().parent / "configs" / "fact_rules"
            candidates.append(package_rules / path)
            if path.suffix == "":
                candidates.append(package_rules / f"{rule_file}.yaml")
                candidates.append(package_rules / f"{rule_file}.yml")

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0]) if candidates else rule_file

    def evaluate(
        self,
        flat_views: Dict[ViewKey, DataFrame],
        raw_stats: Any,
    ) -> Dict[ViewKey, List[AnalysisFact]]:
        if not self.rules:
            return {}

        raw_stats_dict = _as_dict(raw_stats)
        all_facts: Dict[ViewKey, List[AnalysisFact]] = {}
        for view_key, view_df in flat_views.items():
            source_view = view_key[-1]
            view_rules = [rule for rule in self.rules if rule.matches_view(source_view)]
            if not view_rules:
                continue
            facts = self._evaluate_view_rules(
                rule_df=view_df,
                rules=view_rules,
                raw_stats=raw_stats_dict,
                view_key=view_key,
            )
            if facts:
                all_facts[view_key] = facts
        if all_facts:
            logger.info(
                "facts.evaluate.done",
                total_facts=sum(len(v) for v in all_facts.values()),
                view_keys=list(all_facts.keys()),
            )
        return all_facts

    def _evaluate_view_rules(
        self,
        rule_df: DataFrame,
        rules: List[FactRule],
        raw_stats: Dict[str, Any],
        view_key: ViewKey,
    ) -> List[AnalysisFact]:
        run_id = raw_stats.get("run_id")
        view_type = view_key[-1]
        time_granularity = _to_float_or_none(raw_stats.get("time_granularity"))
        time_resolution = _to_float_or_none(raw_stats.get("time_resolution"))
        facts: List[AnalysisFact] = []

        for rule in rules:
            projected_columns = [column for column in rule.projected_columns() if column in rule_df.columns]
            missing_required = [column for column in rule.required_metrics if column not in projected_columns]
            if missing_required:
                logger.debug("facts.rule.skip", rule_id=rule.id, reason="missing_metrics", missing=missing_required)
                continue

            working_df = rule_df[projected_columns]

            try:
                for metric_name, expression in rule.derived_metrics.items():
                    working_df[metric_name] = _eval_expr(working_df, expression)
            except Exception:
                logger.warning("facts.rule.derived_metric_error", rule_id=rule.id, exc_info=True)
                continue

            # Log derived metric values for debugging rule evaluation
            if rule.derived_metrics:
                try:
                    derived_sample = _first_row(working_df, rule.derived_metrics)
                except Exception:
                    derived_sample = {}
                logger.debug(
                    "facts.rule.evaluate",
                    rule_id=rule.id,
                    derived_metrics=derived_sample,
                    when_result="pending",
                )

            unresolved_identifiers = [
                name
                for name in rule.referenced_columns
                if name not in working_df.columns and name not in RESERVED_IDENTIFIERS
            ]
            if unresolved_identifiers:
                logger.debug(
                    "facts.rule.skip_unresolved_identifiers",
                    rule_id=rule.id,
                    unresolved=sorted(unresolved_identifiers),
                )
                continue

            condition_series = _eval_expr(working_df, rule.when)
            condition_mask = condition_series.fillna(False).astype("bool")
            if not condition_mask.any():
                try:
                    derived_vals = _first_row(working_df, rule.derived_metrics)
                except Exception:
                    derived_vals = {}
                logger.debug(
                    "facts.rule.evaluate",
                    rule_id=rule.id,
                    derived_metrics=derived_vals,
                    when_result=False,
                    when=rule.when,
                    rows=working_df.num_rows,
                )
                continue

            severity_series = _eval_expr(working_df, rule.severity_score).clip(lower=0.0, upper=1.0)
            confidence_series = None
            if rule.confidence:
                confidence_series = _eval_expr(working_df, rule.confidence).clip(lower=0.0, upper=1.0)

            provenance = FactProvenance(
                rule_id=rule.id, rule_version=rule.rule_version, view_key=list(view_key)
            )
            rule_facts = self.emitter.emit(
                emit_mode=rule.emit_mode or self.default_emit_mode,
                fact_type=rule.fact_type,
                scope_layer=rule.scope_layer,
                view_type=view_type,
                view_key=view_key,
                run_id=run_id,
                working_df=working_df,
                condition_mask=condition_mask,
                severity_series=severity_series,
                confidence_series=confidence_series,
                evidence_metric_cols=list(rule.required_metrics) + list(rule.derived_metrics),
                opportunity_tags=rule.opportunity_tags,
                suppresses_tags=rule.suppresses_tags,
                provenance=provenance,
                trigger="rule_eval",
                severity_method="rule_expr",
                time_granularity=time_granularity,
                time_resolution=time_resolution,
            )
            for fact in rule_facts:
                logger.debug(
                    "facts.rule.fired",
                    rule_id=rule.id,
                    fact_type=fact.fact_type,
                    severity_score=fact.severity.score,
                    opportunity_tags=fact.opportunity_tags,
                )
            facts.extend(rule_facts)
        return facts


class FactEmitter:
    """Builds AnalysisFacts from a per-entity (mask, severity) evaluation.

    Shared by the rule builder and the metric builder so both produce
    identical-shaped facts (aggregate rollup vs per-entity detail); only the
    provenance and severity.method differ between the two sources.
    """

    def __init__(self, top_k: int = DEFAULT_AGG_TOP_K):
        self.top_k = top_k

    def emit(
        self,
        *,
        emit_mode: str,
        fact_type: str,
        scope_layer: Optional[str],
        view_type: str,
        view_key: ViewKey,
        run_id,
        working_df: DataFrame,
        condition_mask: Series,
        severity_series: Series,
        confidence_series,
        evidence_metric_cols: List[str],
        opportunity_tags: List[str],
        suppresses_tags: List[str],
        provenance: FactProvenance,
        trigger: str,
        severity_method: str = "rule_expr",
        time_granularity: Optional[float] = None,
        time_resolution: Optional[float] = None,
    ) -> List[AnalysisFact]:
        if not condition_mask.any():
            return []
        # Temporal (window) views never collapse to an aggregate -- that would
        # collapse the time series the diagnoser needs. They always emit per row.
        if emit_mode == "aggregate" and view_type not in TEMPORAL_VIEW_TYPES:
            fact = self._aggregate(
                fact_type=fact_type, scope_layer=scope_layer, view_type=view_type,
                run_id=run_id, working_df=working_df, condition_mask=condition_mask,
                severity_series=severity_series, confidence_series=confidence_series,
                evidence_metric_cols=evidence_metric_cols, opportunity_tags=opportunity_tags,
                suppresses_tags=suppresses_tags, provenance=provenance, trigger=trigger,
                severity_method=severity_method,
            )
            return [fact] if fact is not None else []
        return self._detail(
            fact_type=fact_type, scope_layer=scope_layer, view_type=view_type,
            run_id=run_id, working_df=working_df, condition_mask=condition_mask,
            severity_series=severity_series, confidence_series=confidence_series,
            evidence_metric_cols=evidence_metric_cols, opportunity_tags=opportunity_tags,
            suppresses_tags=suppresses_tags, provenance=provenance, trigger=trigger,
            severity_method=severity_method, time_granularity=time_granularity,
            time_resolution=time_resolution,
        )

    def _matched_rows(self, working_df, condition_mask, severity_series, confidence_series):
        """The rows of `working_df` where `condition_mask` holds, in row order, and the names of their label columns.

        The index levels become the label columns; a frame with no index has none. Severity
        and confidence come along as columns, NaN as missing; `_POSITION` is the row number.
        """
        scored = working_df.copy()
        scored[_SEVERITY] = nan_as_null(severity_series, as_float=True)
        if confidence_series is not None:
            scored[_CONFIDENCE] = nan_as_null(confidence_series, as_float=True)
        scored[_POSITION] = Series(list(range(scored.num_rows)), dtype="int64")
        matched = scored.filter(condition_mask)
        if not working_df.index_names():
            return matched, []
        # `reset_index` names an unnamed level `index`, which `index_names` does not report.
        matched = matched.reset_index()
        return matched, [c for c in matched.columns if c not in scored.columns]

    @staticmethod
    def _entity(row: dict[str, Any], label_columns: list[str]) -> Any:
        """The row's index label; the row number for a frame with no index, as the pandas default index."""
        if not label_columns:
            return row[_POSITION]
        if len(label_columns) == 1:
            return row[label_columns[0]]
        return tuple(row[name] for name in label_columns)

    def _top_k(self, matched: DataFrame) -> DataFrame:
        """The `top_k` rows by severity as pandas `Series.nlargest` picks them: ties in row order, missing last."""
        # `top_k` ranks lexicographically; the negated row number keeps the first of equal severities.
        ranked = matched.filter(matched[_SEVERITY].is_not_null())
        ranked[_RANK] = ranked[_POSITION] * -1
        ranked = ranked.top_k(self.top_k, by=[_SEVERITY, _RANK]).drop(columns=[_RANK])
        if ranked.num_rows < self.top_k:
            missing = matched.filter(matched[_SEVERITY].is_null()).head(self.top_k - ranked.num_rows)
            ranked = ranked.concat(missing)
        return ranked

    def _aggregate(
        self, *, fact_type, scope_layer, view_type, run_id, working_df, condition_mask,
        severity_series, confidence_series, evidence_metric_cols, opportunity_tags,
        suppresses_tags, provenance, trigger, severity_method,
    ) -> Optional[AnalysisFact]:
        """Roll the per-entity evaluation up into one aggregate fact for the
        whole view: count/fraction over threshold, the worst-K entities, and the
        aggregate (max) severity. Reuses the row-wise mask/severity — no second pass."""
        matched, label_columns = self._matched_rows(working_df, condition_mask, severity_series, confidence_series)
        affected_count = matched.num_rows
        total_count = working_df.num_rows
        if affected_count == 0:
            return None

        agg_severity = _to_float_or_none(matched[_SEVERITY].max())
        agg_severity = 0.0 if agg_severity is None else agg_severity
        agg_confidence = 1.0
        if confidence_series is not None:
            conf = _to_float_or_none(matched[_CONFIDENCE].max())
            agg_confidence = 1.0 if conf is None else conf

        metric_cols = [m for m in evidence_metric_cols if m in working_df.columns]
        top_k = []
        for row in self._top_k(matched).to_dicts():
            top_k.append({
                "entity": str(self._entity(row, label_columns)),
                "severity": _to_float_or_none(row[_SEVERITY]),
                "metrics": {m: _to_scalar(row[m]) for m in metric_cols},
            })

        fact = AnalysisFact(
            fact_type=fact_type,
            window=FactWindow(run_id=run_id, view_type=view_type, trigger=trigger),
            # entity=None marks this as the whole-view aggregate (vs a per-entity detail fact)
            scope=FactScope(layer=scope_layer, entity=None, rank_set="all", node=None),
            evidence={
                "metrics": {
                    "affected_count": affected_count,
                    "total_count": total_count,
                    "affected_fraction": affected_count / total_count if total_count else 0.0,
                },
                "top_k": top_k,
            },
            severity=FactSeverity(score=agg_severity, label=_severity_label(agg_severity), method=severity_method),
            confidence=agg_confidence,
            opportunity_tags=list(opportunity_tags),
            suppresses_tags=list(suppresses_tags) if suppresses_tags else [],
            provenance=provenance,
        )
        fact.finalize_id()
        return fact

    def _detail(
        self, *, fact_type, scope_layer, view_type, run_id, working_df, condition_mask,
        severity_series, confidence_series, evidence_metric_cols, opportunity_tags,
        suppresses_tags, provenance, trigger, severity_method, time_granularity, time_resolution,
    ) -> List[AnalysisFact]:
        """Emit one fact per matched entity (the per-entity drill-down form)."""
        columns = set(working_df.columns)
        matched, label_columns = self._matched_rows(working_df, condition_mask, severity_series, confidence_series)
        facts: List[AnalysisFact] = []
        for row in matched.to_dicts():
            row_index = self._entity(row, label_columns)
            severity_score = _to_float_or_none(row[_SEVERITY])
            severity_score = 0.0 if severity_score is None else severity_score
            confidence = 1.0
            if confidence_series is not None:
                confidence = _to_float_or_none(row[_CONFIDENCE])
                confidence = 1.0 if confidence is None else confidence
            evidence_metrics = {m: _to_scalar(row[m]) for m in evidence_metric_cols if m in columns}

            window_index = epoch = step = time_bucket = t0_ns = t1_ns = None
            # Each temporal view's row index IS its longitudinal coordinate, stamped
            # into that view's NATURAL field: window->window_index, epoch->epoch,
            # step->step, time_range->time_bucket. epoch/step are ALSO stamped as
            # metadata on other views when present as row columns. The diagnoser
            # selects the coordinate per view_type. See
            # docs/window-as-longitudinal-axis.md.
            if view_type == "window":
                window_index = _to_int_or_none(row_index)
            if view_type == "epoch":
                epoch = _to_int_or_none(row_index)
            elif "epoch" in columns and not is_missing(row["epoch"]):
                epoch = _to_int_or_none(row["epoch"])
            if view_type == "step":
                step = _to_int_or_none(row_index)
            elif "step" in columns and not is_missing(row["step"]):
                step = _to_int_or_none(row["step"])
            if view_type == "time_range":
                time_bucket = _to_int_or_none(row_index)
                if time_bucket is None and "time_range" in columns:
                    time_bucket = _to_int_or_none(row["time_range"])
                if time_bucket is not None and time_granularity is not None and time_resolution is not None:
                    interval_us = time_granularity * time_resolution
                    if interval_us > 0:
                        t0_ns = int(time_bucket * interval_us * 1_000)
                        t1_ns = int((time_bucket + 1) * interval_us * 1_000)

            # Invariant: a temporal fact must carry its axis coordinate, else the
            # diagnoser cannot track it longitudinally. Skip the row if missing.
            temporal_coord = {"window": window_index, "epoch": epoch,
                              "step": step, "time_range": time_bucket}.get(view_type)
            if view_type in TEMPORAL_VIEW_TYPES and temporal_coord is None:
                logger.warning("facts.temporal.missing_coordinate",
                               view_type=view_type, row_index=str(row_index))
                continue

            # Temporal rows go into the window (the coordinate above), not the
            # scope -> entity-free, stable scope (layer:window) so it recurs across
            # windows. Spatial rows keep the entity in the scope.
            scope_entity = None if view_type in TEMPORAL_VIEW_TYPES else str(row_index)
            scope_node = None
            if view_type in {"source_node", "host_hash"}:
                scope_node = str(row_index)
                scope_entity = view_type

            fact = AnalysisFact(
                fact_type=fact_type,
                window=FactWindow(
                    run_id=run_id, view_type=view_type, window_index=window_index,
                    epoch=epoch, step=step, time_bucket=time_bucket,
                    t0_ns=t0_ns, t1_ns=t1_ns, trigger=trigger,
                ),
                scope=FactScope(layer=scope_layer, entity=scope_entity, rank_set="all", node=scope_node),
                evidence={"metrics": evidence_metrics},
                severity=FactSeverity(score=severity_score, label=_severity_label(severity_score), method=severity_method),
                confidence=confidence,
                opportunity_tags=list(opportunity_tags),
                suppresses_tags=list(suppresses_tags) if suppresses_tags else [],
                provenance=provenance,
            )
            fact.finalize_id()
            facts.append(fact)
        return facts


SLOPE_SUFFIX = "_ops_slope"


class MetricFactBuilder:
    """Metric-driven (slope-based) fact builder — WISIO's promise.

    For each ``*_ops_slope`` column (``time_share / op_share``), severity =
    ``normalize_slope`` in [0,1]; emits a fact for every entity above a fixed
    floor. No rules, no thresholds to tune — automatic, exhaustive detection on
    the same severity scale as the rule engine, flowing into the same emitter.
    """

    # facts below this normalized-slope severity are not worth emitting (~MEDIUM)
    SEVERITY_FLOOR = 0.30

    def __init__(self, layers: List[str], default_emit_mode: str = "aggregate",
                 severity_floor: Optional[float] = None):
        # longest-prefix-first so "reader_posix" beats "reader" when splitting columns
        self.layers = sorted(layers or [], key=len, reverse=True)
        self.default_emit_mode = default_emit_mode
        self.severity_floor = self.SEVERITY_FLOOR if severity_floor is None else severity_floor
        self.emitter = FactEmitter()

    def evaluate(
        self,
        flat_views: Dict[ViewKey, DataFrame],
        raw_stats: Any,
    ) -> Dict[ViewKey, List[AnalysisFact]]:
        raw_stats_dict = _as_dict(raw_stats)
        run_id = raw_stats_dict.get("run_id")
        all_facts: Dict[ViewKey, List[AnalysisFact]] = {}
        for view_key, view_df in flat_views.items():
            facts = self._evaluate_view(view_df, view_key[-1], view_key, run_id)
            if facts:
                all_facts[view_key] = facts
        if all_facts:
            logger.info(
                "facts.metric.done",
                total_facts=sum(len(v) for v in all_facts.values()),
                view_keys=list(all_facts.keys()),
            )
        return all_facts

    def _evaluate_view(self, view_df, view_type, view_key, run_id) -> List[AnalysisFact]:
        slope_cols = [c for c in view_df.columns if c.endswith(SLOPE_SUFFIX)]
        if not slope_cols:
            return []
        facts: List[AnalysisFact] = []
        for slope_col in slope_cols:
            severity_series = normalize_slope(view_df[slope_col].astype("float64"))
            condition_mask = severity_series.fillna(0.0) >= self.severity_floor
            if not condition_mask.any():
                continue
            scope_layer, operation = self._split_layer(slope_col)
            fact_type = f"{operation}_slope" if operation else slope_col
            provenance = FactProvenance(
                rule_id="", rule_version="metric", metric_source=slope_col, view_key=list(view_key)
            )
            facts.extend(self.emitter.emit(
                emit_mode=self.default_emit_mode,
                fact_type=fact_type,
                scope_layer=scope_layer,
                view_type=view_type,
                view_key=view_key,
                run_id=run_id,
                working_df=view_df,
                condition_mask=condition_mask,
                severity_series=severity_series,
                confidence_series=None,
                evidence_metric_cols=[slope_col],
                opportunity_tags=[],
                suppresses_tags=[],
                provenance=provenance,
                trigger="metric_eval",
                severity_method="metric_slope",
            ))
        return facts

    def _split_layer(self, slope_col: str):
        base = slope_col[: -len(SLOPE_SUFFIX)]
        for layer in self.layers:
            if base == layer or base.startswith(layer + "_"):
                return layer, base[len(layer):].lstrip("_")
        return None, base


class FactPipeline:
    """Selects exactly one fact builder by eval_mode and runs it."""

    def __init__(self, builder):
        self.builder = builder  # exposes .evaluate(flat_views, raw_stats)

    def build(self, flat_views, raw_stats) -> Dict[ViewKey, List[AnalysisFact]]:
        if self.builder is None:
            return {}
        return self.builder.evaluate(flat_views, raw_stats)

    @classmethod
    def from_facts_config(
        cls,
        facts_config,
        *,
        layers: List[str],
        strict_time_semantics: bool,
        allow_mixed_time_aggregates: bool,
    ) -> "FactPipeline":
        eval_mode = getattr(facts_config, "eval_mode", "rule")
        emit_mode = getattr(facts_config, "emit_mode", "aggregate")
        if eval_mode == "metric":
            builder = MetricFactBuilder(layers=list(layers or []), default_emit_mode=emit_mode)
        else:
            builder = FactEngine.from_rule_file(
                getattr(facts_config, "eval_rule_file", ""),
                strict_time_semantics=strict_time_semantics,
                allow_mixed_time_aggregates=allow_mixed_time_aggregates,
                default_emit_mode=emit_mode,
            )
        return cls(builder)
