"""Tests for the metric-driven (slope-based) fact builder + continuous scoring."""
import math

import pytest
from dftracer.analyzer.fact_engine import FactPipeline, MetricFactBuilder
from dftracer.analyzer.scoring import normalize_slope
from dftracer.utils import Series

from .frame_helpers import flat_view

TAN67_5 = math.tan(math.radians(67.5))   # ~2.414 -> severity 0.5
BIG_SLOPE = 1e9                            # ~90deg -> severity ~1.0


def test_normalize_slope_is_continuous_0_1():
    assert normalize_slope(1.0) == pytest.approx(0.0, abs=1e-6)    # proportional -> not a bottleneck
    assert normalize_slope(0.5) == 0.0                             # below proportional -> 0
    assert normalize_slope(TAN67_5) == pytest.approx(0.5, abs=1e-6)
    assert normalize_slope(BIG_SLOPE) == pytest.approx(1.0, abs=1e-6)


def test_normalize_slope_on_series():
    s = Series([0.5, 1.0, TAN67_5, BIG_SLOPE])
    out = normalize_slope(s)
    assert out.round(3).to_list() == [0.0, 0.0, 0.5, 1.0]


def _flat_view():
    # index = entities (files); one *_ops_slope column for the reader_posix layer.
    # slope 1.0 (proportional) -> 0; 2.414 -> 0.5; 1e9 -> ~1.0
    return flat_view(
        "file_name", ["/d/a.npz", "/d/b.npz", "/d/c.npz"], reader_posix_read_ops_slope=[1.0, TAN67_5, BIG_SLOPE]
    )


def test_metric_builder_aggregate_fact():
    builder = MetricFactBuilder(layers=["reader_posix", "app"], default_emit_mode="aggregate")
    facts = builder.evaluate(
        {("reader_posix", "file_name"): _flat_view()},
        raw_stats={"run_id": "r1"},
    )[("reader_posix", "file_name")]

    assert len(facts) == 1
    f = facts[0]
    assert f.fact_type == "read_slope"
    assert f.scope.layer == "reader_posix"
    assert f.scope.entity is None                       # aggregate
    assert f.severity.method == "metric_slope"
    assert 0.0 <= f.severity.score <= 1.0
    assert f.severity.score == pytest.approx(1.0)       # max severity (c.npz)
    assert f.provenance.metric_source == "reader_posix_read_ops_slope"
    # 2 of 3 entities are above the 0.30 floor (b=0.5, c=1.0; a=0.2 excluded)
    assert f.evidence["metrics"]["affected_count"] == 2
    assert f.evidence["metrics"]["total_count"] == 3


def test_metric_builder_detail_facts():
    builder = MetricFactBuilder(layers=["reader_posix"], default_emit_mode="detail")
    facts = builder.evaluate(
        {("reader_posix", "file_name"): _flat_view()},
        raw_stats={"run_id": "r1"},
    )[("reader_posix", "file_name")]

    assert len(facts) == 2                               # b and c above floor
    entities = sorted(f.scope.entity for f in facts)
    assert entities == ["/d/b.npz", "/d/c.npz"]
    assert all(f.scope.layer == "reader_posix" for f in facts)
    assert all(f.severity.method == "metric_slope" for f in facts)


def test_metric_builder_no_slope_columns_yields_nothing():
    builder = MetricFactBuilder(layers=["reader_posix"])
    facts = builder.evaluate(
        {("reader_posix", "file_name"): flat_view("file_name", ["/a", "/b"], x=[1, 2])},
        raw_stats={"run_id": "r1"},
    )
    assert facts == {}


def test_pipeline_selects_metric_builder():
    from dftracer.analyzer.config import FactsConfig

    cfg = FactsConfig(enabled=True, eval_mode="metric", emit_mode="aggregate")
    pipeline = FactPipeline.from_facts_config(
        cfg, layers=["reader_posix"],
        strict_time_semantics=True, allow_mixed_time_aggregates=False,
    )
    assert isinstance(pipeline.builder, MetricFactBuilder)
    facts = pipeline.build({("reader_posix", "file_name"): _flat_view()}, {"run_id": "r1"})
    assert facts[("reader_posix", "file_name")][0].fact_type == "read_slope"


def test_window_view_emits_per_row_temporal_facts():
    # The window view is the temporal axis: even in aggregate mode it emits ONE
    # fact per window row (never collapsed), with an entity-free scope and the
    # window number stamped as window.window.
    df = flat_view("window", [0, 1, 2], reader_posix_read_ops_slope=[BIG_SLOPE, BIG_SLOPE, BIG_SLOPE])
    builder = MetricFactBuilder(layers=["reader_posix"], default_emit_mode="aggregate")
    facts = builder.evaluate(
        {("reader_posix", "window"): df}, raw_stats={"run_id": "r1"}
    )[("reader_posix", "window")]

    assert len(facts) == 3  # per-row, NOT collapsed to one aggregate
    for f, w in zip(sorted(facts, key=lambda x: x.window.window_index), [0, 1, 2]):
        assert f.scope.entity is None          # entity-free: the window is the axis, not an entity
        assert f.window.window_index == w      # the window coordinate is stamped
        assert f.window.view_type == "window"


def test_pipeline_selects_rule_builder_by_default():
    from dftracer.analyzer.config import FactsConfig
    from dftracer.analyzer.fact_engine import FactEngine

    cfg = FactsConfig(enabled=True, eval_mode="rule", eval_rule_file="")
    pipeline = FactPipeline.from_facts_config(
        cfg, layers=["reader_posix"],
        strict_time_semantics=True, allow_mixed_time_aggregates=False,
    )
    assert isinstance(pipeline.builder, FactEngine)
