import pytest
from dftracer.analyzer.fact_engine import FactEngine
from dftracer.analyzer.fact_rules import build_fact_rule

from .frame_helpers import flat_view

pytestmark = [pytest.mark.smoke, pytest.mark.full]


def test_fact_engine_emits_detail_and_aggregate_facts():
    detail_rule = build_fact_rule(
        raw_rule={
            "id": "rule.fetch.pressure",
            "priority": 10,
            "view_types": ["epoch"],
            "emit_mode": "detail",
            "fact_type": "fetch_pressure",
            "required_metrics": ["fetch_iter_time_frac_parent"],
            "when": "fetch_iter_time_frac_parent >= 0.5",
            "severity_score": "clip01(fetch_iter_time_frac_parent)",
            "confidence": "0.9",
            "opportunity_tags": ["reader_parallelism"],
        }
    )
    aggregate_rule = build_fact_rule(
        raw_rule={
            "id": "rule.epoch.straggler",
            "priority": 20,
            "view_types": ["proc_name"],
            "emit_mode": "aggregate",
            "fact_type": "epoch_straggler",
            "required_metrics": ["epoch_time_max"],
            "derived_metrics": {
                "epoch_straggler_ratio": "epoch_time_max / 3.0",
            },
            "when": "epoch_straggler_ratio > 2.0",
            "severity_score": "clip01((epoch_straggler_ratio - 1.0) / 2.0)",
            "confidence": "0.85",
            "opportunity_tags": ["rank_balance"],
        }
    )
    engine = FactEngine([detail_rule, aggregate_rule])
    flat_views = {
        ("epoch",): flat_view("epoch", [1, 2, 3], fetch_iter_time_frac_parent=[0.20, 0.55, 0.80]),
        ("proc_name",): flat_view("proc_name", ["r0", "r1", "r2"], epoch_time_max=[10.0, 2.0, 5.0]),
    }

    facts = engine.evaluate(flat_views=flat_views, raw_stats={"run_id": "run-123"})

    # epoch is a temporal view: one entity-free fact per epoch, keyed by the epoch
    # coordinate (window.epoch) -- not a per-entity (spatial) fact.
    epoch_facts = facts[("epoch",)]
    assert len(epoch_facts) == 2
    assert {fact.window.epoch for fact in epoch_facts} == {2, 3}
    assert all(fact.scope.entity is None for fact in epoch_facts)
    assert all(fact.confidence == pytest.approx(0.9) for fact in epoch_facts)
    assert all(fact.fact_id and fact.fact_id.startswith("af_") for fact in epoch_facts)

    # aggregate mode: one rolled-up fact for the whole view
    proc_facts = facts[("proc_name",)]
    assert len(proc_facts) == 1
    agg = proc_facts[0]
    assert agg.scope.entity is None  # whole-view aggregate
    assert agg.fact_type == "epoch_straggler"
    assert agg.evidence["metrics"]["affected_count"] == 1
    assert agg.evidence["metrics"]["total_count"] == 3
    assert agg.evidence["top_k"][0]["entity"] == "r0"
    assert agg.evidence["top_k"][0]["metrics"]["epoch_straggler_ratio"] == pytest.approx(10.0 / 3.0)


def test_fact_engine_skips_rule_with_unresolved_identifier():
    unresolved_rule = build_fact_rule(
        raw_rule={
            "id": "rule.unresolved.identifier",
            "priority": 10,
            "source_view": "epoch",
            "fact_type": "invalid_fact",
            "required_metrics": ["epoch_time_max"],
            "when": "missing_metric > 0",
            "severity_score": "0.5",
        }
    )
    engine = FactEngine([unresolved_rule])
    flat_views = {("epoch",): flat_view("epoch", [1, 2], epoch_time_max=[1.0, 2.0])}

    facts = engine.evaluate(flat_views=flat_views, raw_stats={"run_id": "run-123"})

    assert facts == {}


def test_fact_engine_can_load_packaged_dlio_rules_by_alias():
    engine = FactEngine.from_rule_file(
        "dlio",
        strict_time_semantics=True,
        allow_mixed_time_aggregates=False,
    )

    assert len(engine.rules) >= 5
    assert any(rule.matches_view("epoch") for rule in engine.rules)
    assert any(rule.matches_view("proc_name") for rule in engine.rules)


def test_fact_engine_sets_time_range_and_step_window_metadata():
    time_range_rule = build_fact_rule(
        raw_rule={
            "id": "rule.time.range.window",
            "priority": 10,
            "source_view": "time_range",
            "emit_mode": "detail",
            "fact_type": "fetch_interval_pressure",
            "required_metrics": ["fetch_iter_time_frac_parent"],
            "when": "fetch_iter_time_frac_parent > 0.5",
            "severity_score": "0.8",
            "confidence": "0.7",
        }
    )
    step_rule = build_fact_rule(
        raw_rule={
            "id": "rule.step.window",
            "priority": 10,
            "source_view": "step",
            "emit_mode": "detail",
            "fact_type": "step_compute_pressure",
            "required_metrics": ["compute_time_frac_parent"],
            "when": "compute_time_frac_parent >= 0.6",
            "severity_score": "0.6",
            "confidence": "0.9",
        }
    )
    engine = FactEngine([time_range_rule, step_rule])
    flat_views = {
        ("time_range",): flat_view("time_range", [0, 1], fetch_iter_time_frac_parent=[0.2, 0.7]),
        ("step",): flat_view("step", [3, 4], compute_time_frac_parent=[0.4, 0.65]),
    }

    facts = engine.evaluate(
        flat_views=flat_views,
        raw_stats={
            "run_id": "run-xyz",
            "time_granularity": 10.0,
            "time_resolution": 1_000_000.0,
        },
    )

    tr_fact = facts[("time_range",)][0]
    assert tr_fact.window.view_type == "time_range"
    assert tr_fact.window.epoch is None
    assert tr_fact.window.step is None
    assert tr_fact.window.t0_ns == 10_000_000_000
    assert tr_fact.window.t1_ns == 20_000_000_000

    step_fact = facts[("step",)][0]
    assert step_fact.window.view_type == "step"
    assert step_fact.window.step == 4
    assert step_fact.window.t0_ns is None
    assert step_fact.window.t1_ns is None


def test_fact_engine_emits_layer_scoped_facts_with_fillna0_support():
    rule = build_fact_rule(
        raw_rule={
            "id": "rule.reader.operation.imbalance",
            "priority": 10,
            "source_view": "epoch",
            "emit_mode": "detail",
            "fact_type": "operation_imbalance",
            "scope_layer": "reader_posix",
            "required_metrics": [
                "reader_posix_read_count_sum",
                "reader_posix_write_count_sum",
            ],
            "derived_metrics": {
                "reader_posix_operation_imbalance_ratio": (
                    "abs(fillna0(reader_posix_read_count_sum) - fillna0(reader_posix_write_count_sum)) "
                    "/ max(min(fillna0(reader_posix_read_count_sum), fillna0(reader_posix_write_count_sum)), 1e-9)"
                ),
            },
            "when": "reader_posix_operation_imbalance_ratio > 0.10",
            "severity_score": "clip01((reader_posix_operation_imbalance_ratio - 0.10) / 0.90)",
            "confidence": "0.8",
            "opportunity_tags": ["read_write_rebalancing"],
        }
    )
    engine = FactEngine([rule])
    flat_views = {
        ("epoch",): flat_view("epoch", [3], reader_posix_read_count_sum=[10.0], reader_posix_write_count_sum=[None])
    }

    facts = engine.evaluate(flat_views=flat_views, raw_stats={"run_id": "run-123"})

    epoch_facts = facts[("epoch",)]
    assert len(epoch_facts) == 1
    assert epoch_facts[0].scope.layer == "reader_posix"
    assert epoch_facts[0].window.epoch == 3
    assert epoch_facts[0].evidence["metrics"]["reader_posix_operation_imbalance_ratio"] > 1e6
