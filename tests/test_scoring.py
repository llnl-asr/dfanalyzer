"""Stage 1 leaf tests: continuous slope-severity scoring + fact dataclass shape.
(The MetricFactBuilder that consumes these lands with the fact engine, Stage 2.)"""
import math

import pytest
from dftracer.utils import Series

from dftracer.analyzer.scoring import normalize_slope
from dftracer.analyzer.types import (
    AnalysisFact, FactEnvelope, FactScope, FactSeverity, FactWindow,
)


TAN67_5 = math.tan(math.radians(67.5))   # ~2.414 -> severity 0.5
BIG_SLOPE = 1e9                            # ~90deg -> severity ~1.0


def test_normalize_slope_is_continuous_0_1():
    assert normalize_slope(1.0) == pytest.approx(0.0, abs=1e-6)   # proportional -> not a bottleneck
    assert normalize_slope(0.5) == 0.0                            # below proportional -> 0
    assert normalize_slope(TAN67_5) == pytest.approx(0.5, abs=1e-6)
    assert normalize_slope(BIG_SLOPE) == pytest.approx(1.0, abs=1e-6)


def test_normalize_slope_on_series():
    s = Series([0.5, 1.0, TAN67_5, BIG_SLOPE])
    assert normalize_slope(s).round(3).to_list() == [0.0, 0.0, 0.5, 1.0]


def test_fact_dataclasses_and_envelope_roundtrip():
    fact = AnalysisFact(
        fact_type="fetch_pressure",
        window=FactWindow(run_id="r1", view_type="epoch", epoch=0, trigger="epoch.block"),
        scope=FactScope(layer="app", entity=None, rank_set="all"),
        evidence={"metrics": {"fetch_frac": 0.9}},
        severity=FactSeverity(score=1.0, label="critical"),
        confidence=0.8,
        opportunity_tags=["dataloader_prefetch"],
    )
    assert fact.finalize_id().startswith("af_")
    assert fact.severity.method == "rule_expr"          # default label
    env = FactEnvelope(facts=[fact])
    assert env.schema_version == "analyzer.fact-envelope.v1"
    import json
    assert json.loads(env.to_json())["facts"][0]["fact_type"] == "fetch_pressure"
