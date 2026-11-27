"""The fact rule helpers on library frames against the pandas helpers they replace."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dftracer.analyzer.fact_engine import FACT_HELPERS, _eval_expr
from dftracer.analyzer.fact_rules import build_fact_rule
from dftracer.utils import DataFrame, Series
from omegaconf import OmegaConf

pytestmark = pytest.mark.smoke

RTOL = 1e-12
RULES_DIR = Path(__file__).resolve().parents[1] / "python" / "dftracer" / "analyzer" / "configs" / "fact_rules"


def _to_float_series(x):
    if isinstance(x, pd.Series):
        return x.astype("Float64").to_numpy(dtype="float64", na_value=float("nan"))
    return x


def _na_safe_max(*args):
    result = args[0]
    for arg in args[1:]:
        if isinstance(result, pd.Series) or isinstance(arg, pd.Series):
            idx = result.index if isinstance(result, pd.Series) else arg.index
            result = pd.Series(np.fmax(_to_float_series(result), _to_float_series(arg)), index=idx)
        else:
            result = max(result, arg)
    return result


def _na_safe_min(*args):
    result = args[0]
    for arg in args[1:]:
        if isinstance(result, pd.Series) or isinstance(arg, pd.Series):
            idx = result.index if isinstance(result, pd.Series) else arg.index
            result = pd.Series(np.fmin(_to_float_series(result), _to_float_series(arg)), index=idx)
        else:
            result = min(result, arg)
    return result


def _clip01(value):
    if isinstance(value, pd.Series):
        return value.clip(lower=0.0, upper=1.0)
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return value


def _fillna0(value):
    if isinstance(value, pd.Series):
        return value.fillna(0.0)
    if value is None:
        return 0.0
    try:
        return 0.0 if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


PANDAS_HELPERS = {
    "max": _na_safe_max,
    "min": _na_safe_min,
    "abs": abs,
    "clip01": _clip01,
    "fillna0": _fillna0,
    "math": math,
}


def _pandas_eval_expr(df: pd.DataFrame, expression: str) -> pd.Series:
    try:
        result = df.eval(expression, engine="python")
    except (AttributeError, NameError, NotImplementedError, TypeError, ValueError):
        local_env = {column: df[column] for column in df.columns}
        local_env.update(PANDAS_HELPERS)
        result = eval(expression, {"__builtins__": {}}, local_env)
    if isinstance(result, pd.Series):
        return result
    return pd.Series([result] * len(df), index=df.index)


def _cells(values) -> list:
    """Values of a library Series, a pandas Series or a scalar, a missing value or NaN as None."""
    if isinstance(values, Series):
        values = values.to_list()
    elif isinstance(values, pd.Series):
        values = values.astype(object).tolist()
    elif not isinstance(values, list):
        values = [values]
    out = []
    for v in values:
        if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
            out.append(None)
        else:
            out.append(v)
    return out


def _assert_same(got, expected) -> None:
    got, expected = _cells(got), _cells(expected)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        if e is None or g is None:
            assert g is None and e is None, (got, expected)
        elif isinstance(e, (bool, np.bool_)):
            assert bool(g) == bool(e), (got, expected)
        else:
            assert g == pytest.approx(float(e), rel=RTOL, abs=0.0), (got, expected)


COLUMNS = {
    "a": [1.5, None, -3.0, 0.0, None],
    "b": [2.0, 4.0, None, -1.0, None],
    "empty": [None, None, None, None, None],
}


def _frames():
    lib = DataFrame.from_dict(
        {
            "proc_name": Series(["p0", "p1", "p2", "p3", "p4"]),
            **{c: Series(v, dtype="float64") for c, v in COLUMNS.items()},
        }
    ).set_index(["proc_name"])
    pdf = pd.DataFrame(
        {c: pd.array(v, dtype="Float64") for c, v in COLUMNS.items()}, index=["p0", "p1", "p2", "p3", "p4"]
    )
    return lib, pdf


@pytest.mark.parametrize(
    "helper, args",
    [
        ("max", ("a",)),
        ("max", ("a", "b")),
        ("max", ("a", 0.5)),
        ("max", (0.5, "a")),
        ("max", ("a", "empty")),
        ("max", ("empty", "empty")),
        ("max", ("empty", 1e-9)),
        ("max", (2.0, 3.0)),
        ("min", ("a",)),
        ("min", ("a", "b")),
        ("min", ("b", -0.5)),
        ("min", (-0.5, "b")),
        ("min", ("empty", "a")),
        ("min", ("empty", "empty")),
        ("min", ("empty", 1.0)),
        ("min", (2.0, 3.0)),
        ("abs", ("a",)),
        ("abs", ("empty",)),
        ("abs", (-2.5,)),
        ("clip01", ("a",)),
        ("clip01", ("empty",)),
        ("clip01", (1.7,)),
        ("clip01", (-0.2,)),
        ("fillna0", ("a",)),
        ("fillna0", ("empty",)),
        ("fillna0", (None,)),
        ("fillna0", (float("nan"),)),
        ("fillna0", (3.0,)),
    ],
)
def test_helper_matches_pandas(helper, args):
    lib, pdf = _frames()
    got = FACT_HELPERS[helper](*[lib[a] if isinstance(a, str) else a for a in args])
    expected = PANDAS_HELPERS[helper](*[pdf[a] if isinstance(a, str) else a for a in args])
    assert isinstance(got, Series) == isinstance(expected, pd.Series)
    _assert_same(got, expected)


def _packaged_expressions():
    for path in sorted(RULES_DIR.glob("*.yaml")):
        config = OmegaConf.to_object(OmegaConf.load(path))
        defaults = dict(config.get("defaults", {}))
        for raw in config.get("rules", []):
            rule = build_fact_rule(
                raw_rule=raw, defaults=defaults, strict_time_semantics=False, allow_mixed_time_aggregates=True
            )
            expressions = list(rule.derived_metrics.items())
            expressions += [("when", rule.when), ("severity", rule.severity_score)]
            if rule.confidence:
                expressions.append(("confidence", rule.confidence))
            yield pytest.param(rule, expressions, id=f"{path.stem}:{rule.id}")


@pytest.mark.parametrize("rule, expressions", list(_packaged_expressions()))
def test_packaged_rule_expressions_match_pandas(rule, expressions):
    """Every expression of the packaged rules, on columns with numbers, missing values and an all-missing column."""
    names = sorted(set(rule.referenced_columns) | set(rule.required_metrics))
    names = [n for n in names if n not in FACT_HELPERS and n not in rule.derived_metrics]
    rows = 6
    base = [0.25, None, 0.9, 1048576.0 * 2, 3.0, 0.0]
    columns = {}
    for i, name in enumerate(names):
        columns[name] = (
            [None] * rows if i == len(names) - 1 and len(names) > 1 else base[i % rows :] + base[: i % rows]
        )
    lib = DataFrame.from_dict(
        {"epoch": Series(list(range(rows))), **{c: Series(v, dtype="float64") for c, v in columns.items()}}
    ).set_index(["epoch"])
    pdf = pd.DataFrame({c: pd.array(v, dtype="Float64") for c, v in columns.items()}, index=list(range(rows)))
    for target, expression in expressions:
        got = _eval_expr(lib, expression)
        expected = _pandas_eval_expr(pdf, expression)
        _assert_same(got, expected)
        if target in rule.derived_metrics:
            lib[target] = got
            pdf[target] = expected
