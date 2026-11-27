"""Preset, layer and logical-view strings give the same result on library frames as pandas eval."""

import dataclasses as dc

import pandas as pd
import pytest
from dftracer.analyzer import config
from dftracer.analyzer.frames.expr import evaluate, query
from dftracer.utils import DataFrame, Series

pytestmark = [pytest.mark.smoke, pytest.mark.full]

ROWS = {
    "cat": ["posix", "stdio", "posix_lustre", "reader", "reader.posix", "app", "dataloader", None, "checkpoint"],
    "func_name": ["open64", "read", "closedir", "fork", "__getitem__", "epoch_1", "train", "x", "close"],
    "io_cat": [3, 1, 3, 6, 1, 2, 7, 3, 3],
    "proc_name": [
        "app#h1#10#1",
        "app#h1#10#2",
        "app#h2#11#1",
        "app#h2#11#2",
        "app#h3#12#1",
        "app#h3#12#2",
        "app#h4#13#1",
        "app#h4#13#2",
        "app#h5#14#1",
    ],
    "time_range": [0, 1, 2, 3, 4, 5, 6, 7, 8],
}


def _preset_strings() -> list[str]:
    out = set()
    for cls in vars(config).values():
        if not (isinstance(cls, type) and issubclass(cls, config.AnalyzerPresetConfig) and dc.is_dataclass(cls)):
            continue
        fields = {f.name: f for f in dc.fields(cls)}
        for name in ("layer_defs", "derived_metrics", "logical_views"):
            factory = fields[name].default_factory
            if factory is dc.MISSING:
                continue
            for value in factory().values():
                if isinstance(value, str):
                    out.add(value)
                elif isinstance(value, dict):
                    out.update(v for v in value.values() if isinstance(v, str))
    return sorted(out)


def _same(native, expected: pd.Series, strict: bool = False) -> None:
    got = [None if pd.isna(v) else v for v in native.to_list()]
    want = [None if pd.isna(v) else v for v in expected.tolist()]
    if not strict and all(isinstance(v, bool) or v is None for v in got + want):
        # Every caller uses a condition as a mask with missing read as false.
        got = [bool(v) for v in got]
        want = [bool(v) for v in want]
    assert got == want


@pytest.mark.parametrize("expr", _preset_strings())
def test_preset_string_matches_pandas(expr: str) -> None:
    frame = DataFrame(ROWS)
    expected = pd.DataFrame(ROWS).astype({"cat": "string", "func_name": "string", "proc_name": "string"})
    _same(evaluate(frame, expr), expected.eval(expr))


def test_preset_strings_are_found() -> None:
    strings = _preset_strings()
    assert any(".isin(" in s for s in strings)
    assert any(".str.split(" in s for s in strings)


def test_assignment_adds_a_column() -> None:
    frame = DataFrame(ROWS)
    out = evaluate(frame, 'host_name = proc_name.str.split("#").str[1]')
    assert out["host_name"].to_list() == ["h1", "h1", "h2", "h2", "h3", "h3", "h4", "h4", "h5"]
    assert "host_name" not in frame.columns


def test_query_with_local_names_matches_pandas() -> None:
    frame = DataFrame(ROWS)
    keys = [1, 4, 8]
    got = query(frame, "time_range in @keys", {"keys": keys})
    assert got["time_range"].to_list() == keys


def test_query_drops_missing_as_false() -> None:
    frame = DataFrame(ROWS)
    got = query(frame, 'cat.str.contains("posix|stdio")')
    expected = pd.DataFrame(ROWS).astype({"cat": "string"}).query('cat.str.contains("posix|stdio", na=False)')
    assert got["time_range"].to_list() == expected["time_range"].tolist()


LOCAL_ROWS = {
    "x": [1, 2],
    "a": [1.5, None],
    "s": ["ab", None],
}


def _local_frames():
    frame = DataFrame.from_pandas(
        pd.DataFrame(
            {
                "x": pd.array(LOCAL_ROWS["x"], dtype="Int64"),
                "a": pd.array(LOCAL_ROWS["a"], dtype="Float64"),
                "s": pd.array(LOCAL_ROWS["s"], dtype="string"),
            }
        )
    )
    expected = frame.to_pandas(nullable=True)
    return frame, expected


@pytest.mark.parametrize(
    "expr",
    [
        "x == @x",
        "x != @x",
        "(a > 0) and True",
        "True and (a > 0)",
        "(a > 0) or False",
        "False or (a > 0)",
        "(a > 0) & False",
        "True | (a > 0)",
        "s.str[0]",
        "s.str[-1]",
        "s.str[5]",
    ],
)
def test_expression_with_locals_and_mixed_operands_matches_pandas(expr: str) -> None:
    frame, expected = _local_frames()
    want = expected.eval(expr, engine="python", local_dict={"x": 2})
    _same(evaluate(frame, expr, {"x": 2}), want)
    if "@" not in expr:
        # Without names the native eval runs first, so its answer must match pandas too.
        _same(evaluate(frame, expr), want)


MIXED_ROWS = {
    "x": [1, None],
    "s": ["ab", None],
    "t": ["ab", "cd"],
    "u": ["αβ", "a"],
    "v": ["z\n", "\n"],
    "b": [True, None],
}


def _mixed_frames():
    expected = pd.DataFrame(
        {
            "x": pd.array(MIXED_ROWS["x"], dtype="Int64"),
            "s": pd.array(MIXED_ROWS["s"], dtype="string"),
            "t": pd.array(MIXED_ROWS["t"], dtype="string"),
            "u": pd.array(MIXED_ROWS["u"], dtype="string"),
            "v": pd.array(MIXED_ROWS["v"], dtype="string"),
            "b": pd.array(MIXED_ROWS["b"], dtype="boolean"),
        }
    )
    return DataFrame.from_pandas(expected), expected


@pytest.mark.parametrize(
    "expr",
    [
        'x == "a"',
        'x != "a"',
        "s == 1",
        "s != 1",
        "s == t",
        "s != t",
        '"a" in s',
        '"ab" in s',
        '"ab" not in s',
        "u.str[0]",
        "u.str[1]",
        "u.str[-1]",
        "u.str[2]",
        "v.str[-1]",
        "v.str[0]",
        "x == None",
        "x != None",
        "s == None",
        "s != None",
        "None == x",
        "b == True",
        "b == 1",
        "b != 1.5",
        "b < 1",
        "1 > b",
        "s == x",
        "s != x",
        "s == b",
        "s in ['ab', None]",
        "s not in ['ab', None]",
        "x in [1, None]",
        "s.isin(['ab', None])",
        "~s.isin(['ab', None])",
        "x.isin([1, None])",
        "s.isin([None])",
        "b == x",
        "b == b",
        "b != b",
        "s == (x + 1)",
        "s.str.upper() != x",
        "s.isin(values=['ab', None])",
        "s == @nan",
        "x == @nan",
    ],
)
def test_mixed_type_comparisons_and_text_indexing_match_pandas(expr: str) -> None:
    frame, expected = _mixed_frames()
    local = {"nan": float("nan")}
    want = expected.eval(expr, engine="python", local_dict=local)
    # Strict: a missing value must stay missing where pandas keeps it.
    if "@" not in expr:
        _same(evaluate(frame, expr), want, strict=True)
    _same(evaluate(frame, expr, local), want, strict=True)


def test_a_text_index_level_compared_with_a_number_column_matches_pandas() -> None:
    frame, expected = _mixed_frames()
    expected = expected.set_index("s")
    frame = frame.set_index(["s"])
    for expr in ("s == x", "s != x"):
        _same(evaluate(frame, expr), expected.eval(expr, engine="python"), strict=True)


def test_a_column_named_like_the_local_prefix_is_a_column() -> None:
    frame = DataFrame.from_dict({"__local__x": Series([1, 2], dtype="int64")})
    assert evaluate(frame, "__local__x == 1", {"unused": 0}).to_list() == [True, False]


def test_scalar_membership_matches_pandas() -> None:
    frame, expected = _local_frames()
    for expr in ("1 in [1, 2]", "3 in [1, 2]", "1 not in [1, 2]"):
        assert evaluate(frame, expr, {"unused": 0}) == expected.eval(expr, engine="python"), expr


def test_assignment_from_a_local_matches_pandas() -> None:
    frame, expected = _local_frames()
    got = evaluate(frame, "y = @x", {"x": 2})
    assert got["y"].to_list() == expected.eval("y = @x", engine="python", local_dict={"x": 2})["y"].tolist()


def test_string_comparisons_treat_a_missing_value_as_unequal_like_pandas():
    """pandas eval compares a text column with a string as objects, also for the `string` dtype the old
    pipeline read: a missing value is not equal, so `!=` keeps it and `==` drops it."""
    values = ["a", "b", None]
    frame = DataFrame.from_dict({"x": Series(values, dtype="string")})
    expected = pd.DataFrame({"x": pd.array(values, dtype="string")})
    for expr in ('x != "a"', 'x == "a"', '~(x == "a")', '"a" != x'):
        got = [None if v is None else v for v in query(frame, expr)["x"].to_list()]
        want = [None if pd.isna(v) else v for v in expected.query(expr)["x"].tolist()]
        assert got == want, expr
        _same(evaluate(frame, expr), expected.eval(expr))


def test_expression_names_lists_the_columns_an_expression_reads():
    from dftracer.analyzer.frames.expr import expression_names

    assert expression_names('cat == "posix" & io_cat.isin([1, 2]) & x == @x') == {"cat", "io_cat", "x"}
    assert expression_names("y = a + b") == {"a", "b"}


def test_a_number_in_a_column_raises_because_pandas_tests_the_index() -> None:
    frame, expected = _mixed_frames()
    assert expected.eval("0 in x", engine="python") is True  # the row label 0, not a value
    with pytest.raises(TypeError, match="index labels"):
        evaluate(frame, "0 in x")


def test_a_chained_comparison_uses_the_exact_operand() -> None:
    values = {"b": [True, False], "x": [9007199254740993, 1], "s": ["ab", "a"]}
    expected = pd.DataFrame(
        {
            "b": pd.array(values["b"], dtype="boolean"),
            "x": pd.array(values["x"], dtype="Int64"),
            "s": pd.array(values["s"], dtype="string"),
        }
    )
    expr = 'b < x <= 9007199254740992 and s == "ab"'
    _same(evaluate(DataFrame.from_pandas(expected), expr), expected.eval(expr, engine="python"), strict=True)


def test_isin_with_a_string_raises_like_pandas() -> None:
    frame, expected = _mixed_frames()
    with pytest.raises(TypeError):
        expected.eval('s.isin("ab")', engine="python")
    with pytest.raises(TypeError):
        evaluate(frame, 's.isin("ab")')


@pytest.mark.parametrize("items", [[float("nan")], ["ab", float("nan")], []])
def test_isin_with_nan_on_a_text_column_matches_pandas(items) -> None:
    frame, expected = _mixed_frames()
    for expr in ("s.isin(@v)", "~s.isin(@v)", "s in @v"):
        want = expected.eval(expr, engine="python", local_dict={"v": items})
        _same(evaluate(frame, expr, {"v": items}), want, strict=True)
