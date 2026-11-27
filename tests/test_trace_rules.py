"""`TraceRules` transforms on small frames."""

import pytest
from dftracer.analyzer.trace_rules import DEFAULT_TRACE_RULES, TraceRules, drop_matching_files
from dftracer.utils import DataFrame, Series

pytestmark = [pytest.mark.smoke, pytest.mark.full]

RULES = TraceRules(
    posix_cat_rules=(("/data", "_reader"), ("/lustre", "_lustre")),
    ignored_file_patterns=("/proc/", "__pycache__"),
    ignored_func_names=("A.__init__",),
    ignored_func_patterns=(".save_state",),
)


def test_drop_ignored_keeps_rows_without_a_file():
    df = DataFrame.from_dict(
        {
            "file_name": Series(["/proc/self/stat", None, "/scratch/x", "/a/__pycache__/m.pyc"], dtype="string"),
            "func_name": ["read", "op", "read", "read"],
        }
    )
    out = RULES.drop_ignored(df)
    assert out["file_name"].to_list() == [None, "/scratch/x"]


def test_drop_ignored_keeps_rows_without_a_function():
    df = DataFrame.from_dict({"func_name": Series(["A.__init__", None, "m.save_state", "keep"], dtype="string")})
    assert RULES.drop_ignored(df)["func_name"].to_list() == [None, "keep"]


def test_drop_ignored_matches_names_exactly_and_patterns_as_text():
    df = DataFrame.from_dict({"func_name": ["A.__init__", "A.__init__x", "m.save_state", "mXsave_state", "keep"]})
    out = RULES.drop_ignored(df)
    assert out["func_name"].to_list() == ["A.__init__x", "mXsave_state", "keep"]


def test_drop_ignored_without_rules_is_identity():
    df = DataFrame.from_dict({"file_name": ["/proc/x"], "func_name": ["A.__init__"]})
    out = TraceRules().drop_ignored(df)
    assert out.equals(df)


def test_suffix_posix_cat_applies_rules_in_order_on_posix_and_stdio_only():
    df = DataFrame.from_dict(
        {
            "cat": ["posix", "stdio", "POSIX", "posix", "ai", "posix"],
            "file_name": ["/lustre/data/f", "/data/g", "/data/h", "/scratch/i", "/data/j", None],
        }
    )
    out = RULES.suffix_posix_cat(df)
    assert out["cat"].to_list() == ["posix_reader_lustre", "stdio_reader", "POSIX", "posix", "ai", "posix"]


def test_default_rules_order_matches_the_read_and_hlm_fold():
    assert [p for p, _ in DEFAULT_TRACE_RULES.posix_cat_rules] == ["/data", "/checkpoint", "/lustre", "/ssd"]
    assert "/proc/" in DEFAULT_TRACE_RULES.ignored_file_patterns


def test_drop_matching_files_matches_folded_and_unfolded_names():
    """The drop runs after the read, so it must match whether the read kept the
    full path or folded it to the bucket the ignore pattern supplied."""
    df = DataFrame.from_dict(
        {"file_name": ["/data/a.npz", "/data", "/tmp/junk.log", "/tmp", None], "count": [1, 2, 3, 4, 5]}
    )
    out = drop_matching_files(df, ("/tmp",))
    assert out["count"].to_list() == [1, 2, 5]
    assert len(drop_matching_files(df, ())) == 5
    assert (
        drop_matching_files(df.rename(columns={"file_name": "file_path"}), ("/tmp",), "file_path")["count"].sum() == 8
    )


def test_patterns_match_their_literal_text_only():
    df = DataFrame.from_dict(
        {
            "cat": ["posix"] * 4,
            "file_name": ["/in/a.bin", "/in/[ab].bin", "/in/b.bin", "/in/aXbin"],
            "func_name": ["f", "g[ab]", "ga", "gb"],
            "count": [1, 2, 3, 4],
        }
    )
    assert drop_matching_files(df, ("/in/[ab].bin",))["count"].to_list() == [1, 3, 4]
    rules = TraceRules(ignored_func_patterns=("g[ab]",), posix_cat_rules=(("/in/[ab].bin", "_x"),))
    assert rules.drop_ignored(df)["count"].to_list() == [1, 3, 4]
    assert rules.suffix_posix_cat(df.copy())["cat"].to_list() == ["posix", "posix_x", "posix", "posix"]
