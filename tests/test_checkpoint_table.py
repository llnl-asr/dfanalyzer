"""Checkpoint tables round-trip library frames through the gzip JSON-lines format."""

import gzip
import json
import math

import pandas as pd
import pytest
from dftracer.analyzer.checkpoint_table import FORMAT_TAG, read_table, write_table
from dftracer.utils import DataFrame, DType, Series
from dftracer.utils.errors import DFTUtilsParseError, DFTUtilsValueError

pytestmark = [pytest.mark.smoke, pytest.mark.full]


def _cells(values: list) -> list:
    """`values` with NaN as "nan", also inside list cells, so equal cells compare equal."""
    return [
        _cells(v) if isinstance(v, list) else "nan" if isinstance(v, float) and math.isnan(v) else v for v in values
    ]


def _assert_same(got: DataFrame, expected: DataFrame) -> None:
    assert got.index_names() == expected.index_names()
    assert got.columns == expected.columns
    # get_level_values does not find an index named __index_level_0__, so the levels are read as columns.
    got, expected = got.reset_index(), expected.reset_index()
    assert got.columns == expected.columns
    assert got.dtypes == expected.dtypes
    for name in expected.columns:
        assert _cells(got[name].to_list()) == _cells(expected[name].to_list()), name


def _round_trip(frame: DataFrame, tmp_path) -> tuple[str, DataFrame]:
    path = str(tmp_path / "t.pfw.gz")
    write_table(frame, path, "t")
    back = read_table(path)
    _assert_same(back, frame)
    return path, back


def test_every_dtype_and_missing_value_with_a_named_index(tmp_path):
    frame = DataFrame.from_dict(
        {
            "proc_name": Series(["a", "b", None], dtype="string"),
            "time_range": Series([0, 1, 2], dtype="int64"),
            "count": Series([1, None, 2**60], dtype="int64"),
            "time": Series([0.1, None, 1e-300], dtype="float64"),
            "raw": Series([math.nan, math.inf, -math.inf], dtype="float64"),
            "exact": Series([0.1 + 0.2, 1 / 3, 2.0], dtype="float64"),
            "flag": Series([True, None, False], dtype="bool"),
            "word": Series(["x", "y", "x"]).dictionary_encode(),
            "file_name": Series([["/a", "/b"], [], None]),
        }
    ).set_index(["proc_name", "time_range"])
    _round_trip(frame, tmp_path)


def test_unnamed_named_and_default_indexes(tmp_path):
    _round_trip(DataFrame.from_dict({"x": [1, 2]}), tmp_path)
    _, back = _round_trip(DataFrame.from_pandas(pd.DataFrame({"x": [1, 2]}, index=pd.Index(["p", "q"]))), tmp_path)
    assert back.to_pandas().index.name is None and list(back.to_pandas().index) == ["p", "q"]
    _round_trip(DataFrame.from_dict({"k": ["p", "q"], "x": [1, 2]}).set_index(["k"]), tmp_path)


def test_empty_frame(tmp_path):
    frame = DataFrame.from_dict({"k": Series([], dtype="string"), "x": Series([], dtype="int64")}).set_index(["k"])
    _round_trip(frame, tmp_path)
    lists = DataFrame.from_dict({"l": Series([["a", "b"]]), "n": Series([[1, 2]]), "x": Series([1], dtype="int64")})
    _, back = _round_trip(lists.head(0), tmp_path)
    assert back["l"].dtype == DType.LIST and back["n"].dtype == DType.LIST
    assert back["l"].child(0).dtype == DType.STRING and back["n"].child(0).dtype == DType.INT64


def test_list_cells_of_numbers_and_missing_values_round_trip(tmp_path):
    frame = DataFrame.from_dict(
        {
            "ints": Series([[1, 2], [], None, [3]]),
            "floats": Series([[1.5, -2.0], [], None, [0.0]]),
            "words": Series([["a", None], [], None, [""]]),
        }
    )
    _, back = _round_trip(frame, tmp_path)
    assert back["ints"].child(0).dtype == DType.INT64
    assert back["floats"].to_list() == [[1.5, -2.0], [], None, [0.0]]
    assert back["words"].to_list() == [["a", None], [], None, [""]]
    empty = DataFrame.from_dict({"ints": Series([[1]]), "x": [1]}).with_columns(ints=Series([[1], []]).tail(1))
    _, back = _round_trip(empty, tmp_path)
    assert back["ints"].child(0).dtype == DType.INT64


def test_a_missing_dictionary_value_is_written_as_null(tmp_path):
    frame = DataFrame.from_dict({"s": Series(["a", None, "NaN"]).dictionary_encode(), "i": Series([1, None, 2])})
    path, back = _round_trip(frame, tmp_path)
    with gzip.open(path, "rt") as f:
        rows = [json.loads(line)["args"] for line in f][1:]
    assert [r["s"] for r in rows] == ["a", None, "NaN"]
    assert [r["i"] for r in rows] == [1, None, 2]
    assert back["s"].is_null().to_list() == [False, True, False]


def test_list_cells_keep_empty_lists_empty_values_and_missing_cells_apart(tmp_path):
    cells = [["b", "a"], [], None, [""], ["", ""], ["x\x1fy"], [None, "z"]]
    frame = DataFrame.from_dict({"l": cells, "x": ["p", None, "q", "", "r", "s", "t"]})
    path, back = _round_trip(frame, tmp_path)
    assert back["l"].dtype == DType.LIST
    assert back["l"].to_list() == cells
    with gzip.open(path, "rt") as f:
        fields = json.loads(f.readline())["args"]["fields"]
    assert [f.get("elem_dtype") for f in fields] == ["string", None]


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_a_list_value_with_no_json_number_form_is_refused(tmp_path, value):
    """The view sets never hold NaN or inf: a NaN key is left out of its set."""
    frame = DataFrame.from_dict({"floats": Series([[1.5, value]])})
    with pytest.raises(DFTUtilsValueError, match="floats"):
        write_table(frame, str(tmp_path / "t.pfw.gz"), "t")


def test_lines_are_dftracer_records(tmp_path):
    frame = DataFrame.from_dict({"k": [5, 6], "x": Series([1, None], dtype="int64")}).set_index(["k"])
    path, _ = _round_trip(frame, tmp_path)
    with gzip.open(path, "rt") as f:
        header, *rows = [json.loads(line) for line in f]
    assert header["ph"] == 4 and header["args"]["format"] == "dftu.table" and header["args"]["nrows"] == 2
    assert sorted(f["name"] for f in header["args"]["fields"]) == ["k", "x"]
    assert rows[1] == {"name": "t", "cat": "dftu", "pid": 0, "tid": 0, "ts": 1, "ph": 3, "args": {"k": 6, "x": None}}


def test_reader_refuses_an_unknown_version_and_a_short_file(tmp_path):
    path = str(tmp_path / "t.pfw.gz")
    write_table(DataFrame.from_dict({"x": [1, 2]}), path, "t")
    with gzip.open(path, "rt") as f:
        lines = f.read().splitlines()

    header = json.loads(lines[0])
    header["args"]["format_version"] = "2.0"
    with gzip.open(path, "wt") as f:
        f.write("\n".join([json.dumps(header), *lines[1:]]) + "\n")
    with pytest.raises(DFTUtilsParseError, match="format_version"):
        read_table(path)

    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines[:2]) + "\n")
    with pytest.raises(DFTUtilsParseError, match="rows"):
        read_table(path)


def test_nan_and_null_stay_apart_in_a_float64_column(tmp_path):
    frame = DataFrame.from_dict({"x": Series([math.nan, None, 1.5, math.inf], dtype="float64")})
    _, back = _round_trip(frame, tmp_path)
    values = back["x"].to_list()
    assert math.isnan(values[0]) and values[1] is None and values[3] == math.inf
    _round_trip(DataFrame.from_dict({"x": Series([], dtype="float64")}), tmp_path)


def _analyzer(tmp_path):
    from dftracer.analyzer.dftracer import DFTracerAnalyzer

    analyzer = object.__new__(DFTracerAnalyzer)
    analyzer.checkpoint = True
    analyzer.checkpoint_dir = str(tmp_path)
    analyzer.time_granularity = 5
    return analyzer


def test_a_checkpoint_of_the_pandas_build_is_not_read(tmp_path):
    """The pandas build named a checkpoint without the format tag; its table is recomputed, not read."""
    analyzer = _analyzer(tmp_path)
    name = analyzer.get_checkpoint_name("hlm", "proc_name")
    assert name.endswith(f"_{FORMAT_TAG}")
    old = pd.DataFrame({"x": [9]}, index=pd.Index(["old"], name="k"))
    DataFrame.from_pandas(old).write_table(f"{tmp_path}/hlm_proc_name_5.pfw.gz", name="old", mode="overwrite")
    fresh = DataFrame.from_dict({"k": ["new"], "x": [1]}).set_index(["k"])
    calls = []

    def fallback():
        calls.append(1)
        return fresh

    assert analyzer.restore_view(name, fallback=fallback) is fresh
    assert analyzer.restore_view(name, fallback=fallback).get_level_values("k").to_list() == ["new"]
    assert calls == [1]


@pytest.mark.smoke
def test_restore_view_writes_the_checkpoint_here_not_in_a_dask_graph(tmp_path):
    """A Dask graph would hold a second copy of the frame until the write ends."""
    from types import SimpleNamespace

    from dftracer.analyzer.analyzer import Analyzer

    class NoDask:
        def __getattr__(self, name):
            raise AssertionError(f"checkpoint went through the Dask client ({name})")

    written = {}
    stub = SimpleNamespace(
        checkpoint=True,
        dask_client=NoDask(),
        has_checkpoint=lambda name: False,
        write_checkpoint=lambda view, name: written.setdefault(name, view),
    )
    view = DataFrame.from_dict({"x": [1, 2]})
    assert Analyzer.restore_view(stub, "v", fallback=lambda: view) is view
    assert written["v"] is view


@pytest.mark.parametrize("kind", ["list_", "large_list"])
@pytest.mark.parametrize("item", ["int32", "float32", "large_string", "bool_"])
@pytest.mark.parametrize("cells", [[None, None], [[], []], []], ids=["all_missing", "all_empty", "no_rows"])
def test_a_list_column_with_no_values_keeps_its_item_type(tmp_path, kind, item, cells):
    pa = pytest.importorskip("pyarrow")
    frame = DataFrame.from_arrow(pa.table({"v": pa.array(cells, type=getattr(pa, kind)(getattr(pa, item)()))}))
    path = str(tmp_path / "t.pfw.gz")
    write_table(frame, path, "t")
    back = read_table(path)
    family = {"int32": DType.INT32, "float32": DType.FLOAT32, "large_string": DType.STRING, "bool_": DType.BOOL}
    assert back.dtypes == [DType.LIST]
    assert back["v"].child(0).dtype == family[item]
    assert back["v"].to_list() == cells


@pytest.mark.parametrize("kind", ["list_", "large_list"])
@pytest.mark.parametrize("item", ["date32", "timestamp", "decimal", "float16"])
@pytest.mark.parametrize("valued", [False, True])
def test_a_list_column_of_a_type_a_table_cannot_store_is_refused(tmp_path, kind, item, valued):
    import datetime
    import decimal

    pa = pytest.importorskip("pyarrow")
    item_type, value = {
        "date32": (pa.date32(), datetime.date(2026, 1, 1)),
        "timestamp": (pa.timestamp("us"), datetime.datetime(2026, 1, 1)),
        "decimal": (pa.decimal128(5, 2), decimal.Decimal("1.50")),
        "float16": (pa.float16(), 1.5),
    }[item]
    cells = [[value] if valued else [], None]
    frame = DataFrame.from_arrow(pa.table({"v": pa.array(cells, type=getattr(pa, kind)(item_type))}))
    with pytest.raises(TypeError, match="'v'"):
        write_table(frame, str(tmp_path / "t.pfw.gz"), "t")


@pytest.mark.parametrize("value", [math.inf, math.nan])
def test_a_view_the_table_cannot_store_is_kept_without_a_checkpoint(tmp_path, value):
    """A caller time range can hold inf, which a table list cannot; the checkpoint is a cache, so it is skipped."""
    analyzer = _analyzer(tmp_path)
    name = analyzer.get_checkpoint_name("hlm", "proc_name")
    old = DataFrame.from_dict({"k": [1, 2], "time_range": Series([[1.0], [2.0]])})
    assert analyzer.restore_view(name, fallback=lambda: old) is old
    assert analyzer.has_checkpoint(name)
    view = DataFrame.from_dict({"k": [1, 2], "time_range": Series([[1.0], [value]])})
    assert analyzer.restore_view(name, fallback=lambda: view, force=True) is view
    assert not analyzer.has_checkpoint(name)


def test_a_checkpoint_write_error_that_is_not_a_list_value_still_raises(tmp_path, monkeypatch):
    import dftracer.analyzer.analyzer as analyzer_module

    def refuse(*args, **kwargs):
        raise DFTUtilsValueError("column x does not fit the dtype of its header field")

    monkeypatch.setattr(analyzer_module, "write_table", refuse)
    analyzer = _analyzer(tmp_path)
    view = DataFrame.from_dict({"x": [1]})
    with pytest.raises(DFTUtilsValueError, match="header field"):
        analyzer.restore_view(analyzer.get_checkpoint_name("hlm", "proc_name"), fallback=lambda: view)


@pytest.mark.parametrize("kind", ["ListArray", "LargeListArray"])
@pytest.mark.parametrize("item", ["float32", "float64"])
def test_a_value_hidden_under_a_null_list_cell_keeps_the_checkpoint(tmp_path, kind, item):
    """Arrow can keep values under a null list cell; the writer skips them, so the view is stored."""
    pa = pytest.importorskip("pyarrow")
    offsets = pa.array([0, 1, 2], type=pa.int32() if kind == "ListArray" else pa.int64())
    values = pa.array([math.inf, 1.0], type=getattr(pa, item)())
    lists = getattr(pa, kind).from_arrays(offsets, values, mask=pa.array([True, False]))
    view = DataFrame.from_dict({"k": [1, 2], "time_range": Series.from_arrow(lists)})
    analyzer = _analyzer(tmp_path)
    name = analyzer.get_checkpoint_name("hlm", "proc_name")
    assert analyzer.restore_view(name, fallback=lambda: view) is view
    assert analyzer.has_checkpoint(name)
    assert analyzer.read_checkpoint(name)["time_range"].to_list() == [None, [1.0]]
