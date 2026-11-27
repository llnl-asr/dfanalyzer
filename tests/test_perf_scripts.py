"""The perf scripts under scripts/perf."""

import importlib.util
import pathlib
import pickle

import pandas as pd
import pytest

pytestmark = pytest.mark.smoke

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "perf"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture(path: pathlib.Path, flat: pd.DataFrame) -> None:
    path.mkdir()
    payload = {"raw_stats": {"job_time": 1.0}, "layers": ["posix"], "view_types": ["proc_name"]}
    payload["flat_views"] = {("proc_name",): flat}
    (path / "capture.pkl").write_bytes(pickle.dumps(payload))


def test_equivalence_compares_values_not_the_dtype_flavor(tmp_path: pathlib.Path, capsys) -> None:
    """A pandas-build capture (object index, numpy dtypes) equals a library-build one (nullable dtypes)."""
    equivalence = _load("equivalence")
    old = pd.DataFrame({"t": [1.0, 2.5], "n": [3, 4]}, index=pd.Index(["a", "b"], name="proc_name", dtype=object))
    new = pd.DataFrame(
        {"t": pd.array([1.0, 2.5], dtype="Float64"), "n": pd.array([3, 4], dtype="Int64")},
        index=pd.Index(["a", "b"], name="proc_name", dtype="string"),
    )
    _capture(tmp_path / "old", old)
    _capture(tmp_path / "new", new)
    equivalence.compare(str(tmp_path / "old"), str(tmp_path / "new"))
    assert "IDENTICAL" in capsys.readouterr().out

    _capture(tmp_path / "changed", new.assign(t=pd.array([1.0, 2.6], dtype="Float64")))
    with pytest.raises(SystemExit):
        equivalence.compare(str(tmp_path / "old"), str(tmp_path / "changed"))


def test_equivalence_keeps_the_dtype_family(tmp_path: pathlib.Path, capsys) -> None:
    """A float column that turned into an int one differs; two all-missing float columns are equal."""
    equivalence = _load("equivalence")
    index = pd.MultiIndex.from_arrays([[0, 1], ["a", "b"]], names=[None, "proc_name"])
    old = pd.DataFrame({"t": [1.0, 2.0], "m": [float("nan")] * 2}, index=index)
    new = pd.DataFrame(
        {"t": pd.array([1.0, 2.0], dtype="Float64"), "m": pd.array([None] * 2, dtype="Float64")}, index=index
    )
    _capture(tmp_path / "old", old)
    _capture(tmp_path / "new", new)
    equivalence.compare(str(tmp_path / "old"), str(tmp_path / "new"))
    assert "IDENTICAL" in capsys.readouterr().out
    assert list(equivalence._normalized(old).index.names) == [None, "proc_name"]

    _capture(tmp_path / "int", new.assign(t=pd.array([1, 2], dtype="Int64")))
    with pytest.raises(SystemExit):
        equivalence.compare(str(tmp_path / "old"), str(tmp_path / "int"))


def test_equivalence_normalizes_list_columns() -> None:
    equivalence = _load("equivalence")
    frame = pd.DataFrame({"paths": [["/a", "/b"], []]})
    assert equivalence._normalized(frame)["paths"].tolist() == [["/a", "/b"], []]


def test_equivalence_normalizes_empty_sized_and_categorical_columns() -> None:
    equivalence = _load("equivalence")
    empty = equivalence._normalized(
        pd.DataFrame({"t": pd.Series([], dtype="float64"), "n": pd.Series([], dtype="int64")})
    )
    assert {c: str(t) for c, t in empty.dtypes.items()} == {"t": "Float64", "n": "Int64"}
    sized = equivalence._normalized(pd.DataFrame({"a": pd.array([1], dtype="Int32"), "b": [1]}))
    assert str(sized["a"].dtype) == str(sized["b"].dtype) == "Int64"
    cat = equivalence._normalized(pd.DataFrame({"c": pd.Categorical([1, None]), "t": pd.Categorical(["a", "b"])}))
    assert cat["c"].tolist()[0] == 1 and str(cat["c"].dtype) == "Int64" and str(cat["t"].dtype) == "string"


def test_equivalence_keeps_the_family_of_empty_categoricals() -> None:
    equivalence = _load("equivalence")
    frame = pd.DataFrame({"c": pd.Categorical([], categories=pd.Index([], dtype="int64"))})
    assert str(equivalence._normalized(frame)["c"].dtype) == "Int64"


def test_equivalence_keeps_mixed_categories_apart() -> None:
    equivalence = _load("equivalence")
    out = equivalence._normalized(pd.DataFrame({"c": pd.Categorical([1, "1"])}))
    assert out["c"].tolist() == [1, "1"]
