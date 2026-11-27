import numpy as np
import pandas as pd
import pytest
from betterset import BetterSet as S

from dftracer.utils import DataFrame, col

from .pandas_reference import set_layer_metrics

pytestmark = [pytest.mark.smoke, pytest.mark.full]


def _atoms(cell) -> list:
    """The values of a set cell, or of one value, as a list; a missing value is an empty list."""
    if isinstance(cell, (set, frozenset, list, tuple)):
        return [v.item() if hasattr(v, "item") else v for v in cell if not pd.isna(v)]
    return [] if pd.isna(cell) else [cell.item() if hasattr(cell, "item") else cell]


DERIVED_METRICS = {
    "read": "io_cat == 1",
    "write": "io_cat == 2",
    "metadata": "io_cat == 3",
}

SIZE_DERIVED_METRICS = ["read", "write"]


def _build_hlm_df(n_rows: int = 30_000) -> pd.DataFrame:
    io_cat = np.tile(np.array([1, 2, 3, 1, 2], dtype=np.int64), int(np.ceil(n_rows / 5)))[:n_rows]
    idx = np.arange(n_rows, dtype=np.int64)
    return pd.DataFrame(
        {
            "io_cat": io_cat,
            "count": (idx % 17) + 1,
            "time": ((idx % 23) + 1).astype(float),
            "size": ((idx % 101) + 1) * 4096,
            "size_bin_0_4kb": (idx % 2).astype(np.int64),
            "func_name": np.where(io_cat == 1, "read", np.where(io_cat == 2, "write", "metadata")),
        }
    )


def test_set_layer_metrics_correctness() -> None:
    hlm = _build_hlm_df(n_rows=2_000)
    out = set_layer_metrics(
        hlm=hlm,
        derived_metrics=DERIVED_METRICS,
        size_derived_metrics=SIZE_DERIVED_METRICS,
    )

    # Size columns should only be created for metrics explicitly listed in size_derived_metrics.
    assert "read_size" in out.columns
    assert "write_size" in out.columns
    assert "metadata_size" not in out.columns

    read_mask = hlm["io_cat"] == 1
    write_mask = hlm["io_cat"] == 2
    metadata_mask = hlm["io_cat"] == 3

    assert np.allclose(
        out.loc[read_mask, "read_count"].astype(float),
        pd.to_numeric(hlm.loc[read_mask, "count"], errors="coerce").astype(float),
        equal_nan=True,
    )
    assert out.loc[~read_mask, "read_count"].isna().all()
    assert str(out["read_count"].dtype) == "Int64"

    assert np.allclose(
        out.loc[write_mask, "write_time"].astype(float),
        pd.to_numeric(hlm.loc[write_mask, "time"], errors="coerce").astype(float),
        equal_nan=True,
    )
    assert out.loc[~write_mask, "write_time"].isna().all()
    assert str(out["write_time"].dtype) == "Float64"

    # String-derived columns carry original values for matching rows and missing values otherwise.
    # The set union downstream skips missing values.
    assert (out.loc[read_mask, "read_func_name"] == hlm.loc[read_mask, "func_name"]).all()
    assert out.loc[~read_mask, "read_func_name"].isna().all()
    assert (out.loc[metadata_mask, "metadata_func_name"] == hlm.loc[metadata_mask, "func_name"]).all()


def test_set_layer_metrics_preserves_betterset_columns() -> None:
    hlm = pd.DataFrame(
        {
            "group": ["g0", "g0", "g1", "g1"],
            "io_cat": pd.Series([1, 2, 1, 3], dtype="Int64"),
            "count": pd.Series([1, 2, 3, 4], dtype="Int64"),
            "file_name": pd.Series(
                [S(["a"]), S(["b"]), S(["c"]), S(["d"])],
                dtype="object",
            ),
        }
    )
    out = set_layer_metrics(
        hlm=hlm,
        derived_metrics=DERIVED_METRICS,
        size_derived_metrics=SIZE_DERIVED_METRICS,
    )

    read_mask = hlm["io_cat"] == 1
    for idx in hlm.index[read_mask]:
        assert out.at[idx, "read_file_name"] == hlm.at[idx, "file_name"]
    assert out.loc[~read_mask, "read_file_name"].isna().all()

    cells = DataFrame.from_dict(
        {"group": out["group"].tolist(), "read_file_name": [_atoms(v) for v in out["read_file_name"]]}
    )
    aggregated = (
        cells.group_by(["group"])
        .agg(read_file_name=col("read_file_name").set_union(typed=True))
        .to_pandas(index=["group"])["read_file_name"]
    )
    assert set(aggregated.loc["g0"]) == {"a"}
    assert set(aggregated.loc["g1"]) == {"c"}


def test_set_layer_metrics_perf_smoke() -> None:
    hlm = _build_hlm_df(n_rows=50_000)
    out = None
    for _ in range(8):
        out = set_layer_metrics(
            hlm=hlm,
            derived_metrics=DERIVED_METRICS,
            size_derived_metrics=SIZE_DERIVED_METRICS,
        )
    assert out is not None
    assert int(out["read_count"].notna().sum()) > 0
