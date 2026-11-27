"""Checkpoint tables as dftracer-utils ``dftu.table`` files (gzip JSON lines with a typed header)."""

from __future__ import annotations

from dftracer.utils import DataFrame
from dftracer.utils.trace import TraceViewer

FORMAT_TAG = "frames2"
"""Part of every checkpoint name, so a checkpoint written in another frame format is not read."""


def write_table(frame: DataFrame, path: str, name: str, meta: dict | None = None) -> None:
    """Write `frame` to `path` (one ``.pfw.gz`` file) as a table named `name`, replacing any file there.

    The header keeps the dtypes, the index and its names, so `read_table` returns an equal frame.
    Raises TypeError for a list column whose item type a table cannot store (such as dates), even when empty,
    and DFTUtilsValueError for NaN or inf inside a float list cell.
    """
    frame.write_table(path, name=name, meta=meta or {}, mode="overwrite")


def read_table(path: str) -> DataFrame:
    """Read a table written by `write_table`, with its dtypes and index restored."""
    return TraceViewer(path).collect()
