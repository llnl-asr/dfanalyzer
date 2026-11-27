"""Resolve trace inputs and build or find the dftracer-utils index."""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dftracer.utils.dask import DaskTraceViewer, DistributedIndexResult, distributed_index, register_auto_thread_plugin
from dftracer.utils.index import AggregationConfig, Indexer
from dftracer.utils.trace import discover_trace_files

if TYPE_CHECKING:
    from dask.distributed import Client


def _resolve_inputs(
    trace_path: str,
    trace_groups: list[str] | None,
) -> tuple[str, list[str] | None]:
    """Resolve a trace path into ``(directory, files)`` for the Indexer.

    A directory with a ``manifest.json`` selects the subdirectories of
    `trace_groups` (all groups when unset) and returns ``("", files)``. A plain
    directory returns ``(directory, None)``. A file or glob returns ``("", files)``.
    Raises FileNotFoundError when `trace_groups` is set and no manifest exists,
    and KeyError for a group the manifest does not name.
    """
    if not os.path.isdir(trace_path):
        return "", discover_trace_files(trace_path)

    manifest_path = os.path.join(trace_path, "manifest.json")
    if not os.path.isfile(manifest_path):
        if trace_groups:
            raise FileNotFoundError(
                f"trace_groups={trace_groups} requested but no manifest.json at "
                f"{manifest_path}. Provide a manifest.json, or unset "
                "trace_groups."
            )
        if not discover_trace_files(trace_path):
            return "", []
        return trace_path, None

    with open(manifest_path) as f:
        manifest = json.load(f)
    group_map = manifest.get("groups") or {}

    selected = trace_groups if trace_groups else sorted(group_map.keys())
    missing = [g for g in selected if g not in group_map]
    if missing:
        raise KeyError(
            f"trace_groups {missing} not found in manifest at {manifest_path}; "
            f"available groups: {sorted(group_map.keys())}"
        )

    files: list[str] = []
    for g in selected:
        files.extend(discover_trace_files(os.path.join(trace_path, group_map[g])))
    return "", files


def _index_dir_for(trace_path: str) -> str:
    """The ``.dftindex`` directory beside the traces.

    For a directory that is ``<trace_path>/.dftindex``. For a file or glob it is
    ``.dftindex`` in the directory of the file (or of the first match).
    """
    if os.path.isdir(trace_path):
        return os.path.join(trace_path, ".dftindex")
    if "*" in trace_path:
        matches = sorted(glob.glob(trace_path))
        if matches:
            return os.path.join(os.path.dirname(matches[0]), ".dftindex")
    return os.path.join(os.path.dirname(trace_path) or ".", ".dftindex")


def aggregation_config(time_interval_ms: float | str, group_by_file: bool = True) -> AggregationConfig:
    """The one aggregation config of a run.

    The index root holds one aggregation config and rebuilds its tier when an
    open asks for another, so every open of a run must pass this one.
    DFAnalyzer reads no index percentiles, so the tier skips them.
    """
    return AggregationConfig(time_interval_ms=time_interval_ms, compute_percentiles=False, group_by_file=group_by_file)


def active_client() -> Client | None:
    """The active Dask client, or None when ``dask.distributed`` is missing or no client is active."""
    try:
        from dask.distributed import get_client
    except ImportError:
        return None
    try:
        return get_client()
    except ValueError:
        return None


def build_index_distributed(
    files: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    index_dir: str = "",
    client: Client | None = None,
    aggregation: AggregationConfig | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> DistributedIndexResult:
    """Build the index across a Dask cluster, or inline and serially with no `client`.

    `progress` is called with ``(files_done, total_files, phase)``.
    """
    if client is not None:
        register_auto_thread_plugin()
    return distributed_index(
        files,
        index_dir=index_dir,
        client=client,
        aggregation_config=aggregation,
        progress=progress,
    )


@dataclass(frozen=True)
class TraceIndex:
    """The trace inputs of a run and the index directory beside them.

    `files` is a directory (the Indexer lists it) or the matched files.
    """

    files: str | tuple[str, ...]
    index_dir: str

    @classmethod
    def resolve(cls, trace_path: str, trace_groups: list[str] | None) -> TraceIndex:
        """Resolve `trace_path` without touching the index.

        Raises FileNotFoundError when `trace_groups` is set and no manifest
        exists, and KeyError for a group the manifest does not name.
        """
        directory, files = _resolve_inputs(trace_path, trace_groups)
        return cls(directory or tuple(files or ()), _index_dir_for(trace_path))

    @property
    def is_empty(self) -> bool:
        return not self.files

    def trace_files(self) -> list[str]:
        """The trace files, listing a directory with the indexer's file rule."""
        if not isinstance(self.files, str):
            return list(self.files)
        return discover_trace_files(self.files)

    def ensure(
        self,
        time_interval_ms: float | str,
        client: Client | None = None,
        progress: Callable[[int, int, str], None] | None = None,
        group_by_file: bool = True,
    ) -> None:
        """Build or refresh the index with the run's aggregation config.

        Runs on `client`, or inline with none. Idempotent: dftracer-utils skips
        files whose tiers exist. `group_by_file` is part of the index identity.
        """
        if self.is_empty:
            return
        build_index_distributed(
            self.files if isinstance(self.files, str) else list(self.files),
            index_dir=self.index_dir,
            client=client,
            aggregation=aggregation_config(time_interval_ms, group_by_file),
            progress=progress,
        )


@dataclass(frozen=True)
class TraceRun:
    """The data files, the opened index and the client of one read."""

    files: tuple[str, ...]
    index_dir: str
    client: Client

    def viewer(self) -> DaskTraceViewer:
        return DaskTraceViewer(list(self.files), self.index_dir, client=self.client)

    def count_files(self, ignored_patterns: tuple[str, ...] = ()) -> int:
        """Distinct data files in the index, skipping paths that contain any of `ignored_patterns`.

        Counts the index's ``files`` row set rather than the read, so it stays exact
        when the read is folded to a grain with no per-file rows. A caller that
        filters files out of the analysis must pass the same patterns.
        """
        with Indexer(list(self.files), index_dir=self.index_dir, require_checkpoint=False, require_bloom=False) as ix:
            rows = ix.rowset("files")
        table = {str(k): str(v) for k, v in zip(rows["fhash"].to_list(), rows["path"].to_list())}
        return sum(1 for name in table.values() if not any(p in name for p in ignored_patterns))
