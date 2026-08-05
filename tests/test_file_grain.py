"""Coverage for the two file-derived behaviours the folded read must preserve.

Both were untested and both failed silently while the read was being folded to
a coarser grain: ignored files stopped being dropped, and the distinct-file
count was computed from a frame that no longer had one row per file.
"""

import gzip
import json
import pathlib

import pytest
from dask.distributed import LocalCluster

from dftracer.analyzer import init_with_hydra

pytestmark = [pytest.mark.smoke, pytest.mark.full]

HHASH = "abc123def456"
PID = TID = 2000

# Two files the analysis keeps, two matching IGNORED_FILE_PATTERNS.
KEPT_FILES = ["/scratch/input/a.bin", "/scratch/input/b.bin"]
IGNORED_FILES = ["/usr/lib/python3.12/site.py", "/proc/self/stat"]


@pytest.fixture(scope="session")
def dask_cluster():
    cluster = LocalCluster(processes=False, protocol="tcp", worker_class="distributed.nanny.Nanny")
    yield cluster
    cluster.close()


def _write_trace(trace_dir: pathlib.Path) -> str:
    """One POSIX read per file, so per-file counts are unambiguous."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    files = KEPT_FILES + IGNORED_FILES
    lines = [
        "[",
        json.dumps(
            {"name": "HH", "cat": "dftracer", "pid": PID, "tid": TID, "ph": "M",
             "args": {"hhash": HHASH, "name": "node0", "value": HHASH}}
        ),
    ]
    for i, path in enumerate(files):
        lines.append(
            json.dumps(
                {"name": "FH", "cat": "dftracer", "pid": PID, "tid": TID, "ph": "M",
                 "args": {"hhash": HHASH, "name": path, "value": f"fh{i}"}}
            )
        )
    lines.append(
        json.dumps(
            {"id": 1, "name": "start", "cat": "dftracer", "pid": PID, "tid": TID,
             "ts": 1000000, "dur": 0, "ph": "X", "args": {"hhash": HHASH, "ppid": 999}}
        )
    )
    ts, eid = 1000100, 2
    for i, _ in enumerate(files):
        for _ in range(5):
            lines.append(
                json.dumps(
                    {"id": eid, "name": "read", "cat": "POSIX", "pid": PID, "tid": TID,
                     "ts": ts, "dur": 100, "ph": "X",
                     "args": {"hhash": HHASH, "fhash": f"fh{i}", "ret": 4096, "offset": 0}}
                )
            )
            ts += 110
            eid += 1
    with gzip.open(trace_dir / "trace-1_chunk0.pfw.gz", "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return str(trace_dir)


@pytest.fixture
def mixed_file_trace(tmp_path: pathlib.Path) -> str:
    return _write_trace(tmp_path / "mixed")


def _analyze(trace_path, tmp_path, dask_cluster, view_types="[time_range]"):
    dfa = init_with_hydra(hydra_overrides=[
        "analyzer=dftracer",
        "analyzer/preset=auto",
        "analyzer.checkpoint=False",
        f"analyzer.checkpoint_dir={tmp_path}/checkpoints",
        "cluster=external",
        "cluster.restart_on_connect=True",
        f"cluster.scheduler_address={dask_cluster.scheduler_address}",
        f"hydra.run.dir={tmp_path}",
        f"hydra.runtime.output_dir={tmp_path}",
        f"trace_path={trace_path}",
        f"view_types={view_types}",
    ])
    return dfa, dfa.analyze_trace()


def test_ignored_files_are_dropped_from_the_analysis(
    mixed_file_trace: str, tmp_path: pathlib.Path, dask_cluster: LocalCluster
) -> None:
    """IGNORED_FILE_PATTERNS must still drop rows once the read folds files
    into buckets: the fold sees the full path, so it can still match."""
    dfa, result = _analyze(mixed_file_trace, tmp_path, dask_cluster)

    flat_view = result.flat_views[next(iter(result.flat_views))]
    counts = {
        layer: int(flat_view[f"{layer}_count_sum"].sum())
        for layer in result.layers
        if f"{layer}_count_sum" in flat_view.columns
    }
    assert counts, "no layer counts reached the views"
    # 5 reads per kept file; the ignored files' 10 reads must not be counted.
    assert counts.get("posix") == len(KEPT_FILES) * 5, f"ignored files leaked: {counts}"

    dfa.shutdown()


def test_unique_file_count_excludes_ignored_files(
    mixed_file_trace: str, tmp_path: pathlib.Path, dask_cluster: LocalCluster
) -> None:
    """The distinct-file count reports files the analysis actually used, so it
    must not include files dropped by IGNORED_FILE_PATTERNS."""
    dfa, result = _analyze(mixed_file_trace, tmp_path, dask_cluster)

    assert int(result.raw_stats.unique_file_count) == len(KEPT_FILES)

    dfa.shutdown()
