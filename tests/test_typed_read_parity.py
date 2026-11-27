# SPDX-License-Identifier: MIT
# ruff: noqa: UP031
"""Parity checks for the View-based dfanalyzer read path.

These tests run `read_frames`, the read the analyzer uses, and pin the schema
mapping, the fold of files into buckets and the per-worker split.
"""

import gzip

import dftracer.utils.index as dftu_utils
import pytest

try:
    import dask  # noqa: F401

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


def _read(files, idx, client=None, **settings):
    """`read_frames` on `files` as pandas {events, profiles, system}, an absent family as an empty frame."""
    import pandas as pd
    from dftracer.analyzer.cluster import NullClient
    from dftracer.analyzer.trace_index import TraceRun
    from dftracer.analyzer.typed_read import ReadSettings, read_frames
    from dftracer.utils import DataFrame

    run = TraceRun(tuple(files), idx, client or NullClient())
    fr = read_frames(run, ReadSettings(**settings))
    out = {}
    for name, key in (("events", "traces"), ("profiles", "profiles"), ("system", "system")):
        frame = fr[key]
        assert frame is None or isinstance(frame, DataFrame), (name, type(frame))
        out[name] = frame.to_pandas() if frame is not None else pd.DataFrame()
    return out


def _make_trace(path):
    with gzip.open(path, "wt") as f:
        for i in range(300):
            name, dur, ret = [("read", 10 + i, 100), ("write", 20 + i, 200), ("open", 5, 0)][i % 3]
            extra = "" if name == "open" else ', "ret": %d' % ret
            f.write(
                '{"ph":"X","name":"%s","cat":"POSIX","pid":1,"tid":1,'
                '"ts":%d,"dur":%d,"args":{"fhash":"f%d"%s}}\n' % (name, 1000 + i, dur, i % 2, extra)
            )


def test_workers_return_library_frames(tmp_path):
    """Each worker sends {events, profiles, system} as library frames (or None),
    so the driver concatenates them natively."""
    from dftracer.analyzer.cluster import NullClient
    from dftracer.analyzer.trace_index import TraceRun
    from dftracer.analyzer.typed_read import ReadSettings, submit_typed_read
    from dftracer.utils import DataFrame
    from dftracer.utils.index import AggregationConfig

    files = [str(tmp_path / f"t{k}.pfw.gz") for k in range(2)]
    for p in files:
        _make_trace(p)
    idx = str(tmp_path / "idx")
    with dftu_utils.Indexer(
        files=files,
        index_dir=idx,
        aggregation_config=AggregationConfig(time_interval_ms=100000),
    ) as ix:
        ix.ensure()

    client = NullClient()
    run = TraceRun(tuple(files), idx, client)
    # Granularity must match the tier's interval (100000 ms == 100 s): coarsening
    # only reads coarser than the stored bucket, never finer.
    (frames,) = client.gather(submit_typed_read(run, ReadSettings(time_granularity=100.0)))
    assert set(frames) == {"events", "profiles", "system"}
    assert all(f is None or isinstance(f, DataFrame) for f in frames.values())
    ev = frames["events"]
    assert ev is not None
    assert {"cat", "func_name", "count", "time", "time_start"}.issubset(ev.columns)
    assert int(ev["count"].sum()) == int(_read(files, idx, time_granularity=100.0)["events"]["count"].sum()) > 0


def test_read_maps_all_three_to_dfanalyzer_schema(tmp_path):
    """The read maps the index records' regular/aggregated/counters onto
    the dfanalyzer event/profile/system frames in one pass."""
    pytest.importorskip("pyarrow")
    import gzip

    from dftracer.utils.index import AggregationConfig

    p = str(tmp_path / "t.pfw.gz")
    with gzip.open(p, "wt") as f:
        for i in range(20):
            f.write(
                '{"ph":"X","name":"read","cat":"POSIX","pid":1,"tid":1,'
                '"ts":%d,"dur":%d,"args":{"ret":%d}}\n' % (1000 + i, 10 + i, 100 + i)
            )
        for e in range(3):
            f.write(
                '{"ph":"C","name":"train","cat":"app","pid":1,"tid":1,'
                '"ts":%d,"dur":5,"args":{"epoch":%d,"step":0}}\n' % (2000 + e, e)
            )
        for i in range(4):
            f.write(
                '{"ph":"C","name":"cpu","cat":"sys","pid":0,"tid":0,'
                '"ts":%d,"args":{"user_pct":%d}}\n' % (3000 + i, 40 + i)
            )
    idx = str(tmp_path / "idx")
    with dftu_utils.Indexer(
        files=[p],
        index_dir=idx,
        aggregation_config=AggregationConfig(time_interval_ms=100000),
    ) as ix:
        ix.ensure()

    fr = _read([p], idx, time_granularity=100.0)
    ev, prof, sysf = fr["events"], fr["profiles"], fr["system"]

    assert {"cat", "func_name", "count", "time", "time_start", "time_bucket"}.issubset(ev.columns)
    assert ev["func_name"].tolist() == ["read"] and int(ev["count"].iloc[0]) == 20
    # proc_name/file_hash/host_hash derived for the base HLM.
    assert {"proc_name", "file_hash", "host_hash"}.issubset(ev.columns)
    assert ev["proc_name"].iloc[0].startswith("app#") and ev["proc_name"].iloc[0].endswith("#1#1")
    assert prof["func_name"].tolist() == ["train"] and int(prof["count"].iloc[0]) == 3
    assert sysf["name"].tolist() == ["cpu"]
    assert abs(float(sysf["user_pct"].iloc[0]) - 41.5) < 1e-9


@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
def test_read_distributed_matches_single_node(tmp_path):
    """The Dask-fanned typed read (disjoint shard ranges per worker) equals the
    single-node read, for events, profiles, and system."""
    pytest.importorskip("pyarrow")
    import gzip

    from dask.distributed import Client, LocalCluster
    from dftracer.utils.index import AggregationConfig

    p = str(tmp_path / "t.pfw.gz")
    with gzip.open(p, "wt") as f:
        for i in range(300):
            f.write(
                '{"ph":"X","name":"read%d","cat":"POSIX","pid":%d,"tid":1,'
                '"ts":%d,"dur":%d}\n' % (i % 9, i % 6, 1000 + i, 10 + i)
            )
        for i in range(30):
            f.write(
                '{"ph":"C","name":"cpu","cat":"sys","pid":0,"tid":0,"ts":%d,'
                '"args":{"user_pct":%d,"hhash":"h%d"}}\n' % (3000 + i * 100, 40 + i, i % 4)
            )
    idx = str(tmp_path / "idx")
    with dftu_utils.Indexer(
        files=[p],
        index_dir=idx,
        aggregation_config=AggregationConfig(time_interval_ms=100000),
    ) as ix:
        ix.ensure()

    single = _read([p], idx, time_granularity=100.0)
    cluster = LocalCluster(processes=False, n_workers=3, threads_per_worker=1)
    client = Client(cluster)
    try:
        dist = _read([p], idx, client, time_granularity=100.0)
    finally:
        client.close()
        cluster.close()

    for name in ("events", "system"):
        assert len(single[name]) == len(dist[name]), name
    assert int(single["events"]["count"].sum()) == int(dist["events"]["count"].sum())
    assert set(single["system"]["name"]) == set(dist["system"]["name"])


def test_read_folds_files_into_buckets(tmp_path):
    """group_by_file=False collapses the per-file rows into one, preserving the
    additive metrics and the global extremes, and still emits the file columns
    (empty) that dfanalyzer's dask meta declares."""
    pytest.importorskip("pyarrow")
    from dftracer.analyzer.typed_read import typed_group_keys
    from dftracer.utils.index import AggregationConfig

    n_files = 8
    p = str(tmp_path / "t.pfw.gz")
    with gzip.open(p, "wt") as f:
        for k in range(n_files):
            f.write('{"name":"FH","ph":"M","pid":1,"tid":1,"args":{"name":"/data/f%d.dat","value":"h%d"}}\n' % (k, k))
        for k in range(n_files):
            for i in range(5):
                f.write(
                    '{"ph":"X","name":"read","cat":"POSIX","pid":1,"tid":1,'
                    '"ts":%d,"dur":%d,"args":{"ret":%d,"fhash":"h%d"}}\n' % (1000 + i, 10 + k, 100 + k, k)
                )
    idx = str(tmp_path / "idx")
    with dftu_utils.Indexer(
        files=[p],
        index_dir=idx,
        aggregation_config=AggregationConfig(time_interval_ms=100000),
    ) as ix:
        ix.ensure()

    fine = _read([p], idx, time_granularity=100.0)["events"]
    folded = _read([p], idx, time_granularity=100.0, group_keys=typed_group_keys(("/data", "/ckpt")))["events"]

    assert len(fine) == n_files, f"expected one row per file, got {len(fine)}"
    assert len(folded) == 1, f"expected the files folded into one row, got {len(folded)}"

    for col in ("count", "time", "size"):
        assert abs(fine[col].sum() - folded[col].sum()) <= 1e-9 * max(1.0, abs(fine[col].sum()))
    assert fine["time_min"].min() == folded["time_min"].min()
    assert fine["time_max"].max() == folded["time_max"].max()

    assert set(fine.columns) == set(folded.columns)
    # file_name carries the matched bucket, not the full path and not blank:
    # the caller's file-name substring rules still have to fire on it.
    assert set(folded["file_name"]) <= {"/data", "/ckpt"}
    assert folded["file_name"].str.contains("/data").any()
    assert set(fine["file_name"]) != {""}


def test_profile_on_a_bucket_boundary_keeps_its_bucket(tmp_path):
    """A profile row at a bucket's first microsecond belongs to that bucket,
    as an event does."""
    pytest.importorskip("pyarrow")
    import gzip

    from dftracer.utils.index import AggregationConfig

    p = str(tmp_path / "t.pfw.gz")
    with gzip.open(p, "wt") as f:
        f.write('{"ph":"X","name":"read","cat":"POSIX","pid":1,"tid":1,"ts":1000,"dur":10}\n')
        f.write('{"ph":"C","name":"train","cat":"app","pid":1,"tid":1,"ts":5000000,"dur":5,"args":{"count":1}}\n')
    idx = str(tmp_path / "idx")
    with dftu_utils.Indexer(
        files=[p], index_dir=idx, aggregation_config=AggregationConfig(time_interval_ms=5000)
    ) as ix:
        ix.ensure()
    prof = _read([p], idx, time_granularity=5.0, profile_time_granularity=5.0)["profiles"]
    assert prof["func_name"].tolist() == ["train"]
    assert int(prof["time_start"].iloc[0]) == 5_000_000
    assert int(prof["time_end"].iloc[0]) == 10_000_000
    assert int(prof["time_range"].iloc[0]) == 1


_NO_PYARROW_SCRIPT = r"""
import gzip, json, sys
if sys.argv[3] == "block":
    sys.modules["pyarrow"] = None

from dftracer.analyzer.cluster import NullClient
from dftracer.analyzer.trace_hlm import HLMConfig, hlm
from dftracer.analyzer.trace_index import TraceRun
from dftracer.analyzer.typed_read import ReadSettings, read_frames
from dftracer.utils.index import AggregationConfig, Indexer

path, idx = sys.argv[1], sys.argv[2]
with gzip.open(path, "wt") as f:
    for i in range(20):
        f.write(json.dumps({"ph": "X", "name": "read", "cat": "POSIX", "pid": 1, "tid": 1,
                            "ts": 1000 + i, "dur": 10 + i, "args": {"ret": 100 + i}}) + "\n")
    for e in range(3):
        f.write(json.dumps({"ph": "C", "name": "train", "cat": "app", "pid": 1, "tid": 1,
                            "ts": 2000 + e, "dur": 5, "args": {"epoch": e, "step": 0}}) + "\n")
    for i in range(4):
        f.write(json.dumps({"ph": "C", "name": "cpu", "cat": "sys", "pid": 0, "tid": 0,
                            "ts": 3000 + i, "args": {"user_pct": 40 + i}}) + "\n")
with Indexer(files=[path], index_dir=idx, aggregation_config=AggregationConfig(time_interval_ms=100000)) as ix:
    ix.ensure()
run = TraceRun((path,), idx, NullClient())
frames = read_frames(run, ReadSettings(time_granularity=100.0))
cfg = HLMConfig(time_granularity=100.0)
events = hlm(run, ["proc_name", "time_range"], cfg)
profiles = hlm(run, ["proc_name", "time_range"], cfg, profiles=True)
print(json.dumps({
    "traces": frames["traces"].num_rows,
    "profiles": frames["profiles"].num_rows,
    "system": frames["system"].num_rows,
    "hlm": events.num_rows,
    "profile_hlm": profiles.num_rows,
    "files": run.count_files(),
    "pyarrow": sys.modules.get("pyarrow") is not None,
}))
"""


def test_read_path_runs_without_pyarrow(tmp_path):
    """The read and both HLMs run in a process where pyarrow cannot be imported,
    and give the same row counts as a process that has it."""
    import json
    import subprocess
    import sys

    reports = {}
    for mode in ("block", "allow"):
        out = tmp_path / mode
        out.mkdir()
        run = subprocess.run(
            [sys.executable, "-c", _NO_PYARROW_SCRIPT, str(out / "t.pfw.gz"), str(out / "idx"), mode],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert run.returncode == 0, run.stderr[-3000:]
        reports[mode] = json.loads(run.stdout.strip().splitlines()[-1])
    blocked = reports["block"]
    assert blocked.pop("pyarrow") is False
    reports["allow"].pop("pyarrow")
    assert blocked == reports["allow"]
    assert all(blocked[k] > 0 for k in ("traces", "profiles", "system", "hlm", "profile_hlm")), blocked
