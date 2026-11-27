import gzip
import json
import pytest
from dask.distributed import Client, LocalCluster
from omegaconf import OmegaConf

from dftracer.analyzer.config import AnalyzerPresetConfigAI, AnalyzerPresetConfigPOSIX
from dftracer.analyzer.dftracer import DFTracerAnalyzer

pytestmark = [pytest.mark.smoke, pytest.mark.full]


def _write_trace_lines(path, lines):
    """Write trace as gzipped JSONL (.pfw.gz). Stray JSON-array brackets ('[',
    ']') from legacy-style fixtures are skipped since the indexer expects
    one JSON value per line."""
    with gzip.open(str(path), "wt", encoding="utf-8") as f:
        for line in lines:
            if line in ("[", "]"):
                continue
            f.write(line + "\n")


def _write_trace(path, events):
    """Write a list of event dicts as a gzipped JSONL trace (.pfw.gz)."""
    _write_trace_lines(path, [json.dumps(e) for e in events])


TRACE_CONTENT = [
    "[",
    json.dumps(
        {
            "id": 1,
            "name": "HH",
            "cat": "dftracer",
            "pid": 1,
            "tid": 1,
            "ph": "M",
            "args": {"hhash": "h1", "name": "hostA", "value": "h1"},
        }
    ),
    json.dumps(
        {
            "id": 2,
            "name": "FH",
            "cat": "dftracer",
            "pid": 1,
            "tid": 1,
            "ph": "M",
            "args": {"hhash": "h1", "name": "/tmp/data/file.bin", "value": "f1"},
        }
    ),
    json.dumps(
        {
            "id": 3,
            "name": "read",
            "cat": "POSIX",
            "pid": 1,
            "tid": 1,
            "ph": "X",
            "ts": 6000123,
            "dur": 200,
            "args": {"hhash": "h1", "fhash": "f1", "ret": 4096, "offset": 0},
        }
    ),
    json.dumps(
        {
            "id": 4,
            "name": "open64",
            "cat": "POSIX",
            "pid": 1,
            "tid": 1,
            "ph": "C",
            "ts": 10000000,
            "args": {"hhash": "h1", "fhash": "f1", "dft_cnt": 3, "dur_sum": 300},
        }
    ),
    "]",
]

TRACE_NO_HOST_METADATA_CONTENT = [
    "[",
    json.dumps(
        {
            "id": 1,
            "name": "FH",
            "cat": "dftracer",
            "pid": 1,
            "tid": 1,
            "ph": "M",
            "args": {"hhash": "h1", "name": "/tmp/data/file.bin", "value": "f1"},
        }
    ),
    json.dumps(
        {
            "id": 2,
            "name": "open64",
            "cat": "POSIX",
            "pid": 1,
            "tid": 1,
            "ph": "C",
            "ts": 10000000,
            "args": {"hhash": "h1", "fhash": "f1", "dft_cnt": 3, "dur_sum": 300},
        }
    ),
    "]",
]

FULL_TRACE_CONTENT = [
    "[",
    json.dumps(
        {
            "id": 1,
            "name": "HH",
            "cat": "dftracer",
            "pid": 1,
            "tid": 1,
            "ph": "M",
            "args": {"hhash": "h1", "name": "hostA", "value": "h1"},
        }
    ),
    json.dumps(
        {
            "id": 2,
            "name": "FH",
            "cat": "dftracer",
            "pid": 1,
            "tid": 1,
            "ph": "M",
            "args": {"hhash": "h1", "name": "/tmp/data/file.bin", "value": "f1"},
        }
    ),
    json.dumps(
        {
            "id": 3,
            "name": "read",
            "cat": "POSIX",
            "pid": 1,
            "tid": 1,
            "ph": "X",
            "ts": 6000123,
            "dur": 200,
            "args": {"hhash": "h1", "fhash": "f1", "ret": 4096, "offset": 0},
        }
    ),
    json.dumps(
        {
            "id": 4,
            "name": "ai_root",
            "cat": "ai_root",
            "pid": 1,
            "tid": 1,
            "ph": "C",
            "ts": 10000000,
            "args": {"hhash": "h1", "dft_cnt": 1, "dur_sum": 5000000},
        }
    ),
    json.dumps(
        {
            "id": 5,
            "name": "train",
            "cat": "pipeline",
            "pid": 1,
            "tid": 1,
            "ph": "C",
            "ts": 10000000,
            "args": {"hhash": "h1", "dft_cnt": 1, "dur_sum": 5000000},
        }
    ),
    json.dumps(
        {
            "id": 6,
            "name": "epoch.1",
            "cat": "pipeline",
            "pid": 1,
            "tid": 1,
            "ph": "C",
            "ts": 10000000,
            "args": {"hhash": "h1", "dft_cnt": 1, "dur_sum": 5000000},
        }
    ),
    json.dumps(
        {
            "id": 7,
            "name": "compute",
            "cat": "compute",
            "pid": 1,
            "tid": 1,
            "ph": "C",
            "ts": 10000000,
            "args": {"hhash": "h1", "dft_cnt": 2, "dur_sum": 4000000},
        }
    ),
    json.dumps(
        {
            "id": 8,
            "name": "fetch.iter",
            "cat": "pipeline",
            "pid": 1,
            "tid": 1,
            "ph": "C",
            "ts": 10000000,
            "args": {"hhash": "h1", "dft_cnt": 1, "dur_sum": 1000000},
        }
    ),
    json.dumps(
        {
            "id": 9,
            "name": "item",
            "cat": "data",
            "pid": 1,
            "tid": 1,
            "ph": "C",
            "ts": 10000000,
            "args": {"hhash": "h1", "dft_cnt": 2, "dur_sum": 2000000},
        }
    ),
    json.dumps(
        {
            "id": 10,
            "name": "item",
            "cat": "data",
            "pid": 1,
            "tid": 1,
            "ph": "C",
            "ts": 10000000,
            "args": {"hhash": "h1", "dft_cnt": 1, "dur": 500000},
        }
    ),
    json.dumps(
        {
            "id": 11,
            "name": "open64",
            "cat": "POSIX",
            "pid": 1,
            "tid": 1,
            "ph": "C",
            "ts": 10000000,
            "args": {"hhash": "h1", "fhash": "f1", "dft_cnt": 3, "dur_sum": 300},
        }
    ),
    "]",
]


@pytest.fixture
def dask_client():
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
        scheduler_port=0,
        silence_logs="error",
    )
    client = Client(cluster)
    try:
        yield client
    finally:
        client.close()
        cluster.close()


@pytest.fixture
def hybrid_trace_path(tmp_path):
    trace_path = tmp_path / "hybrid.pfw.gz"
    _write_trace_lines(trace_path, TRACE_CONTENT)
    return trace_path


@pytest.fixture
def full_hybrid_trace_path(tmp_path):
    trace_path = tmp_path / "full-hybrid.pfw.gz"
    _write_trace_lines(trace_path, FULL_TRACE_CONTENT)
    return trace_path


@pytest.fixture
def no_host_metadata_trace_path(tmp_path):
    trace_path = tmp_path / "no-host-hybrid.pfw.gz"
    _write_trace_lines(trace_path, TRACE_NO_HOST_METADATA_CONTENT)
    return trace_path


def make_analyzer(tmp_path, time_granularity, preset=None):
    return DFTracerAnalyzer(
        preset=OmegaConf.structured(preset or AnalyzerPresetConfigPOSIX()),
        checkpoint=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        debug=False,
        quantile_stats=False,
        time_approximate=True,
        time_granularity=time_granularity,
        time_resolution=10**6,
        time_sliced=False,
        verbose=False,
    )


def test_read_trace_standardizes_profiles_and_aligns_to_5s_grid(dask_client, hybrid_trace_path, tmp_path):
    analyzer = make_analyzer(tmp_path=tmp_path, time_granularity=5)

    read_result = analyzer.read_trace(
        trace_path=str(hybrid_trace_path),
        extra_columns=None,
        extra_columns_fn=None,
    )

    traces = read_result.traces
    profiles = read_result.profiles

    assert len(traces) == 1
    assert len(profiles) == 1

    profile = profiles.iloc[0]
    assert profile["func_name"] == "open64"
    assert profile["proc_name"] == "app#hostA#1#1"
    assert profile["file_name"] == "/tmp/data/file.bin"
    assert profile["count"] == 3
    assert profile["time"] == pytest.approx(0.0003)
    assert profile["size"] is None or str(profile["size"]) == "<NA>"
    # time_range is relative to the trace origin (first populated bucket).
    # The read at 6.0s is in bucket 0; the profile at 10.0s is one bucket later.
    assert profile["time_start"] == 5_000_000
    assert profile["time_end"] == 10_000_000
    assert profile["time_range"] == 1
    assert profile["time_start"] % 5_000_000 == 0


def test_analyze_trace_rejects_non_aligned_profile_granularity(dask_client, hybrid_trace_path, tmp_path):
    analyzer = make_analyzer(tmp_path=tmp_path, time_granularity=7)

    with pytest.raises(ValueError, match="must evenly divide"):
        analyzer.analyze_trace(
            trace_path=str(hybrid_trace_path),
            view_types=["proc_name", "time_range"],
        )


def test_read_trace_uses_host_hash_in_proc_name_when_host_name_is_missing(
    dask_client,
    no_host_metadata_trace_path,
    tmp_path,
):
    analyzer = make_analyzer(tmp_path=tmp_path, time_granularity=5)

    read_result = analyzer.read_trace(
        trace_path=str(no_host_metadata_trace_path),
        extra_columns=None,
        extra_columns_fn=None,
    )

    profiles = read_result.profiles

    assert len(profiles) == 1
    assert profiles.iloc[0]["host_name"] is None or str(profiles.iloc[0]["host_name"]) == "<NA>"
    assert profiles.iloc[0]["proc_name"] == "app#h1#1#1"


def test_read_trace_coalesces_duplicate_full_profile_rows(dask_client, full_hybrid_trace_path, tmp_path):
    analyzer = make_analyzer(
        tmp_path=tmp_path,
        time_granularity=5,
        preset=AnalyzerPresetConfigAI(),
    )

    read_result = analyzer.read_trace(
        trace_path=str(full_hybrid_trace_path),
        extra_columns=None,
        extra_columns_fn=None,
    )

    profiles = read_result.profiles
    item_profiles = profiles[profiles["func_name"] == "item"].reset_index(drop=True)

    assert len(item_profiles) == 1
    assert int(item_profiles.loc[0, "count"]) == 3
    assert float(item_profiles.loc[0, "time"]) == pytest.approx(2.5)
    assert item_profiles.loc[0, "size"] is None or str(item_profiles.loc[0, "size"]) == "<NA>"


def test_analyze_trace_reconciles_hybrid_hlm_and_tracks_raw_stats(dask_client, hybrid_trace_path, tmp_path):
    analyzer = make_analyzer(tmp_path=tmp_path, time_granularity=5)

    result = analyzer.analyze_trace(
        trace_path=str(hybrid_trace_path),
        view_types=["file_name", "proc_name", "time_range"],
    )

    assert result.profiles is not None
    assert int(result.raw_stats.trace_event_count) == 1
    assert int(result.raw_stats.profile_event_count) == 3
    assert int(result.raw_stats.total_event_count) == 4

    posix_hlm = result.get_hlm("posix").reset_index()

    assert set(posix_hlm["func_name"]) == {"read", "open64"}
    by_func = posix_hlm.set_index("func_name")
    assert int(by_func.loc["read", "count"]) == 1
    assert int(by_func.loc["open64", "count"]) == 3


def test_analyze_trace_reconciles_profiles_per_layer_for_full_hybrid_case(
    dask_client,
    full_hybrid_trace_path,
    tmp_path,
):
    analyzer = make_analyzer(
        tmp_path=tmp_path,
        time_granularity=5,
        preset=AnalyzerPresetConfigAI(),
    )

    result = analyzer.analyze_trace(
        trace_path=str(full_hybrid_trace_path),
        view_types=["proc_name", "time_range"],
    )

    assert result.profiles is not None

    app_hlm = result.get_hlm("app").reset_index()
    compute_hlm = result.get_hlm("compute").reset_index()
    data_hlm = result.get_hlm("data_loader").reset_index()
    posix_hlm = result.get_hlm("posix").reset_index()

    assert set(app_hlm["func_name"]) == {"ai_root"}
    assert set(compute_hlm["func_name"]) == {"compute"}
    assert set(data_hlm["func_name"]) == {"item"}
    assert set(posix_hlm["func_name"]) == {"read", "open64"}

    assert int(compute_hlm.iloc[0]["count"]) == 2
    assert int(data_hlm.iloc[0]["count"]) == 3
    assert int(posix_hlm.set_index("func_name").loc["open64", "count"]) == 3


def test_analyze_trace_expands_profiles_when_granularity_finer_than_bucket(dask_client, hybrid_trace_path, tmp_path):
    """When analysis granularity (1s) < profile bucket (5s), each profile row
    should be expanded into 5 sub-bucket rows with measures distributed."""
    analyzer = make_analyzer(tmp_path=tmp_path, time_granularity=1)

    result = analyzer.analyze_trace(
        trace_path=str(hybrid_trace_path),
        view_types=["proc_name", "time_range"],
    )

    profiles = result.profiles

    # Original had 1 profile row; with expansion_factor=5 we expect 5 rows
    assert len(profiles) == 5

    # All rows should share the same identity
    assert set(profiles["func_name"]) == {"open64"}
    assert set(profiles["proc_name"]) == {"app#hostA#1#1"}

    # Each sub-bucket should have a distinct time_range
    assert len(profiles["time_range"].unique()) == 5

    # Total count across sub-buckets should equal original count (3)
    assert int(profiles["count"].sum()) == 3

    # Total time across sub-buckets should equal original time
    assert float(profiles["time"].sum()) == pytest.approx(0.0003)

    # time_start should advance by 1s (1_000_000 us) per sub-bucket
    sorted_profiles = profiles.sort_values("time_start").reset_index(drop=True)
    starts = sorted_profiles["time_start"].values
    for i in range(1, len(starts)):
        assert int(starts[i] - starts[i - 1]) == 1_000_000

    # time_end - time_start should be 1s for each sub-bucket
    for _, row in sorted_profiles.iterrows():
        assert int(row["time_end"] - row["time_start"]) == 1_000_000


def test_analyze_trace_expands_profiles_with_weighted_distribution(dask_client, tmp_path):
    """Weighted distribution should concentrate more time in later sub-buckets
    when time_max > time_min."""
    trace_content = [
        "[",
        json.dumps({
            "id": 1, "name": "HH", "cat": "dftracer", "pid": 1, "tid": 1,
            "ph": "M", "args": {"hhash": "h1", "name": "hostA", "value": "h1"},
        }),
        json.dumps({
            "id": 2, "name": "read", "cat": "POSIX", "pid": 1, "tid": 1,
            "ph": "C", "ts": 5000000,
            "args": {
                "hhash": "h1", "dft_cnt": 10,
                "dur_sum": 1000, "dur_min": 10, "dur_max": 190,
                "ret_sum": 40960, "ret_min": 4096, "ret_max": 4096,
            },
        }),
        "]",
    ]
    trace_path = tmp_path / "weighted.pfw.gz"
    _write_trace_lines(trace_path, trace_content)

    analyzer = DFTracerAnalyzer(
        preset=OmegaConf.structured(AnalyzerPresetConfigPOSIX()),
        checkpoint=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        debug=False,
        profile_distribution="weighted",
        quantile_stats=False,
        time_approximate=True,
        time_granularity=1,
        time_resolution=10**6,
        time_sliced=False,
        verbose=False,
    )

    result = analyzer.analyze_trace(
        trace_path=str(trace_path),
        view_types=["proc_name", "time_range"],
    )

    profiles = result.profiles
    assert len(profiles) == 5

    sorted_profiles = profiles.sort_values("time_start").reset_index(drop=True)

    # With weighted distribution and dur_min=10, dur_max=190:
    # later sub-buckets should get more time than earlier ones
    first_time = float(sorted_profiles.iloc[0]["time"])
    last_time = float(sorted_profiles.iloc[-1]["time"])
    assert last_time > first_time

    # Total time should still be conserved
    assert float(profiles["time"].sum()) == pytest.approx(0.001)

    # Total count still conserved
    assert int(profiles["count"].sum()) == 10


def test_analyze_trace_rejects_non_divisible_finer_granularity(dask_client, hybrid_trace_path, tmp_path):
    """3s doesn't evenly divide 5s, so it should be rejected."""
    analyzer = make_analyzer(tmp_path=tmp_path, time_granularity=3)

    with pytest.raises(ValueError, match="must evenly divide"):
        analyzer.analyze_trace(
            trace_path=str(hybrid_trace_path),
            view_types=["proc_name", "time_range"],
        )


def test_analyze_trace_accepts_finer_aligned_granularity(dask_client, hybrid_trace_path, tmp_path):
    """2.5s evenly divides 5s (factor=2), should be accepted."""
    analyzer = make_analyzer(tmp_path=tmp_path, time_granularity=2.5)

    result = analyzer.analyze_trace(
        trace_path=str(hybrid_trace_path),
        view_types=["proc_name", "time_range"],
    )

    profiles = result.profiles
    # 5s / 2.5s = 2 sub-buckets per original profile row
    assert len(profiles) == 2
    assert int(profiles["count"].sum()) == 3


# ---------------------------------------------------------------------------
# Edge-case data fixtures
# ---------------------------------------------------------------------------



METADATA_EVENTS = [
    {"id": 1, "name": "HH", "cat": "dftracer", "pid": 1, "tid": 1,
     "ph": "M", "args": {"hhash": "h1", "name": "hostA", "value": "h1"}},
    {"id": 2, "name": "FH", "cat": "dftracer", "pid": 1, "tid": 1,
     "ph": "M", "args": {"hhash": "h1", "name": "/tmp/data/file.bin", "value": "f1"}},
]


# ---------------------------------------------------------------------------
# Test: C row with bare `dur` instead of `dur_sum` (37% of production data)
# ---------------------------------------------------------------------------

def test_read_trace_handles_dur_field_without_dur_sum(dask_client, tmp_path):
    events = METADATA_EVENTS + [
        {"id": 3, "name": "open64", "cat": "POSIX", "pid": 1, "tid": 1,
         "ph": "C", "ts": 5000000,
         "args": {"hhash": "h1", "fhash": "f1", "dft_cnt": 1, "dur": 150}},
    ]
    path = tmp_path / "dur-only.pfw.gz"
    _write_trace(path, events)
    analyzer = make_analyzer(tmp_path=tmp_path, time_granularity=5)

    profiles = analyzer.read_trace(
        trace_path=str(path), extra_columns=None, extra_columns_fn=None,
    ).profiles

    assert len(profiles) == 1
    p = profiles.iloc[0]
    assert p["func_name"] == "open64"
    assert p["count"] == 1
    # dur=150 us → time=150/1e6 s
    assert float(p["time"]) == pytest.approx(150 / 1e6)
    # min/max should fall back to dur when dur_min/dur_max absent
    assert float(p["time_min"]) == pytest.approx(150 / 1e6)
    assert float(p["time_max"]) == pytest.approx(150 / 1e6)


# ---------------------------------------------------------------------------
# Test: C row missing fhash (non-file events: compute, comm, device, etc.)
# ---------------------------------------------------------------------------

def test_read_trace_handles_profile_without_fhash(dask_client, tmp_path):
    events = [
        {"id": 1, "name": "HH", "cat": "dftracer", "pid": 1, "tid": 1,
         "ph": "M", "args": {"hhash": "h1", "name": "hostA", "value": "h1"}},
        # X event so traces aren't empty
        {"id": 2, "name": "read", "cat": "POSIX", "pid": 1, "tid": 1,
         "ph": "X", "ts": 5000100, "dur": 50,
         "args": {"hhash": "h1", "fhash": "f1", "ret": 100}},
        # C event without fhash — like compute:forward in real data
        {"id": 3, "name": "forward", "cat": "compute", "pid": 1, "tid": 1,
         "ph": "C", "ts": 5000000,
         "args": {"hhash": "h1", "dft_cnt": 5, "dur_sum": 4000000}},
    ]
    path = tmp_path / "no-fhash.pfw.gz"
    _write_trace(path, events)
    analyzer = make_analyzer(tmp_path=tmp_path, time_granularity=5)

    profiles = analyzer.read_trace(
        trace_path=str(path), extra_columns=None, extra_columns_fn=None,
    ).profiles

    assert len(profiles) == 1
    p = profiles.iloc[0]
    assert p["func_name"] == "forward"
    assert p["cat"] == "compute"
    assert p["count"] == 5
    # file_name should be NA for non-file events
    assert p["file_name"] is None or str(p["file_name"]) == "<NA>"


# ---------------------------------------------------------------------------
# Test: Full aggregation fields (ret_min/max, offset_min/max) are preserved
# ---------------------------------------------------------------------------

def test_read_trace_preserves_stat_columns_from_aggregation_fields(dask_client, tmp_path):
    events = METADATA_EVENTS + [
        {"id": 3, "name": "read", "cat": "POSIX", "pid": 1, "tid": 1,
         "ph": "C", "ts": 5000000,
         "args": {
             "hhash": "h1", "fhash": "f1",
             "dft_cnt": 10,
             "dur_sum": 5000, "dur_min": 100, "dur_max": 900,
             "ret_sum": 40960, "ret_min": 1024, "ret_max": 8192,
             "offset_sum": 100000, "offset_min": 0, "offset_max": 50000,
         }},
    ]
    path = tmp_path / "full-agg.pfw.gz"
    _write_trace(path, events)
    analyzer = make_analyzer(tmp_path=tmp_path, time_granularity=5)

    profiles = analyzer.read_trace(
        trace_path=str(path), extra_columns=None, extra_columns_fn=None,
    ).profiles

    assert len(profiles) == 1
    p = profiles.iloc[0]
    assert float(p["time_min"]) == pytest.approx(100 / 1e6)
    assert float(p["time_max"]) == pytest.approx(900 / 1e6)
    assert int(p["size_min"]) == 1024
    assert int(p["size_max"]) == 8192
    assert int(p["offset_min"]) == 0
    assert int(p["offset_max"]) == 50000
    assert int(p["size"]) == 40960
    assert int(p["count"]) == 10


# ---------------------------------------------------------------------------
# Test: Trace-only input — no C events, profiles should be None
# ---------------------------------------------------------------------------

def test_read_trace_returns_none_profiles_when_no_counter_events(dask_client, tmp_path):
    events = METADATA_EVENTS + [
        {"id": 3, "name": "read", "cat": "POSIX", "pid": 1, "tid": 1,
         "ph": "X", "ts": 5000100, "dur": 200,
         "args": {"hhash": "h1", "fhash": "f1", "ret": 4096, "offset": 0}},
    ]
    path = tmp_path / "no-profiles.pfw.gz"
    _write_trace(path, events)
    analyzer = make_analyzer(tmp_path=tmp_path, time_granularity=5)

    result = analyzer.read_trace(
        trace_path=str(path), extra_columns=None, extra_columns_fn=None,
    )

    assert result.profiles is None
    assert result.profile_time_granularity is None
    traces = result.traces
    assert len(traces) == 1


# ---------------------------------------------------------------------------
# Test: Multiple profile buckets at different timestamps
# ---------------------------------------------------------------------------

def test_read_trace_handles_multiple_profile_buckets(dask_client, tmp_path):
    events = METADATA_EVENTS + [
        {"id": 3, "name": "read", "cat": "POSIX", "pid": 1, "tid": 1,
         "ph": "C", "ts": 5000000,
         "args": {"hhash": "h1", "fhash": "f1", "dft_cnt": 3, "dur_sum": 600,
                  "ret_sum": 12288, "ret_min": 4096, "ret_max": 4096}},
        {"id": 4, "name": "read", "cat": "POSIX", "pid": 1, "tid": 1,
         "ph": "C", "ts": 10000000,
         "args": {"hhash": "h1", "fhash": "f1", "dft_cnt": 7, "dur_sum": 1400,
                  "ret_sum": 28672, "ret_min": 4096, "ret_max": 4096}},
        {"id": 5, "name": "write", "cat": "POSIX", "pid": 1, "tid": 1,
         "ph": "C", "ts": 5000000,
         "args": {"hhash": "h1", "fhash": "f1", "dft_cnt": 2, "dur_sum": 800,
                  "ret_sum": 8192, "ret_min": 4096, "ret_max": 4096}},
    ]
    path = tmp_path / "multi-bucket.pfw.gz"
    _write_trace(path, events)
    analyzer = make_analyzer(tmp_path=tmp_path, time_granularity=5)

    profiles = analyzer.read_trace(
        trace_path=str(path), extra_columns=None, extra_columns_fn=None,
    ).profiles

    # 2 read buckets at different ts + 1 write bucket = 3 rows
    assert len(profiles) == 3

    reads = profiles[profiles["func_name"] == "read"].sort_values("time_range").reset_index(drop=True)
    writes = profiles[profiles["func_name"] == "write"]
    assert len(reads) == 2
    assert len(writes) == 1
    assert int(reads["count"].sum()) == 10
    assert int(writes.iloc[0]["count"]) == 2


# ---------------------------------------------------------------------------
# Test: Coalescing preserves min/max correctly across duplicate rows
# ---------------------------------------------------------------------------

def test_coalesce_takes_min_of_mins_and_max_of_maxes(dask_client, tmp_path):
    """Two C rows for same (func, ts, pid, ...) key with different min/max
    should coalesce to the true min and true max."""
    events = METADATA_EVENTS + [
        {"id": 3, "name": "item", "cat": "data", "pid": 1, "tid": 1,
         "ph": "C", "ts": 10000000,
         "args": {"hhash": "h1", "dft_cnt": 5, "dur_sum": 2000000,
                  "dur_min": 200000, "dur_max": 600000}},
        {"id": 4, "name": "item", "cat": "data", "pid": 1, "tid": 1,
         "ph": "C", "ts": 10000000,
         "args": {"hhash": "h1", "dft_cnt": 3, "dur_sum": 900000,
                  "dur_min": 100000, "dur_max": 500000}},
    ]
    path = tmp_path / "coalesce-minmax.pfw.gz"
    _write_trace(path, events)
    analyzer = make_analyzer(
        tmp_path=tmp_path, time_granularity=5,
        preset=AnalyzerPresetConfigAI(),
    )

    profiles = analyzer.read_trace(
        trace_path=str(path), extra_columns=None, extra_columns_fn=None,
    ).profiles

    assert len(profiles) == 1
    p = profiles.iloc[0]
    assert int(p["count"]) == 8
    assert float(p["time"]) == pytest.approx(2.9)
    # min should be the smallest across both rows
    assert float(p["time_min"]) == pytest.approx(100000 / 1e6)
    # max should be the largest across both rows
    assert float(p["time_max"]) == pytest.approx(600000 / 1e6)


# ---------------------------------------------------------------------------
# Test: Expansion preserves stat columns unchanged across sub-buckets
# ---------------------------------------------------------------------------

def test_expansion_preserves_stat_columns_across_sub_buckets(dask_client, tmp_path):
    events = METADATA_EVENTS + [
        {"id": 3, "name": "read", "cat": "POSIX", "pid": 1, "tid": 1,
         "ph": "C", "ts": 5000000,
         "args": {
             "hhash": "h1", "fhash": "f1",
             "dft_cnt": 10,
             "dur_sum": 5000, "dur_min": 100, "dur_max": 900,
             "ret_sum": 40960, "ret_min": 1024, "ret_max": 8192,
             "offset_min": 0, "offset_max": 50000,
         }},
    ]
    path = tmp_path / "expand-stats.pfw.gz"
    _write_trace(path, events)
    analyzer = make_analyzer(tmp_path=tmp_path, time_granularity=1)

    result = analyzer.analyze_trace(
        trace_path=str(path),
        view_types=["proc_name", "time_range"],
    )

    profiles = result.profiles
    assert len(profiles) == 5

    # Every sub-bucket should carry the same stat bounds
    for _, row in profiles.iterrows():
        assert float(row["time_min"]) == pytest.approx(100 / 1e6)
        assert float(row["time_max"]) == pytest.approx(900 / 1e6)
        assert int(row["size_min"]) == 1024
        assert int(row["size_max"]) == 8192
        assert int(row["offset_min"]) == 0
        assert int(row["offset_max"]) == 50000


# ---------------------------------------------------------------------------
# Test: Non-POSIX profile categories reconcile into correct layers
# ---------------------------------------------------------------------------

def test_analyze_trace_non_posix_profiles_map_to_correct_layers(dask_client, tmp_path):
    events = [
        {"id": 1, "name": "HH", "cat": "dftracer", "pid": 1, "tid": 1,
         "ph": "M", "args": {"hhash": "h1", "name": "hostA", "value": "h1"}},
        # Need at least one X event for traces
        {"id": 2, "name": "read", "cat": "POSIX", "pid": 1, "tid": 1,
         "ph": "X", "ts": 6000100, "dur": 50,
         "args": {"hhash": "h1", "fhash": "f1", "ret": 100}},
        # comm layer
        {"id": 3, "name": "all_reduce", "cat": "comm", "pid": 1, "tid": 1,
         "ph": "C", "ts": 5000000,
         "args": {"hhash": "h1", "dft_cnt": 4, "dur_sum": 2000000}},
        # device layer
        {"id": 4, "name": "transfer", "cat": "device", "pid": 1, "tid": 1,
         "ph": "C", "ts": 5000000,
         "args": {"hhash": "h1", "dft_cnt": 2, "dur_sum": 500000}},
        # compute layer
        {"id": 5, "name": "backward", "cat": "compute", "pid": 1, "tid": 1,
         "ph": "C", "ts": 5000000,
         "args": {"hhash": "h1", "dft_cnt": 1, "dur_sum": 3000000}},
        # dataloader layer
        {"id": 6, "name": "fetch.block", "cat": "dataloader", "pid": 1, "tid": 1,
         "ph": "C", "ts": 5000000,
         "args": {"hhash": "h1", "dft_cnt": 3, "dur_sum": 1500000}},
    ]
    path = tmp_path / "non-posix.pfw.gz"
    _write_trace(path, events)
    analyzer = make_analyzer(
        tmp_path=tmp_path, time_granularity=5,
        preset=AnalyzerPresetConfigAI(),
    )

    result = analyzer.analyze_trace(
        trace_path=str(path),
        view_types=["proc_name", "time_range"],
    )

    comm_hlm = result.get_hlm("comm").reset_index()
    device_hlm = result.get_hlm("device").reset_index()
    compute_hlm = result.get_hlm("compute").reset_index()
    data_hlm = result.get_hlm("data_loader").reset_index()

    assert set(comm_hlm["func_name"]) == {"all_reduce"}
    assert int(comm_hlm.iloc[0]["count"]) == 4

    assert set(device_hlm["func_name"]) == {"transfer"}
    assert int(device_hlm.iloc[0]["count"]) == 2

    assert set(compute_hlm["func_name"]) == {"backward"}
    assert int(compute_hlm.iloc[0]["count"]) == 1

    assert set(data_hlm["func_name"]) == {"fetch.block"}
    assert int(data_hlm.iloc[0]["count"]) == 3


# ---------------------------------------------------------------------------
# Test: Profile-only trace (all C events, no X events)
# ---------------------------------------------------------------------------

def test_analyze_trace_works_with_profile_only_input(dask_client, tmp_path):
    events = METADATA_EVENTS + [
        {"id": 3, "name": "open64", "cat": "POSIX", "pid": 1, "tid": 1,
         "ph": "C", "ts": 5000000,
         "args": {"hhash": "h1", "fhash": "f1", "dft_cnt": 5, "dur_sum": 1000}},
    ]
    path = tmp_path / "profile-only.pfw.gz"
    _write_trace(path, events)
    analyzer = make_analyzer(tmp_path=tmp_path, time_granularity=5)

    result = analyzer.analyze_trace(
        trace_path=str(path),
        view_types=["proc_name", "time_range"],
    )

    posix_hlm = result.get_hlm("posix").reset_index()
    assert set(posix_hlm["func_name"]) == {"open64"}
    assert int(posix_hlm.iloc[0]["count"]) == 5


# ---------------------------------------------------------------------------
# Test: read_trace sets profile_time_granularity on ReadTraceResult
# ---------------------------------------------------------------------------

def test_read_trace_result_carries_profile_time_granularity(dask_client, hybrid_trace_path, tmp_path):
    analyzer = make_analyzer(tmp_path=tmp_path, time_granularity=5)

    result = analyzer.read_trace(
        trace_path=str(hybrid_trace_path),
        extra_columns=None,
        extra_columns_fn=None,
    )

    assert result.profile_time_granularity == 5


# ---------------------------------------------------------------------------
# Test: Custom profile_time_granularity from config propagates correctly
# ---------------------------------------------------------------------------

def test_custom_profile_time_granularity_from_config(dask_client, tmp_path):
    """When profile_time_granularity=10, a 10s bucket at ts=10_000_000
    should produce time_end - time_start = 10_000_000 us."""
    events = METADATA_EVENTS + [
        {"id": 3, "name": "read", "cat": "POSIX", "pid": 1, "tid": 1,
         "ph": "C", "ts": 10000000,
         "args": {"hhash": "h1", "fhash": "f1", "dft_cnt": 2, "dur_sum": 400,
                  "ret_sum": 8192, "ret_min": 4096, "ret_max": 4096}},
    ]
    path = tmp_path / "custom-ptg.pfw.gz"
    _write_trace(path, events)
    analyzer = DFTracerAnalyzer(
        preset=OmegaConf.structured(AnalyzerPresetConfigPOSIX()),
        checkpoint=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        debug=False,
        profile_time_granularity=10,
        quantile_stats=False,
        time_approximate=True,
        time_granularity=5,
        time_resolution=10**6,
        time_sliced=False,
        verbose=False,
    )

    result = analyzer.read_trace(
        trace_path=str(path), extra_columns=None, extra_columns_fn=None,
    )

    assert result.profile_time_granularity == 10
    profiles = result.profiles
    assert len(profiles) == 1
    p = profiles.iloc[0]
    # 10s bucket → time_end - time_start = 10_000_000 us
    assert int(p["time_end"] - p["time_start"]) == 10_000_000


# ---------------------------------------------------------------------------
# Test: profile expansion matches the pandas path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("distribution", ["uniform", "weighted"])
@pytest.mark.parametrize("factor", [1, 2, 5, 9])
def test_expand_profile_buckets_matches_pandas(distribution, factor):
    """Counts, sizes, times and the sub-bucket grid equal the pandas path (rtol 1e-12, NaN as missing)."""
    import numpy as np
    import pandas as pd
    from dftracer.utils import DataFrame, Series

    from . import pandas_reference

    profiles = DataFrame.from_dict(
        {
            "func_name": ["read", "write", "open", "close"],
            "count": Series([7, 3, None, 12], dtype="int64"),
            "time": [0.5, 0.0, 1.25, 3.0],
            "size": Series([1001, None, 0, 4096], dtype="int64"),
            "time_min": [0.01, None, 0.2, 0.3],
            "time_max": [0.09, 0.4, None, 0.3],
            "time_range": Series([1, 2, 3, 4], dtype="int64"),
            "time_start": Series([5_000_000, 10_000_000, 15_000_000, 20_000_000], dtype="int64"),
            "time_end": Series([10_000_000, 15_000_000, 20_000_000, 25_000_000], dtype="int64"),
        }
    )
    args = (factor, 1e6, 5.0 / factor, distribution)
    got = DFTracerAnalyzer._expand_profile_buckets(profiles, *args).to_pandas()
    expected = pandas_reference.expand_profile_buckets(profiles.to_pandas(nullable=True), *args)

    assert list(got.columns) == list(expected.columns)
    assert len(got) == len(expected) == 4 * factor
    for col in expected.columns:
        g, e = got[col], expected[col]
        assert g.isna().tolist() == e.isna().tolist(), col
        if pd.api.types.is_numeric_dtype(e.dtype):
            assert pd.api.types.is_integer_dtype(g.dtype) == pd.api.types.is_integer_dtype(e.dtype), col
            np.testing.assert_allclose(
                g[g.notna()].to_numpy(dtype=float), e[e.notna()].to_numpy(dtype=float), rtol=1e-12, atol=0
            )
        else:
            assert g.tolist() == e.tolist(), col


def test_expand_profile_buckets_truncates_negative_sizes_as_pandas():
    import pandas as pd
    from dftracer.utils import DataFrame, Series

    from . import pandas_reference

    profiles = DataFrame.from_dict(
        {
            "count": Series([3, 1], dtype="int64"),
            "time": [1.0, 2.0],
            "size": Series([-3, -1], dtype="int64"),
            "time_min": [0.1, 0.2],
            "time_max": [0.1, 0.2],
            "time_range": Series([0, 1], dtype="int64"),
            "time_start": Series([0, 5_000_000], dtype="int64"),
            "time_end": Series([5_000_000, 10_000_000], dtype="int64"),
        }
    )
    args = (2, 1e6, 2.5, "uniform")
    got = DFTracerAnalyzer._expand_profile_buckets(profiles, *args).to_pandas()
    expected = pandas_reference.expand_profile_buckets(profiles.to_pandas(nullable=True), *args)
    assert got["size"].tolist() == expected["size"].astype("int64").tolist() == [-1, -1, 0, 0]
    assert pd.api.types.is_integer_dtype(got["size"].dtype)
