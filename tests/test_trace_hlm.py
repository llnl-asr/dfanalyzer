import pytest
from dask.distributed import Client, LocalCluster
from dftracer.analyzer.analyzer import Analyzer
from dftracer.analyzer.frames.main_view import reconcile
from dftracer.analyzer.trace_hlm import HLMConfig, _to_hlm
from dftracer.utils import DataFrame

from .test_hybrid_profiles import TRACE_CONTENT, _write_trace_lines, make_analyzer
from .test_utils_frame_parity import TRACE_ROWS, _pandas_reconcile, _rows, as_frame

pytestmark = pytest.mark.smoke

VIEW_TYPES = ["proc_name", "time_range"]


def _frame(buckets, min_ts=None):
    n = len(buckets)
    return DataFrame.from_dict(
        {
            "pid": [1] * n,
            "tid": [1] * n,
            "host_name": ["h"] * n,
            "hhash": ["h"] * n,
            "cat": ["POSIX"] * n,
            "io_cat": [1] * n,
            "name": ["read"] * n,
            "time_bucket": buckets,
            "min_ts": min_ts or buckets,
            "count": [1] * n,
            "sum_dur": [1] * n,
            "sum_size": [1] * n,
            "sumsq_dur": [1] * n,
            "sumsq_size": [1] * n,
        }
    )


def test_empty_hlm_index_dtypes_match_non_empty():
    cfg = HLMConfig()
    empty = _to_hlm(None, VIEW_TYPES, cfg)
    full = _to_hlm(_frame([0]), VIEW_TYPES, cfg)
    assert empty.num_rows == 0
    assert empty.index_names() == full.index_names()
    assert [empty.get_level_values(n).dtype for n in empty.index_names()] == [
        full.get_level_values(n).dtype for n in full.index_names()
    ]
    assert empty.columns == full.columns
    assert empty.dtypes == full.dtypes
    # The pandas stages read the HLM through the same conversion.
    assert empty.to_pandas(nullable=True).dtypes.equals(full.to_pandas(nullable=True).dtypes)


def test_reconcile_with_empty_profile_hlm_keeps_trace_rows():
    trace = _rows(TRACE_ROWS)
    profile = _to_hlm(None, VIEW_TYPES, HLMConfig())
    got, overlap = reconcile(as_frame(trace), profile, VIEW_TYPES)
    expected, expected_overlap = _pandas_reconcile(trace, profile.to_pandas(nullable=True), VIEW_TYPES)
    assert overlap == expected_overlap == 0
    assert len(got) == len(expected) == len(TRACE_ROWS)
    assert sorted(got["count"].to_list()) == sorted(r[4] for r in TRACE_ROWS)


def test_time_range_follows_the_grouped_bucket_not_the_event_time():
    cfg = HLMConfig(time_granularity=5.0, time_resolution=1e6)
    df = _frame([0, 5_000_000], min_ts=[4_900_000, 5_100_000])
    hlm = _to_hlm(df, VIEW_TYPES, cfg)
    keys = hlm.reset_index()[hlm.index_names()]
    assert keys.drop_duplicates().num_rows == keys.num_rows
    assert hlm.get_level_values("time_range").to_list() == [0, 1]


def test_time_origin_is_shared_between_frames():
    cfg = HLMConfig(time_granularity=5.0, time_resolution=1e6, time_origin=5_000_000)
    hlm = _to_hlm(_frame([10_000_000]), VIEW_TYPES, cfg)
    assert hlm.get_level_values("time_range").to_list() == [1]


@pytest.fixture
def dask_client():
    cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=False, scheduler_port=0, silence_logs="error")
    client = Client(cluster)
    try:
        yield client
    finally:
        client.close()
        cluster.close()


def test_hybrid_profile_hlm_time_range_matches_read_result(dask_client, tmp_path, monkeypatch):
    path = tmp_path / "hybrid.pfw.gz"
    _write_trace_lines(path, TRACE_CONTENT)
    analyzer = make_analyzer(tmp_path=tmp_path, time_granularity=5)

    captured = []
    original = Analyzer._reconcile_hlm

    def spy(self, trace_hlm, profile_hlm, view_types, layer=None):
        captured.append((trace_hlm.copy(), profile_hlm.copy()))
        return original(self, trace_hlm, profile_hlm, view_types, layer)

    monkeypatch.setattr(Analyzer, "_reconcile_hlm", spy)
    result = analyzer.analyze_trace(trace_path=str(path), view_types=VIEW_TYPES)

    trace_hlm, profile_hlm = captured[0]
    assert trace_hlm.get_level_values("time_range").to_list() == [0]
    assert profile_hlm.get_level_values("time_range").to_list() == result.profiles["time_range"].to_list() == [1]
