"""Console transfer sizes must describe data calls, independent of metadata volume."""

import pandas as pd
import pytest

from dftracer.analyzer.constants import HUMANIZED_LAYERS, MiB
from dftracer.analyzer.output import ConsoleOutput
from dftracer.analyzer.types import AnalysisResult, RawStats


pytestmark = [pytest.mark.smoke, pytest.mark.full]


def _summarize(flat_view, layer, view_type):
    result = AnalysisResult(
        additional_metrics={}, checkpoint_dir='', flat_views={(view_type,): flat_view},
        layers=[layer], raw_stats=RawStats(20, 5, 1000, 0, 0, 0, 1, 1, 2),
        view_types=[view_type], views={}, _hlms={}, _main_views={}, _metric_boundaries={},
    )
    output = ConsoleOutput()
    summary = output._create_summary(result, (view_type,), raw_stats=result.raw_stats)
    table = output._create_summary_table(summary, (view_type,))
    rows = dict(zip(table.columns[0]._cells, table.columns[2]._cells))
    return summary.layer_metrics[layer], rows


@pytest.mark.parametrize('layer', ['posix', 'reader_posix', 'checkpoint_posix'])
@pytest.mark.parametrize('view_type', ['proc_name', 'time_range'])
@pytest.mark.parametrize('metadata_count', [0, 200, 10000])
def test_transfer_size_excludes_metadata(layer, view_type, metadata_count):
    time_col = 'time_sum' if view_type == 'proc_name' else 'time_proc_max'
    # Read/write counts include contributions from multiple rows (and can
    # represent aggregate profile counts rather than one event per operation).
    flat_view = pd.DataFrame({
        f'{layer}_{time_col}': [2.0, 2.0],
        f'{layer}_count_sum': [400 + metadata_count, 400],
        f'{layer}_data_count_sum': pd.array([400, 400], dtype='Int64'),
        f'{layer}_size_sum': [100 * MiB, 100 * MiB],
    })
    metrics, rows = _summarize(flat_view, layer, view_type)
    assert metrics.count == 800 + metadata_count
    assert metrics.data_count == 800
    assert next(v for k, v in rows.items() if k.endswith(' Avg Transfer Size')) == '0.250'
    assert rows[f'{HUMANIZED_LAYERS[layer]} Count'] == f'{800 + metadata_count:,}'


@pytest.mark.parametrize('data_counts', [None, [0, 0], [pd.NA, pd.NA]])
def test_transfer_size_without_data_count_is_unavailable(data_counts):
    flat_view = pd.DataFrame({
        'posix_time_proc_max': [2.0, 2.0],
        'posix_count_sum': [500, 500],
        'posix_size_sum': [100 * MiB, 100 * MiB],
    })
    if data_counts is not None:
        flat_view['posix_data_count_sum'] = pd.array(data_counts, dtype='Int64')
    _, rows = _summarize(flat_view, 'posix', 'time_range')
    assert rows['POSIX - All Avg Transfer Size'] == 'N/A'
    assert rows['POSIX - All Count'] == '1,000'
    assert rows['POSIX - All Size'] == '200.000'
