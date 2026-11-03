"""Expose duration-based overlap separately from operation and byte totals."""

from dataclasses import replace

import pytest
from rich.text import Text

from dftracer.analyzer.constants import MiB
from dftracer.analyzer.output import ConsoleOutput, JSONOutput, OutputLayerMetrics, OutputSummary


pytestmark = [pytest.mark.smoke, pytest.mark.full]


def _metrics(**overrides):
    metrics = OutputLayerMetrics(
        time=10.0, count=1000, size=200 * MiB, ops=100.0,
        bandwidth=20 * MiB, num_files=1, num_processes=1,
        u_time=1.0, u_count=1000, u_size=200 * MiB,
    )
    return replace(metrics, **overrides)


def _summary(layer_metrics):
    return OutputSummary(
        job_time=20.0, layer_metrics=layer_metrics, layers=list(layer_metrics),
        time_granularity=5.0, time_resolution=1000,
        trace_event_count=1000, profile_event_count=0, total_event_count=1000,
        unique_file_count=1, unique_host_count=1, unique_process_count=1,
    )


def _columns(table):
    return {
        column.header: [Text.from_markup(cell).plain for cell in column._cells]
        for column in table.columns
    }


def test_time_overlap_is_separate_from_operation_and_size_totals():
    # This group has 90% estimated time overlap, but all operations and bytes
    # belong to a group with positive exposed time. Their totals stay plain.
    summary = _summary({'compute': _metrics(u_time=None), 'posix': _metrics()})
    table = ConsoleOutput()._create_layer_breakdown_table(summary, ('time_range',))
    columns = _columns(table)
    assert table.title == 'Layer Breakdown'
    assert columns['Est. compute overlap'] == ['N/A', '90%']
    assert columns['Time (s)'] == ['10.000', '10.000']
    assert columns['Ops'] == ['1,000', '1,000']
    assert columns['Size (MB)'] == ['200.000', '200.000']
    assert columns['Ops/sec'] == ['100.000', '100.000']
    assert columns['Bandwidth (MB/s)'] == ['20.000', '20.000']
    assert 'grouped durations' in table.caption
    # Console presentation does not remove existing machine-readable metrics.
    payload = JSONOutput()._create_summary_payload(summary)['layers']['posix']
    assert payload['u_time_s'] == 1.0
    assert payload['u_count'] == 1000
    assert payload['u_size_bytes'] == 200 * MiB


@pytest.mark.parametrize('time,u_time,expected', [
    (10.0, 0.0, '100%'),
    (10.0, 10.0, '0%'),
    (10.0, 4.5, '55%'),
    (0.0, 0.0, 'N/A'),
    (float('nan'), 0.0, 'N/A'),
    (10.0, float('nan'), 'N/A'),
])
def test_overlap_endpoints_and_unavailable_time(time, u_time, expected):
    summary = _summary({'posix': _metrics(time=time, u_time=u_time)})
    table = ConsoleOutput()._create_layer_breakdown_table(summary, ('time_range',))
    assert _columns(table)['Est. compute overlap'] == [expected]


@pytest.mark.parametrize('layers', [['posix'], ['app', 'posix']])
def test_no_overlap_column_without_available_metrics(layers):
    summary = _summary({layer: _metrics(u_time=None) for layer in layers})
    table = ConsoleOutput()._create_layer_breakdown_table(summary, ('proc_name',))
    assert 'Est. compute overlap' not in _columns(table)
    assert table.caption is None


def test_empty_layers_do_not_add_overlap_column():
    summary = _summary({'posix': _metrics(u_time=None), 'reader_posix': _metrics(count=0)})
    table = ConsoleOutput()._create_layer_breakdown_table(summary, ('time_range',))
    assert _columns(table)['Layer'] == ['POSIX - All']
    assert 'Est. compute overlap' not in _columns(table)
