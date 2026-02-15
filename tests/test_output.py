import io
import multiprocessing
import pandas as pd
import pytest

from dftracer.analyzer.output import MofkaOutput
from dftracer.analyzer.streaming.mofka_io import open_consumer
from dftracer.analyzer.types import AnalyzerResultType, RawStats


pytestmark = [pytest.mark.smoke, pytest.mark.full]


def test_mofka_output_roundtrip(bedrock_mofka):
    group_file, topic_name = bedrock_mofka

    flat_view = pd.DataFrame(
        {
            "epoch_time_sum": [10.0],
            "fetch_data_time_frac_epoch": [0.75],
        },
        index=pd.Index([7], name="epoch"),
    )
    result = AnalyzerResultType(
        _hlms={},
        _main_views={},
        _metric_boundaries={},
        checkpoint_dir=".",
        flat_views={("epoch",): flat_view},
        layers=["epoch"],
        raw_stats=RawStats(
            job_time=10,
            time_granularity=1,
            time_resolution=1,
            total_event_count=1,
            unique_file_count=1,
            unique_host_count=1,
            unique_process_count=1,
        ),
        view_types=["epoch"],
        views={},
    )

    def run_producer():
        output = MofkaOutput(group_file=group_file, topic_name=topic_name)
        output.handle_result(result)
        del output

    proc = multiprocessing.Process(target=run_producer, daemon=True)
    proc.start()

    driver, consumer = open_consumer(group_file, topic_name)
    try:
        future = consumer.pull()
        event = future.wait(timeout_ms=5000)
        assert event is not None, "No event received from MofkaOutput"

        metadata = event.metadata
        assert isinstance(metadata, dict)
        assert metadata.get("view_type") == "epoch"
        assert metadata.get("view_len") == len(flat_view)

        payload = event.data
        assert payload is not None, "No data payload received"
        if isinstance(payload, list):
            assert payload, "Empty data payload list"
            assert all(isinstance(item, (bytes, bytearray)) for item in payload), "Unexpected payload chunk type"
            payload = b"".join(payload)
        restored = pd.read_parquet(io.BytesIO(payload))
        assert len(restored) == len(flat_view)
        assert list(restored.columns) == list(flat_view.columns)
        assert list(restored.index) == list(flat_view.index)
        assert restored.index.name == flat_view.index.name
        event.acknowledge()
    finally:
        del consumer
        del driver
        proc.join(timeout=5)
