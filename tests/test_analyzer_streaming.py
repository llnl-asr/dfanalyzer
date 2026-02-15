import pathlib
import pytest
import threading
import time
import zmq
from dftracer.analyzer import init_with_hydra
from dftracer.analyzer.utils.streaming import is_streaming_available
from typing import List

# Ensure this module runs in both smoke and full CI modes
pytestmark = [pytest.mark.smoke, pytest.mark.full]


@pytest.mark.skipif(not is_streaming_available, reason="streamz not installed")
def test_analyzer_dftracer_read_zmq(
    tmp_path: pathlib.Path,
    dftracer_ai_logging_posix_events: List[List[str]],
) -> None:
    """Test DFTracerAnalyzer.read_zmq by publishing toy JSON lines over ZMQ.

    This verifies the stream parses incoming JSON lines into dictionaries
    using the analyzer's JSON loader.
    """
    zmq_port = 5570

    dfa = init_with_hydra(
        hydra_overrides=[
            "analyzer=dftracer",
            "analyzer/preset=dlio-ailogging",
            f"analyzer.checkpoint={False}",
            f"analyzer.checkpoint_dir={tmp_path}/checkpoints",
            f"cluster.processes={False}",
            f"hydra.run.dir={tmp_path}",
            f"hydra.runtime.output_dir={tmp_path}",
            "input=zmq",
            f"input.address=tcp://*:{zmq_port}",
            "view_types=[epoch]",
        ]
    )

    # use realistic epoch lines extracted from the test data
    events = dftracer_ai_logging_posix_events[0]

    def publisher():
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUSH)
        sock.connect(f"tcp://localhost:{zmq_port}")
        time.sleep(0.1)
        for event in events:
            sock.send_string(event)
            time.sleep(0.01)
        sock.close()
        ctx.term()

    pub_thread = threading.Thread(target=publisher, daemon=True)

    try:
        collected = []

        extra_columns = {"epoch": "Int8"}

        def extra_columns_fn(json_dict):
            return {"epoch": json_dict.get("epoch", None)}

        read_stream = dfa.analyzer.read_zmq(
            trace_address=f"tcp://*:{zmq_port}",
            extra_columns=extra_columns,
            extra_columns_fn=extra_columns_fn,
        )
        read_stream.sink(collected.append)

        # start the stream then publisher (give stream time to bind)
        read_stream.start()
        time.sleep(0.5)
        pub_thread.start()

        # wait for data
        timeout = 10.0
        start = time.time()
        while time.time() - start < timeout and len(collected) == 0:
            time.sleep(0.05)

        # Stop stream
        try:
            read_stream.stop()
        except Exception:
            pass

        assert len(collected) > 0, "No parsed messages collected from read_zmq"
        # messages should be dicts with at least 'name' and 'pid' for our toy msgs
        assert all(isinstance(x, dict) for x in collected), "Parsed items are not dicts"
        names = [x.get("name") for x in collected]
        assert "epoch.start" in names or "epoch.block" in names or "read" in names
    finally:
        try:
            dfa.shutdown()
        except Exception:
            pass
