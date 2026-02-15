import pandas as pd
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
def test_e2e_zmq(
    tmp_path: pathlib.Path,
    dftracer_ai_logging_posix_events: List[List[str]],
) -> None:
    """Synchronous test: parse toy JSON events and run the analyzer pipeline directly.

    This avoids network flakiness while exercising the same parsing and
    aggregation code paths used by `analyze_zmq`.
    """

    zmq_port = 5561

    # Initialize analyzer instance via hydra same as real usage
    dfa = init_with_hydra(
        hydra_overrides=[
            "analyzer=dftracer",
            "analyzer/preset=posix",
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

    extra_columns = {"epoch": "Int8"}
    extra_columns_fn = lambda json_dict: {"epoch": json_dict.get("epoch", None)}

    # Use realistic epoch lines from the test fixture (first epoch)
    events = dftracer_ai_logging_posix_events[0]

    # Publisher: connect and push the toy messages
    def publisher():
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUSH)
        sock.connect(f"tcp://localhost:{zmq_port}")
        time.sleep(0.2)
        for event in events:
            sock.send_string(event)
            # print("send line", event)
            time.sleep(0.01)
        sock.close()
        ctx.term()

    pub_thread = threading.Thread(target=publisher, daemon=True)

    # Collect emitted AnalyzerResultType objects
    collected = []

    # analyze_zmq returns a Stream object; sink receives AnalyzerResultType per epoch
    analysis_stream = dfa.analyze_zmq(extra_columns=extra_columns, extra_columns_fn=extra_columns_fn)
    assert analysis_stream is not None
    assert hasattr(analysis_stream, "sink")

    analysis_stream.sink(collected.append)

    # Start the stream
    analysis_stream.start()

    # Start the publisher once the stream is running and bound
    pub_thread.start()

    # Wait for the publisher to finish
    pub_thread.join()

    # Wait for analysis result (stream processing may be async)
    timeout = 10.0
    start = time.time()
    while time.time() - start < timeout and len(collected) == 0:
        time.sleep(0.1)

    # Stop stream and cleanup
    try:
        analysis_stream.stop()
    except Exception:
        pass

    # Basic assertions: we got at least one analysis result
    assert len(collected) > 0, "No analysis results emitted from analyze_zmq"
    result = collected[0]

    # Validate the AnalyzerResultType shape
    assert hasattr(result, "flat_views"), "Result missing flat_views"
    assert isinstance(result.flat_views, dict), "flat_views not a dict"
    assert len(result.flat_views) > 0, "flat_views is empty"

    # At least one flat view must be a non-empty pandas DataFrame
    any_df = False
    for k, v in result.flat_views.items():
        assert isinstance(k, tuple), "flat_views keys should be tuples (view_key)"
        assert isinstance(v, pd.DataFrame), f"flat_views[{k}] is not a pandas DataFrame"
        any_df = any_df or (getattr(v, "shape", (0, 0))[0] > 0 and getattr(v, "shape", (0, 0))[1] > 0)

    assert any_df, "No non-empty pandas DataFrame found in flat_views"

    # Additional sanity checks: traces and layers present
    assert hasattr(result, "layers")
    assert isinstance(result.layers, list)

    try:
        dfa.shutdown()
    except Exception:
        pass
