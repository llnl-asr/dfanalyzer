import json
import multiprocessing
import time

import pandas as pd
import pytest

from dftracer.analyzer import init_with_hydra
from dftracer.analyzer.streaming.mofka_io import open_producer


pytestmark = [pytest.mark.smoke, pytest.mark.full]


def test_e2e_mofka(
    tmp_path,
    bedrock_mofka,
    dftracer_ai_logging_posix_events,
):
    group_file, topic_name = bedrock_mofka

    dfa = init_with_hydra(
        hydra_overrides=[
            "analyzer=dftracer",
            "analyzer/preset=dlio",
            "analyzer.preset.time_boundary_layer=epoch",
            "analyzer.checkpoint=False",
            f"analyzer.checkpoint_dir={tmp_path}/checkpoints",
            "cluster.processes=False",
            f"hydra.run.dir={tmp_path}",
            f"hydra.runtime.output_dir={tmp_path}",
            "input=mofka",
            f"input.group_file={group_file}",
            f"input.topic_name={topic_name}",
            "output=console",
            "view_types=[epoch]",
            "debug=True",
        ]
    )

    events_raw = dftracer_ai_logging_posix_events[0]
    events_raw.append(
        '{"id":478,"name":"end","cat":"dftracer","pid":4103080,"tid":4103080,"ts":1753300246416070,"dur":0,"ph":"X","args":{"hhash":"03089b0f8c47cc3d","p_idx":476,"num_events":477,"level":6}}'
    )
    events = [json.loads(line) for line in events_raw]

    collected = []

    def run_producer():
        p_driver, producer = open_producer(group_file, topic_name)
        for event in events:
            producer.push(event)
        producer.flush()
        del producer
        del p_driver

    proc = multiprocessing.Process(target=run_producer, daemon=True)
    proc.start()

    extra_columns = {"epoch": "Int8"}
    extra_columns_fn = lambda json_dict: {"epoch": json_dict.get("epoch", None)}

    dfa.analyzer.analyze_mofka(
        group_file=group_file,
        topic_name=topic_name,
        view_types=["epoch"],
        extra_columns=extra_columns,
        extra_columns_fn=extra_columns_fn,
        output_handler=collected.append,
    )

    assert len(collected) > 0, "No analysis results emitted from analyze_mofka"
    result = collected[0]

    assert hasattr(result, "flat_views"), "Result missing flat_views"
    assert isinstance(result.flat_views, dict), "flat_views not a dict"
    assert len(result.flat_views) > 0, "flat_views is empty"

    any_df = False
    for k, v in result.flat_views.items():
        assert isinstance(k, tuple), "flat_views keys should be tuples (view_key)"
        assert isinstance(v, pd.DataFrame), f"flat_views[{k}] is not a pandas DataFrame"
        any_df = any_df or (getattr(v, "shape", (0, 0))[0] > 0 and getattr(v, "shape", (0, 0))[1] > 0)

    assert any_df, "No non-empty pandas DataFrame found in flat_views"

    try:
        dfa.shutdown()
    except Exception:
        pass

    proc.join(timeout=5)
