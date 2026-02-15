import json

import pytest

from dftracer.analyzer.streaming.mofka_io import open_consumer, open_producer


pytestmark = [pytest.mark.smoke, pytest.mark.full]


def test_mofka_producer_consumer_roundtrip(bedrock_mofka):
    group_file, topic_name = bedrock_mofka

    p_driver, producer = open_producer(group_file, topic_name)
    c_driver, consumer = open_consumer(group_file, topic_name)

    payload = json.dumps({"name": "epoch.start", "pid": 1})
    producer.push(payload)
    producer.flush()

    event = consumer.pull().wait(timeout_ms=5000)
    assert event is not None
    assert isinstance(event.metadata, dict)
    assert event.metadata.get("name") == "epoch.start"
    assert event.metadata.get("pid") == 1

    del producer
    del consumer
    del p_driver
    del c_driver
