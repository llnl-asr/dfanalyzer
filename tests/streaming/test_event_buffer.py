import pytest

from dftracer.analyzer.streaming.epoch_buffer import EpochBuffer


pytestmark = [pytest.mark.smoke, pytest.mark.full]


def _push_all(buffer: EpochBuffer, events):
    emitted = []
    for event in events:
        out = buffer.push(event)
        if out:
            emitted.append(out)
    return emitted


def test_epoch_buffer_buffers_until_epoch_block():
    buffer = EpochBuffer()
    emitted = _push_all(
        buffer,
        [
            {"name": "data", "value": 0, "pid": 1},
            {"name": "epoch.start", "pid": 1},
            {"name": "data", "value": 1, "pid": 1},
            {"name": "epoch.block", "pid": 1},
        ],
    )

    assert len(emitted) == 1
    buf = emitted[0]
    assert isinstance(buf, list)
    assert len(buf) == 3
    assert all(item.get("epoch") == 1 for item in buf)


def test_epoch_buffer_multiple_epochs():
    buffer = EpochBuffer()
    emitted = _push_all(
        buffer,
        [
            {"name": "epoch.start", "pid": 1},
            {"name": "data", "value": 10, "pid": 1},
            {"name": "epoch.block", "pid": 1},
            {"name": "epoch.start", "pid": 1},
            {"name": "data", "value": 20, "pid": 1},
            {"name": "data", "value": 21, "pid": 1},
            {"name": "epoch.block", "pid": 1},
        ],
    )

    assert len(emitted) == 2
    assert all(item.get("epoch") == 1 for item in emitted[0])
    assert all(item.get("epoch") == 2 for item in emitted[1])


def test_epoch_buffer_pid_waits_for_all_pids():
    buffer = EpochBuffer()

    assert buffer.push({"name": "epoch.start", "pid": 10}) is None
    assert buffer.push({"name": "epoch.start", "pid": 20}) is None
    assert buffer.push({"name": "data", "value": 1, "pid": 10}) is None
    assert buffer.push({"name": "data", "value": 2, "pid": 20}) is None

    assert buffer.push({"name": "epoch.block", "pid": 10}) is None
    emitted = buffer.push({"name": "epoch.block", "pid": 20})
    assert emitted is not None
    assert all(item.get("epoch") == 1 for item in emitted)


def test_epoch_buffer_missing_pid_raises():
    buffer = EpochBuffer()
    buffer.push({"name": "epoch.start", "pid": 1})
    buffer.push({"name": "data", "value": 1, "pid": 1})

    with pytest.raises(ValueError):
        buffer.push({"name": "epoch.block"})

def test_pid_end_without_prior_seen_raises():
    buffer = EpochBuffer()
    with pytest.raises(ValueError):
        buffer.push({"name": "epoch.block", "pid": 99})


def test_data_without_pid_raises():
    buffer = EpochBuffer()
    with pytest.raises(ValueError):
        buffer.push({"name": "data", "value": 123})


def test_overlapping_epochs_fast_process_starts_next_before_slow_finishes():
    buffer = EpochBuffer()
    emitted = _push_all(
        buffer,
        [
            {"name": "epoch.start", "pid": 1},
            {"name": "data", "pid": 1, "value": "a1"},
            {"name": "epoch.start", "pid": 2},
            {"name": "data", "pid": 2, "value": "b1"},
            {"name": "epoch.block", "pid": 2},
            {"name": "epoch.start", "pid": 1},
            {"name": "data", "pid": 1, "value": "a2"},
            {"name": "epoch.block", "pid": 1},
        ],
    )

    assert len(emitted) == 1
    vals = [item.get("value") for item in emitted[0]]
    assert "a1" in vals and "b1" in vals
    assert "a2" not in vals


def test_epoch_buffer_custom_names():
    buffer = EpochBuffer(epoch_start_name="start", epoch_end_name="done", process_key="rank")
    emitted = _push_all(
        buffer,
        [
            {"name": "start", "rank": 7},
            {"name": "data", "rank": 7, "value": 1},
            {"name": "done", "rank": 7},
        ],
    )
    assert len(emitted) == 1
    assert all(item.get("epoch") == 1 for item in emitted[0])
