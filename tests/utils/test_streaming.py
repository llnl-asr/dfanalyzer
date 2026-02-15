import pytest

from dftracer.analyzer.utils.streaming import is_streaming_available


pytestmark = [pytest.mark.smoke, pytest.mark.full]


@pytest.mark.skipif(not is_streaming_available, reason="streamz not installed")
def test_epoch_window_buffers_until_epoch_block():
    """Verify incoming dicts are buffered after epoch.start and emitted on epoch.block."""
    from streamz import Stream

    source = Stream()
    window = source.epoch_window_via_dict()
    collected = []
    window.sink(collected.append)

    # Events before epoch.start must not be buffered
    source.emit({"name": "data", "value": 0, "pid": 1})
    assert collected == []

    # Start epoch and emit some events for pid=1
    source.emit({"name": "epoch.start", "pid": 1})
    source.emit({"name": "data", "value": 1, "pid": 1})
    source.emit({"name": "data", "value": 2, "pid": 1})

    # epoch.block should flush the buffer for pid=1
    source.emit({"name": "epoch.block", "pid": 1})

    assert len(collected) == 1
    buffer = collected[0]
    assert isinstance(buffer, list)
    # epoch.start + 2 data + epoch.block => 4 items
    assert len(buffer) == 4
    # All items in the emitted buffer should have epoch set to 1
    assert all(item.get("epoch") == 1 for item in buffer)


@pytest.mark.skipif(not is_streaming_available, reason="streamz not installed")
def test_epoch_window_multiple_epochs():
    """Verify multiple epochs produce separate buffers with incremented epoch numbers."""
    from streamz import Stream

    source = Stream()
    window = source.epoch_window_via_dict()
    collected = []
    window.sink(collected.append)

    # First epoch for pid=1
    source.emit({"name": "epoch.start", "pid": 1})
    source.emit({"name": "data", "value": 10, "pid": 1})
    source.emit({"name": "epoch.block", "pid": 1})

    # Second epoch for pid=1
    source.emit({"name": "epoch.start", "pid": 1})
    source.emit({"name": "data", "value": 20, "pid": 1})
    source.emit({"name": "data", "value": 21, "pid": 1})
    source.emit({"name": "epoch.block", "pid": 1})

    assert len(collected) == 2
    first_buf, second_buf = collected
    assert all(item.get("epoch") == 1 for item in first_buf)
    assert all(item.get("epoch") == 2 for item in second_buf)


@pytest.mark.skipif(not is_streaming_available, reason="streamz not installed")
def test_epoch_window_pid_waits_for_all_pids():
    """Verify that the pid-aware window waits until all seen pids send epoch.block."""
    from streamz import Stream

    source = Stream()
    window = source.epoch_window_via_dict()
    collected = []
    window.sink(collected.append)

    # Start epoch and emit events from two pids
    # For per-pid epoch numbering, signal start per pid
    source.emit({"name": "epoch.start", "pid": 10})
    source.emit({"name": "epoch.start", "pid": 20})
    source.emit({"name": "data", "value": 1, "pid": 10})
    source.emit({"name": "data", "value": 2, "pid": 20})

    # epoch.block from pid 10 only -> should NOT emit yet
    source.emit({"name": "epoch.block", "pid": 10})
    assert collected == []

    # epoch.block from pid 20 -> now emit
    source.emit({"name": "epoch.block", "pid": 20})
    assert len(collected) == 1
    buf = collected[0]
    assert all(item.get("epoch") == 1 for item in buf)


@pytest.mark.skipif(not is_streaming_available, reason="streamz not installed")
def test_epoch_window_pid_missing_pid_raises():
    """If epoch.block has no pid, it should raise an error (pid required)."""
    from streamz import Stream

    source = Stream()
    window = source.epoch_window_via_dict()
    collected = []
    window.sink(collected.append)

    # start and data must include pid
    source.emit({"name": "epoch.start", "pid": 1})
    source.emit({"name": "data", "value": 1, "pid": 1})

    # epoch.block without pid should raise
    with pytest.raises(ValueError):
        source.emit({"name": "epoch.block"})


@pytest.mark.skipif(not is_streaming_available, reason="streamz not installed")
def test_repeat_epoch_start_increments_epoch():
    """Starting twice for same pid increments its epoch counter and buffers split by epoch."""
    from streamz import Stream

    source = Stream()
    window = source.epoch_window_via_dict()
    collected = []
    window.sink(collected.append)

    # pid 1 starts epoch1
    source.emit({"name": "epoch.start", "pid": 1})
    source.emit({"name": "data", "pid": 1, "value": "x1"})

    # pid 1 starts epoch2 before ending epoch1
    source.emit({"name": "epoch.start", "pid": 1})
    source.emit({"name": "data", "pid": 1, "value": "x2"})

    # ending should close epoch1 first (earliest seen)
    source.emit({"name": "epoch.block", "pid": 1})
    assert len(collected) == 1
    emitted = collected[0]
    vals = [item.get("value") for item in emitted]
    assert "x1" in vals
    assert "x2" not in vals


@pytest.mark.skipif(not is_streaming_available, reason="streamz not installed")
def test_pid_end_without_prior_seen_no_emit():
    """If a pid sends epoch.block without prior start/seen, nothing should emit."""
    from streamz import Stream

    source = Stream()
    window = source.epoch_window_via_dict()
    collected = []
    window.sink(collected.append)

    # pid 99 never started or sent any data -> still raises because no seen epoch
    with pytest.raises(ValueError):
        source.emit({"name": "epoch.block", "pid": 99})


@pytest.mark.skipif(not is_streaming_available, reason="streamz not installed")
def test_data_without_pid_raises():
    """Any non-start/end event without pid should raise, pid is mandatory."""
    from streamz import Stream

    source = Stream()
    window = source.epoch_window_via_dict()

    with pytest.raises(ValueError):
        source.emit({"name": "data", "value": 123})


@pytest.mark.skipif(not is_streaming_available, reason="streamz not installed")
def test_overlapping_epochs_fast_process_waits_for_slowest_block():
    """Fast process can start next epoch while slow finishes previous; emission uses slowest end."""
    from streamz import Stream

    source = Stream()
    window = source.epoch_window_via_dict()
    collected = []
    window.sink(collected.append)

    # pid1 and pid2 start epoch1
    source.emit({"name": "epoch.start", "pid": 1})
    source.emit({"name": "data", "pid": 1, "value": "a1"})
    source.emit({"name": "epoch.start", "pid": 2})
    source.emit({"name": "data", "pid": 2, "value": "b1"})

    # pid2 ends quickly
    source.emit({"name": "epoch.block", "pid": 2})
    assert collected == []

    # pid1 starts next epoch before ending previous
    source.emit({"name": "epoch.start", "pid": 1})
    source.emit({"name": "data", "pid": 1, "value": "a2"})

    # now pid1 ends epoch1 -> should trigger emission for epoch1 only
    source.emit({"name": "epoch.block", "pid": 1})

    assert len(collected) == 1
    emitted = collected[0]
    # emitted buffer should contain the epoch1 items (a1 and b1) but not a2
    vals = [item.get("value") for item in emitted]
    assert "a1" in vals and "b1" in vals
    assert "a2" not in vals
