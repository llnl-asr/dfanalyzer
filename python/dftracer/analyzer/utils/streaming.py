try:
    import zmq  # noqa: F401
    import streamz_zmq  # noqa: F401
    import structlog
    from streamz import Stream

    is_streaming_available = True

    logger = structlog.get_logger()

    @Stream.register_api()
    class epoch_window_via_dict(Stream):
        """
        Groups incoming elements into buffers that are emitted when an
        element matching the configured end name is received.

        Parameters (constructor kwargs / call args on the Stream API):
        - epoch_start_name: name of the start event (default: "epoch.start")
        - epoch_end_name: name of the end event (default: "epoch.block")
        - process_key: key in the incoming dict to use as the pid (default: "pid").

        The emitted value is the buffered list of elements for the epoch.
        """

        _graphviz_shape = 'diamond'

        def __init__(
            self,
            upstream,
            epoch_start_name: str = "epoch.start",
            epoch_end_name: str = "epoch.block",
            process_key: str = "pid",
            **kwargs,
        ):
            self._buffer = []
            # per-pid epoch counters
            self.epoch_numbers = {}  # pid -> int
            # track seen/ended pids per epoch number (do not clear)
            self.seen_pids_by_epoch = {}  # epoch -> set(pid)
            self.ended_pids_by_epoch = {}  # epoch -> set(pid)
            # buffered events per epoch
            self.buffer_by_epoch = {}  # epoch -> list
            self.epoch_start_name = epoch_start_name
            self.epoch_end_name = epoch_end_name
            self.process_key = process_key
            Stream.__init__(self, upstream, **kwargs)

        def update(self, x, who=None, metadata=None):
            # Track epoch boundaries and per-pid completion
            if x.get('name') == self.epoch_start_name:
                pid_start = x.get(self.process_key)
                if pid_start is None:
                    logger.error('epoch.start.no_pid')
                    raise ValueError('Missing pid in epoch.start event')
                # increment per-pid epoch counter
                self.epoch_numbers[pid_start] = self.epoch_numbers.get(pid_start, 0) + 1
                epoch_for_pid = self.epoch_numbers[pid_start]
                # record this pid as seen for that epoch
                self.seen_pids_by_epoch.setdefault(epoch_for_pid, set()).add(pid_start)
                logger.info('epoch.start', pid=pid_start, epoch=epoch_for_pid)

            # Assign epoch per-pid when available, otherwise use global epoch
            pid = x.get(self.process_key)
            epoch_assigned = 0
            if pid is not None:
                epoch_for_pid = self.epoch_numbers.get(pid, 0)
                if epoch_for_pid > 0:
                    x['epoch'] = epoch_for_pid
                    epoch_assigned = epoch_for_pid
                    # record pid seen for that epoch (idempotent)
                    self.seen_pids_by_epoch.setdefault(epoch_for_pid, set()).add(pid)
            else:
                # events must carry pid
                pid = x.get(self.process_key)
                if pid is None:
                    logger.error('event.no_pid', name=x.get('name'))
                    raise ValueError(f"Missing pid on event: {x.get('name')}")

            # Buffer only once some epoch has started (per-pid or global)
            if epoch_assigned > 0:
                self.buffer_by_epoch.setdefault(epoch_assigned, []).append(x)

            if x.get('name') == self.epoch_end_name:
                # pid-aware emission: find the earliest seen epoch for this pid
                pid = x.get(self.process_key)
                if pid is not None:
                    candidate_epoch = None
                    for e in sorted(self.seen_pids_by_epoch.keys()):
                        if pid in self.seen_pids_by_epoch.get(e, set()):
                            ended_set = self.ended_pids_by_epoch.get(e, set())
                            if pid not in ended_set:
                                candidate_epoch = e
                                break
                    if candidate_epoch is None:
                        # nothing to close
                        logger.error('epoch.end.no_seen_epoch', pid=pid)
                        raise ValueError(f'No seen epoch for pid {pid}')
                    self.ended_pids_by_epoch.setdefault(candidate_epoch, set()).add(pid)
                    logger.debug('epoch.end.from_pid', pid=pid, epoch=candidate_epoch)
                    seen = self.seen_pids_by_epoch.get(candidate_epoch, set())
                    ended = self.ended_pids_by_epoch.get(candidate_epoch, set())
                    if seen and ended >= seen:
                        data_to_emit = self.buffer_by_epoch.get(candidate_epoch, [])
                        ret = self._emit(data_to_emit)
                        self.buffer_by_epoch.pop(candidate_epoch, None)
                        return ret
                    return []
                else:
                    # No pid provided: cannot attribute end
                    logger.error('epoch.end.no_pid', name=x.get('name'))
                    raise ValueError('Missing pid in epoch.end event')

            return []


except ImportError:
    Stream = None

    is_streaming_available = False
