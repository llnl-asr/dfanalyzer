import structlog
from typing import Dict, List, Optional


logger = structlog.get_logger()


class EpochBuffer:
    """
    Buffer events by epoch using epoch.start / epoch.block markers.

    This mirrors the streamz epoch_window_via_dict logic but is synchronous
    and returns completed epoch buffers directly.
    """

    def __init__(
        self,
        epoch_start_name: str = "epoch.start",
        epoch_end_name: str = "epoch.block",
        process_key: str = "pid",
    ):
        self.epoch_start_name = epoch_start_name
        self.epoch_end_name = epoch_end_name
        self.process_key = process_key
        self.epoch_numbers: Dict[int, int] = {}
        self.seen_pids_by_epoch: Dict[int, set] = {}
        self.ended_pids_by_epoch: Dict[int, set] = {}
        self.buffer_by_epoch: Dict[int, List[dict]] = {}

    def push(self, event: dict) -> Optional[List[dict]]:
        if event.get("name") == self.epoch_start_name:
            pid_start = event.get(self.process_key)
            if pid_start is None:
                logger.error("epoch.start.no_pid")
                raise ValueError("Missing pid in epoch.start event")
            self.epoch_numbers[pid_start] = self.epoch_numbers.get(pid_start, 0) + 1
            epoch_for_pid = self.epoch_numbers[pid_start]
            self.seen_pids_by_epoch.setdefault(epoch_for_pid, set()).add(pid_start)
            logger.info("epoch.start", pid=pid_start, epoch=epoch_for_pid)

        pid = event.get(self.process_key)
        if pid is None:
            logger.error("event.no_pid", name=event.get("name"))
            raise ValueError(f"Missing pid on event: {event.get('name')}")

        epoch_for_pid = self.epoch_numbers.get(pid, 0)
        if epoch_for_pid > 0:
            event["epoch"] = epoch_for_pid
            self.seen_pids_by_epoch.setdefault(epoch_for_pid, set()).add(pid)
            self.buffer_by_epoch.setdefault(epoch_for_pid, []).append(event)

        if event.get("name") == self.epoch_end_name:
            candidate_epoch = None
            for epoch in sorted(self.seen_pids_by_epoch.keys()):
                if pid in self.seen_pids_by_epoch.get(epoch, set()):
                    ended_set = self.ended_pids_by_epoch.get(epoch, set())
                    if pid not in ended_set:
                        candidate_epoch = epoch
                        break
            if candidate_epoch is None:
                logger.error("epoch.end.no_seen_epoch", pid=pid)
                raise ValueError(f"No seen epoch for pid {pid}")

            self.ended_pids_by_epoch.setdefault(candidate_epoch, set()).add(pid)
            logger.debug("epoch.end.from_pid", pid=pid, epoch=candidate_epoch)
            seen = self.seen_pids_by_epoch.get(candidate_epoch, set())
            ended = self.ended_pids_by_epoch.get(candidate_epoch, set())
            if seen and ended >= seen:
                data_to_emit = self.buffer_by_epoch.get(candidate_epoch, [])
                self.buffer_by_epoch.pop(candidate_epoch, None)
                return data_to_emit
        return None
