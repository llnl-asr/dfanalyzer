import hydra
import signal
import structlog
from dask_jobqueue import LSFCluster, PBSCluster, SLURMCluster
from dataclasses import asdict, dataclass, field
from distributed import Client, LocalCluster, wait
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING
from typing import Any, List, Optional, Union

from .config import (
    AutoClusterConfig,
    ClusterConfig,
    CustomHelpConfig,
    CustomJobConfig,
    InProcessClusterConfig,
    LocalClusterConfig,
    LSFClusterConfig,
    PBSClusterConfig,
    SLURMClusterConfig,
)


logger = structlog.get_logger()


@dataclass
class ExternalCluster:
    restart_on_connect: Optional[bool]
    scheduler_address: str
    local_directory: Optional[str]


@dataclass
class AutoCluster:
    """Marker for `cluster=auto` -- see `AutoClusterConfig`.

    Holds the settings a `LocalCluster` would be built with, in case the trace
    turns out to be too large to analyse in one process.
    """

    max_bytes: int = 256 * 1024 * 1024
    host: Optional[str] = None
    memory_limit: Optional[int] = None
    n_workers: Optional[int] = None
    processes: Optional[bool] = True
    silence_logs: Optional[int] = None
    local_directory: Optional[str] = None

    def build_cluster(self) -> LocalCluster:
        kwargs = {
            "host": self.host,
            "memory_limit": self.memory_limit,
            "n_workers": self.n_workers,
            "processes": self.processes,
            "local_directory": self.local_directory,
        }
        if self.silence_logs is not None:
            kwargs["silence_logs"] = self.silence_logs
        return LocalCluster(**{k: v for k, v in kwargs.items() if v is not None})

    def close(self) -> None:
        """The cluster, if one was ever built, is owned and closed by AutoClient."""


@dataclass
class InProcessCluster:
    """Marker for "no cluster at all" -- see `InProcessClusterConfig`.

    Carries no scheduler, so `dfanalyzer-cluster` cannot serve it; it exists
    only so `cluster=none` composes like any other cluster choice.
    """

    local_directory: Optional[str] = None

    def close(self) -> None:
        """Nothing was started, so nothing needs stopping."""


# Keyword arguments `Client.submit` accepts to place and schedule a task. They
# describe *where* work runs, which is meaningless once it runs here, so
# NullClient drops them rather than passing them to the function.
_SCHEDULING_KWARGS = frozenset(
    {
        "actor",
        "actors",
        "allow_other_workers",
        "fifo_timeout",
        "key",
        "priority",
        "pure",
        "resources",
        "retries",
        "workers",
    }
)


class NullClient:
    """A `Client`-shaped object that runs work here and now.

    The analyzer talks to a Dask client in a handful of places -- submitting
    scans, gathering partials, scheduling checkpoint writes. Rather than teach
    each of those to ask "is there a cluster?", this satisfies the same calls
    synchronously, so one code path serves both modes.

    `submit` returns the function's result rather than a future. That is what
    lets the shim disappear: Dask resolves futures to values before calling a
    function, so a caller that submits a "future" here and passes it to another
    `submit` sees exactly what it would have seen from a real client. `gather`
    is then the identity.

    `is_distributed` is the one thing callers may legitimately branch on --
    reading a trace, for instance, has to scan locally instead of fanning the
    work out to workers that do not exist.
    """

    is_distributed = False

    def submit(self, fn, *args, **kwargs):
        for key in _SCHEDULING_KWARGS:
            kwargs.pop(key, None)
        return fn(*args, **kwargs)

    def gather(self, futures, **kwargs):
        """Identity: `submit` already returned values, not futures."""
        if isinstance(futures, (list, tuple)):
            return list(futures)
        return futures

    def compute(self, collections, sync=True, **kwargs):
        """Compute now, whatever `sync` says -- there is nowhere to defer to."""
        import dask

        return dask.compute(collections)[0]

    def cancel(self, *args, **kwargs) -> None:
        """Work is finished by the time anything could cancel it."""

    def nthreads(self, *args, **kwargs) -> dict:
        """No workers, so no per-worker thread counts."""
        return {}

    def run(self, fn, *args, **kwargs) -> dict:
        """Run a worker-setup function against this process instead."""
        return {"in-process": fn(*args, **kwargs)}

    def restart(self, *args, **kwargs) -> None:
        """No workers to restart."""

    def close(self, *args, **kwargs) -> None:
        """Nothing to disconnect from."""

    def shutdown(self, *args, **kwargs) -> None:
        """Nothing to shut down."""


class AutoClient:
    """Runs in-process until the trace proves too big, then starts a cluster.

    Begins as a `NullClient` and forwards everything to it. `promote()` builds a
    `LocalCluster`, connects a real `Client`, and swaps it in -- *in place*, so
    every reference already handed out (notably `Analyzer.dask_client`) keeps
    working and starts talking to the cluster.

    The caller decides when to promote, because only it can see how much data
    the scan is producing; `max_bytes` is carried here so it has the budget to
    compare against.
    """

    can_promote = True

    # How much larger than the budget the traces may be on disk before the
    # in-process attempt is abandoned without scanning at all. Deliberately
    # loose: across the bundled fixtures the aggregated result runs between 0.3x
    # and 2.8x the bytes on disk, so this sits an order of magnitude above
    # anything observed. It guards against exhausting memory during the scan
    # itself; it is not an attempt to predict the aggregated size.
    on_disk_budget_factor = 50

    def __init__(self, cluster: AutoCluster) -> None:
        self.max_bytes = cluster.max_bytes
        self._spec = cluster
        self._impl = NullClient()
        self._cluster: Optional[LocalCluster] = None

    @property
    def is_distributed(self) -> bool:
        return self._cluster is not None

    def accepts_estimate(self, on_disk_bytes: int) -> bool:
        """Whether a scan is worth attempting here, judged before doing one.

        Callers pass whatever cheap upper bound they have. Rejecting promotes,
        so the caller can simply stop and submit its work as usual.
        """
        if on_disk_bytes <= self.max_bytes * self.on_disk_budget_factor:
            return True
        logger.info(
            "traces far larger than the in-process budget, starting a cluster unscanned",
            on_disk_bytes=on_disk_bytes,
            max_bytes=self.max_bytes,
        )
        self.promote()
        return False

    def accepts(self, nbytes: int) -> bool:
        """Whether this much aggregated data should be handled in process.

        The workload fits only when its peak (~3x the aggregated bytes: gathered
        partials + groupby intermediate + result) stays under available memory,
        i.e. the aggregated bytes are under ~1/3 of it. `max_bytes` is an extra
        hard cap. Rejecting promotes, so a caller that has already scanned can
        drop what it holds and re-submit through the cluster; the warning tells a
        manual-cluster user how much memory or how many nodes they need.
        """
        from dftracer.utils.dftracer_utils_ext import memory_budget_advice

        advice = memory_budget_advice(nbytes)
        if not advice["fits"]:
            logger.warning(
                advice["warning"],
                scanned_bytes=nbytes,
                suggested_nodes=advice["suggested_nodes"],
            )
        if advice["fits"] and nbytes <= self.max_bytes:
            logger.info("analysing in process", scanned_bytes=nbytes, max_bytes=self.max_bytes)
            return True
        logger.info(
            "trace too large to analyse in process, starting a cluster",
            scanned_bytes=nbytes,
            max_bytes=self.max_bytes,
        )
        self.promote()
        return False

    def promote(self):
        """Start a real cluster and route everything to it from here on."""
        if self._cluster is None:
            self._cluster = self._spec.build_cluster()
            self._impl = Client(self._cluster)
        return self._impl

    def close(self, *args, **kwargs) -> None:
        self._impl.close(*args, **kwargs)
        if self._cluster is not None:
            self._cluster.close()
            self._cluster = None

    def __getattr__(self, name):
        # Only consulted for names not found on the instance or class, so the
        # attributes above keep their own meaning and the rest -- submit,
        # gather, compute, cancel, nthreads, run -- follow whichever client is
        # currently in charge.
        return getattr(self._impl, name)


def wait_for(client, items) -> None:
    """Block until `items` are finished, where finishing means anything.

    This asks the *client*, not the frames. `distributed.wait` needs a
    scheduler tracking futures, and whether one exists is a property of how
    execution was set up -- not of whether a given collection is a Dask
    collection. The two come apart exactly here: under `cluster=none` the
    frames are still genuine Dask DataFrames, so anything that dispatched on
    the frame would conclude "Dask, therefore wait" and raise `No clients
    found`. There is nothing to wait for either way, because `persist` and
    `compute` have already run to completion inline.
    """
    if getattr(client, "is_distributed", True):
        wait(items)


@dataclass
class Config:
    defaults: List[Any] = field(
        default_factory=lambda: [
            {"hydra/job": "custom"},
            {"cluster": "local"},
            "_self_",
            {"override hydra/help": "custom"},
        ]
    )
    cluster: ClusterConfig = MISSING
    debug: Optional[bool] = False
    verbose: Optional[bool] = False


cs = ConfigStore.instance()
cs.store(group="hydra/help", name="custom", node=asdict(CustomHelpConfig()))
cs.store(group="hydra/job", name="custom", node=CustomJobConfig)
cs.store(name="config", node=Config)
cs.store(group="cluster", name="auto", node=AutoClusterConfig)
cs.store(group="cluster", name="local", node=LocalClusterConfig)
cs.store(group="cluster", name="none", node=InProcessClusterConfig)
cs.store(group="cluster", name="lsf", node=LSFClusterConfig)
cs.store(group="cluster", name="pbs", node=PBSClusterConfig)
cs.store(group="cluster", name="slurm", node=SLURMClusterConfig)


ClusterType = Union[AutoCluster, ExternalCluster, InProcessCluster, LocalCluster, LSFCluster, PBSCluster, SLURMCluster]


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    cluster: ClusterType = instantiate(cfg.cluster)
    if not hasattr(cluster, "scheduler"):
        raise ValueError(
            f"cluster={type(cluster).__name__} has no scheduler to serve. "
            "dfanalyzer-cluster starts a cluster for other processes to connect to; "
            "choose cluster=local, slurm, lsf or pbs."
        )
    print(cluster.scheduler.address, flush=True)
    try:
        signal.pause()
    except KeyboardInterrupt:
        print("Shutting down the Dask cluster...")
    finally:
        cluster.close()
        print("Dask cluster is shut down.")


if __name__ == "__main__":
    main()
