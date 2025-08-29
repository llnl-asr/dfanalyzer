import hydra
import signal
from dask_jobqueue import LSFCluster, PBSCluster, SLURMCluster
from dataclasses import asdict, dataclass, field
from distributed import LocalCluster
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING
from typing import Any, List, Optional, Union

from .config import (
    ClusterConfig,
    CustomHelpConfig,
    CustomJobConfig,
    LocalClusterConfig,
    LSFClusterConfig,
    PBSClusterConfig,
    SLURMClusterConfig,
)


@dataclass
class ExternalCluster:
    restart_on_connect: Optional[bool]
    scheduler_address: str
    local_directory: Optional[str]


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
cs.store(group="cluster", name="local", node=LocalClusterConfig)
cs.store(group="cluster", name="lsf", node=LSFClusterConfig)
cs.store(group="cluster", name="pbs", node=PBSClusterConfig)
cs.store(group="cluster", name="slurm", node=SLURMClusterConfig)


ClusterType = Union[ExternalCluster, LocalCluster, LSFCluster, PBSCluster, SLURMCluster]


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    cluster: ClusterType = instantiate(cfg.cluster)
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
