import dask
import structlog
from dataclasses import dataclass
from distributed import Client
from hydra import compose, initialize
from hydra.core.hydra_config import DictConfig, HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
from typing import Callable, Dict, List, Union, Optional

from .analyzer import Analyzer
from .cluster import ClusterType, ExternalCluster
from .config import CLUSTER_RESTART_TIMEOUT_SECONDS, init_hydra_config_store
from .dftracer import DFTracerAnalyzer
from .output import ConsoleOutput, CSVOutput, SQLiteOutput
from .recorder import RecorderAnalyzer
from .types import ViewType
from .utils.log_utils import configure_logging, log_block
from .utils.warning_utils import filter_warnings

filter_warnings()

# TODO(izzet): Suppress Dask warnings that are not relevant to the user
dask.config.set({"dataframe.query-planning-warning": False})

try:
    from .darshan import DarshanAnalyzer
except ModuleNotFoundError:
    DarshanAnalyzer = Analyzer

AnalyzerType = Union[DarshanAnalyzer, DFTracerAnalyzer, RecorderAnalyzer]
OutputType = Union[ConsoleOutput, CSVOutput, SQLiteOutput]


@dataclass
class DFAnalyzerInstance:
    analyzer: Analyzer
    client: Client
    cluster: ClusterType
    hydra_config: DictConfig
    output: OutputType

    def analyze_trace(
        self,
        view_types: Optional[List[ViewType]] = None,
        extra_columns: Optional[Dict[str, str]] = None,
        extra_columns_fn: Optional[Callable[[dict], dict]] = None,
        trace_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """Analyze the trace using the configured analyzer."""
        return self.analyzer.analyze_trace(
            exclude_characteristics=self.hydra_config.exclude_characteristics,
            extra_columns=extra_columns,
            extra_columns_fn=extra_columns_fn,
            logical_view_types=self.hydra_config.logical_view_types,
            metric_boundaries=OmegaConf.to_object(self.hydra_config.metric_boundaries),
            trace_path=trace_path or self.hydra_config.trace_path,
            unoverlapped_posix_only=self.hydra_config.unoverlapped_posix_only,
            view_types=self.hydra_config.view_types if not view_types else view_types,
            checkpoint_dir=checkpoint_dir,
        )

    def shutdown(self):
        """Shutdown the Dask client and cluster."""
        self.client.close()
        if hasattr(self.cluster, 'close'):
            self.cluster.close()


def init_with_hydra(hydra_overrides: List[str]):
    # Init Hydra config
    with initialize(version_base=None, config_path=None):
        init_hydra_config_store()
        hydra_config = compose(
            config_name="config",
            overrides=hydra_overrides,
            return_hydra_config=True,
        )
    HydraConfig.instance().set_config(hydra_config)

    # Configure structlog + stdlib logging
    log_file = f"{hydra_config.hydra.run.dir}/{hydra_config.hydra.job.name}.log"
    log_level = "debug" if hydra_config.debug else "info"
    configure_logging(log_file=log_file, level=log_level)
    log = structlog.get_logger()
    log.info("Starting dfanalyzer")

    # Setup cluster
    with log_block("Cluster setup"):
        cluster = instantiate(hydra_config.cluster)
        if isinstance(cluster, ExternalCluster):
            client = Client(cluster.scheduler_address)
            if cluster.restart_on_connect:
                client.restart(timeout=CLUSTER_RESTART_TIMEOUT_SECONDS)
        else:
            client = Client(cluster)

    # Setup cluster logging
    with log_block("Configuring logging on all Dask workers"):
        client.run(configure_logging, log_file=log_file, level=log_level)

    # Setup analyzer
    with log_block("Analyzer setup"):
        analyzer = instantiate(
            hydra_config.analyzer,
            debug=hydra_config.debug,
            verbose=hydra_config.verbose,
        )

    # Setup output
    output = instantiate(hydra_config.output)

    return DFAnalyzerInstance(
        analyzer=analyzer,
        client=client,
        cluster=cluster,
        hydra_config=hydra_config,
        output=output,
    )
