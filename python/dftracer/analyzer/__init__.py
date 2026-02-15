import dftracer.analyzer.utils.warning_utils  # noqa: F401
import dask
import structlog
from dataclasses import dataclass
from distributed import Client
from hydra import compose, initialize
from hydra.core.hydra_config import DictConfig, HydraConfig
from hydra.utils import instantiate
from importlib.metadata import PackageNotFoundError, version
from omegaconf import OmegaConf
from typing import Callable, Dict, List, Union, Optional

from .analyzer import Analyzer
from .cluster import ClusterType, ExternalCluster
from .config import CLUSTER_RESTART_TIMEOUT_SECONDS, init_hydra_config_store
from .dftracer import DFTracerAnalyzer
from .input import FileInput, MofkaInput, ZMQInput
from .output import ConsoleOutput, CSVOutput, MofkaOutput, SQLiteOutput, ZMQOutput
from .recorder import RecorderAnalyzer
from .types import AnalyzerResultType, ViewType
from .utils.log_utils import configure_logging, log_block

try:
    __version__ = version("dftracer-analyzer")
except PackageNotFoundError:
    # package is not installed
    pass

# TODO(izzet): Suppress Dask warnings that are not relevant to the user
dask.config.set({"dataframe.query-planning-warning": False})

try:
    from .darshan import DarshanAnalyzer
except ModuleNotFoundError:
    DarshanAnalyzer = Analyzer

AnalyzerType = Union[DarshanAnalyzer, DFTracerAnalyzer, RecorderAnalyzer]
InputType = Union[FileInput, ZMQInput, MofkaInput]
OutputType = Union[ConsoleOutput, CSVOutput, SQLiteOutput, ZMQOutput, MofkaOutput]


@dataclass
class DFAnalyzerInstance:
    analyzer: Analyzer
    client: Client
    cluster: ClusterType
    hydra_config: DictConfig
    input: InputType
    output: OutputType

    def analyze_file(
        self,
        view_types: Optional[List[ViewType]] = None,
        extra_columns: Optional[Dict[str, str]] = None,
        extra_columns_fn: Optional[Callable[[dict], dict]] = None,
    ):
        """Analyze the trace using the configured analyzer."""
        return self.analyzer.analyze_file(
            exclude_characteristics=self.hydra_config.exclude_characteristics,
            extra_columns=extra_columns,
            extra_columns_fn=extra_columns_fn,
            logical_view_types=self.hydra_config.logical_view_types,
            metric_boundaries=OmegaConf.to_object(self.hydra_config.metric_boundaries),
            path=self.input.path,
            view_types=self.hydra_config.view_types if not view_types else view_types,
        )

    def analyze_zmq(
        self,
        view_types: Optional[List[ViewType]] = None,
        extra_columns: Optional[Dict[str, str]] = None,
        extra_columns_fn: Optional[Callable[[dict], dict]] = None,
    ):
        """Analyze the ZMQ trace using the configured analyzer."""
        return self.analyzer.analyze_zmq(
            address=self.input.address,
            exclude_characteristics=self.hydra_config.exclude_characteristics,
            extra_columns=extra_columns,
            extra_columns_fn=extra_columns_fn,
            logical_view_types=self.hydra_config.logical_view_types,
            metric_boundaries=OmegaConf.to_object(self.hydra_config.metric_boundaries),
            view_types=self.hydra_config.view_types if not view_types else view_types,
        )

    def analyze_mofka(
        self,
        view_types: Optional[List[ViewType]] = None,
        extra_columns: Optional[Dict[str, str]] = None,
        extra_columns_fn: Optional[Callable[[dict], dict]] = None,
        output_handler: Optional[Callable[[AnalyzerResultType], None]] = None,
    ):
        """Analyze the Mofka trace using the configured analyzer."""
        if not isinstance(self.input, MofkaInput):
            raise ValueError("Input is not MofkaInput")
        return self.analyzer.analyze_mofka(
            exclude_characteristics=self.hydra_config.exclude_characteristics,
            extra_columns_fn=extra_columns_fn,
            extra_columns=extra_columns,
            group_file=self.input.group_file,
            logical_view_types=self.hydra_config.logical_view_types,
            metric_boundaries=OmegaConf.to_object(self.hydra_config.metric_boundaries),
            output_handler=output_handler,
            topic_name=self.input.topic_name,
            view_types=self.hydra_config.view_types if not view_types else view_types,
        )

    def shutdown(self):
        """Shutdown the Dask client and cluster."""
        self.client.close()
        if hasattr(self.cluster, "close"):
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

    # Setup instance
    return DFAnalyzerInstance(
        analyzer=analyzer,
        client=client,
        cluster=cluster,
        hydra_config=hydra_config,
        input=instantiate(hydra_config.input),
        output=instantiate(hydra_config.output),
    )
