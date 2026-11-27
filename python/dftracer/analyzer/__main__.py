import hydra
import structlog
from distributed import Client
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from . import AnalyzerType, ClusterType, OutputType
from .cluster import AutoClient, AutoCluster, ExternalCluster, InProcessCluster, NullClient, connect_external
from .config import Config, init_hydra_config_store
from .utils.log_utils import configure_library_logging, configure_logging, console_block, log_block
from .utils.preset_utils import apply_detected_preset
from .utils.warning_utils import filter_warnings

filter_warnings()
init_hydra_config_store()


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    # Configure structlog + stdlib logging
    hydra_config = HydraConfig.get()
    log_file = f"{hydra_config.runtime.output_dir}/{hydra_config.job.name}.log"
    log_level = "debug" if cfg.debug else "info"
    configure_logging(log_file=log_file, level=log_level)
    library_log_level = "debug" if cfg.debug else cfg.log_level
    configure_library_logging(library_log_level)
    log = structlog.get_logger()
    log.info("Starting dfanalyzer")

    # Setup cluster
    with console_block("Cluster setup"):
        cluster: ClusterType = instantiate(cfg.cluster)
        if isinstance(cluster, AutoCluster):
            client = AutoClient(cluster)
        elif isinstance(cluster, InProcessCluster):
            client = NullClient()
        elif isinstance(cluster, ExternalCluster):
            client = connect_external(cluster)
        else:
            client = Client(cluster)

    # Setup cluster logging
    with log_block("Configuring logging on all Dask workers"):
        client.run(configure_logging, log_file=log_file, level=log_level)
        client.run(configure_library_logging, library_log_level)

    # Setup analyzer
    with console_block("Analyzer setup"):
        with log_block("detect_preset"):
            apply_detected_preset(cfg, hydra_config.overrides.task)
        analyzer: AnalyzerType = instantiate(
            cfg.analyzer,
            debug=cfg.debug,
            verbose=cfg.verbose,
            facts_config=cfg.facts,
        )
        analyzer.use_client(client)

    # Analyze trace
    result = analyzer.analyze_trace(
        exclude_characteristics=cfg.exclude_characteristics,
        logical_view_types=cfg.logical_view_types,
        metric_boundaries=OmegaConf.to_object(cfg.metric_boundaries),
        trace_path=cfg.trace_path,
        unoverlapped_posix_only=cfg.unoverlapped_posix_only,
        view_types=cfg.view_types,
    )

    # Handle result
    with console_block("Output"):
        output: OutputType = instantiate(cfg.output)
        output.handle_result(result=result)

    # Teardown cluster
    with console_block("Cluster teardown"):
        client.close()
        if not isinstance(cluster, ExternalCluster):
            cluster.close()  # type: ignore


if __name__ == "__main__":
    main()
