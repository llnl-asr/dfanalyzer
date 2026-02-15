import dftracer.analyzer.utils.warning_utils  # noqa: F401
import hydra
import signal
import structlog
from distributed import Client
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from . import AnalyzerType, ClusterType, InputType, OutputType
from .cluster import ExternalCluster
from .config import CLUSTER_RESTART_TIMEOUT_SECONDS, Config, init_hydra_config_store
from .input import FileInput, MofkaInput, ZMQInput
from .utils.log_utils import configure_logging, console_block, log_block

init_hydra_config_store()


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    # Configure structlog + stdlib logging
    hydra_config = HydraConfig.get()
    log_file = f"{hydra_config.runtime.output_dir}/{hydra_config.job.name}.log"
    log_level = "debug" if cfg.debug else "info"
    configure_logging(log_file=log_file, level=log_level)
    log = structlog.get_logger()
    log.info("Starting dfanalyzer")

    # Setup cluster
    with console_block("Cluster setup"):
        cluster: ClusterType = instantiate(cfg.cluster)
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
    with console_block("Analyzer setup"):
        analyzer: AnalyzerType = instantiate(
            cfg.analyzer,
            debug=cfg.debug,
            verbose=cfg.verbose,
        )

    input: InputType = instantiate(cfg.input)
    output: OutputType = instantiate(cfg.output)

    if isinstance(input, FileInput):
        # Analyze trace
        result = analyzer.analyze_file(
            exclude_characteristics=cfg.exclude_characteristics,
            logical_view_types=cfg.logical_view_types,
            metric_boundaries=OmegaConf.to_object(cfg.metric_boundaries),
            path=cfg.input.path,
            view_types=cfg.view_types,
        )
        with console_block("Output"):
            # Handle result
            output.handle_result(result=result)
    elif isinstance(input, ZMQInput):
        print(f"Starting stream analysis from: {input.address}")
        analysis_stream = analyzer.analyze_zmq(
            address=input.address,
            exclude_characteristics=cfg.exclude_characteristics,
            logical_view_types=cfg.logical_view_types,
            metric_boundaries=OmegaConf.to_object(cfg.metric_boundaries),
            view_types=cfg.view_types,
        )
        analysis_stream = analysis_stream.map(lambda result: result.flat_views[("epoch",)].to_json(orient="index"))
        analysis_stream.sink(print)
        analysis_stream.to_zmq(output.address)
        analysis_stream.visualize("analysis")
        analysis_stream.start()
        print("Streaming analysis started. Press Ctrl+C to exit.")
        try:
            signal.pause()
        except KeyboardInterrupt:
            print("\nShutting down streaming analysis...")
    elif isinstance(input, MofkaInput):
        if not hasattr(output, "handle_result"):
            raise ValueError("Output does not support handle_result for Mofka input")
        analyzer.analyze_mofka(
            group_file=input.group_file,
            topic_name=input.topic_name,
            exclude_characteristics=cfg.exclude_characteristics,
            logical_view_types=cfg.logical_view_types,
            metric_boundaries=OmegaConf.to_object(cfg.metric_boundaries),
            view_types=cfg.view_types,
            output_handler=output.handle_result,
        )
    else:
        raise ValueError(f"Unsupported input configuration type: {type(cfg.input)}")

    # Teardown cluster
    with console_block("Cluster teardown"):
        client.close()
        if not isinstance(cluster, ExternalCluster):
            cluster.close()  # type: ignore


if __name__ == "__main__":
    main()
