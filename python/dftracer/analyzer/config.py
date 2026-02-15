import dataclasses as dc
import logging
import socket
from hydra.core.config_store import ConfigStore
from hydra.conf import HelpConf, JobConf
from omegaconf import MISSING
from typing import Any, Dict, List, Optional

from .constants import COL_TIME_RANGE
from .types import ViewMetricBoundaries
from .utils.env_utils import get_bool_env_var, get_int_env_var


CHECKPOINT_VIEWS = get_bool_env_var("DFANALYZER_CHECKPOINT_VIEWS", False)
CLUSTER_RESTART_TIMEOUT_SECONDS = get_int_env_var("DFANALYZER_CLUSTER_RESTART_TIMEOUT_SECONDS", 120)
DERIVED_POSIX_METRICS = {
    'data': 'io_cat == 1 or io_cat == 2',
    'read': 'io_cat == 1',
    'write': 'io_cat == 2',
    'metadata': 'io_cat == 3',
    'close': 'io_cat == 3 and func_name.str.contains("close") and ~func_name.str.contains("dir")',
    'open': 'io_cat == 3 and func_name.str.contains("open") and ~func_name.str.contains("dir")',
    'seek': 'io_cat == 3 and func_name.str.contains("seek")',
    'stat': 'io_cat == 3 and func_name.str.contains("stat")',
    'other': 'io_cat == 6',
    'sync': 'io_cat == 7',
}
HASH_CHECKPOINT_NAMES = get_bool_env_var("DFANALYZER_HASH_CHECKPOINT_NAMES", False)


@dc.dataclass
class AnalyzerPresetConfig:
    additional_metrics: Optional[Dict[str, Optional[str]]] = dc.field(default_factory=dict)
    async_layers: Optional[List[str]] = dc.field(default_factory=list)
    derived_metrics: Optional[Dict[str, Dict[str, str]]] = dc.field(default_factory=dict)
    layer_defs: Dict[str, Optional[str]] = MISSING
    layer_deps: Optional[Dict[str, Optional[str]]] = dc.field(default_factory=dict)
    logical_views: Optional[Dict[str, Dict[str, Optional[str]]]] = dc.field(default_factory=dict)
    name: str = MISSING
    time_boundary_layer: str = MISSING
    unscored_metrics: Optional[List[str]] = dc.field(default_factory=list)


@dc.dataclass
class AnalyzerPresetConfigPOSIX(AnalyzerPresetConfig):
    derived_metrics: Optional[Dict[str, Dict[str, str]]] = dc.field(
        default_factory=lambda: {
            'posix': DERIVED_POSIX_METRICS,
        }
    )
    layer_defs: Dict[str, Optional[str]] = dc.field(
        default_factory=lambda: {
            'posix': 'cat.str.contains("posix|stdio")',
        }
    )
    logical_views: Optional[Dict[str, Dict[str, Optional[str]]]] = dc.field(
        default_factory=lambda: {
            'file_name': {
                'file_dir': None,
                'file_pattern': None,
            },
            'proc_name': {
                'host_name': 'proc_name.str.split("#").str[1]',
                'proc_id': 'proc_name.str.split("#").str[2]',
                'thread_id': 'proc_name.str.split("#").str[3]',
            },
        }
    )
    name: str = "posix"
    time_boundary_layer: str = "posix"


@dc.dataclass
class AnalyzerPresetConfigDLIO(AnalyzerPresetConfig):
    async_layers: Optional[List[str]] = dc.field(
        default_factory=lambda: [
            'data_loader',
            'data_loader_fork',
            'reader',
            'reader_posix',
            'posix',
            # 'reader_posix_lustre',
            # 'reader_posix_ssd',
            # 'checkpoint_posix_lustre',
            # 'checkpoint_posix_ssd',
        ]
    )
    derived_metrics: Optional[Dict[str, Dict[str, str]]] = dc.field(
        default_factory=lambda: {
            'app': {},
            'training': {},
            'epoch': {},
            'compute': {},
            'fetch_data': {},
            'checkpoint': {},
            'data_loader': {
                'init': 'func_name.str.contains("init")',
                'item': 'func_name.str.contains("item")',
            },
            'data_loader_fork': {},
            'reader': {
                'close': 'func_name.str.contains(".close")',
                'open': 'func_name.str.contains(".open")',  # e.g. NPZReader.open
                'preprocess': 'func_name.str.contains(".preprocess")',
                'sample': 'func_name.str.contains(".get_sample")',
            },
            'posix': DERIVED_POSIX_METRICS,
            'reader_posix': DERIVED_POSIX_METRICS,
            # 'reader_posix_lustre': DERIVED_POSIX_METRICS,
            # 'reader_posix_ssd': DERIVED_POSIX_METRICS,
            'checkpoint_posix': DERIVED_POSIX_METRICS,
            # 'checkpoint_posix_lustre': DERIVED_POSIX_METRICS,
            # 'checkpoint_posix_ssd': DERIVED_POSIX_METRICS,
            'other_posix': DERIVED_POSIX_METRICS,
            # 'other_posix_lustre': DERIVED_POSIX_METRICS,
            # 'other_posix_ssd': DERIVED_POSIX_METRICS,
        }
    )
    layer_defs: Dict[str, Optional[str]] = dc.field(
        default_factory=lambda: {
            'app': 'func_name.isin(["DLIOBenchmark.initialize", "DLIOBenchmark.run"])',
            'training': 'func_name == "DLIOBenchmark.run"',
            'epoch': 'func_name == "DLIOBenchmark._train"',
            'compute': 'cat == "ai_framework"',
            'fetch_data': 'func_name.isin(["<module>.iter", "fetch-data.iter", "loop.iter"])',
            'checkpoint': 'cat == "checkpoint"',
            'data_loader': 'cat == "data_loader" & ~func_name.isin(["loop.iter", "loop.yield"])',
            'data_loader_fork': 'cat == "posix" & func_name == "fork"',
            'reader': 'cat == "reader"',
            'posix': 'cat.str.contains("posix|stdio")',
            'reader_posix': 'cat.str.contains("posix|stdio") & cat.str.contains("_reader")',
            # 'reader_posix_lustre': 'cat.str.contains("posix|stdio") & cat.str.contains("_reader_lustre")',
            # 'reader_posix_ssd': 'cat.str.contains("posix|stdio") & cat.str.contains("_reader_ssd")',
            'checkpoint_posix': 'cat.str.contains("posix|stdio") & cat.str.contains("_checkpoint")',
            # 'checkpoint_posix_lustre': 'cat.str.contains("posix|stdio") & cat.str.contains("_checkpoint_lustre")',
            # 'checkpoint_posix_ssd': 'cat.str.contains("posix|stdio") & cat.str.contains("_checkpoint_ssd")',
            # 'other_posix': 'cat.isin(["posix", "stdio"]) & func_name != "fork"',
            # 'other_posix_lustre': 'cat.isin(["posix_lustre", "stdio_lustre"])',
            # 'other_posix_ssd': 'cat.isin(["posix_ssd", "stdio_ssd"])',
        }
    )
    layer_deps: Optional[Dict[str, Optional[str]]] = dc.field(
        default_factory=lambda: {
            'app': None,
            'training': 'app',
            'epoch': 'training',
            'compute': 'epoch',
            'fetch_data': 'epoch',
            'checkpoint': 'epoch',
            'data_loader': 'fetch_data',
            'data_loader_fork': 'fetch_data',
            'reader': 'data_loader',
            'posix': None,
            'reader_posix': 'reader',
            # 'reader_posix_lustre': 'reader',
            # 'reader_posix_ssd': 'reader',
            'checkpoint_posix': 'checkpoint',
            # 'checkpoint_posix_lustre': 'checkpoint',
            # 'checkpoint_posix_ssd': 'checkpoint',
            # 'other_posix': None,
            # 'other_posix_lustre': 'other_posix',
            # 'other_posix_ssd': 'other_posix',
        }
    )
    logical_views: Optional[Dict[str, Dict[str, Optional[str]]]] = dc.field(
        default_factory=lambda: {
            'file_name': {
                'file_dir': None,
                'file_pattern': None,
            },
            'proc_name': {
                'host_name': 'proc_name.str.split("#").str[1]',
                'proc_id': 'proc_name.str.split("#").str[2]',
                'thread_id': 'proc_name.str.split("#").str[3]',
            },
        }
    )
    name: str = "dlio-pre-ai-logging"
    time_boundary_layer: str = "app"


@dc.dataclass
class AnalyzerPresetConfigDLIOAILogging(AnalyzerPresetConfigDLIO):
    layer_defs: Dict[str, Optional[str]] = dc.field(
        default_factory=lambda: {
            'app': 'func_name == "ai_root"',
            'training': 'cat == "pipeline" & func_name == "train"',
            'epoch': 'cat == "pipeline" & func_name.str.startswith("epoch")',
            'compute': 'cat == "compute" & func_name == "compute"',
            'fetch_data': 'func_name == "fetch.iter"',
            'checkpoint': 'cat == "checkpoint"',
            'data_loader': 'cat == "data"',
            'data_loader_fork': 'cat == "posix" & func_name == "fork"',
            'reader': 'cat == "reader" or func_name == "preprocess"',
            'posix': 'cat.str.contains("posix|stdio")',
            'reader_posix': 'cat.str.contains("posix|stdio") & cat.str.contains("_reader")',
            # 'reader_posix_lustre': 'cat.str.contains("posix|stdio") & cat.str.contains("_reader_lustre")',
            # 'reader_posix_ssd': 'cat.str.contains("posix|stdio") & cat.str.contains("_reader_ssd")',
            'checkpoint_posix': 'cat.str.contains("posix|stdio") & cat.str.contains("_checkpoint")',
            # 'checkpoint_posix_lustre': 'cat.str.contains("posix|stdio") & cat.str.contains("_checkpoint_lustre")',
            # 'checkpoint_posix_ssd': 'cat.str.contains("posix|stdio") & cat.str.contains("_checkpoint_ssd")',
            # 'other_posix': 'cat.isin(["posix", "stdio"]) & func_name != "fork"',
            # 'other_posix_lustre': 'cat.isin(["posix_lustre", "stdio_lustre"])',
            # 'other_posix_ssd': 'cat.isin(["posix_ssd", "stdio_ssd"])',
        }
    )
    name: str = "dlio"


@dc.dataclass
class AnalyzerConfig:
    checkpoint: Optional[bool] = False
    checkpoint_dir: Optional[str] = "${hydra:run.dir}/checkpoints"
    preset: Optional[AnalyzerPresetConfig] = MISSING
    quantile_stats: Optional[bool] = False
    time_approximate: Optional[bool] = True
    time_granularity: Optional[float] = MISSING
    time_resolution: Optional[float] = MISSING
    time_sliced: Optional[bool] = False


@dc.dataclass
class DarshanAnalyzerConfig(AnalyzerConfig):
    _target_: str = "dftracer.analyzer.darshan.DarshanAnalyzer"
    time_granularity: Optional[float] = 1
    time_resolution: Optional[float] = 1e3


@dc.dataclass
class DFTracerAnalyzerConfig(AnalyzerConfig):
    _target_: str = "dftracer.analyzer.dftracer.DFTracerAnalyzer"
    assign_epochs: Optional[bool] = False
    time_granularity: Optional[float] = 1
    time_resolution: Optional[float] = 1e6


@dc.dataclass
class RecorderAnalyzerConfig(AnalyzerConfig):
    _target_: str = "dftracer.analyzer.recorder.RecorderAnalyzer"
    time_granularity: Optional[float] = 1
    time_resolution: Optional[float] = 1e7


@dc.dataclass
class ClusterConfig:
    local_directory: Optional[str] = "/tmp/${hydra:job.name}-${oc.env:USER}/${oc.select:hydra.job.id,0}"


@dc.dataclass
class ExternalClusterConfig(ClusterConfig):
    _target_: str = "dftracer.analyzer.cluster.ExternalCluster"
    restart_on_connect: Optional[bool] = False
    scheduler_address: Optional[str] = MISSING


@dc.dataclass
class JobQueueClusterSchedulerConfig:
    dashboard_address: Optional[str] = None
    host: Optional[str] = dc.field(default_factory=socket.gethostname)


@dc.dataclass
class JobQueueClusterConfig(ClusterConfig):
    cores: int = 16  # ncores
    death_timeout: Optional[int] = 60
    job_directives_skip: Optional[List[str]] = dc.field(default_factory=list)
    job_extra_directives: Optional[List[str]] = dc.field(default_factory=list)
    log_directory: Optional[str] = ""
    memory: Optional[str] = None
    processes: Optional[int] = 1  # nnodes
    scheduler_options: Optional[JobQueueClusterSchedulerConfig] = dc.field(
        default_factory=JobQueueClusterSchedulerConfig
    )


@dc.dataclass
class LocalClusterConfig(ClusterConfig):
    _target_: str = "dask.distributed.LocalCluster"
    host: Optional[str] = None
    memory_limit: Optional[int] = None
    n_workers: Optional[int] = None
    processes: Optional[bool] = True
    silence_logs: Optional[int] = logging.CRITICAL


@dc.dataclass
class LSFClusterConfig(JobQueueClusterConfig):
    _target_: str = "dask_jobqueue.LSFCluster"
    use_stdin: Optional[bool] = True


@dc.dataclass
class PBSClusterConfig(JobQueueClusterConfig):
    _target_: str = "dask_jobqueue.PBSCluster"


@dc.dataclass
class SLURMClusterConfig(JobQueueClusterConfig):
    _target_: str = "dask_jobqueue.SLURMCluster"


@dc.dataclass
class InputConfig:
    pass


@dc.dataclass
class FileInputConfig(InputConfig):
    _target_: str = "dftracer.analyzer.input.FileInput"
    path: str = MISSING


@dc.dataclass
class ZMQInputConfig(InputConfig):
    _target_: str = "dftracer.analyzer.input.ZMQInput"
    address: str = MISSING


@dc.dataclass
class MofkaInputConfig(InputConfig):
    _target_: str = "dftracer.analyzer.input.MofkaInput"
    group_file: str = MISSING
    topic_name: str = MISSING


@dc.dataclass
class OutputConfig:
    pass


@dc.dataclass
class FileOutputConfig(OutputConfig):
    compact: Optional[bool] = False
    name: Optional[str] = ""
    root_only: Optional[bool] = True
    view_names: Optional[List[str]] = dc.field(default_factory=list)


@dc.dataclass
class ConsoleOutputConfig(OutputConfig):
    _target_: str = "dftracer.analyzer.output.ConsoleOutput"
    show_debug: Optional[bool] = False
    show_header: Optional[bool] = True


@dc.dataclass
class CSVOutputConfig(OutputConfig):
    _target_: str = "dftracer.analyzer.output.CSVOutput"


@dc.dataclass
class SQLiteOutputConfig(OutputConfig):
    _target_: str = "dftracer.analyzer.output.SQLiteOutput"
    run_db_path: Optional[str] = ""


@dc.dataclass
class ZMQOutputConfig(OutputConfig):
    _target_: str = "dfanalyzer.output.ZMQOutput"
    address: str = MISSING


@dc.dataclass
class MofkaOutputConfig(OutputConfig):
    _target_: str = "dftracer.analyzer.output.MofkaOutput"
    group_file: str = MISSING
    topic_name: str = MISSING


@dc.dataclass
class CustomJobConfig(JobConf):
    name: str = "dftracer.analyzer"


@dc.dataclass
class CustomHelpConfig(HelpConf):
    app_name: str = "DFAnalyzer"
    header: str = "${hydra:help.app_name}: Data Flow Analyzer"
    footer: str = dc.field(
        default_factory=lambda: """
Powered by Hydra (https://hydra.cc)

Use --hydra-help to view Hydra specific help
    """.strip()
    )
    template: str = dc.field(
        default_factory=lambda: """
${hydra:help.header}

== Configuration groups ==

Compose your configuration from those groups (group=option)

$APP_CONFIG_GROUPS
== Config ==

Override anything in the config (foo.bar=value)

$CONFIG
${hydra:help.footer}
    """.strip()
    )


@dc.dataclass
class Config:
    defaults: List[Any] = dc.field(
        default_factory=lambda: [
            {"analyzer": "dftracer"},
            {"analyzer/preset": "posix"},
            {"hydra/job": "custom"},
            {"cluster": "local"},
            {"input": "file"},
            {"output": "console"},
            "_self_",
            {"override hydra/help": "custom"},
        ]
    )
    analyzer: AnalyzerConfig = MISSING
    cluster: ClusterConfig = MISSING
    debug: Optional[bool] = False
    exclude_characteristics: Optional[List[str]] = dc.field(default_factory=list)
    input: InputConfig = MISSING
    logical_view_types: Optional[bool] = False
    metric_boundaries: Optional[ViewMetricBoundaries] = dc.field(default_factory=dict)
    output: OutputConfig = MISSING
    verbose: Optional[bool] = False
    view_types: Optional[List[str]] = dc.field(default_factory=lambda: [COL_TIME_RANGE])


def init_hydra_config_store() -> ConfigStore:
    cs = ConfigStore.instance()
    cs.store(group="hydra/help", name="custom", node=dc.asdict(CustomHelpConfig()))
    cs.store(group="hydra/job", name="custom", node=CustomJobConfig)
    cs.store(name="config", node=Config)
    cs.store(group="analyzer", name="darshan", node=DarshanAnalyzerConfig)
    cs.store(group="analyzer", name="dftracer", node=DFTracerAnalyzerConfig)
    cs.store(group="analyzer", name="recorder", node=RecorderAnalyzerConfig)
    cs.store(group="analyzer/preset", name="posix", node=AnalyzerPresetConfigPOSIX)
    cs.store(group="analyzer/preset", name="dlio-pre-ai-logging", node=AnalyzerPresetConfigDLIO)
    cs.store(group="analyzer/preset", name="dlio", node=AnalyzerPresetConfigDLIOAILogging)
    cs.store(group="cluster", name="external", node=ExternalClusterConfig)
    cs.store(group="cluster", name="local", node=LocalClusterConfig)
    cs.store(group="cluster", name="lsf", node=LSFClusterConfig)
    cs.store(group="cluster", name="pbs", node=PBSClusterConfig)
    cs.store(group="cluster", name="slurm", node=SLURMClusterConfig)
    cs.store(group="input", name="file", node=FileInputConfig)
    cs.store(group="input", name="zmq", node=ZMQInputConfig)
    cs.store(group="input", name="mofka", node=MofkaInputConfig)
    cs.store(group="output", name="console", node=ConsoleOutputConfig)
    cs.store(group="output", name="csv", node=CSVOutputConfig)
    cs.store(group="output", name="sqlite", node=SQLiteOutputConfig)
    cs.store(group="output", name="zmq", node=ZMQOutputConfig)
    cs.store(group="output", name="mofka", node=MofkaOutputConfig)
    return cs
