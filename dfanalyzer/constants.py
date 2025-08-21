import numpy as np
from enum import Enum, auto
from strenum import StrEnum


class AccessPattern(Enum):
    SEQUENTIAL = 0
    RANDOM = 1


class EventType(StrEnum):
    ATTACH_REASONS = auto()
    COMPUTE_HLM = auto()
    COMPUTE_MAIN_VIEW = auto()
    COMPUTE_METRIC_BOUNDARIES = auto()
    COMPUTE_PERSPECTIVES = auto()
    COMPUTE_VIEW = auto()
    DETECT_CHARACTERISTICS = auto()
    EVALUATE_VIEW = auto()
    READ_TRACES = auto()
    SAVE_VIEWS = auto()


class IOCategory(Enum):
    READ = 1
    WRITE = 2
    METADATA = 3
    PCTL = 4
    IPC = 5
    OTHER = 6
    SYNC = 7


class Layer(StrEnum):
    APP = auto()
    DATALOADER = auto()
    NETCDF = auto()
    PNETCDF = auto()
    HDF5 = auto()
    MPI = auto()
    POSIX = auto()


COL_ACC_PAT = 'acc_pat'
COL_APP_NAME = 'app_name'
COL_BEHAVIOR = 'behavior'
COL_CATEGORY = 'cat'
COL_COUNT = 'count'
COL_EPOCH = 'epoch'
COL_FILE_DIR = 'file_dir'
COL_FILE_NAME = 'file_name'
COL_FILE_PATTERN = 'file_pattern'
COL_FUNC_NAME = 'func_name'
COL_HOST_NAME = 'host_name'
COL_IO_CAT = 'io_cat'
COL_NODE_NAME = 'node_name'
COL_PROC_ID = 'proc_id'
COL_PROC_NAME = 'proc_name'
COL_RANK = 'rank'
COL_SIZE = 'size'
COL_TIME = 'time'
COL_TIME_OVERALL = 'time_overall'
COL_TIME_RANGE = 'time_range'
COL_TIME_START = 'time_start'
COL_TIME_END = 'time_end'


LOGICAL_VIEW_TYPES = [
    ('file_name', 'file_dir'),
    ('file_name', 'file_pattern'),
    ('proc_name', 'app_name'),
    ('proc_name', 'host_name'),
    ('proc_name', 'node_name'),
    ('proc_name', 'proc_id'),
    ('proc_name', 'rank'),
    ('proc_name', 'thread_id'),
]
VIEW_TYPES = [
    'time_range',
]

ACC_PAT_SUFFIXES = ['time', 'size', 'count']
DERIVED_MD_OPS = ['close', 'open', 'seek', 'stat']
IO_CATS = [io_cat.value for io_cat in list(IOCategory)]
IO_TYPES = ['read', 'write', 'metadata']
COMPACT_IO_TYPES = ['R', 'W', 'M']


# todo(izzet): add mmap
POSIX_IO_CAT_FUNCTIONS = {
    IOCategory.READ: {
        'fread',
        'pread',
        'preadv',
        'read',
        'readv',
    },
    IOCategory.WRITE: {
        'fwrite',
        'pwrite',
        'pwritev',
        'write',
        'writev',
    },
    IOCategory.SYNC: {
        'fsync',  # Forces write of all dirty pages for a file to disk
        'fdatasync',  # Similar to fsync but doesn't flush metadata
        'msync',  # Memory-mapped file sync
        'sync',  # System-wide sync
    },
    IOCategory.METADATA: {
        '__fxstat',
        '__fxstat64',
        '__lxstat',
        '__lxstat64',
        '__xstat',
        '__xstat64',
        'access',
        'close',
        'closedir',
        'fclose',
        'fcntl',
        'fopen',
        'fopen64',
        'fseek',
        'fstat',
        'fstatat',
        'ftell',
        'ftruncate',
        'link',
        'lseek',
        'lseek64',
        'mkdir',
        'open',
        'open64',
        'opendir',
        'readdir',
        'readlink',
        'remove',
        'rename',
        'rmdir',
        'seek',
        'stat',
        'unlink',
    },
    IOCategory.PCTL: {
        'exec',
        'exit',
        'fork',
        'kill',
        'pipe',
        'wait',
    },
    IOCategory.IPC: {
        'msgctl',
        'msgget',
        'msgrcv',
        'msgsnd',
        'semctl',
        'semget',
        'semop',
        'shmat',
        'shmctl',
        'shmdt',
        'shmget',
    },
}
POSIX_IO_CAT_MAPPING = {func: category for category, functions in POSIX_IO_CAT_FUNCTIONS.items() for func in functions}
POSIX_METADATA_FUNCTIONS = POSIX_IO_CAT_FUNCTIONS[IOCategory.METADATA]


FILE_PATTERN_PLACEHOLDER = '[0-9]'
PROC_NAME_SEPARATOR = '#'

HUMANIZED_COLS = dict(
    acc_pat='Access Pattern',
    app_io_time='Application I/O Time',
    app_name='Application',
    behavior='Behavior',
    cat='Category',
    checkpoint_io_time='Checkpoint I/O Time',
    compute_time='Compute Time',
    count='Count',
    file_dir='File Directory',
    file_name='File',
    file_pattern='File Pattern',
    func_name='Function Name',
    host_name='Host',
    io_cat='I/O Category',
    io_time='I/O Time',
    node_name='Node',
    proc_name='Process',
    rank='Rank',
    read_io_time='Read I/O Time',
    size='Size',
    time='Time',
    time_range='Time Period',
    u_app_compute_time='Unoverlapped Application Compute Time',
    u_app_io_time='Unoverlapped Application I/O Time',
    u_checkpoint_io_time='Unoverlapped Checkpoint I/O Time',
    u_compute_time='Unoverlapped Compute Time',
    u_io_time='Unoverlapped I/O Time',
    u_read_io_time='Unoverlapped Read I/O Time',
)
HUMANIZED_METRICS = dict(
    bw='I/O Bandwidth',
    intensity='I/O Intensity',
    iops='I/O Operations per Second',
    time='I/O Time',
)
HUMANIZED_VIEW_TYPES = dict(
    app_name='App',
    file_dir='File Directory',
    file_name='File',
    file_pattern='File Pattern',
    node_name='Node',
    proc_name='Process',
    rank='Rank',
    time_range='Time Period',
)

KiB = 1024.0
MiB = KiB * KiB
GiB = KiB * MiB

SIZE_BINS = [
    -np.inf,
    4 * KiB,
    16 * KiB,
    64 * KiB,
    256 * KiB,
    1 * MiB,
    4 * MiB,
    16 * MiB,
    64 * MiB,
    256 * MiB,
    1 * GiB,
    4 * GiB,
    np.inf,
]
SIZE_BIN_LABELS = [
    '<4 KiB',
    '4 KiB - 16 KiB',
    '16 KiB - 64 KiB',
    '64 KiB - 256 KiB',
    '256 KiB - 1 MiB',
    '1 MiB - 4 MiB',
    '4 MiB - 16 MiB',
    '16 MiB - 64 MiB',
    '64 MiB - 256 MiB',
    '256 MiB - 1 GiB',
    '1 GiB - 4 GiB',
    '>4 GiB',
]
SIZE_BIN_NAMES = [
    '<4 kiB',
    "4 KiB",
    "16 KiB",
    "64 KiB",
    "256 KiB",
    "1 MiB",
    "4 MiB",
    "16 MiB",
    "64 MiB",
    "256 MiB",
    "1 GiB",
    ">4 GiB",
]
SIZE_BIN_SUFFIXES = [
    "0_4kib",
    "4kib_16kib",
    "16kib_64kib",
    "64kib_256kib",
    "256kib_1mib",
    "1mib_4mib",
    "4mib_16mib",
    "16mib_64mib",
    "64mib_256mib",
    "256mib_1gib",
    "1gib_4gib",
    "4gib_plus",
]

NUM_SIZE_BINS = len(SIZE_BINS) - 1
assert len(SIZE_BIN_LABELS) == NUM_SIZE_BINS, (
    f"SIZE_BIN_LABELS length mismatch: expected {NUM_SIZE_BINS}, got {len(SIZE_BIN_LABELS)}"
)
assert len(SIZE_BIN_NAMES) == NUM_SIZE_BINS, (
    f"SIZE_BIN_NAMES length mismatch: expected {NUM_SIZE_BINS}, got {len(SIZE_BIN_NAMES)}"
)
assert len(SIZE_BIN_SUFFIXES) == NUM_SIZE_BINS, (
    f"SIZE_BIN_SUFFIXES length mismatch: expected {NUM_SIZE_BINS}, got {len(SIZE_BIN_SUFFIXES)}"
)

EVENT_ATT_REASONS = 'attach_reasons'
EVENT_COMP_HLM = 'compute_hlm'
EVENT_COMP_MAIN_VIEW = 'compute_main_view'
EVENT_COMP_METBD = 'compute_metric_boundaries'
EVENT_COMP_PERS = 'compute_perspectives'
EVENT_COMP_ROOT_VIEWS = 'compute_root_views'
EVENT_COMP_VIEW = 'compute_view'
EVENT_DET_CHAR = 'detect_characteristics'
EVENT_READ_TRACES = 'read_traces'
EVENT_SAVE_VIEWS = 'save_views'

#########################################################
# CONSTANTS 
# Copied from dftracer/logger/ai.py
#########################################################

# namespacing for AI DFTracer
class AIDFTracer:
    ROOT_NAME = "ai_root"
    ROOT_CAT = "ai_root"
    ITER_COUNT_NAME = "count"
    INIT_NAME = "init"
    ITER_COUNT_NAME = "count"
    INIT_NAME = "init"
    BLOCK_NAME = "block"
    ITER_NAME = "iter"
    CTX_SEPARATOR = "."

    class Category(StrEnum):
        COMPUTE = "compute"
        DATA = "data"
        DATALOADER = "dataloader"
        COMM = "comm"
        DEVICE = "device"
        CHECKPOINT = "checkpoint"
        PIPELINE = "pipeline"

    class Compute(StrEnum):
        FORWARD = "forward"
        BACKWARD = "backward"
        STEP = "step"

    class Data(StrEnum):
        PREPROCESS = "preprocess"
        ITEM = "item"

    class DataLoader(StrEnum):
        FETCH = "fetch"

    class Communication(StrEnum):
        SEND = "send"
        RECEIVE = "receive"
        BARRIER = "barrier"
        BCAST = "bcast"
        REDUCE = "reduce"
        ALL_REDUCE = "all_reduce"
        GATHER = "gather"
        ALL_GATHER = "all_gather"
        SCATTER = "scatter"
        REDUCE_SCATTER = "reduce_scatter"
        ALL_TO_ALL = "all_to_all"

    class Device(StrEnum):
        TRANSFER = "transfer"

    class Checkpoint(StrEnum):
        CAPTURE = "capture"
        RESTART = "restart"

    class Pipeline(StrEnum):
        EPOCH = "epoch"
        TRAIN = "train"
        EVALUATE = "evaluate"
        TEST = "test"

    @staticmethod
    def get_block(func_name: str) -> str:
        return f"{func_name}{AIDFTracer.CTX_SEPARATOR}{AIDFTracer.BLOCK_NAME}"
    
    @staticmethod
    def get_iter(func_name: str) -> str:
        return f"{func_name}{AIDFTracer.CTX_SEPARATOR}{AIDFTracer.ITER_NAME}"
    
    @staticmethod
    def get_init(func_name: str) -> str:
        return f"{func_name}{AIDFTracer.CTX_SEPARATOR}{AIDFTracer.INIT_NAME}"
    
    @staticmethod
    def get_epoch_query() -> str:
        return f'(cat == "{AIDFTracer.Category.PIPELINE}" & (func_name == "{AIDFTracer.get_block(AIDFTracer.Pipeline.EPOCH)}" | func_name == "{AIDFTracer.Pipeline.EPOCH}"))'


#########################################################


