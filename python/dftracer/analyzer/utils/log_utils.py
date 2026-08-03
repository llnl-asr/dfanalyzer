import logging
import logging.config
import structlog
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Callable, Optional
from rich.console import Console

from .notebook_utils import IN_JUPYTER

console = Console()

# Updater of the innermost active console_progress_block, so code running inside
# it (e.g. the shard scan) can report progress without threading a callback
# through every layer.
_active_progress: ContextVar[Optional[Callable[..., None]]] = ContextVar(
    "dfanalyzer_active_progress", default=None
)


def current_progress() -> Optional[Callable[..., None]]:
    """The ``update(done, total, detail="")`` of the enclosing
    console_progress_block, or None when not inside one."""
    return _active_progress.get()


def configure_logging(log_file: str, level: str = "info") -> None:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    pre_chain = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
    ]

    logging_config = {
        "version": 1,
        "handlers": {
            "json_file": {
                "class": "logging.FileHandler",
                "filename": log_file,
                "level": level.upper(),
                "formatter": "json",
            },
        },
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),
                "foreign_pre_chain": pre_chain,
            },
        },
        "loggers": {
            "": {
                "handlers": ["json_file"],
                "level": level.upper(),
                "propagate": False,
            }
        },
    }

    logging.config.dictConfig(logging_config)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_logger_name,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


@contextmanager
def console_block(message: str, level: str = "info", logger=None, **kwargs):
    """
    A context manager that logs the elapsed time of the block to the console.
    """
    logger = logger or structlog.get_logger()
    start = time.perf_counter()
    if IN_JUPYTER:
        yield
        elapsed = time.perf_counter() - start
        getattr(logger, level)(message, elapsed=elapsed, **kwargs)
        return
    with console.status(f"{message}...", spinner="dots"):
        try:
            getattr(logger, level)(f"▶ {message}...", **kwargs)
            yield
        finally:
            elapsed = time.perf_counter() - start
            console.print(f"✓ {message} [i]({elapsed:.3f}s)[/i]")
            getattr(logger, level)(f"✓ {message}", elapsed=elapsed, **kwargs)


@contextmanager
def console_progress_block(message: str, level: str = "info", logger=None, **kwargs):
    """Like console_block but shows a Rich progress bar instead of a spinner.

    Publishes an ``update(done, total, detail="")`` via ``current_progress()``
    so nested work can drive the bar; total 0/None leaves it pulsing.
    """
    logger = logger or structlog.get_logger()
    start = time.perf_counter()
    if IN_JUPYTER:

        def _noop(done: int = 0, total=None, detail: str = "") -> None:
            pass

        token = _active_progress.set(_noop)
        try:
            yield _noop
        finally:
            _active_progress.reset(token)
            elapsed = time.perf_counter() - start
            getattr(logger, level)(message, elapsed=elapsed, **kwargs)
        return

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prog:
        task = prog.add_task(message, total=None)

        def update(done: int = 0, total=None, detail: str = "") -> None:
            prog.update(
                task,
                description=(f"{message} {detail}" if detail else message),
                completed=done,
                total=(total or None),
            )

        token = _active_progress.set(update)
        try:
            getattr(logger, level)(f"▶ {message}...", **kwargs)
            yield update
        finally:
            _active_progress.reset(token)
            elapsed = time.perf_counter() - start
            console.print(f"✓ {message} [i]({elapsed:.3f}s)[/i]")
            getattr(logger, level)(f"✓ {message}", elapsed=elapsed, **kwargs)


@contextmanager
def log_block(message: str, level: str = "info", logger=None, **kwargs):
    """
    A context manager that logs the elapsed time of the block.
    """
    logger = logger or structlog.get_logger()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if kwargs:
            logger = logger.bind(**kwargs)
        getattr(logger, level)(message, elapsed=elapsed)
