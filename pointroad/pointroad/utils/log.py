from __future__ import annotations

from loguru import logger
from pathlib import Path
import sys


def setup_logger(log_dir: Path | None = None, level: str = "INFO") -> None:
    """Configure loguru logger with file + stderr sinks.

    Args:
        log_dir: Optional directory to write logs into.
        level: Log level string.
    """
    logger.remove()
    logger.add(sys.stderr, level=level, enqueue=True, backtrace=False, diagnose=False)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "pointroad.log",
            level=level,
            rotation="5 MB",
            retention=5,
            enqueue=True,
        )


__all__ = ["logger", "setup_logger"]



