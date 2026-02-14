"""
Logging configuration for Aerovision-V1-Server.
"""

import logging
import sys
from typing import Literal

from pythonjsonlogger import jsonlogger

from app.core.config import get_settings


def setup_logging(
    level: str | None = None,
    format_type: Literal["json", "text"] | None = None
) -> logging.Logger:
    """
    Set up application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_type: Log format type (json or text)

    Returns:
        Configured logger instance
    """
    settings = get_settings()
    log_level = level or settings.log_level
    log_format = format_type or settings.log_format

    # Get root logger
    logger = logging.getLogger("aerovision")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper()))

    # Create formatter
    if log_format == "json":
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            timestamp=True
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Suppress noisy third-party logs
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("paddleocr").setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(f"aerovision.{name}")


# Initialize logger on module import
logger = setup_logging()
