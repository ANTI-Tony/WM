"""Logging utilities."""

import logging
import sys
from pathlib import Path


def setup_logger(name: str, log_file: Path = None, level=logging.INFO):
    """Setup logger with console and optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def log_metrics(*metric_dicts, epoch=None):
    """Log metrics to wandb if available."""
    try:
        import wandb
        merged = {}
        for d in metric_dicts:
            merged.update(d)
        if epoch is not None:
            merged["epoch"] = epoch
        wandb.log(merged)
    except ImportError:
        pass
