"""
cryptovit/utils/logging_setup.py
=================================
Centralised logging configuration for CryptoViT.

Call ``setup_logging`` once at program entry; all subsequent
``logging.getLogger(...)`` calls will inherit the configuration.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Configure root logger with console (and optionally file) handlers.

    Parameters
    ----------
    level:
        Logging level string, e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
    log_file:
        Optional path to a ``.log`` file.  The directory is created if needed.
    fmt:
        Log record format string.
    datefmt:
        Date/time format for the log record.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    # Avoid duplicate handlers if called multiple times
    root.handlers.clear()

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # --- File handler (optional) ---
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    logging.getLogger(__name__).info(
        "Logging initialised (level=%s, file=%s).", level, log_file
    )
