# core/app_logging.py
from __future__ import annotations
import logging
from mangetamain.config import ROOT_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"

LOG_DIR = ROOT_DIR / "logs"
APP_LOG = LOG_DIR / "app.log"
ERR_LOG = LOG_DIR / "error.log"

def setup_logging() -> None:
    """
    Configure global logging:
    - logs/app.log (INFO+)
    - logs/error.log (ERROR+)
    - console (INFO+)
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clean existing handlers to prevent Streamlit duplicate logs
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(_LOG_FORMAT)

    # File: INFO+
    fh_info = logging.FileHandler(APP_LOG, encoding="utf-8")
    fh_info.setLevel(logging.INFO)
    fh_info.setFormatter(fmt)
    logger.addHandler(fh_info)

    # File: ERROR+
    fh_err = logging.FileHandler(ERR_LOG, encoding="utf-8")
    fh_err.setLevel(logging.ERROR)
    fh_err.setFormatter(fmt)
    logger.addHandler(fh_err)

    # Console handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)


def get_logger(name: str = "mangetamain") -> logging.Logger:
    """Return a named logger (default: 'mangetamain')."""
    return logging.getLogger(name)
