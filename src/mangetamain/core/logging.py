# src/core/app_logging.py
from __future__ import annotations
from pathlib import Path
import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

def setup_logging(root: Path) -> None:
    """
    Configure le logging global :
    - logs/app.log (niveau INFO+)
    - logs/error.log (niveau ERROR+)
    - console (INFO+)
    """
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    app_path = log_dir / "app.log"
    err_path = log_dir / "error.log"

    # Logger racine
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Nettoyer les handlers existants pour éviter doublons
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(_LOG_FORMAT)

    # Fichier INFO+
    fh_info = logging.FileHandler(app_path, encoding="utf-8")
    fh_info.setLevel(logging.INFO)
    fh_info.setFormatter(fmt)
    logger.addHandler(fh_info)

    # Fichier ERROR+
    fh_err = logging.FileHandler(err_path, encoding="utf-8")
    fh_err.setLevel(logging.ERROR)
    fh_err.setFormatter(fmt)
    logger.addHandler(fh_err)

    # Console
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)


def get_logger(name: str = "mangetamain") -> logging.Logger:
    """Récupère un logger par nom."""
    return logging.getLogger(name)
