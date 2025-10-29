import logging
import os
from datetime import datetime

def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO):
    """Configure global logging."""
    os.makedirs(log_dir, exist_ok=True)
    # log_filename = f"recipes_analyzer_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_filename = "recipes_analyzer.log"
    log_file = os.path.join(log_dir, log_filename)

    # Define a global format
    log_format = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"

    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            # no StreamHandler → stdout stays clean
        ],
    )

    return log_file
