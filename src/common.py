"""Shared paths, logging and timing helpers for the pipeline, API and eval."""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", ROOT / "cache"))
LOGS_DIR = Path(os.environ.get("LOGS_DIR", ROOT / "logs"))

# Never touch the network while processing a video: weights come from models/ only.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def setup_logging(name: str = "pipeline", level: int = logging.INFO) -> logging.Logger:
    """Configure INFO logging to stderr and to logs/<name>.log; return the logger."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(LOGS_DIR / f"{name}.log")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


@contextmanager
def timed(logger: logging.Logger, stage: str, timings: dict | None = None):
    """Log the wall time of a pipeline stage; optionally store it in `timings[stage]`."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if timings is not None:
            timings[stage] = round(dt, 3)
        logger.info("%s: %.2fs", stage, dt)
