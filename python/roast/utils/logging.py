"""Console output.

The MATLAB code prints progress with ``disp`` and framed banners; the Python
port routes the same messages through :mod:`logging` so they can be silenced or
redirected, while keeping the familiar banners on screen.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "roast") -> logging.Logger:
    """Return the package logger, attaching a plain stdout handler once."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _CONFIGURED = True
    return logger


def banner(*lines: str, width: int = 54, logger: logging.Logger | None = None) -> None:
    """Print a framed block, the way the MATLAB code frames its milestones."""
    logger = logger or get_logger()
    rule = "=" * width
    logger.info(rule)
    for line in lines:
        logger.info(line.center(width).rstrip())
    logger.info(rule)


def step_banner(step: int | None, total: int | None, text: str,
                logger: logging.Logger | None = None) -> None:
    """``STEP 3 (out of 6): ELECTRODE PLACEMENT...``"""
    if step is None:
        banner(text, logger=logger)
    else:
        banner(f"STEP {step} (out of {total}): {text}", logger=logger)
