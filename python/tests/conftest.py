"""Shared pytest setup.

Puts the package on the path so the suite runs straight from a checkout without
installing it, and selects a non-interactive matplotlib backend so the tests
that build figures do not try to open a window.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
