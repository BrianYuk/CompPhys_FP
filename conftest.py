"""Make the repo-root modules (ising, config, observables, plotting) importable
from tests regardless of pytest's working directory or import mode."""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
