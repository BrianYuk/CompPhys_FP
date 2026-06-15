"""Shared constants and builders for the Ising experiment scripts.

This module is pure: importing it performs no filesystem I/O. Output directories
are only created when a caller explicitly invokes `ensure_dir`.
"""

import os

import numpy as np

# Onsager's exact critical temperature for the 2D square-lattice Ising model,
# in reduced units: T_c/J = 2 / ln(1 + sqrt(2)) ≈ 2.26919.
T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))

# Root directory for all generated artifacts (plots, .npz data, reports).
OUTPUTS_ROOT = "./outputs"

# Common Metropolis run parameters. Scripts that need heavier sampling (e.g. the
# baseline and the L=100 extended run) pass their own values explicitly rather
# than mutating these shared defaults.
DEFAULT_N_EQUIL = 2000
DEFAULT_N_MEAS = 5000
DEFAULT_SAMPLE_EVERY = 5
DEFAULT_SEED = 42


def standard_temperature_grid():
    """Return the 31-point T/J grid shared by every experiment, dense near T_c.

    Three concatenated segments: a coarse ordered-phase tail below the transition,
    a fine band bracketing the Onsager point so the susceptibility peak is well
    resolved, and a coarse disordered-phase tail above it. A fresh array is
    returned on each call so callers can never mutate shared state.
    """
    ordered_tail = np.linspace(1.50, 2.10, 7, endpoint=False)
    near_critical = np.linspace(2.10, 2.45, 15)
    disordered_tail = np.linspace(2.50, 3.30, 9)
    return np.concatenate([ordered_tail, near_critical, disordered_tail])


def output_dir(*parts):
    """Build a path under the outputs root, e.g. output_dir("baseline")."""
    return os.path.join(OUTPUTS_ROOT, *parts)


def ensure_dir(path):
    """Create `path` (and any parents) if absent; return it for chaining."""
    os.makedirs(path, exist_ok=True)
    return path
