"""One-shot generator for the committed golden reference files.

Run this ONCE against the CURRENT, UNMODIFIED engine, before refactoring:

    python3 tests/_generate_golden.py

It stores the exact observable arrays the refactor must reproduce. Never
regenerate from refactored code — that would defeat the purpose of the lock.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)   # so `import ising` works when run as a script
sys.path.insert(0, _HERE)        # so `import conftest` works

from conftest import (  # noqa: E402
    GOLDEN_OBSERVABLES,
    TINY_L,
    TINY_N_EQUIL,
    TINY_N_MEAS,
    TINY_SAMPLE_EVERY,
    TINY_TEMPERATURES,
)
from ising_mc.ising import record_trajectory, run_at_temperature, sweep_temperatures  # noqa: E402

GOLDEN_DIR = os.path.join(_HERE, "golden")


def _sweep_observable_arrays(seed):
    """Run the canonical tiny seeded sweep and return {observable: array}."""
    results, _ = sweep_temperatures(
        L=TINY_L,
        T_over_J_array=TINY_TEMPERATURES.copy(),
        h_over_J=0.0,
        n_equil=TINY_N_EQUIL,
        n_meas=TINY_N_MEAS,
        sample_every=TINY_SAMPLE_EVERY,
        anneal=True,
        seed=seed,
    )
    return {key: np.array([r[key] for r in results]) for key in GOLDEN_OBSERVABLES}


def _single_point_observables(seed):
    """Run one run_at_temperature point from a seeded random lattice."""
    rng = np.random.default_rng(seed)
    lattice = rng.choice(np.array([-1, 1], dtype=np.int8), size=(TINY_L, TINY_L))
    result = run_at_temperature(
        lattice, T_over_J=2.27, h_over_J=0.0,
        n_equil=TINY_N_EQUIL, n_meas=TINY_N_MEAS,
        sample_every=TINY_SAMPLE_EVERY, rng=rng,
    )
    return {key: np.array(result[key]) for key in GOLDEN_OBSERVABLES}


def main():
    os.makedirs(GOLDEN_DIR, exist_ok=True)

    for seed in (0, 42):
        np.savez(os.path.join(GOLDEN_DIR, f"sweep_L4_seed{seed}.npz"),
                 **_sweep_observable_arrays(seed))

    np.savez(os.path.join(GOLDEN_DIR, "run_point_seed0.npz"),
             **_single_point_observables(0))

    frames = record_trajectory(L=6, T_over_J=2.27, h_over_J=0.0,
                               n_equil=10, n_frames=4, sweeps_per_frame=2, seed=0)
    np.savez(os.path.join(GOLDEN_DIR, "trajectory_L6_seed0.npz"), frames=frames)

    print(f"Golden files written to {GOLDEN_DIR}/")


if __name__ == "__main__":
    main()
