"""Exact-value characterization: the engine must reproduce the committed golden
observable arrays bit-for-bit (up to last-ULP float noise). If the refactor shifts
the random-number stream, observables move by O(1) and these tests fail loudly.
"""

import os

import numpy as np

from conftest import (
    GOLDEN_OBSERVABLES,
    TINY_L,
    TINY_N_EQUIL,
    TINY_N_MEAS,
    TINY_SAMPLE_EVERY,
    TINY_TEMPERATURES,
)
from ising_mc.ising import run_at_temperature, sweep_temperatures

# Tight enough to catch any real numerical change; tolerant of last-ULP jitter.
RTOL = 1e-12
ATOL = 1e-12


def _load_golden(golden_dir, name):
    return np.load(os.path.join(golden_dir, name))


def test_sweep_matches_golden_seed0(golden_dir, tiny_sweep_kwargs):
    golden = _load_golden(golden_dir, "sweep_L4_seed0.npz")
    results, _ = sweep_temperatures(seed=0, **tiny_sweep_kwargs)
    for key in GOLDEN_OBSERVABLES:
        actual = np.array([r[key] for r in results])
        np.testing.assert_allclose(actual, golden[key], rtol=RTOL, atol=ATOL,
                                   err_msg=f"observable '{key}' drifted from golden")


def test_run_at_temperature_matches_golden(golden_dir):
    golden = _load_golden(golden_dir, "run_point_seed0.npz")
    rng = np.random.default_rng(0)
    lattice = rng.choice(np.array([-1, 1], dtype=np.int8), size=(TINY_L, TINY_L))
    result = run_at_temperature(
        lattice, T_over_J=2.27, h_over_J=0.0,
        n_equil=TINY_N_EQUIL, n_meas=TINY_N_MEAS,
        sample_every=TINY_SAMPLE_EVERY, rng=rng,
    )
    for key in GOLDEN_OBSERVABLES:
        np.testing.assert_allclose(result[key], golden[key], rtol=RTOL, atol=ATOL,
                                   err_msg=f"observable '{key}' drifted from golden")
