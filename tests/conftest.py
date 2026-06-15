"""Shared fixtures for the characterization test suite.

These tests lock the engine's current behavior so the clean-code refactor cannot
silently change the Monte Carlo results or shift the random-number stream.
"""

import os

import numpy as np
import pytest


# Tiny, fully seeded run parameters — fast enough that the whole suite stays well
# under a few seconds, but exercise every code path the scripts rely on.
TINY_L = 4
TINY_TEMPERATURES = np.array([1.5, 2.27, 3.3])   # spans ordered / near-Tc / disordered
TINY_N_EQUIL = 20
TINY_N_MEAS = 40
TINY_SAMPLE_EVERY = 5

# Scalar observables stored as golden reference arrays (one value per temperature).
GOLDEN_OBSERVABLES = (
    "T_over_J", "m_mean", "abs_m", "m2", "energy", "chi_abs", "chi_signed", "cv",
)


@pytest.fixture
def golden_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")


@pytest.fixture
def tiny_sweep_kwargs():
    """Keyword args for the canonical tiny seeded sweep (seed supplied per test)."""
    return dict(
        L=TINY_L,
        T_over_J_array=TINY_TEMPERATURES.copy(),
        h_over_J=0.0,
        n_equil=TINY_N_EQUIL,
        n_meas=TINY_N_MEAS,
        sample_every=TINY_SAMPLE_EVERY,
        anneal=True,
    )
