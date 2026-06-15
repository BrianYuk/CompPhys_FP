"""Tests for LiveSimulation (the GUI-free interactive driver).

The Tkinter app itself needs a display and is not unit-tested; all of its simulation
logic lives in LiveSimulation so it can be checked headlessly.
"""

import numpy as np

from ising_mc.interactive import LiveSimulation


def test_step_preserves_shape_and_spin_values():
    sim = LiveSimulation(L=16, seed=0)
    lattice, _ = sim.step(2.5, 0.0, 3)
    assert lattice.shape == (16, 16)
    assert lattice.dtype == np.int8
    assert set(np.unique(lattice)).issubset({-1, 1})


def test_abs_m_is_a_valid_order_parameter():
    sim = LiveSimulation(L=16, seed=1)
    # |m| is a per-site magnitude, so it must always lie in [0, 1].
    _, abs_m = sim.step(0.5, 0.0, 50)
    assert 0.0 <= abs_m <= 1.0


def test_same_seed_is_deterministic():
    a = LiveSimulation(L=12, seed=42)
    b = LiveSimulation(L=12, seed=42)
    for _ in range(3):
        la, _ = a.step(2.27, 0.0, 2)
        lb, _ = b.step(2.27, 0.0, 2)
    np.testing.assert_array_equal(la, lb)


def test_reset_rerandomizes_lattice():
    sim = LiveSimulation(L=20, seed=7)
    before = sim.lattice.copy()
    sim.reset()
    # A fresh 20x20 random draw is astronomically unlikely to equal the previous one.
    assert not np.array_equal(before, sim.lattice)
    assert sim.lattice.shape == before.shape
