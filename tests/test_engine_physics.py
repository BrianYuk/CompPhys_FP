"""Physics sanity checks. Looser than the golden locks — these catch gross
regressions even if a golden were regenerated incorrectly, and assert the
Ising model's known qualitative behavior.
"""

import numpy as np

from ising_mc.ising import run_at_temperature


def _run_point(T_over_J, L=8, n_equil=400, n_meas=800, seed=0):
    rng = np.random.default_rng(seed)
    lattice = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
    return run_at_temperature(lattice, T_over_J=T_over_J, h_over_J=0.0,
                              n_equil=n_equil, n_meas=n_meas,
                              sample_every=5, rng=rng)


def test_low_temperature_orders():
    # Deep in the ordered phase (T/J = 1.0 << T_c) spins align: <|m|> -> 1.
    assert _run_point(1.0)["abs_m"] > 0.85


def test_high_temperature_disorders():
    # Well above T_c (T/J = 5.0) thermal noise wins: <|m|> -> 0.
    assert _run_point(5.0)["abs_m"] < 0.30


def test_abs_m_within_unit_interval():
    # |m| per site is bounded by the saturation magnetization.
    for T in (1.0, 2.27, 5.0):
        abs_m = _run_point(T)["abs_m"]
        assert 0.0 <= abs_m <= 1.0


def test_energy_within_bonds_bound():
    # At h=0 each site has 2 counted nearest-neighbour bonds, so e/J in [-2, 2].
    for T in (1.0, 2.27, 5.0):
        energy = _run_point(T)["energy"]
        assert -2.0 <= energy <= 2.0


def test_susceptibility_and_heat_capacity_nonnegative():
    # chi and Cv are variances divided by positive powers of T — never negative.
    for T in (1.0, 2.27, 5.0):
        result = _run_point(T)
        assert result["chi_abs"] >= 0.0
        assert result["cv"] >= 0.0
