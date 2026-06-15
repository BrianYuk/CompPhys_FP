"""Tests for the shared helper modules (config, observables).

The grid test is load-bearing: it proves the single `standard_temperature_grid()`
reproduces the per-script grids exactly, so consolidating them changes no results.
"""

import numpy as np

from ising_mc import config, observables


def test_standard_grid_matches_legacy_concatenation():
    legacy = np.concatenate([
        np.linspace(1.50, 2.10, 7, endpoint=False),
        np.linspace(2.10, 2.45, 15),
        np.linspace(2.50, 3.30, 9),
    ])
    grid = config.standard_temperature_grid()
    np.testing.assert_array_equal(grid, legacy)
    # The baseline script wrote 2.45 + 0.05 for the disordered-tail start; confirm
    # that is bit-identical to the 2.50 the others used, so all 31-point grids agree.
    assert 2.45 + 0.05 == 2.50
    assert grid.size == 31
    assert grid.min() == 1.50
    assert grid.max() == 3.30


def test_standard_grid_returns_fresh_array():
    a = config.standard_temperature_grid()
    a[0] = -999.0
    assert config.standard_temperature_grid()[0] == 1.50


def test_extract_stacks_observable():
    results = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    np.testing.assert_array_equal(observables.extract(results, "x"),
                                  np.array([1.0, 2.0, 3.0]))


def test_peak_info_finds_maximum():
    T = np.array([1.0, 2.0, 3.0])
    chi = np.array([0.5, 4.0, 1.0])
    assert observables.peak_info(T, chi) == (2.0, 4.0)


def test_peak_parabolic_recovers_subgrid_vertex():
    # Exact parabola chi = 10 - 400*(T - 2.27)^2 whose vertex (2.27, 10) lies
    # between grid points; the fit must recover it even though argmax cannot.
    T = np.array([2.20, 2.25, 2.30, 2.35, 2.40])
    chi = 10.0 - 400.0 * (T - 2.27) ** 2
    T_peak, chi_peak = observables.peak_parabolic(T, chi)
    assert np.isclose(T_peak, 2.27)
    assert np.isclose(chi_peak, 10.0)
    # The raw grid maximum is offset, proving the fit added sub-grid resolution.
    assert observables.peak_info(T, chi)[0] != T_peak


def test_peak_parabolic_falls_back_at_edge():
    # Monotone-rising chi: the maximum is unbracketed (last point), so there is
    # no concave vertex inside the data and we must return the grid argmax.
    T = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    chi = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert observables.peak_parabolic(T, chi) == (3.0, 5.0)


def test_peak_parabolic_falls_back_on_short_array():
    T = np.array([2.0, 2.5])
    chi = np.array([3.0, 4.0])
    assert observables.peak_parabolic(T, chi) == (2.5, 4.0)


def test_mean_and_sem_two_runs():
    # Two repeats; each column's values differ by 2, so sample std (ddof=1) is
    # sqrt(2) and sem = std / sqrt(2) = 1.
    stack = np.array([[1.0, 2.0, 3.0],
                      [3.0, 4.0, 5.0]])
    mean, sem = observables.mean_and_sem(stack)
    np.testing.assert_allclose(mean, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(sem, [1.0, 1.0, 1.0])


def test_mean_and_sem_single_run_has_zero_sem():
    # One run -> no spread to measure -> sem returned as zeros (not "certain").
    stack = np.array([[1.0, 2.0, 3.0]])
    mean, sem = observables.mean_and_sem(stack)
    np.testing.assert_allclose(mean, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(sem, np.zeros(3))


def test_edge_abs_m_returns_first_and_last():
    T = np.array([1.0, 2.0, 3.0])
    m = np.array([0.9, 0.5, 0.1])
    assert observables.edge_abs_m(T, m) == (0.9, 0.1)


def test_estimate_T_half_interpolates_crossing():
    T = np.array([1.0, 2.0, 3.0])
    m = np.array([0.9, 0.6, 0.2])   # crosses 0.5 between T=2 and T=3
    # frac = (0.6 - 0.5)/(0.6 - 0.2) = 0.25 -> 2 + 0.25*(3-2) = 2.25
    assert observables.estimate_T_half(T, m) == 2.25


def test_estimate_T_half_nan_when_already_below():
    T = np.array([1.0, 2.0, 3.0])
    m = np.array([0.4, 0.3, 0.1])
    assert np.isnan(observables.estimate_T_half(T, m))


def test_estimate_T_half_nan_when_never_crosses():
    T = np.array([1.0, 2.0, 3.0])
    m = np.array([0.95, 0.9, 0.8])
    assert np.isnan(observables.estimate_T_half(T, m))


def test_observables_from_samples_matches_inline_formulas():
    rng = np.random.default_rng(0)
    mags = rng.uniform(-1.0, 1.0, size=50)
    energies = rng.uniform(-2.0, 0.0, size=50)
    n_sites, T = 400, 2.27

    result = observables.observables_from_samples(mags, energies, n_sites, T)

    # Recompute exactly as the engine did inline, to lock numerical equivalence.
    m_mean = mags.mean()
    abs_m = np.abs(mags).mean()
    m2 = (mags ** 2).mean()
    e_mean = energies.mean()
    e2 = (energies ** 2).mean()
    assert result["m_mean"] == m_mean
    assert result["abs_m"] == abs_m
    assert result["m2"] == m2
    assert result["energy"] == e_mean
    assert result["chi_abs"] == n_sites * (m2 - abs_m ** 2) / T
    assert result["chi_signed"] == n_sites * (m2 - m_mean ** 2) / T
    assert result["cv"] == n_sites * (e2 - e_mean ** 2) / T ** 2
