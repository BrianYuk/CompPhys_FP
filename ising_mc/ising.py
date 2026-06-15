"""
2D Ising Model - Monte Carlo simulation (Metropolis algorithm)
==============================================================

Hamiltonian:    H = -J * sum_<ij> s_i s_j  -  h * sum_i s_i
Spin values:    s_i in {-1, +1}
Boundary:       periodic (toroidal)

Reduced units
-------------
We parametrize everything in T/J and h/J. The Metropolis acceptance only
depends on these ratios, since for a flip s -> -s:

    dE = 2 * J * s_i * (sum of 4 neighbors)  +  2 * h * s_i
    dE / T = 2 * s_i * (neighbors + h/J) / (T/J)

So the physics is entirely determined by T/J and h/J. The value of J only
sets the overall energy scale (and the *absolute* temperature T = (T/J) * J).
At h = 0 in particular, all curves of m, chi, Cv versus T/J are independent
of J — this is a useful sanity check.

Observables (per site, reported in reduced units)
-------------------------------------------------
  e/J        = <H>/(N J)
  |m|        = <|sum_i s_i|>/N
  chi * J    = N * (<m^2> - <|m|>^2) / (T/J)
  Cv         = N * (<e^2> - <e>^2) / (T/J)^2     (per site, dimensionless)

Public API
----------
  advance(...)            advance a lattice by N Metropolis sweeps (the stepping
                          primitive; a caller may change T/J or h/J between calls)
  run_at_temperature(...) equilibrate + measure observables at one (T/J, h/J)
  sweep_temperatures(...) scan a T/J grid with optional annealing
  record_trajectory(...)  collect a frame sequence for animation
"""

import numpy as np
from numba import njit

from .observables import observables_from_samples


# ----- Numba-accelerated kernels ------------------------------------------

@njit(cache=True, fastmath=True)
def _metropolis_sweep(lattice: np.ndarray, T_over_J: float, h_over_J: float,
                      L: int, rand_i: np.ndarray, rand_j: np.ndarray,
                      rand_u: np.ndarray) -> None:
    """
    One Metropolis sweep = L*L single-spin-flip attempts.
    Pre-generated random arrays are passed in for reproducibility & speed.
    """
    inv_TJ = 1.0 / T_over_J
    for k in range(L * L):
        i = rand_i[k]
        j = rand_j[k]
        s = lattice[i, j]
        nb = (lattice[(i + 1) % L, j] + lattice[(i - 1) % L, j]
              + lattice[i, (j + 1) % L] + lattice[i, (j - 1) % L])
        # Energy cost of flipping this spin, in units of J: dE/J = 2 s (nb + h/J).
        dE_over_J = 2.0 * s * (nb + h_over_J)
        dE_over_T = dE_over_J * inv_TJ
        # Metropolis rule: always accept a downhill flip; accept an uphill flip
        # with probability exp(-dE/T).
        if dE_over_T <= 0.0 or rand_u[k] < np.exp(-dE_over_T):
            lattice[i, j] = -s


@njit(cache=True, fastmath=True)
def _energy_per_site(lattice: np.ndarray, h_over_J: float, L: int) -> float:
    """Energy per site, in units of J (so this returns e/J)."""
    E = 0.0
    for i in range(L):
        for j in range(L):
            s = lattice[i, j]
            # Count each bond once: only the right and down neighbours, so the
            # nearest-neighbour sum is not double counted.
            E -= s * (lattice[(i + 1) % L, j] + lattice[i, (j + 1) % L])
            E -= h_over_J * s
    return E / (L * L)


@njit(cache=True, fastmath=True)
def _magnetization_per_site(lattice: np.ndarray, L: int) -> float:
    """Signed magnetization per site, m = (sum_i s_i) / N."""
    s = 0.0
    for i in range(L):
        for j in range(L):
            s += lattice[i, j]
    return s / (L * L)


# ----- Stepping primitive ---------------------------------------------------

def advance(lattice: np.ndarray, T_over_J: float, h_over_J: float,
            n_sweeps: int, rng: np.random.Generator) -> None:
    """Advance `lattice` in place by `n_sweeps` Metropolis sweeps at (T/J, h/J).

    This is the single stepping primitive used everywhere. A caller may change
    T/J or h/J between calls (e.g. an annealing schedule, or an interactive UI)
    and resume from the current lattice state.

    Fresh random index/threshold arrays are drawn per sweep. The draw order
    (row indices, then column indices, then acceptance thresholds) is fixed: it
    defines the reproducible random stream for a given seed.
    """
    L = lattice.shape[0]
    n_sites = L * L
    for _ in range(n_sweeps):
        rand_i = rng.integers(0, L, size=n_sites)
        rand_j = rng.integers(0, L, size=n_sites)
        rand_u = rng.random(size=n_sites)
        _metropolis_sweep(lattice, T_over_J, h_over_J, L, rand_i, rand_j, rand_u)


# ----- High-level driver ---------------------------------------------------

def measure_at_temperature(lattice: np.ndarray, T_over_J: float, h_over_J: float,
                           n_equil: int = 1500, n_meas: int = 4000,
                           sample_every: int = 5,
                           rng: np.random.Generator | None = None
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Equilibrate, then sample the per-site magnetization and energy time series.

    Modifies `lattice` in place. Returns (magnetizations, energies): 1-D float64
    arrays holding the signed per-site magnetization and energy at each sample.
    Sampling every `sample_every` sweeps decorrelates successive measurements.
    """
    if rng is None:
        rng = np.random.default_rng()

    L = lattice.shape[0]
    advance(lattice, T_over_J, h_over_J, n_equil, rng)   # reach equilibrium first

    n_samples = n_meas // sample_every
    magnetizations = np.empty(n_samples, dtype=np.float64)
    energies = np.empty_like(magnetizations)
    idx = 0
    for step in range(n_meas):
        advance(lattice, T_over_J, h_over_J, 1, rng)
        if step % sample_every == 0 and idx < magnetizations.size:
            magnetizations[idx] = _magnetization_per_site(lattice, L)
            energies[idx] = _energy_per_site(lattice, h_over_J, L)
            idx += 1

    return magnetizations[:idx], energies[:idx]


def run_at_temperature(lattice: np.ndarray, T_over_J: float, h_over_J: float,
                       n_equil: int = 1500, n_meas: int = 4000,
                       sample_every: int = 5,
                       rng: np.random.Generator | None = None) -> dict:
    """
    Run MC at a single (T/J, h/J) starting from `lattice` (modified in-place).
    Returns a dict of measured observables and the underlying time-series.
    """
    if rng is None:
        rng = np.random.default_rng()

    magnetizations, energies = measure_at_temperature(
        lattice, T_over_J, h_over_J,
        n_equil=n_equil, n_meas=n_meas, sample_every=sample_every, rng=rng,
    )
    n_sites = lattice.shape[0] ** 2
    observables = observables_from_samples(magnetizations, energies, n_sites, T_over_J)

    return {
        "T_over_J": T_over_J,
        "h_over_J": h_over_J,
        "L": lattice.shape[0],
        **observables,           # m_mean, abs_m, m2, energy, chi_abs, chi_signed, cv
        "mags": magnetizations,
        "energies": energies,
    }


def _random_lattice(L: int, rng: np.random.Generator) -> np.ndarray:
    """Draw an L×L lattice of ±1 spins (a disordered, high-T-like start state)."""
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))


def _closest_sweep_temperature_for_targets(temperatures_descending: np.ndarray,
                                           targets: list) -> dict:
    """Map each requested target T/J to the closest simulated T/J (by argmin).

    Requested snapshot temperatures rarely equal a grid point exactly, so we key
    the result by the simulated float and remember which target it satisfies.
    """
    target_for_sweep_temperature = {}
    for target in targets:
        idx = int(np.argmin(np.abs(temperatures_descending - target)))
        target_for_sweep_temperature[float(temperatures_descending[idx])] = target
    return target_for_sweep_temperature


def sweep_temperatures(L: int, T_over_J_array: np.ndarray, h_over_J: float,
                       n_equil: int = 1500, n_meas: int = 4000,
                       sample_every: int = 5, anneal: bool = True,
                       seed: int = 0, return_lattices_at: list | None = None
                       ) -> tuple[list, dict]:
    """
    Sweep over an array of T/J values. If anneal=True, the lattice from the
    previous T is reused as the starting state for the next T (descending T).
    This speeds equilibration enormously in the ordered phase.

    `return_lattices_at`: optional list of T/J values; the final lattice at
    the closest sweep point will be saved and returned alongside results.
    """
    rng = np.random.default_rng(seed)

    # Sort descending so we anneal from disordered (high T) toward ordered (low T):
    # reusing the previous configuration is far cheaper than re-equilibrating from
    # a random start at each temperature.
    descending_order = np.argsort(T_over_J_array)[::-1]
    temperatures_descending = T_over_J_array[descending_order]

    lattice = _random_lattice(L, rng)

    target_for_sweep_temperature = {}
    if return_lattices_at:
        target_for_sweep_temperature = _closest_sweep_temperature_for_targets(
            temperatures_descending, return_lattices_at)

    results = []
    saved_lattices = {}
    for T in temperatures_descending:
        if not anneal:
            lattice = _random_lattice(L, rng)
        res = run_at_temperature(
            lattice, T, h_over_J,
            n_equil=n_equil, n_meas=n_meas, sample_every=sample_every, rng=rng,
        )
        results.append(res)
        if float(T) in target_for_sweep_temperature:
            saved_lattices[target_for_sweep_temperature[float(T)]] = lattice.copy()

    # Re-sort results into the original (ascending) order for cleaner plots.
    results_sorted = sorted(results, key=lambda r: r["T_over_J"])
    return results_sorted, saved_lattices


def record_trajectory(L: int, T_over_J: float, h_over_J: float,
                      n_equil: int = 500, n_frames: int = 120,
                      sweeps_per_frame: int = 2, seed: int = 0) -> np.ndarray:
    """
    Run a single simulation and save lattice snapshots every `sweeps_per_frame`
    sweeps after equilibration. Useful for the animation.
    """
    rng = np.random.default_rng(seed)
    lattice = _random_lattice(L, rng)
    advance(lattice, T_over_J, h_over_J, n_equil, rng)   # equilibrate first

    frames = np.empty((n_frames, L, L), dtype=np.int8)
    for f in range(n_frames):
        advance(lattice, T_over_J, h_over_J, sweeps_per_frame, rng)
        frames[f] = lattice
    return frames
