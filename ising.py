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
"""

import numpy as np
from numba import njit


# ----- Numba-accelerated kernels ------------------------------------------

@njit(cache=True, fastmath=True)
def _metropolis_sweep(lattice, T_over_J, h_over_J, L, rand_i, rand_j, rand_u):
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
        # dE in units of J: dE/J = 2 s (nb + h/J)
        dE_over_J = 2.0 * s * (nb + h_over_J)
        dE_over_T = dE_over_J * inv_TJ
        if dE_over_T <= 0.0 or rand_u[k] < np.exp(-dE_over_T):
            lattice[i, j] = -s


@njit(cache=True, fastmath=True)
def _energy_per_site(lattice, h_over_J, L):
    """Energy per site, in units of J (so this returns e/J)."""
    E = 0.0
    for i in range(L):
        for j in range(L):
            s = lattice[i, j]
            # Count each pair once: only right and down neighbours
            E -= s * (lattice[(i + 1) % L, j] + lattice[i, (j + 1) % L])
            E -= h_over_J * s
    return E / (L * L)


@njit(cache=True, fastmath=True)
def _magnetization_per_site(lattice, L):
    s = 0.0
    for i in range(L):
        for j in range(L):
            s += lattice[i, j]
    return s / (L * L)


# ----- High-level driver ---------------------------------------------------

def run_at_temperature(lattice, T_over_J, h_over_J,
                       n_equil=1500, n_meas=4000, sample_every=5, rng=None):
    """
    Run MC at a single (T/J, h/J) starting from `lattice` (modified in-place).
    Returns a dict of measured observables and the time-series.
    """
    if rng is None:
        rng = np.random.default_rng()
    L = lattice.shape[0]
    N = L * L

    # equilibration
    for _ in range(n_equil):
        ri = rng.integers(0, L, size=N)
        rj = rng.integers(0, L, size=N)
        ru = rng.random(size=N)
        _metropolis_sweep(lattice, T_over_J, h_over_J, L, ri, rj, ru)

    # measurement
    mags = np.empty(n_meas // sample_every, dtype=np.float64)
    energies = np.empty_like(mags)
    idx = 0
    for step in range(n_meas):
        ri = rng.integers(0, L, size=N)
        rj = rng.integers(0, L, size=N)
        ru = rng.random(size=N)
        _metropolis_sweep(lattice, T_over_J, h_over_J, L, ri, rj, ru)
        if step % sample_every == 0 and idx < mags.size:
            mags[idx] = _magnetization_per_site(lattice, L)
            energies[idx] = _energy_per_site(lattice, h_over_J, L)
            idx += 1

    mags = mags[:idx]
    energies = energies[:idx]

    abs_m = np.abs(mags).mean()
    m2 = (mags ** 2).mean()
    e_mean = energies.mean()
    e2 = (energies ** 2).mean()

    chi = N * (m2 - abs_m ** 2) / T_over_J            # chi * J
    cv = N * (e2 - e_mean ** 2) / T_over_J ** 2       # heat capacity per site

    return {
        "T_over_J": T_over_J,
        "h_over_J": h_over_J,
        "L": L,
        "abs_m": abs_m,
        "m2": m2,
        "energy": e_mean,
        "chi": chi,
        "cv": cv,
        "mags": mags,
        "energies": energies,
    }


def sweep_temperatures(L, T_over_J_array, h_over_J,
                       n_equil=1500, n_meas=4000, sample_every=5,
                       anneal=True, seed=0, return_lattices_at=None):
    """
    Sweep over an array of T/J values. If anneal=True, the lattice from the
    previous T is reused as the starting state for the next T (descending T).
    This speeds equilibration enormously in the ordered phase.

    `return_lattices_at`: optional list of T/J values; the final lattice at
    the closest sweep point will be saved and returned alongside results.
    """
    rng = np.random.default_rng(seed)

    # Sort descending so we anneal from disordered -> ordered
    T_order = np.argsort(T_over_J_array)[::-1]
    T_sorted = T_over_J_array[T_order]

    # Start from random (high-T-like) configuration
    lattice = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))

    results = []
    saved_lattices = {}
    targets = set(return_lattices_at) if return_lattices_at else set()

    for T in T_sorted:
        if not anneal:
            lattice = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
        res = run_at_temperature(
            lattice, T, h_over_J,
            n_equil=n_equil, n_meas=n_meas, sample_every=sample_every, rng=rng,
        )
        results.append(res)
        if T in targets:
            saved_lattices[T] = lattice.copy()

    # Re-sort results into the original (ascending) order for cleaner plots
    results_sorted = sorted(results, key=lambda r: r["T_over_J"])
    return results_sorted, saved_lattices


def record_trajectory(L, T_over_J, h_over_J,
                      n_equil=500, n_frames=120, sweeps_per_frame=2, seed=0):
    """
    Run a single simulation and save lattice snapshots every `sweeps_per_frame`
    sweeps after equilibration. Useful for the animation.
    """
    rng = np.random.default_rng(seed)
    lattice = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
    N = L * L

    # equilibrate
    for _ in range(n_equil):
        ri = rng.integers(0, L, size=N)
        rj = rng.integers(0, L, size=N)
        ru = rng.random(size=N)
        _metropolis_sweep(lattice, T_over_J, h_over_J, L, ri, rj, ru)

    frames = np.empty((n_frames, L, L), dtype=np.int8)
    for f in range(n_frames):
        for _ in range(sweeps_per_frame):
            ri = rng.integers(0, L, size=N)
            rj = rng.integers(0, L, size=N)
            ru = rng.random(size=N)
            _metropolis_sweep(lattice, T_over_J, h_over_J, L, ri, rj, ru)
        frames[f] = lattice
    return frames
