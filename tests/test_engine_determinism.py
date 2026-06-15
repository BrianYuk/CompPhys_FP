"""RNG-determinism contract. This is the primary guard for the engine refactor:
extracting the `advance()` primitive must not reorder or resize the random draws.
"""

import os

import numpy as np

from conftest import GOLDEN_OBSERVABLES
from ising_mc.ising import record_trajectory, sweep_temperatures


def test_same_seed_is_identical(tiny_sweep_kwargs):
    a, _ = sweep_temperatures(seed=7, **tiny_sweep_kwargs)
    b, _ = sweep_temperatures(seed=7, **tiny_sweep_kwargs)
    for key in GOLDEN_OBSERVABLES:
        np.testing.assert_array_equal(
            np.array([r[key] for r in a]), np.array([r[key] for r in b]))


def test_different_seed_differs(tiny_sweep_kwargs):
    a, _ = sweep_temperatures(seed=0, **tiny_sweep_kwargs)
    b, _ = sweep_temperatures(seed=1, **tiny_sweep_kwargs)
    differs = any(
        not np.array_equal(np.array([r[key] for r in a]),
                           np.array([r[key] for r in b]))
        for key in GOLDEN_OBSERVABLES
    )
    assert differs, "different seeds produced identical results — seed is ignored"


def test_seed42_matches_golden(golden_dir, tiny_sweep_kwargs):
    # The experiment scripts all use seed=42; lock that production path too.
    golden = np.load(os.path.join(golden_dir, "sweep_L4_seed42.npz"))
    results, _ = sweep_temperatures(seed=42, **tiny_sweep_kwargs)
    for key in GOLDEN_OBSERVABLES:
        actual = np.array([r[key] for r in results])
        np.testing.assert_allclose(actual, golden[key], rtol=1e-12, atol=1e-12)


def test_results_sorted_ascending(tiny_sweep_kwargs):
    # Every script relies on sweep_temperatures returning ascending-T results.
    results, _ = sweep_temperatures(seed=0, **tiny_sweep_kwargs)
    temperatures = [r["T_over_J"] for r in results]
    assert temperatures == sorted(temperatures)


def test_return_lattices_keyed_by_requested_target(tiny_sweep_kwargs):
    target = 1.5
    _, saved = sweep_temperatures(seed=0, return_lattices_at=[target],
                                  **tiny_sweep_kwargs)
    assert list(saved.keys()) == [target]
    assert saved[target].shape == (tiny_sweep_kwargs["L"], tiny_sweep_kwargs["L"])


def test_record_trajectory_matches_golden(golden_dir):
    golden = np.load(os.path.join(golden_dir, "trajectory_L6_seed0.npz"))["frames"]
    frames = record_trajectory(L=6, T_over_J=2.27, h_over_J=0.0,
                               n_equil=10, n_frames=4, sweeps_per_frame=2, seed=0)
    np.testing.assert_array_equal(frames, golden)
    assert frames.shape == (4, 6, 6)
    assert frames.dtype == np.int8
    assert set(np.unique(frames)).issubset({-1, 1})


def test_record_trajectory_deterministic():
    kwargs = dict(L=6, T_over_J=2.27, h_over_J=0.0,
                  n_equil=10, n_frames=4, sweeps_per_frame=2, seed=0)
    np.testing.assert_array_equal(record_trajectory(**kwargs),
                                  record_trajectory(**kwargs))
