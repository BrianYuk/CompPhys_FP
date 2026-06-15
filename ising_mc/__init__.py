"""ising_mc — 2D Ising Monte Carlo library.

Importable engine and helpers for the experiment scripts (and the future
interactive interface). Import submodules explicitly so that lightweight,
numba-free modules (config, observables) don't pull in the JIT-compiled engine
unless it's actually needed:

    from ising_mc.ising import advance, sweep_temperatures, run_at_temperature
    from ising_mc import config, observables, plotting
"""
