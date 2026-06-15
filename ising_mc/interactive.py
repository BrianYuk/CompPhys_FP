"""Interactive driver for the 2D Ising engine.

`LiveSimulation` wraps the engine's `advance()` stepping primitive so a UI can hold
one simulation, step it forward a few sweeps at a time, and change T/J or h/J between
steps — the lattice keeps evolving from wherever it is rather than restarting. This
module is deliberately free of any GUI toolkit, so the simulation logic stays
importable and testable without a display; the Tkinter app in
experiments/run_interactive.py is a thin shell over it.

Reduced units throughout (T/J, h/J), matching the engine.
"""

import numpy as np

from ising_mc.ising import advance


class LiveSimulation:
    """A mutable lattice + RNG you can step forward at live-changeable (T/J, h/J).

    The UI calls `step()` once per displayed frame with the current control values and
    renders the returned lattice. State (lattice, rng) persists across calls, so
    changing T/J or h/J between steps evolves the *current* configuration rather than
    starting over.
    """

    def __init__(self, L: int, seed: int = 0):
        self.L = L
        self.rng = np.random.default_rng(seed)
        self.lattice = self._random_lattice()
        self._warm_up()

    def _random_lattice(self) -> np.ndarray:
        # Disordered (high-T-like) start: each site is an independent ±1 spin.
        return self.rng.choice(np.array([-1, 1], dtype=np.int8), size=(self.L, self.L))

    def _warm_up(self) -> None:
        # One throwaway sweep on a scratch lattice so numba compiles the kernel now,
        # off the UI's hot path, instead of stalling the first user-visible frame.
        scratch = self._random_lattice()
        advance(scratch, 2.5, 0.0, 1, self.rng)

    def step(self, T_over_J: float, h_over_J: float, n_sweeps: int):
        """Advance the lattice `n_sweeps` at (T/J, h/J); return (lattice, abs_m).

        abs_m = |⟨s_i⟩| is the per-site |magnetization|, the h=0 order parameter: for
        ±1 spins the lattice mean already equals the signed m, so its magnitude is |m|.
        """
        advance(self.lattice, T_over_J, h_over_J, n_sweeps, self.rng)
        abs_m = float(np.abs(self.lattice.mean()))
        return self.lattice, abs_m

    def reset(self) -> None:
        """Re-randomize to a fresh disordered start (the RNG stream continues)."""
        self.lattice = self._random_lattice()
