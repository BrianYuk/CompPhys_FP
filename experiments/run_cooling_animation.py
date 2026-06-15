"""
Cooling / annealing animation: L=50, J=1.00, h/J=0.00

Unlike the fixed-temperature animations produced by run_baseline.py
(spin_evolution_low_T / _critical / _high_T), this script slowly *cools*
the system from a disordered high-temperature state down through the
critical region into the ordered phase.  The audience can watch ordered
domains nucleate and grow as T/J is lowered.

IMPORTANT: this is a NON-EQUILIBRIUM annealing trajectory, not an
equilibrium measurement.  Each frame runs only a few Metropolis sweeps,
so the lattice lags the instantaneous temperature.  It is a visual aid;
the quantitative phase-transition results come from the fixed-T sweeps in
the other scripts.

Output (outputs/animation/):
    cooling_transition.gif
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from ising_mc import config, plotting
from ising_mc.ising import advance

L = 100
J = 1.00
H_OVER_J = 0.0
T_C = config.T_C    # Onsager exact T_c/J ≈ 2.26919

N_FRAMES = 120
T_START = 3.3
T_END = 1.5
SWEEPS_PER_FRAME = 4     # few sweeps per step, so the lattice lags the temperature
SEED = 42


def cooling_schedule():
    """Cosine ease from T_START down to T_END across N_FRAMES.

    A cosine ramp lingers slightly near the start and end, giving a smoother
    visual cool-down than a bare linear ramp.
    """
    s = np.linspace(0.0, 1.0, N_FRAMES)
    ease = 0.5 * (1.0 - np.cos(np.pi * s))            # 0 -> 1, smooth
    return T_START + (T_END - T_START) * ease         # 3.3 -> 1.5


def record_cooling_frames(T_schedule, rng):
    """Anneal a fresh lattice along `T_schedule`, snapshotting after each step."""
    # Disordered (high-T) starting configuration of random ±1 spins.
    lattice = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
    frames = np.empty((N_FRAMES, L, L), dtype=np.int8)
    for f in range(N_FRAMES):
        advance(lattice, T_schedule[f], H_OVER_J, SWEEPS_PER_FRAME, rng)
        frames[f] = lattice
    return frames


def main():
    out_anim = config.ensure_dir(config.output_dir("animation"))
    T_schedule = cooling_schedule()

    print(f"Cooling L={L} lattice from T/J={T_START} to {T_END} "
          f"over {N_FRAMES} frames ({SWEEPS_PER_FRAME} sweeps/frame) ...")
    rng = np.random.default_rng(SEED)
    frames = record_cooling_frames(T_schedule, rng)

    fig, ax = plt.subplots(figsize=(5.0, 5.4))
    im = ax.imshow(frames[0], cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
    title_obj = ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.suptitle("Cooling / annealing trajectory  (NON-equilibrium)",
                 fontsize=11, fontweight="bold")

    def update(k):
        im.set_array(frames[k])
        T = T_schedule[k]
        phase = "disordered" if T > T_C else "ordered"
        title_obj.set_text(
            f"L={L}, J={J}, h/J={H_OVER_J}  |  T/J = {T:.3f}  ({phase})")
        return [im, title_obj]

    fname = os.path.join(out_anim, "cooling_transition.gif")
    plotting.save_gif(fig, update, N_FRAMES, fname)
    print(f"Saved: {fname}")
    print("Note: this is an annealing trajectory for visual explanation only — "
          "not an equilibrium measurement.")


if __name__ == "__main__":
    main()
