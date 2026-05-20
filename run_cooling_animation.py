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
from matplotlib import animation

from ising import _metropolis_sweep

OUT_ANIM = "./outputs/animation"
os.makedirs(OUT_ANIM, exist_ok=True)

L = 50
J = 1.00
H_OVER_J = 0.0
T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))   # Onsager ≈ 2.26919

N_FRAMES = 120
T_START = 3.3
T_END = 1.5
SWEEPS_PER_FRAME = 4     # small number of sweeps per temperature step
SEED = 42

# ----- smooth cooling schedule (cosine ease, high -> low) ------------------
# A cosine ramp spends a little more "time" near the start and end, giving a
# smoother visual cool-down than a bare linear ramp.
s = np.linspace(0.0, 1.0, N_FRAMES)
ease = 0.5 * (1.0 - np.cos(np.pi * s))            # 0 -> 1, smooth
T_schedule = T_START + (T_END - T_START) * ease   # 3.3 -> 1.5

# ----- run the annealing trajectory ----------------------------------------
rng = np.random.default_rng(SEED)
lattice = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
N = L * L

print(f"Cooling L={L} lattice from T/J={T_START} to {T_END} "
      f"over {N_FRAMES} frames ({SWEEPS_PER_FRAME} sweeps/frame) ...")

frames = np.empty((N_FRAMES, L, L), dtype=np.int8)
for f in range(N_FRAMES):
    T = T_schedule[f]
    for _ in range(SWEEPS_PER_FRAME):
        ri = rng.integers(0, L, size=N)
        rj = rng.integers(0, L, size=N)
        ru = rng.random(size=N)
        _metropolis_sweep(lattice, T, H_OVER_J, L, ri, rj, ru)
    frames[f] = lattice

# ----- build the animation --------------------------------------------------
fig, ax = plt.subplots(figsize=(5.0, 5.4))
im = ax.imshow(frames[0], cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
title_obj = ax.set_title("")
ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Cooling / annealing trajectory  (NON-equilibrium)",
             fontsize=11, fontweight="bold")


def _update(k):
    im.set_array(frames[k])
    T = T_schedule[k]
    phase = "disordered" if T > T_C else "ordered"
    title_obj.set_text(
        f"L={L}, J={J}, h/J={H_OVER_J}  |  T/J = {T:.3f}  ({phase})"
    )
    return [im, title_obj]


anim = animation.FuncAnimation(
    fig, _update, frames=N_FRAMES, interval=80, blit=False,
)
fname = os.path.join(OUT_ANIM, "cooling_transition.gif")
anim.save(fname, writer="pillow", fps=12, dpi=110)
plt.close(fig)

print(f"Saved: {fname}")
print("Note: this is an annealing trajectory for visual explanation only — "
      "not an equilibrium measurement.")
