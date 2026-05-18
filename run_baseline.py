"""
Baseline experiment: L = 20, J = 1.00, h = 0, T/J in [1.5, 3.3].

Produces:
    plot_magnetization =     |m| vs T/J
    plot_energy =            E/J per site vs T/J
    plot_susceptibility =    chi*J vs T/J
    plot_heat_capacity =     Cv per site vs T/J
    snapshot_lowT =          spin lattice at T/J = 1.5
    snapshot_critical =      spin lattice at T/J = 2.27
    snapshot_highT =         spin lattice at T/J = 3.3
    spin_evolution =         animation near T_c
    baseline_results =       raw numerical results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from ising import sweep_temperatures, record_trajectory

# ----- configuration -------------------------------------------------------
OUT = "./outputs"
os.makedirs(OUT, exist_ok=True)

L = 20
J = 1.00           # physical scale; only matters if you want absolute T
H_OVER_J = 0.0
T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))   # Onsager: ≈ 2.26919

# Denser sampling near T_c
T_low  = np.linspace(1.50, 2.10, 7,   endpoint=False)
T_crit = np.linspace(2.10, 2.45, 15)
T_high = np.linspace(2.45 + 0.05, 3.30, 9)
T_over_J = np.concatenate([T_low, T_crit, T_high])
print(f"Sweeping {len(T_over_J)} temperatures from {T_over_J.min():.2f} to {T_over_J.max():.2f}")

snap_targets = [float(T_low[0]), float(T_crit[np.argmin(np.abs(T_crit - T_C))]), float(T_high[-1])]
print(f"Snapshot temperatures: low={snap_targets[0]:.3f}, "
      f"critical={snap_targets[1]:.3f}, high={snap_targets[2]:.3f}")

# ----- run sweep -----------------------------------------------------------
results, snap_lattices = sweep_temperatures(
    L=L,
    T_over_J_array=T_over_J,
    h_over_J=H_OVER_J,
    n_equil=2000,
    n_meas=6000,
    sample_every=5,
    anneal=True,
    seed=42,
    return_lattices_at=snap_targets,
)

T_arr   = np.array([r["T_over_J"] for r in results])
m_arr   = np.array([r["abs_m"]   for r in results])
e_arr   = np.array([r["energy"]  for r in results])
chi_arr = np.array([r["chi"]     for r in results])
cv_arr  = np.array([r["cv"]      for r in results])

np.savez(os.path.join(OUT, "baseline_results.npz"),
         T_over_J=T_arr, abs_m=m_arr, energy=e_arr, chi=chi_arr, cv=cv_arr,
         L=L, J=J, h_over_J=H_OVER_J, T_c_exact=T_C)

# ----- four observable plots ----------------------------------------------
def _format(ax, ylabel, title):
    ax.axvline(T_C, color="grey", ls="--", lw=1, alpha=0.7, label=f"$T_c/J = {T_C:.3f}$")
    ax.set_xlabel(r"$T/J$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(T_arr, m_arr, "o-", color="C0", lw=1.5, ms=5)
_format(ax, r"$\langle|m|\rangle$", f"Magnetization | L={L}, J={J}, h/J={H_OVER_J}")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "plot_magnetization.png"), dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(T_arr, e_arr, "s-", color="C3", lw=1.5, ms=5)
_format(ax, r"$\langle E\rangle/(JN)$", f"Energy per site | L={L}, J={J}, h/J={H_OVER_J}")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "plot_energy.png"), dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(T_arr, chi_arr, "^-", color="C2", lw=1.5, ms=5)
_format(ax, r"$\chi J$", f"Susceptibility | L={L}, J={J}, h/J={H_OVER_J}")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "plot_susceptibility.png"), dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(T_arr, cv_arr, "d-", color="C1", lw=1.5, ms=5)
_format(ax, r"$C_v$ per site", f"Heat capacity | L={L}, J={J}, h/J={H_OVER_J}")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "plot_heat_capacity.png"), dpi=140); plt.close(fig)

# ----- spin lattice snapshots ---------------------------------------------
labels = {snap_targets[0]: ("snapshot_lowT.png", "Low T"),
          snap_targets[1]: ("snapshot_critical.png", "Near $T_c$"),
          snap_targets[2]: ("snapshot_highT.png", "High T")}
for T, lat in snap_lattices.items():
    fname, label = labels[T]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(lat, cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(f"{label}: T/J = {T:.3f} | L={L}, h/J={H_OVER_J}")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=140); plt.close(fig)

# Combined snapshot figure as bonus
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
for ax, T in zip(axes, snap_targets):
    ax.imshow(snap_lattices[T], cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(f"T/J = {T:.3f}")
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Spin configurations | L={L}, J={J}, h/J={H_OVER_J}")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "snapshots_combined.png"), dpi=140); plt.close(fig)

# ----- animation near T_c -------------------------------------------------
print("Recording animation trajectory near T_c ...")
frames = record_trajectory(L=L, T_over_J=T_C, h_over_J=H_OVER_J,
                           n_equil=500, n_frames=120, sweeps_per_frame=2, seed=1)

fig, ax = plt.subplots(figsize=(4.5, 4.5))
im = ax.imshow(frames[0], cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
title = ax.set_title("")
ax.set_xticks([]); ax.set_yticks([])

def _update(k):
    im.set_array(frames[k])
    title.set_text(f"Spin evolution near $T_c$ | sweep {k * 2}")
    return [im, title]

anim = animation.FuncAnimation(fig, _update, frames=len(frames), interval=80, blit=False)
anim.save(os.path.join(OUT, "spin_evolution.gif"), writer="pillow", fps=12, dpi=110)
plt.close(fig)
print("Done.")
