"""
Lattice-size study: L = 10, 20, 50; J = 1.00; h/J = 0.00

Shows how finite-size effects round the phase transition.  Larger L
produces a sharper transition and a taller susceptibility peak.

Outputs (outputs/lattice_size/):
    plot_magnetization_by_L.png
    plot_susceptibility_by_L.png
    lattice_size_results.npz
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from ising import sweep_temperatures

OUT = "./outputs/lattice_size"
os.makedirs(OUT, exist_ok=True)

J = 1.00
H_OVER_J = 0.0
T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))   # Onsager ≈ 2.26919

# Denser sampling near T_c
T_low  = np.linspace(1.50, 2.10, 7,  endpoint=False)
T_crit = np.linspace(2.10, 2.45, 15)
T_high = np.linspace(2.50, 3.30, 9)
T_over_J = np.concatenate([T_low, T_crit, T_high])

SIZES  = [10, 20, 50]
COLORS = ["C0", "C1", "C2"]

# ----- run sweeps ----------------------------------------------------------
all_results = {}
for L in SIZES:
    print(f"Running L={L} ({len(T_over_J)} temperatures) ...")
    results, _ = sweep_temperatures(
        L=L, T_over_J_array=T_over_J, h_over_J=H_OVER_J,
        n_equil=2000, n_meas=5000, sample_every=5,
        anneal=True, seed=42,
    )
    all_results[L] = results

# ----- save raw data -------------------------------------------------------
save_dict = {
    "T_over_J": T_over_J,
    "J": J,
    "h_over_J": H_OVER_J,
    "T_c_exact": T_C,
}
for L in SIZES:
    T_arr   = np.array([r["T_over_J"] for r in all_results[L]])
    m_arr   = np.array([r["abs_m"]    for r in all_results[L]])
    chi_arr = np.array([r["chi_abs"]  for r in all_results[L]])
    save_dict[f"T_arr_L{L}"]   = T_arr
    save_dict[f"abs_m_L{L}"]   = m_arr
    save_dict[f"chi_abs_L{L}"] = chi_arr

np.savez(os.path.join(OUT, "lattice_size_results.npz"), **save_dict)

# ----- plot magnetization --------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for L, c in zip(SIZES, COLORS):
    T_arr = np.array([r["T_over_J"] for r in all_results[L]])
    m_arr = np.array([r["abs_m"]    for r in all_results[L]])
    ax.plot(T_arr, m_arr, "o-", color=c, lw=1.5, ms=4, label=f"L = {L}")
ax.axvline(T_C, color="grey", ls="--", lw=1, alpha=0.7,
           label=f"$T_c/J \\approx {T_C:.3f}$")
ax.set_xlabel(r"$T/J$")
ax.set_ylabel(r"$\langle|m|\rangle$")
ax.set_title(f"Magnetization vs lattice size  (J={J}, h/J={H_OVER_J})")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "plot_magnetization_by_L.png"), dpi=140)
plt.close(fig)

# ----- plot susceptibility -------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for L, c in zip(SIZES, COLORS):
    T_arr   = np.array([r["T_over_J"] for r in all_results[L]])
    chi_arr = np.array([r["chi_abs"]  for r in all_results[L]])
    ax.plot(T_arr, chi_arr, "^-", color=c, lw=1.5, ms=4, label=f"L = {L}")
ax.axvline(T_C, color="grey", ls="--", lw=1, alpha=0.7,
           label=f"$T_c/J \\approx {T_C:.3f}$")
ax.set_xlabel(r"$T/J$")
ax.set_ylabel(r"$\chi J$")
ax.set_title(f"Susceptibility vs lattice size  (J={J}, h/J={H_OVER_J})")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "plot_susceptibility_by_L.png"), dpi=140)
plt.close(fig)

print(f"Done. Results in {OUT}/")
