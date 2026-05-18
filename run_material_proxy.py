"""
Material-proxy study: L=20, J = [0.75, 1.00, 1.25], h/J = 0.00

In reduced units (T/J) the Ising model curves overlap for different J.
Plotting against *absolute* temperature T_abs = (T/J) * J reveals that
higher J shifts the transition to a higher absolute temperature, mimicking
a "stiffer" magnetic material.

This is a simplified proxy — not a real named material.

Expected transition temperatures (Tc ≈ 2.269 * J):
    J = 0.75  ->  Tc_abs ≈ 1.70
    J = 1.00  ->  Tc_abs ≈ 2.27
    J = 1.25  ->  Tc_abs ≈ 2.84

Outputs (outputs/material_proxy/):
    plot_magnetization_vs_absolute_T_by_J.png
    plot_susceptibility_vs_absolute_T_by_J.png
    material_proxy_results.npz
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from ising import sweep_temperatures

OUT = "./outputs/material_proxy"
os.makedirs(OUT, exist_ok=True)

L = 20
H_OVER_J = 0.0
T_C_REDUCED = 2.0 / np.log(1.0 + np.sqrt(2.0))   # ≈ 2.26919

T_low  = np.linspace(1.50, 2.10, 7,  endpoint=False)
T_crit = np.linspace(2.10, 2.45, 15)
T_high = np.linspace(2.50, 3.30, 9)
T_over_J = np.concatenate([T_low, T_crit, T_high])

J_VALUES = [0.75, 1.00, 1.25]
COLORS   = ["C0", "C1", "C2"]

# ----- run sweeps ----------------------------------------------------------
all_results = {}
for J in J_VALUES:
    print(f"Running J={J:.2f} ({len(T_over_J)} temperatures) ...")
    results, _ = sweep_temperatures(
        L=L, T_over_J_array=T_over_J, h_over_J=H_OVER_J,
        n_equil=2000, n_meas=5000, sample_every=5,
        anneal=True, seed=42,
    )
    all_results[J] = results

# ----- save raw data -------------------------------------------------------
save_dict = {"T_over_J": T_over_J, "L": L, "h_over_J": H_OVER_J}
for J in J_VALUES:
    T_arr   = np.array([r["T_over_J"] for r in all_results[J]])
    m_arr   = np.array([r["abs_m"]    for r in all_results[J]])
    chi_arr = np.array([r["chi_abs"]  for r in all_results[J]])
    T_abs   = T_arr * J
    key = f"{J:.2f}"
    save_dict[f"T_abs_J{key}"]   = T_abs
    save_dict[f"abs_m_J{key}"]   = m_arr
    save_dict[f"chi_abs_J{key}"] = chi_arr

np.savez(os.path.join(OUT, "material_proxy_results.npz"), **save_dict)

# ----- plot magnetization vs absolute T ------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for J, c in zip(J_VALUES, COLORS):
    T_arr  = np.array([r["T_over_J"] for r in all_results[J]])
    m_arr  = np.array([r["abs_m"]    for r in all_results[J]])
    T_abs  = T_arr * J
    Tc_abs = T_C_REDUCED * J
    ax.plot(T_abs, m_arr, "o-", color=c, lw=1.5, ms=4,
            label=f"J = {J:.2f}  ($T_c \\approx {Tc_abs:.2f}$)")
ax.set_xlabel(r"Absolute temperature $T_\mathrm{abs} = (T/J)\cdot J$")
ax.set_ylabel(r"$\langle|m|\rangle$")
ax.set_title(f"Magnetization: material-proxy comparison  (L={L}, h/J={H_OVER_J})")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "plot_magnetization_vs_absolute_T_by_J.png"), dpi=140)
plt.close(fig)

# ----- plot susceptibility vs absolute T -----------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for J, c in zip(J_VALUES, COLORS):
    T_arr   = np.array([r["T_over_J"] for r in all_results[J]])
    chi_arr = np.array([r["chi_abs"]  for r in all_results[J]])
    T_abs   = T_arr * J
    Tc_abs  = T_C_REDUCED * J
    ax.plot(T_abs, chi_arr, "^-", color=c, lw=1.5, ms=4,
            label=f"J = {J:.2f}  ($T_c \\approx {Tc_abs:.2f}$)")
ax.set_xlabel(r"Absolute temperature $T_\mathrm{abs} = (T/J)\cdot J$")
ax.set_ylabel(r"$\chi J$")
ax.set_title(f"Susceptibility: material-proxy comparison  (L={L}, h/J={H_OVER_J})")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "plot_susceptibility_vs_absolute_T_by_J.png"), dpi=140)
plt.close(fig)

print(f"Done. Results in {OUT}/")
