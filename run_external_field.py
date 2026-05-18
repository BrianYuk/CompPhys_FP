"""
External-field study: L=20, J=1.00, h/J = [0.00, 0.15]

h/J = 0.15 breaks spin-flip symmetry, biasing magnetization in the positive
direction and rounding the transition.  This is NOT a clean critical-temperature
shift; it is field-biased behavior where the order parameter no longer vanishes
at the h=0 Tc.

Signed magnetization <m> is the appropriate observable here because the field
selects a preferred direction.  chi_signed = N(<m^2> - <m>^2)/T captures
fluctuations around the field-biased mean.

Outputs (outputs/external_field/):
    plot_signed_magnetization_by_h.png
    plot_abs_magnetization_by_h.png
    plot_susceptibility_by_h.png
    external_field_results.npz
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from ising import sweep_temperatures

OUT = "./outputs/external_field"
os.makedirs(OUT, exist_ok=True)

L = 20
J = 1.00
T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))   # Onsager ≈ 2.26919

T_low  = np.linspace(1.50, 2.10, 7,  endpoint=False)
T_crit = np.linspace(2.10, 2.45, 15)
T_high = np.linspace(2.50, 3.30, 9)
T_over_J = np.concatenate([T_low, T_crit, T_high])

H_VALUES = [0.00, 0.15]
COLORS   = ["C0", "C1"]

# ----- run sweeps ----------------------------------------------------------
all_results = {}
for h in H_VALUES:
    print(f"Running h/J={h:.2f} ({len(T_over_J)} temperatures) ...")
    results, _ = sweep_temperatures(
        L=L, T_over_J_array=T_over_J, h_over_J=h,
        n_equil=2000, n_meas=5000, sample_every=5,
        anneal=True, seed=42,
    )
    all_results[h] = results

# ----- save raw data -------------------------------------------------------
save_dict = {"T_over_J": T_over_J, "L": L, "J": J, "T_c_exact": T_C}
for h in H_VALUES:
    T_arr      = np.array([r["T_over_J"]   for r in all_results[h]])
    m_mean     = np.array([r["m_mean"]     for r in all_results[h]])
    abs_m      = np.array([r["abs_m"]      for r in all_results[h]])
    chi_signed = np.array([r["chi_signed"] for r in all_results[h]])
    chi_abs    = np.array([r["chi_abs"]    for r in all_results[h]])
    key = f"{h:.2f}"
    save_dict[f"T_arr_h{key}"]      = T_arr
    save_dict[f"m_mean_h{key}"]     = m_mean
    save_dict[f"abs_m_h{key}"]      = abs_m
    save_dict[f"chi_signed_h{key}"] = chi_signed
    save_dict[f"chi_abs_h{key}"]    = chi_abs

np.savez(os.path.join(OUT, "external_field_results.npz"), **save_dict)

# ----- plot signed magnetization ------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for h, c in zip(H_VALUES, COLORS):
    T_arr  = np.array([r["T_over_J"] for r in all_results[h]])
    m_mean = np.array([r["m_mean"]   for r in all_results[h]])
    ax.plot(T_arr, m_mean, "o-", color=c, lw=1.5, ms=4, label=f"h/J = {h:.2f}")
ax.axvline(T_C, color="grey", ls="--", lw=1, alpha=0.7,
           label=f"$T_c/J \\approx {T_C:.3f}$ (h=0)")
ax.set_xlabel(r"$T/J$")
ax.set_ylabel(r"$\langle m \rangle$")
ax.set_title(f"Signed magnetization  (L={L}, J={J})")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "plot_signed_magnetization_by_h.png"), dpi=140)
plt.close(fig)

# ----- plot absolute magnetization ----------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for h, c in zip(H_VALUES, COLORS):
    T_arr = np.array([r["T_over_J"] for r in all_results[h]])
    abs_m = np.array([r["abs_m"]    for r in all_results[h]])
    ax.plot(T_arr, abs_m, "o-", color=c, lw=1.5, ms=4, label=f"h/J = {h:.2f}")
ax.axvline(T_C, color="grey", ls="--", lw=1, alpha=0.7,
           label=f"$T_c/J \\approx {T_C:.3f}$ (h=0)")
ax.set_xlabel(r"$T/J$")
ax.set_ylabel(r"$\langle|m|\rangle$")
ax.set_title(f"Absolute magnetization  (L={L}, J={J})")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "plot_abs_magnetization_by_h.png"), dpi=140)
plt.close(fig)

# ----- plot susceptibility (signed) ----------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for h, c in zip(H_VALUES, COLORS):
    T_arr      = np.array([r["T_over_J"]   for r in all_results[h]])
    chi_signed = np.array([r["chi_signed"] for r in all_results[h]])
    ax.plot(T_arr, chi_signed, "^-", color=c, lw=1.5, ms=4, label=f"h/J = {h:.2f}")
ax.axvline(T_C, color="grey", ls="--", lw=1, alpha=0.7,
           label=f"$T_c/J \\approx {T_C:.3f}$ (h=0)")
ax.set_xlabel(r"$T/J$")
ax.set_ylabel(r"$\chi_\mathrm{signed}\, J$")
ax.set_title(f"Susceptibility (signed)  (L={L}, J={J})")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "plot_susceptibility_by_h.png"), dpi=140)
plt.close(fig)

print(f"Done. Results in {OUT}/")
