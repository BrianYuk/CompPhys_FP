"""
Extended lattice-size study: L = 10, 20, 50, 100; J = 1.00; h/J = 0.00

This extends run_lattice_sizes.py by adding L = 100 to strengthen the
project's central theme — finite-size effects on the phase transition.
A larger lattice should show a sharper magnetization drop and a taller,
narrower susceptibility peak located closer to the Onsager value
T_c/J = 2.26919.

L = 100 is ONLY added here, to the focused one-variable lattice-size
comparison.  It is deliberately NOT included in the 27-condition full
grid (run_full_grid.py), to keep that verification sweep affordable.

Outputs (outputs/lattice_size_extended/):
    plot_magnetization_by_L_extended.png
    plot_susceptibility_by_L_extended.png
    lattice_size_extended_results.npz
    lattice_size_extended_summary.csv
"""

import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt

from ising import sweep_temperatures

OUT = "./outputs/lattice_size_extended"
os.makedirs(OUT, exist_ok=True)

J = 1.00
H_OVER_J = 0.0
T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))   # Onsager ≈ 2.26919

# Dense sampling near T_c (31 points, same as run_lattice_sizes.py)
T_low  = np.linspace(1.50, 2.10, 7,  endpoint=False)
T_crit = np.linspace(2.10, 2.45, 15)
T_high = np.linspace(2.50, 3.30, 9)
T_over_J = np.concatenate([T_low, T_crit, T_high])

SIZES  = [10, 20, 50, 100]
COLORS = {10: "C0", 20: "C1", 50: "C2", 100: "C3"}

# MC settings; L=100 is heavier so keep n_meas modest for a student laptop.
N_EQUIL = 2500
N_MEAS = 6000
SAMPLE_EVERY = 5
SEED = 42

# ----- run sweeps ----------------------------------------------------------
all_results = {}
runtimes = {}
for L in SIZES:
    print(f"Running L={L} ({len(T_over_J)} temperatures) ...")
    t0 = time.time()
    results, _ = sweep_temperatures(
        L=L, T_over_J_array=T_over_J, h_over_J=H_OVER_J,
        n_equil=N_EQUIL, n_meas=N_MEAS, sample_every=SAMPLE_EVERY,
        anneal=True, seed=SEED,
    )
    runtimes[L] = time.time() - t0
    all_results[L] = results
    print(f"  L={L} done in {runtimes[L]:.1f} s")

# ----- collect arrays + summary scalars ------------------------------------
summary = {}
save_dict = {"T_over_J": T_over_J, "J": J, "h_over_J": H_OVER_J,
             "T_c_exact": T_C, "sizes": np.array(SIZES)}
for L in SIZES:
    T_arr   = np.array([r["T_over_J"] for r in all_results[L]])
    m_arr   = np.array([r["abs_m"]    for r in all_results[L]])
    chi_arr = np.array([r["chi_abs"]  for r in all_results[L]])
    peak_idx = int(np.argmax(chi_arr))
    save_dict[f"T_arr_L{L}"]   = T_arr
    save_dict[f"abs_m_L{L}"]   = m_arr
    save_dict[f"chi_abs_L{L}"] = chi_arr
    summary[L] = {
        "peak_T_over_J": float(T_arr[peak_idx]),
        "max_chi_abs": float(chi_arr.max()),
        "low_T_abs_m": float(m_arr[np.argmin(T_arr)]),
        "high_T_abs_m": float(m_arr[np.argmax(T_arr)]),
        "runtime_s": runtimes[L],
    }

np.savez(os.path.join(OUT, "lattice_size_extended_results.npz"), **save_dict)

# ----- summary CSV ---------------------------------------------------------
csv_path = os.path.join(OUT, "lattice_size_extended_summary.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["L", "J", "h_over_J", "peak_chi_T_over_J", "max_chi_abs",
                "low_T_abs_m", "high_T_abs_m", "runtime_s"])
    for L in SIZES:
        s = summary[L]
        w.writerow([L, f"{J:.2f}", f"{H_OVER_J:.2f}",
                    f"{s['peak_T_over_J']:.4f}", f"{s['max_chi_abs']:.4f}",
                    f"{s['low_T_abs_m']:.4f}", f"{s['high_T_abs_m']:.4f}",
                    f"{s['runtime_s']:.1f}"])

# ----- plot magnetization --------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.2))
for L in SIZES:
    T_arr = np.array([r["T_over_J"] for r in all_results[L]])
    m_arr = np.array([r["abs_m"]    for r in all_results[L]])
    ax.plot(T_arr, m_arr, "o-", color=COLORS[L], lw=1.5, ms=4, label=f"L = {L}")
ax.axvline(T_C, color="grey", ls="--", lw=1, alpha=0.7,
           label=f"$T_c/J \\approx {T_C:.3f}$")
ax.set_xlabel(r"$T/J$")
ax.set_ylabel(r"$\langle|m|\rangle$")
ax.set_title(f"Magnetization vs lattice size — extended with L=100  "
             f"(J={J}, h/J={H_OVER_J})")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "plot_magnetization_by_L_extended.png"), dpi=140)
plt.close(fig)

# ----- plot susceptibility -------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.2))
for L in SIZES:
    T_arr   = np.array([r["T_over_J"] for r in all_results[L]])
    chi_arr = np.array([r["chi_abs"]  for r in all_results[L]])
    ax.plot(T_arr, chi_arr, "^-", color=COLORS[L], lw=1.5, ms=4, label=f"L = {L}")
ax.axvline(T_C, color="grey", ls="--", lw=1, alpha=0.7,
           label=f"$T_c/J \\approx {T_C:.3f}$")
ax.set_xlabel(r"$T/J$")
ax.set_ylabel(r"$\chi J$")
ax.set_title(f"Susceptibility vs lattice size — extended with L=100  "
             f"(J={J}, h/J={H_OVER_J})")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "plot_susceptibility_by_L_extended.png"), dpi=140)
plt.close(fig)

print(f"\nDone. Results in {OUT}/")
for L in SIZES:
    s = summary[L]
    print(f"  L={L:3d}: peak T/J={s['peak_T_over_J']:.3f}  "
          f"max chiJ={s['max_chi_abs']:.2f}  ({s['runtime_s']:.1f} s)")
