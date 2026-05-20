"""
Full 27-condition parameter grid: L x J x (h/J)

    L   = [10, 20, 50]
    J   = [0.75, 1.00, 1.25]
    h/J = [0.00, 0.15, 0.50]   ->  3 x 3 x 3 = 27 conditions

This grid is a VERIFICATION sweep.  The main one-variable-at-a-time
analysis lives in run_lattice_sizes.py / run_material_proxy.py /
run_external_field.py.  Here we confirm the expected qualitative trends
hold jointly:

  * larger L  -> sharper transition / taller susceptibility peak
  * higher J  -> transition shifted to higher *absolute* temperature
  * higher h/J -> magnetization biased, transition rounded / crossover

Interpretation rules:
  * J = 0.75/1.00/1.25 are exchange-coupling *proxies*, not real materials.
  * For h/J = 0.00 the susceptibility peak is a pseudo-critical indicator
    (finite-size rounded Onsager transition, T_c/J = 2.26919).
  * For h/J = 0.15 and 0.50 there is NO true critical temperature; the
    susceptibility peak is a field-biased crossover indicator only.

Outputs (outputs/full_grid/):
    full_grid_results.npz
    full_grid_summary.csv
    plot_full_grid_magnetization_small_multiples.png
    plot_full_grid_susceptibility_small_multiples.png
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from ising import sweep_temperatures

OUT = "./outputs/full_grid"
os.makedirs(OUT, exist_ok=True)

T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))   # Onsager ≈ 2.26919

# Dense temperature sampling near T_c (31 points, same as the other scripts)
T_low  = np.linspace(1.50, 2.10, 7,  endpoint=False)
T_crit = np.linspace(2.10, 2.45, 15)
T_high = np.linspace(2.50, 3.30, 9)
T_OVER_J = np.concatenate([T_low, T_crit, T_high])

L_VALUES = [10, 20, 50]
J_VALUES = [0.75, 1.00, 1.25]
H_VALUES = [0.00, 0.15, 0.50]

# MC settings (kept modest so the 27-cell grid runs on a student laptop)
N_EQUIL = 2000
N_MEAS = 5000
SAMPLE_EVERY = 5
SEED = 42


def estimate_T_half(T_arr, abs_m_arr):
    """
    Interpolated T/J where <|m|> first drops below 0.5 (descending in m).
    Returns NaN if <|m|> never crosses 0.5 within the sampled range
    (e.g. strong-field cases where it stays biased high).
    """
    T_arr = np.asarray(T_arr, dtype=float)
    m = np.asarray(abs_m_arr, dtype=float)
    order = np.argsort(T_arr)
    T_arr, m = T_arr[order], m[order]
    if m[0] < 0.5:
        return np.nan          # already below 0.5 at the lowest sampled T
    below = np.where(m < 0.5)[0]
    if below.size == 0:
        return np.nan          # never drops below 0.5 within the range
    i = below[0]               # first point below 0.5; i-1 is above
    m_hi, m_lo = m[i - 1], m[i]
    T_hi, T_lo = T_arr[i - 1], T_arr[i]
    if m_hi == m_lo:
        return float(T_lo)
    frac = (m_hi - 0.5) / (m_hi - m_lo)
    return float(T_hi + frac * (T_lo - T_hi))


# ----- run the 27-condition grid -------------------------------------------
results = {}          # (L, J, h) -> dict of arrays + scalars
total = len(L_VALUES) * len(J_VALUES) * len(H_VALUES)
n = 0
for h in H_VALUES:
    for J in J_VALUES:
        for L in L_VALUES:
            n += 1
            print(f"[{n:2d}/{total}] L={L:3d}  J={J:.2f}  h/J={h:.2f} ...")
            res, _ = sweep_temperatures(
                L=L, T_over_J_array=T_OVER_J, h_over_J=h,
                n_equil=N_EQUIL, n_meas=N_MEAS, sample_every=SAMPLE_EVERY,
                anneal=True, seed=SEED,
            )
            T_over_J   = np.array([r["T_over_J"]   for r in res])
            abs_m      = np.array([r["abs_m"]      for r in res])
            m_mean     = np.array([r["m_mean"]     for r in res])
            chi_abs    = np.array([r["chi_abs"]    for r in res])
            chi_signed = np.array([r["chi_signed"] for r in res])
            T_abs      = T_over_J * J

            # h=0 -> chi_abs is the meaningful susceptibility; h>0 -> chi_signed
            chi_main = chi_abs if h == 0.0 else chi_signed
            peak_idx = int(np.argmax(chi_main))
            T_half_oj = estimate_T_half(T_over_J, abs_m)

            results[(L, J, h)] = {
                "L": L, "J": J, "h_over_J": h,
                "T_over_J": T_over_J,
                "T_absolute": T_abs,
                "abs_m": abs_m,
                "m_mean": m_mean,
                "chi_abs": chi_abs,
                "chi_signed": chi_signed,
                "estimated_T_half_over_J": T_half_oj,
                "estimated_T_half_absolute": T_half_oj * J,
                "peak_chi_over_J": float(T_over_J[peak_idx]),
                "peak_chi_absolute": float(T_abs[peak_idx]),
                "max_chi_abs": float(chi_abs.max()),
                "max_chi_signed": float(chi_signed.max()),
                "low_T_abs_m": float(abs_m[np.argmin(T_over_J)]),
                "high_T_abs_m": float(abs_m[np.argmax(T_over_J)]),
            }

# ----- save raw data (.npz) ------------------------------------------------
save_dict = {
    "T_over_J": T_OVER_J,
    "L_values": np.array(L_VALUES),
    "J_values": np.array(J_VALUES),
    "h_values": np.array(H_VALUES),
    "T_c_exact": T_C,
}
for (L, J, h), d in results.items():
    key = f"L{L}_J{J:.2f}_h{h:.2f}"
    for field in ("T_over_J", "T_absolute", "abs_m", "m_mean",
                  "chi_abs", "chi_signed"):
        save_dict[f"{key}__{field}"] = d[field]
    for field in ("estimated_T_half_over_J", "estimated_T_half_absolute",
                  "peak_chi_over_J", "peak_chi_absolute",
                  "max_chi_abs", "max_chi_signed",
                  "low_T_abs_m", "high_T_abs_m"):
        save_dict[f"{key}__{field}"] = d[field]
np.savez(os.path.join(OUT, "full_grid_results.npz"), **save_dict)

# ----- save summary CSV ----------------------------------------------------
csv_path = os.path.join(OUT, "full_grid_summary.csv")
fields = ["L", "J", "h_over_J", "transition_type",
          "estimated_T_half_over_J", "estimated_T_half_absolute",
          "peak_chi_over_J", "peak_chi_absolute",
          "max_chi_abs", "max_chi_signed",
          "low_T_abs_m", "high_T_abs_m"]
with open(csv_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(fields)
    for h in H_VALUES:
        for J in J_VALUES:
            for L in L_VALUES:
                d = results[(L, J, h)]
                ttype = ("pseudo_critical" if h == 0.0
                         else "field_biased_crossover")
                w.writerow([
                    L, f"{J:.2f}", f"{h:.2f}", ttype,
                    f"{d['estimated_T_half_over_J']:.4f}",
                    f"{d['estimated_T_half_absolute']:.4f}",
                    f"{d['peak_chi_over_J']:.4f}",
                    f"{d['peak_chi_absolute']:.4f}",
                    f"{d['max_chi_abs']:.4f}",
                    f"{d['max_chi_signed']:.4f}",
                    f"{d['low_T_abs_m']:.4f}",
                    f"{d['high_T_abs_m']:.4f}",
                ])

# ----- small-multiple plots (3 rows h x 3 cols J, 3 lines L) ---------------
L_COLORS = {10: "C0", 20: "C1", 50: "C2"}


def small_multiples(observable_key, ylabel, title, fname, susceptibility=False):
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex="col")
    for row, h in enumerate(H_VALUES):
        for col, J in enumerate(J_VALUES):
            ax = axes[row][col]
            for L in L_VALUES:
                d = results[(L, J, h)]
                key = observable_key
                if susceptibility:
                    # h=0 uses chi_abs; h>0 uses chi_signed
                    key = "chi_abs" if h == 0.0 else "chi_signed"
                ax.plot(d["T_absolute"], d[key], "o-",
                        color=L_COLORS[L], lw=1.3, ms=3, label=f"L={L}")
            ax.axvline(T_C * J, color="grey", ls="--", lw=1, alpha=0.6)
            ax.grid(True, alpha=0.3)
            if susceptibility:
                chi_label = "chi_abs" if h == 0.0 else "chi_signed"
                ttype = ("pseudo-critical" if h == 0.0
                         else "field-biased crossover")
                ax.set_title(f"J={J:.2f}, h/J={h:.2f}\n"
                             f"({chi_label}, {ttype})", fontsize=9)
            else:
                ax.set_title(f"J={J:.2f}, h/J={h:.2f}", fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel(r"$T_\mathrm{abs} = (T/J)\cdot J$")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, fname), dpi=140)
    plt.close(fig)


small_multiples(
    "abs_m", r"$\langle|m|\rangle$",
    "Full grid magnetization  (rows: h/J = 0.00, 0.15, 0.50  |  cols: J)",
    "plot_full_grid_magnetization_small_multiples.png",
)
small_multiples(
    None, r"$\chi J$",
    "Full grid susceptibility  (rows: h/J = 0.00, 0.15, 0.50  |  cols: J)\n"
    "h/J=0 uses chi_abs (pseudo-critical); h/J>0 uses chi_signed (crossover)",
    "plot_full_grid_susceptibility_small_multiples.png",
    susceptibility=True,
)

# ----- console summary ------------------------------------------------------
n_nan = sum(1 for d in results.values()
            if np.isnan(d["estimated_T_half_over_J"]))
print(f"\nDone. 27 conditions written to {OUT}/")
print(f"  estimated_T_half_over_J is NaN for {n_nan} condition(s) "
      f"(<|m|> never crossed 0.5 — strong-field biased cases).")
