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

import csv
import os

import numpy as np

from ising_mc import config, plotting
from ising_mc.ising import sweep_temperatures
from ising_mc.observables import estimate_T_half, extract

T_C = config.T_C    # Onsager exact T_c/J ≈ 2.26919

L_VALUES = [10, 20, 50]
J_VALUES = [0.75, 1.00, 1.25]
H_VALUES = [0.00, 0.15, 0.50]
L_COLORS = {10: "C0", 20: "C1", 50: "C2"}

# MC settings kept modest so the 27-cell grid runs on a student laptop.
N_EQUIL = 2000
N_MEAS = 5000
SAMPLE_EVERY = 5
SEED = 42


def main():
    out = config.ensure_dir(config.output_dir("full_grid"))
    T_over_J = config.standard_temperature_grid()

    results = _run_grid(T_over_J)
    _save_results(out, T_over_J, results)
    _write_summary_csv(out, results)
    _plot_small_multiples(out, results)

    n_nan = sum(1 for d in results.values()
                if np.isnan(d["estimated_T_half_over_J"]))
    print(f"\nDone. 27 conditions written to {out}/")
    print(f"  estimated_T_half_over_J is NaN for {n_nan} condition(s) "
          f"(<|m|> never crossed 0.5 — strong-field biased cases).")


def _run_grid(T_over_J):
    """Run every (L, J, h) condition and reduce each to a dict of arrays + scalars."""
    results = {}
    total = len(L_VALUES) * len(J_VALUES) * len(H_VALUES)
    n = 0
    for h in H_VALUES:
        for J in J_VALUES:
            for L in L_VALUES:
                n += 1
                print(f"[{n:2d}/{total}] L={L:3d}  J={J:.2f}  h/J={h:.2f} ...")
                res, _ = sweep_temperatures(
                    L=L, T_over_J_array=T_over_J, h_over_J=h,
                    n_equil=N_EQUIL, n_meas=N_MEAS, sample_every=SAMPLE_EVERY,
                    anneal=True, seed=SEED,
                )
                results[(L, J, h)] = _summarize_condition(res, L, J, h)
    return results


def _summarize_condition(res, L, J, h):
    T_over_J   = extract(res, "T_over_J")
    abs_m      = extract(res, "abs_m")
    m_mean     = extract(res, "m_mean")
    chi_abs    = extract(res, "chi_abs")
    chi_signed = extract(res, "chi_signed")
    T_abs      = T_over_J * J

    # h=0 -> chi_abs is the meaningful susceptibility; h>0 -> chi_signed.
    chi_main = chi_abs if h == 0.0 else chi_signed
    peak_idx = int(np.argmax(chi_main))
    T_half_oj = estimate_T_half(T_over_J, abs_m)

    return {
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


ARRAY_FIELDS = ("T_over_J", "T_absolute", "abs_m", "m_mean", "chi_abs", "chi_signed")
SCALAR_FIELDS = ("estimated_T_half_over_J", "estimated_T_half_absolute",
                 "peak_chi_over_J", "peak_chi_absolute",
                 "max_chi_abs", "max_chi_signed", "low_T_abs_m", "high_T_abs_m")


def _save_results(out, T_over_J, results):
    save_dict = {
        "T_over_J": T_over_J,
        "L_values": np.array(L_VALUES),
        "J_values": np.array(J_VALUES),
        "h_values": np.array(H_VALUES),
        "T_c_exact": T_C,
    }
    for (L, J, h), d in results.items():
        key = f"L{L}_J{J:.2f}_h{h:.2f}"
        for field in ARRAY_FIELDS + SCALAR_FIELDS:
            save_dict[f"{key}__{field}"] = d[field]
    np.savez(os.path.join(out, "full_grid_results.npz"), **save_dict)


def _write_summary_csv(out, results):
    csv_path = os.path.join(out, "full_grid_summary.csv")
    fields = ["L", "J", "h_over_J", "transition_type",
              "estimated_T_half_over_J", "estimated_T_half_absolute",
              "peak_chi_over_J", "peak_chi_absolute",
              "max_chi_abs", "max_chi_signed", "low_T_abs_m", "high_T_abs_m"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for h in H_VALUES:
            for J in J_VALUES:
                for L in L_VALUES:
                    d = results[(L, J, h)]
                    ttype = ("pseudo_critical" if h == 0.0
                             else "field_biased_crossover")
                    writer.writerow([
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


def _plot_small_multiples(out, results):
    """Two 3x3 grids (rows: h/J, cols: J, lines: L) — magnetization and chi."""
    plotting.small_multiples_grid(
        row_values=H_VALUES, col_values=J_VALUES, series_values=L_VALUES,
        series_color=lambda L: L_COLORS[L], series_label=lambda L: f"L={L}",
        cell_xy=lambda L, J, h: (results[(L, J, h)]["T_absolute"],
                                 results[(L, J, h)]["abs_m"]),
        cell_title=lambda J, h: f"J={J:.2f}, h/J={h:.2f}",
        cell_vline=lambda J: T_C * J,
        xlabel=r"$T_\mathrm{abs} = (T/J)\cdot J$", ylabel=r"$\langle|m|\rangle$",
        suptitle="Full grid magnetization  (rows: h/J = 0.00, 0.15, 0.50  |  cols: J)",
        out_path=os.path.join(out, "plot_full_grid_magnetization_small_multiples.png"),
    )

    def chi_xy(L, J, h):
        d = results[(L, J, h)]
        # h=0 -> chi_abs (pseudo-critical); h>0 -> chi_signed (field-biased crossover).
        chi = d["chi_abs"] if h == 0.0 else d["chi_signed"]
        return d["T_absolute"], chi

    def chi_title(J, h):
        chi_label = "chi_abs" if h == 0.0 else "chi_signed"
        ttype = "pseudo-critical" if h == 0.0 else "field-biased crossover"
        return f"J={J:.2f}, h/J={h:.2f}\n({chi_label}, {ttype})"

    plotting.small_multiples_grid(
        row_values=H_VALUES, col_values=J_VALUES, series_values=L_VALUES,
        series_color=lambda L: L_COLORS[L], series_label=lambda L: f"L={L}",
        cell_xy=chi_xy, cell_title=chi_title, cell_vline=lambda J: T_C * J,
        xlabel=r"$T_\mathrm{abs} = (T/J)\cdot J$", ylabel=r"$\chi J$",
        suptitle="Full grid susceptibility  (rows: h/J = 0.00, 0.15, 0.50  |  cols: J)\n"
                 "h/J=0 uses chi_abs (pseudo-critical); h/J>0 uses chi_signed (crossover)",
        out_path=os.path.join(out, "plot_full_grid_susceptibility_small_multiples.png"),
    )


if __name__ == "__main__":
    main()
