"""
Lattice-size study: L = 10, 20, 50; J = 1.00; h/J = 0.00

The project's central theme — finite-size effects on the phase transition. As L
grows the magnetization drop sharpens and the susceptibility peak grows taller
and narrower. The peak does NOT sit on the Onsager point T_c/J = 2.26919: for a
finite L x L system it sits at a pseudo-critical temperature slightly ABOVE T_c,
approaching it only from above as L -> inf. So a finite lattice whose peak lands
exactly on T_c is a sign of noise/luck, not accuracy.

Near T_c a single Metropolis run is noisy (critical slowing down), and worse for
larger L, so one measured peak can land out of order. To let the true trend show
through *honestly* — no physics changes, no nudging toward the known answer — each
size is run several times with different random seeds and the curves are averaged,
with real error bars from the spread across repeats:

  * one parameter set for every L (so identical L give identical results);
  * a sub-grid parabolic peak fit instead of a grid-snapping argmax;
  * Metropolis sweep budgets that scale up with L to offset critical slowing
    down (at L <= 100 here, the base budget is used);
  * N independent repeats per size, averaged, with mean +- SEM error bars.

The repeat count defaults to 5 but is adjustable from the command line, so a quick
single run is one keystroke away:

    python -m experiments.run_lattice_sizes            # 5 repeats (default)
    python -m experiments.run_lattice_sizes 1          # 1 repeat (quick, no bars)
    python -m experiments.run_lattice_sizes --seeds 8  # 8 repeats

Outputs (outputs/lattice_size/):
    plot_magnetization_by_L.png
    plot_susceptibility_by_L.png
    lattice_size_results.npz
    lattice_size_summary.csv
"""

import csv
import os
import sys
import time

import numpy as np

from ising_mc import config, plotting
from ising_mc.ising import sweep_temperatures
from ising_mc.observables import (edge_abs_m, extract, mean_and_sem, peak_info,
                                  peak_parabolic)

J = 1.00
H_OVER_J = 0.0
T_C = config.T_C    # Onsager exact T_c/J ≈ 2.26919

SIZES = [10, 20, 50]
COLORS = {10: "C0", 20: "C1", 50: "C2"}

# Metropolis budget. Sizes at or below L_REF use the base counts; larger sizes
# scale up (see _mc_sweeps_for). SAMPLE_EVERY decorrelates successive samples.
N_EQUIL_BASE = 2500
N_MEAS_BASE = 6000
N_MEAS_MIN_PER_SEED = 1000   # floor so many repeats can't starve a single run
SAMPLE_EVERY = 5
L_REF = 100

# Independent repeats per size: default, and the seeds used (BASE_SEED + 0..N-1).
N_SEEDS = 5
BASE_SEED = 42


def _parse_n_seeds(argv):
    """Number of independent repeats from the command line, defaulting to N_SEEDS.

    Accepts a bare integer (`... 1`) or `--seeds N`, so a quick single run is one
    keystroke away mid-presentation. Unparseable or <1 values fall back / clamp.
    """
    for token in argv[1:]:
        if token == "--seeds":
            continue
        try:
            return max(1, int(token))
        except ValueError:
            continue
    return N_SEEDS


def _dense_critical_grid():
    """31-point shared grid refined to 0.0125 spacing through the critical band.

    Same three-segment shape as config.standard_temperature_grid(), but the
    near-critical band [2.10, 2.45] is sampled twice as finely (0.0125 vs 0.025)
    so the susceptibility peak — and the parabola fitted to its top — is better
    resolved. The Onsager point T_c/J ≈ 2.269 lies inside this band. The coarse
    ordered/disordered tails are unchanged; they only set the curve shoulders.
    """
    ordered_tail = np.linspace(1.50, 2.10, 7, endpoint=False)
    near_critical = np.linspace(2.10, 2.45, 29)
    disordered_tail = np.linspace(2.50, 3.30, 9)
    return np.concatenate([ordered_tail, near_critical, disordered_tail])


def _mc_sweeps_for(L, n_seeds):
    """(n_equil, n_meas_per_seed) for size L, with the measuring split across repeats.

    Near T_c the Metropolis autocorrelation/relaxation time grows with L because of
    critical slowing down (tau ~ L^z, z ≈ 2.17), so a fixed sweep budget gives
    fewer *independent* samples — and risks under-equilibration — as L grows. The
    total measurement budget is scaled linearly in L above L_REF as an affordable
    partial compensation, then divided across the repeats: averaging n_seeds runs
    of length total/n_seeds matches the precision of one run of length total while
    also yielding an error estimate. Equilibration is re-paid in full per repeat
    (each fresh seed must reach equilibrium first); a floor keeps each repeat usable.
    """
    scale = max(1.0, L / L_REF)
    n_equil = int(round(N_EQUIL_BASE * scale))
    total_meas = int(round(N_MEAS_BASE * scale))
    n_meas = max(N_MEAS_MIN_PER_SEED, total_meas // n_seeds)
    return n_equil, n_meas


def _run_size(L, T_over_J, n_seeds):
    """Run `n_seeds` independent sweeps at size L; return averaged curves + summary.

    Each repeat uses a distinct seed (the engine is untouched — only `seed` varies),
    so the repeats are independent samples of the same physics. The chi/|m| curves
    are averaged with their per-temperature SEM; the peak is taken from the averaged
    chi curve, and its error bar is the spread of the per-repeat peak locations.
    """
    n_equil, n_meas = _mc_sweeps_for(L, n_seeds)
    print(f"Running L={L}: {n_seeds}x (n_equil={n_equil}, n_meas={n_meas}/repeat, "
          f"{len(T_over_J)} temperatures) ...")
    start = time.time()

    chi_runs, m_runs, peak_runs = [], [], []
    for seed in range(BASE_SEED, BASE_SEED + n_seeds):
        results, _ = sweep_temperatures(
            L=L, T_over_J_array=T_over_J, h_over_J=H_OVER_J,
            n_equil=n_equil, n_meas=n_meas, sample_every=SAMPLE_EVERY,
            anneal=True, seed=seed,
        )
        chi_runs.append(extract(results, "chi_abs"))
        m_runs.append(extract(results, "abs_m"))
        peak_runs.append(peak_parabolic(T_over_J, chi_runs[-1])[0])
    runtime_s = time.time() - start
    print(f"  L={L} done in {runtime_s:.1f} s")

    chi_mean, chi_sem = mean_and_sem(chi_runs)
    m_mean, m_sem = mean_and_sem(m_runs)
    peak_T_raw, max_chi = peak_info(T_over_J, chi_mean)
    peak_T_fit, _ = peak_parabolic(T_over_J, chi_mean)
    # Peak uncertainty = spread of the per-repeat peak locations (0 when n_seeds=1).
    _, peak_sem_col = mean_and_sem(np.asarray(peak_runs)[:, None])
    low_m, high_m = edge_abs_m(T_over_J, m_mean)

    curves = {"T": T_over_J, "abs_m": m_mean, "abs_m_sem": m_sem,
              "chi": chi_mean, "chi_sem": chi_sem}
    summary = {"peak_T_raw": peak_T_raw, "peak_T_fit": peak_T_fit,
               "peak_T_sem": float(peak_sem_col[0]), "max_chi_abs": max_chi,
               "low_T_abs_m": low_m, "high_T_abs_m": high_m,
               "n_seeds": n_seeds, "n_meas": n_meas, "runtime_s": runtime_s}
    return curves, summary


def main(argv=None):
    n_seeds = _parse_n_seeds(sys.argv if argv is None else argv)
    out = config.ensure_dir(config.output_dir("lattice_size"))
    T_over_J = _dense_critical_grid()
    print(f"Lattice-size study: {n_seeds} repeat(s) per size "
          f"(seeds {BASE_SEED}..{BASE_SEED + n_seeds - 1}).")

    curves_by_L, summary = {}, {}
    for L in SIZES:
        curves_by_L[L], summary[L] = _run_size(L, T_over_J, n_seeds)

    _save_results(out, T_over_J, curves_by_L)
    _write_summary_csv(out, summary)
    _plot_by_size(out, curves_by_L)
    _print_report(out, summary)


def _save_results(out, T_over_J, curves_by_L):
    # chi_abs_L{L} / abs_m_L{L} hold the across-repeat MEAN curves; the *_sem_L{L}
    # keys carry their standard errors.
    save_dict = {"T_over_J": T_over_J, "J": J, "h_over_J": H_OVER_J,
                 "T_c_exact": T_C, "sizes": np.array(SIZES)}
    for L in SIZES:
        c = curves_by_L[L]
        save_dict[f"T_arr_L{L}"]       = c["T"]
        save_dict[f"abs_m_L{L}"]       = c["abs_m"]
        save_dict[f"abs_m_sem_L{L}"]   = c["abs_m_sem"]
        save_dict[f"chi_abs_L{L}"]     = c["chi"]
        save_dict[f"chi_abs_sem_L{L}"] = c["chi_sem"]
    np.savez(os.path.join(out, "lattice_size_results.npz"), **save_dict)


def _write_summary_csv(out, summary):
    csv_path = os.path.join(out, "lattice_size_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["L", "J", "h_over_J", "peak_T_over_J_raw",
                         "peak_T_over_J_fit", "peak_T_over_J_sem", "max_chi_abs",
                         "low_T_abs_m", "high_T_abs_m", "n_seeds",
                         "n_meas_per_seed", "runtime_s"])
        for L in SIZES:
            s = summary[L]
            writer.writerow([L, f"{J:.2f}", f"{H_OVER_J:.2f}",
                             f"{s['peak_T_raw']:.4f}", f"{s['peak_T_fit']:.4f}",
                             f"{s['peak_T_sem']:.4f}", f"{s['max_chi_abs']:.4f}",
                             f"{s['low_T_abs_m']:.4f}", f"{s['high_T_abs_m']:.4f}",
                             s["n_seeds"], s["n_meas"], f"{s['runtime_s']:.1f}"])


def _band(curve, key, sem_key):
    """(x, mean-sem, mean+sem, ...) inputs for a shaded error envelope."""
    return curve["T"], curve[key] - curve[sem_key], curve[key] + curve[sem_key]


def _plot_by_size(out, curves_by_L):
    mag_curves = [(curves_by_L[L]["T"], curves_by_L[L]["abs_m"], "o-", COLORS[L],
                   f"L = {L}") for L in SIZES]
    mag_bands = [(*_band(curves_by_L[L], "abs_m", "abs_m_sem"), COLORS[L])
                 for L in SIZES]
    plotting.line_plot(
        mag_curves, xlabel=r"$T/J$", ylabel=r"$\langle|m|\rangle$",
        title=f"Magnetization vs lattice size  (J={J}, h/J={H_OVER_J})",
        out_path=os.path.join(out, "plot_magnetization_by_L.png"),
        tc=T_C, tc_label=f"$T_c/J \\approx {T_C:.3f}$", bands=mag_bands,
        figsize=(7.5, 5.2),
    )

    chi_curves = [(curves_by_L[L]["T"], curves_by_L[L]["chi"], "^-", COLORS[L],
                   f"L = {L}") for L in SIZES]
    chi_bands = [(*_band(curves_by_L[L], "chi", "chi_sem"), COLORS[L])
                 for L in SIZES]
    plotting.line_plot(
        chi_curves, xlabel=r"$T/J$", ylabel=r"$\chi J$",
        title=f"Susceptibility vs lattice size  (J={J}, h/J={H_OVER_J})",
        out_path=os.path.join(out, "plot_susceptibility_by_L.png"),
        tc=T_C, tc_label=f"$T_c/J \\approx {T_C:.3f}$", bands=chi_bands,
        figsize=(7.5, 5.2),
    )


def _print_report(out, summary):
    n_seeds = summary[SIZES[0]]["n_seeds"]
    print(f"\nDone. Results in {out}/  ({n_seeds} repeat(s) per size)")
    for L in SIZES:
        s = summary[L]
        bar = f" ± {s['peak_T_sem']:.4f}" if n_seeds >= 2 else "  (no error bar, K=1)"
        print(f"  L={L:3d}: peak T/J fit={s['peak_T_fit']:.4f}{bar}   "
              f"raw={s['peak_T_raw']:.4f}   max chiJ={s['max_chi_abs']:.2f}   "
              f"({s['runtime_s']:.1f} s)")
    print(f"  (Onsager T_c/J = {T_C:.4f})")


if __name__ == "__main__":
    main()
