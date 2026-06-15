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
    plot_overlay_reduced_units.png
    material_proxy_results.npz
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from ising_mc import config, plotting
from ising_mc.ising import sweep_temperatures
from ising_mc.observables import extract

L = 20
H_OVER_J = 0.0
T_C_REDUCED = config.T_C    # Onsager exact T_c/J ≈ 2.26919

J_VALUES = [0.75, 1.00, 1.25]
COLORS = ["C0", "C1", "C2"]


def main():
    out = config.ensure_dir(config.output_dir("material_proxy"))
    T_over_J = config.standard_temperature_grid()

    curves_by_J = {}
    for J in J_VALUES:
        print(f"Running J={J:.2f} ({len(T_over_J)} temperatures) ...")
        results, _ = sweep_temperatures(
            L=L, T_over_J_array=T_over_J, h_over_J=H_OVER_J,
            n_equil=2000, n_meas=5000, sample_every=5, anneal=True, seed=42,
        )
        curves_by_J[J] = {
            "T": extract(results, "T_over_J"),
            "abs_m": extract(results, "abs_m"),
            "chi": extract(results, "chi_abs"),
        }

    _save_results(out, T_over_J, curves_by_J)
    _plot_vs_absolute_temperature(out, curves_by_J)
    _plot_overlay_reduced_units(out, curves_by_J)
    print(f"Done. Results in {out}/")


def _save_results(out, T_over_J, curves_by_J):
    save_dict = {"T_over_J": T_over_J, "L": L, "h_over_J": H_OVER_J}
    for J in J_VALUES:
        key = f"{J:.2f}"
        # Store against absolute temperature T_abs = (T/J)*J so the curves can be
        # read back already separated by material "stiffness".
        save_dict[f"T_abs_J{key}"]   = curves_by_J[J]["T"] * J
        save_dict[f"abs_m_J{key}"]   = curves_by_J[J]["abs_m"]
        save_dict[f"chi_abs_J{key}"] = curves_by_J[J]["chi"]
    np.savez(os.path.join(out, "material_proxy_results.npz"), **save_dict)


def _label_for(J):
    tc_abs = T_C_REDUCED * J   # the transition sits at T_c/J * J in absolute units
    return f"J = {J:.2f}  ($T_c \\approx {tc_abs:.2f}$)"


def _plot_vs_absolute_temperature(out, curves_by_J):
    xlabel = r"Absolute temperature $T_\mathrm{abs} = (T/J)\cdot J$"

    mag_curves = [(curves_by_J[J]["T"] * J, curves_by_J[J]["abs_m"], "o-", c,
                   _label_for(J)) for J, c in zip(J_VALUES, COLORS)]
    plotting.line_plot(
        mag_curves, xlabel=xlabel, ylabel=r"$\langle|m|\rangle$",
        title=f"Magnetization: material-proxy comparison  (L={L}, h/J={H_OVER_J})",
        out_path=os.path.join(out, "plot_magnetization_vs_absolute_T_by_J.png"),
    )

    chi_curves = [(curves_by_J[J]["T"] * J, curves_by_J[J]["chi"], "^-", c,
                   _label_for(J)) for J, c in zip(J_VALUES, COLORS)]
    plotting.line_plot(
        chi_curves, xlabel=xlabel, ylabel=r"$\chi J$",
        title=f"Susceptibility: material-proxy comparison  (L={L}, h/J={H_OVER_J})",
        out_path=os.path.join(out, "plot_susceptibility_vs_absolute_T_by_J.png"),
    )


def _plot_overlay_reduced_units(out, curves_by_J):
    """Sanity check: in reduced units T/J every J curve must collapse onto one.

    A two-panel overlay (magnetization and susceptibility vs T/J) confirms the
    physics is J-independent at fixed T/J — J only rescales the absolute
    temperature axis. This is the only figure unique to the old notebook.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for J, c in zip(J_VALUES, COLORS):
        T = curves_by_J[J]["T"]
        axes[0].plot(T, curves_by_J[J]["abs_m"], "o-", color=c, lw=1.5, ms=3,
                     label=f"J={J:.2f}")
        axes[1].plot(T, curves_by_J[J]["chi"], "^-", color=c, lw=1.5, ms=3,
                     label=f"J={J:.2f}")
    panels = zip(axes,
                 [r"$\langle|m|\rangle$", r"$\chi J$"],
                 ["Magnetization in T/J units", "Susceptibility in T/J units"])
    for ax, ylabel, title in panels:
        ax.axvline(T_C_REDUCED, color="grey", ls="--", lw=1, alpha=0.7)
        ax.set_xlabel(r"$T/J$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Curves overlap in T/J — J only shifts the absolute temperature scale")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "plot_overlay_reduced_units.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
