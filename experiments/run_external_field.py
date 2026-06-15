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

from ising_mc import config, plotting
from ising_mc.ising import sweep_temperatures
from ising_mc.observables import extract

L = 20
J = 1.00
T_C = config.T_C    # Onsager exact T_c/J ≈ 2.26919

H_VALUES = [0.00, 0.15]
COLORS = ["C0", "C1"]

# The dashed marker sits at the h=0 critical point; with a field there is no true
# T_c, so it is only a reference line.
TC_LABEL = f"$T_c/J \\approx {T_C:.3f}$ (h=0)"


def main():
    out = config.ensure_dir(config.output_dir("external_field"))
    T_over_J = config.standard_temperature_grid()

    curves_by_h = {}
    for h in H_VALUES:
        print(f"Running h/J={h:.2f} ({len(T_over_J)} temperatures) ...")
        results, _ = sweep_temperatures(
            L=L, T_over_J_array=T_over_J, h_over_J=h,
            n_equil=2000, n_meas=5000, sample_every=5, anneal=True, seed=42,
        )
        curves_by_h[h] = {
            "T": extract(results, "T_over_J"),
            "m_mean": extract(results, "m_mean"),
            "abs_m": extract(results, "abs_m"),
            "chi_signed": extract(results, "chi_signed"),
            "chi_abs": extract(results, "chi_abs"),
        }

    _save_results(out, T_over_J, curves_by_h)
    _plot_field_observables(out, curves_by_h)
    print(f"Done. Results in {out}/")


def _save_results(out, T_over_J, curves_by_h):
    save_dict = {"T_over_J": T_over_J, "L": L, "J": J, "T_c_exact": T_C}
    for h in H_VALUES:
        key = f"{h:.2f}"
        save_dict[f"T_arr_h{key}"]      = curves_by_h[h]["T"]
        save_dict[f"m_mean_h{key}"]     = curves_by_h[h]["m_mean"]
        save_dict[f"abs_m_h{key}"]      = curves_by_h[h]["abs_m"]
        save_dict[f"chi_signed_h{key}"] = curves_by_h[h]["chi_signed"]
        save_dict[f"chi_abs_h{key}"]    = curves_by_h[h]["chi_abs"]
    np.savez(os.path.join(out, "external_field_results.npz"), **save_dict)


def _plot_field_observables(out, curves_by_h):
    def curves(field, style):
        return [(curves_by_h[h]["T"], curves_by_h[h][field], style, c,
                 f"h/J = {h:.2f}") for h, c in zip(H_VALUES, COLORS)]

    common = dict(xlabel=r"$T/J$", tc=T_C, tc_label=TC_LABEL)

    plotting.line_plot(
        curves("m_mean", "o-"), ylabel=r"$\langle m \rangle$",
        title=f"Signed magnetization  (L={L}, J={J})",
        out_path=os.path.join(out, "plot_signed_magnetization_by_h.png"), **common)
    plotting.line_plot(
        curves("abs_m", "o-"), ylabel=r"$\langle|m|\rangle$",
        title=f"Absolute magnetization  (L={L}, J={J})",
        out_path=os.path.join(out, "plot_abs_magnetization_by_h.png"), **common)
    plotting.line_plot(
        curves("chi_signed", "^-"), ylabel=r"$\chi_\mathrm{signed}\, J$",
        title=f"Susceptibility (signed)  (L={L}, J={J})",
        out_path=os.path.join(out, "plot_susceptibility_by_h.png"), **common)


if __name__ == "__main__":
    main()
