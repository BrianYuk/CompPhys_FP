"""
Baseline experiment: L = 20, J = 1.00, h = 0, T/J between [1.5, 3.3].

Produces:
    plot_magnetization =        |m| vs T/J
    plot_energy =               E/J per site vs T/J
    plot_susceptibility =       chi*J vs T/J
    plot_heat_capacity =        Cv per site vs T/J
    snapshot_lowT =             spin lattice at T/J = 1.5
    snapshot_critical =         spin lattice at T/J = 2.27
    snapshot_highT =            spin lattice at T/J = 3.3
    spin_evolution_low_T =      animation in the ordered phase
    spin_evolution_critical =   animation near T_c
    spin_evolution_high_T =     animation in the disordered phase
    baseline_results =          raw numerical results
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from ising_mc import config, plotting
from ising_mc.ising import record_trajectory, sweep_temperatures
from ising_mc.observables import extract

L = 20
J = 1.00            # physical scale; only matters if you want absolute T
H_OVER_J = 0.0
T_C = config.T_C    # Onsager exact T_c/J ≈ 2.26919


def main():
    out = config.ensure_dir(config.output_dir("baseline"))
    out_snap = config.ensure_dir(config.output_dir("snapshots"))
    out_anim = config.ensure_dir(config.output_dir("animation"))

    T_over_J = config.standard_temperature_grid()
    print(f"Sweeping {len(T_over_J)} temperatures "
          f"from {T_over_J.min():.2f} to {T_over_J.max():.2f}")

    # Snapshots at the coldest point, the grid point nearest T_c, and the hottest.
    snap_targets = [
        float(T_over_J[0]),
        float(T_over_J[(abs(T_over_J - T_C)).argmin()]),
        float(T_over_J[-1]),
    ]
    print(f"Snapshot temperatures: low={snap_targets[0]:.3f}, "
          f"critical={snap_targets[1]:.3f}, high={snap_targets[2]:.3f}")

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

    T_arr      = extract(results, "T_over_J")
    abs_m_arr  = extract(results, "abs_m")
    m_mean_arr = extract(results, "m_mean")
    energy_arr = extract(results, "energy")
    chi_arr    = extract(results, "chi_abs")
    chi_s_arr  = extract(results, "chi_signed")
    cv_arr     = extract(results, "cv")

    np.savez(os.path.join(out, "baseline_results.npz"),
             T_over_J=T_arr, abs_m=abs_m_arr, m_mean=m_mean_arr,
             energy=energy_arr, chi=chi_arr, chi_signed=chi_s_arr, cv=cv_arr,
             L=L, J=J, h_over_J=H_OVER_J, T_c_exact=T_C)

    _plot_observables(out, T_arr, abs_m_arr, energy_arr, chi_arr, cv_arr)
    _plot_snapshots(out_snap, snap_targets, snap_lattices)
    _record_animations(out_anim)
    print("Done.")


def _plot_observables(out, T_arr, abs_m_arr, energy_arr, chi_arr, cv_arr):
    """The four equilibrium observables vs T/J, each with the T_c marker."""
    suffix = f"L={L}, J={J}, h/J={H_OVER_J}"
    tc_label = f"$T_c/J = {T_C:.3f}$"
    common = dict(xlabel=r"$T/J$", tc=T_C, tc_label=tc_label,
                  figsize=(6.5, 4.5), ms=5)

    plotting.line_plot([(T_arr, abs_m_arr, "o-", "C0", None)],
                       ylabel=r"$\langle|m|\rangle$", title=f"Magnetization | {suffix}",
                       out_path=os.path.join(out, "plot_magnetization.png"), **common)
    plotting.line_plot([(T_arr, energy_arr, "s-", "C3", None)],
                       ylabel=r"$\langle E\rangle/(JN)$", title=f"Energy per site | {suffix}",
                       out_path=os.path.join(out, "plot_energy.png"), **common)
    plotting.line_plot([(T_arr, chi_arr, "^-", "C2", None)],
                       ylabel=r"$\chi J$", title=f"Susceptibility | {suffix}",
                       out_path=os.path.join(out, "plot_susceptibility.png"), **common)
    plotting.line_plot([(T_arr, cv_arr, "d-", "C1", None)],
                       ylabel=r"$C_v$ per site", title=f"Heat capacity | {suffix}",
                       out_path=os.path.join(out, "plot_heat_capacity.png"), **common)


def _plot_snapshots(out_snap, snap_targets, snap_lattices):
    """Individual low/critical/high-T spin images plus a combined panel."""
    labels = {snap_targets[0]: ("snapshot_lowT.png", "Low T"),
              snap_targets[1]: ("snapshot_critical.png", "Near $T_c$"),
              snap_targets[2]: ("snapshot_highT.png", "High T")}
    for T, lattice in snap_lattices.items():
        fname, label = labels[T]
        plotting.snapshot(lattice,
                          title=f"{label}: T/J = {T:.3f} | L={L}, h/J={H_OVER_J}",
                          out_path=os.path.join(out_snap, fname))

    plotting.snapshot_panel(
        [snap_lattices[T] for T in snap_targets],
        [f"T/J = {T:.3f}" for T in snap_targets],
        suptitle=f"Spin configurations | L={L}, J={J}, h/J={H_OVER_J}",
        out_path=os.path.join(out_snap, "snapshots_combined.png"),
    )


# Three fixed-temperature animations: ordered, critical, and disordered phases.
# Distinct seeds keep the three trajectories visually independent.
ANIM_CASES = [
    (1.5, "low_T",     "Ordered phase (low $T$)",            1),
    (T_C, "critical",  "Critical point ($T \\approx T_c$)",  2),
    (3.3, "high_T",    "Disordered phase (high $T$)",        3),
]
SWEEPS_PER_FRAME = 2


def _record_animations(out_anim):
    for T_anim, tag, phase_label, seed_val in ANIM_CASES:
        print(f"Recording animation at T/J = {T_anim:.3f} ({tag}) ...")
        frames = record_trajectory(L=L, T_over_J=T_anim, h_over_J=H_OVER_J,
                                   n_equil=500, n_frames=120,
                                   sweeps_per_frame=SWEEPS_PER_FRAME, seed=seed_val)

        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        im = ax.imshow(frames[0], cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
        title_obj = ax.set_title("")
        ax.set_xticks([])
        ax.set_yticks([])

        def update(k, im=im, title_obj=title_obj, frames=frames,
                   phase_label=phase_label, T_anim=T_anim):
            im.set_array(frames[k])
            title_obj.set_text(
                f"{phase_label}  |  T/J = {T_anim:.3f}  |  sweep {k * SWEEPS_PER_FRAME}")
            return [im, title_obj]

        fname = f"spin_evolution_{tag}.gif"
        plotting.save_gif(fig, update, len(frames), os.path.join(out_anim, fname))
        print(f"  Saved: {fname}")


if __name__ == "__main__":
    main()
