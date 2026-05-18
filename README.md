# 2D Ising Model — Monte Carlo Simulation

Metropolis Monte Carlo simulation of the 2D Ising model on a square lattice with periodic boundary conditions.

The Hamiltonian is:

```
H = -J * Σ_{<ij>} s_i s_j  -  h * Σ_i s_i
```

All quantities are expressed in reduced units (T/J, h/J) so the physics is independent of J at fixed T/J. The value of J only sets the absolute temperature scale T_abs = (T/J) · J.

The exact critical temperature (Onsager, h=0) is T_c/J ≈ 2.26919.

---

## Requirements

Python 3.8+ with:

```
numpy
matplotlib
numba
pillow
```

Install with:

```bash
pip install -r requirements.txt
```

Numba JIT-compiles the inner Monte Carlo loop. The first run will be slightly slower while Numba compiles the kernels; subsequent runs use cached bytecode.

---

## Project structure

```
ising.py                  core simulation engine (Metropolis sweep, observables)
run_baseline.py           baseline run: L=20, J=1, h=0
run_lattice_sizes.py      lattice-size study: L=10,20,50
run_material_proxy.py     material-proxy study: J=0.75,1.00,1.25
run_external_field.py     external-field study: h/J=0.00,0.15
requirements.txt
```

---

## How to run

### Baseline

```bash
python run_baseline.py
```

Runs L=20, J=1.00, h/J=0 across 31 temperatures (1.5–3.3), denser near T_c.

### Lattice-size comparison

```bash
python run_lattice_sizes.py
```

Runs L=10, 20, 50 at h/J=0 to show finite-size rounding of the transition.

### Material-proxy comparison

```bash
python run_material_proxy.py
```

Runs J=0.75, 1.00, 1.25 at L=20, h/J=0. Plots against absolute temperature to show that higher J shifts the transition to higher T.

### External magnetic field

```bash
python run_external_field.py
```

Runs h/J=0.00 and h/J=0.15 at L=20, J=1.00. Shows how a field biases and rounds the transition.

---

## Output structure

All outputs are written into subdirectories of `outputs/`:

```
outputs/
  baseline/
    baseline_results.npz          raw arrays (T/J, |m|, <m>, chi, cv, energy)
    plot_magnetization.png        |m| vs T/J
    plot_energy.png               energy per site vs T/J
    plot_susceptibility.png       chi*J vs T/J
    plot_heat_capacity.png        Cv per site vs T/J
  snapshots/
    snapshot_lowT.png             spin lattice at T/J ≈ 1.5
    snapshot_critical.png         spin lattice near T_c
    snapshot_highT.png            spin lattice at T/J ≈ 3.3
    snapshots_combined.png        all three side by side
  animation/
    spin_evolution.gif            spin dynamics near T_c
  lattice_size/
    lattice_size_results.npz
    plot_magnetization_by_L.png   |m| vs T/J for L=10,20,50
    plot_susceptibility_by_L.png  chi vs T/J for L=10,20,50
  material_proxy/
    material_proxy_results.npz
    plot_magnetization_vs_absolute_T_by_J.png
    plot_susceptibility_vs_absolute_T_by_J.png
  external_field/
    external_field_results.npz
    plot_signed_magnetization_by_h.png
    plot_abs_magnetization_by_h.png
    plot_susceptibility_by_h.png
```

---

## What each plot shows

| Plot | What to look for |
|------|-----------------|
| `plot_magnetization.png` | Order parameter drops from ~1 to ~0 at T_c. Dashed line marks the exact T_c. |
| `plot_susceptibility.png` | Peak near T_c where spin fluctuations are largest. |
| `plot_heat_capacity.png` | Broad peak near T_c due to energy fluctuations. |
| `plot_magnetization_by_L.png` | Larger L gives a sharper, less rounded transition. |
| `plot_susceptibility_by_L.png` | Larger L gives a taller, narrower susceptibility peak. |
| `plot_magnetization_vs_absolute_T_by_J.png` | Same reduced-unit curve shifts right in absolute T as J increases. |
| `plot_signed_magnetization_by_h.png` | At h/J=0.15 the magnetization stays positive even above h=0 T_c; the transition is rounded, not shifted. |
| `plot_susceptibility_by_h.png` | The susceptibility peak is suppressed and broadened by the field. |
| `snapshot_*.png` | Ordered domains at low T; disordered at high T; critical clusters at T_c. |

---

## Observables and units

| Symbol | Definition | Notes |
|--------|-----------|-------|
| `<\|m\|>` | mean absolute magnetization per site | used at h=0 |
| `<m>` | mean signed magnetization per site | used when h≠0 |
| `chi_abs` | N(<m²> − <\|m\|>²) / (T/J) | susceptibility based on \|m\| |
| `chi_signed` | N(<m²> − <m>²) / (T/J) | susceptibility based on signed m |
| `energy` | mean energy per site in units of J | |
| `cv` | heat capacity per site (dimensionless) | |
