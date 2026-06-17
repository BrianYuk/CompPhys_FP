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

### Running the tests

The test suite locks the engine's numerical behavior — seeded golden values, the
RNG-determinism contract, and physics sanity checks — so refactors cannot silently
change results.

```bash
pip install -r requirements-dev.txt
pytest
```

---

## Project structure

Files are grouped by role: the importable library and the runnable studies.

```
ising_mc/                       importable library (the engine + helpers)
  ising.py                      core MC engine: advance() sweep primitive + drivers
  config.py                     shared constants, the T/J grid, output paths
  observables.py                pure observable / analysis functions
  plotting.py                   reusable matplotlib figure helpers
experiments/                    runnable studies (run as modules, see below)
  run_baseline.py               baseline run: L=20, J=1, h=0
  run_lattice_sizes.py          lattice-size study: L=10,20,50 (averaged)
  run_material_proxy.py         material comparison: Ni/Fe/Co (J=0.60/1.00/1.34)
  run_external_field.py         external-field study: h/J=0.00,0.15
  run_cooling_animation.py      cooling/annealing animation (presentation)
  run_full_grid.py              full 27-condition L×J×(h/J) verification grid
tests/                          characterization, determinism, and physics tests
requirements.txt                runtime dependencies (pinned)
requirements-dev.txt            adds pytest for running the tests
```

Scripts import the library as a package (`from ising_mc.ising import ...`) and are
run as modules from the repository root, so the engine is found without any install.

---

## How to run

### Baseline

```bash
python -m experiments.run_baseline
```

Runs L=20, J=1.00, h/J=0 across 31 temperatures (1.5–3.3), denser near T_c.

### Lattice-size comparison

```bash
python -m experiments.run_lattice_sizes
```

Runs L=10, 20, 50 at h/J=0 to show finite-size rounding of the transition.
The susceptibility peak sits *above* the Onsager point at every finite L (the
finite-size pseudo-critical temperature) and is located with a sub-grid parabolic
fit — so a finite lattice whose peak lands exactly on T_c is a noise coincidence,
not accuracy. Each size's peak still sits above T_c and moves toward it as L grows.

Each size is run several times with different random seeds and **averaged**, so the
noisy near-T_c peaks settle and the true "bigger L → closer to T_c" trend shows
through honestly; the curves carry mean ± SEM shaded bands. The repeat count
defaults to 5 but is adjustable, so a quick single run is available mid-presentation:

```bash
python -m experiments.run_lattice_sizes            # 5 repeats (~7 min, default)
python -m experiments.run_lattice_sizes 1          # 1 repeat (~3 min, no error bars)
python -m experiments.run_lattice_sizes --seeds 8  # 8 repeats
```

Averaging shrinks the noise like √K but cannot remove it — with single-spin-flip
Metropolis the largest lattice stays the blurriest. The error bars show that
honestly rather than forcing a clean trend.

### Material comparison (Nickel, Iron, Cobalt)

```bash
python -m experiments.run_material_proxy
```

Runs three real ferromagnetic metals — Nickel (J=0.60), Iron (J=1.00), Cobalt (J=1.34) — at L=20, h/J=0. Each J is calibrated so the ratio of J values matches the ratio of the metals' experimental Curie temperatures (Kittel, Ch. 12). Plotting against absolute temperature separates the curves: Cobalt transitions at the highest T, then Iron, then Nickel.

### External magnetic field

```bash
python -m experiments.run_external_field
```

Runs h/J=0.00 and h/J=0.15 at L=20, J=1.00. Shows how a field biases and rounds the transition.

### Interactive explorer

```bash
python -m experiments.run_interactive
```

A live desktop app (Tkinter + matplotlib). Pick a lattice size, a ferromagnetic
material (Nickel, Iron, or Cobalt — each sets its coupling J), and a starting
temperature, then watch the spin grid (red = +1, blue = −1) evolve while you drag
the **temperature** and **magnetic-field** sliders in real time. Temperature and
field are in **absolute** units, so a larger J pushes the transition to higher T
(T_c = 2.269·J) — picking the material actually changes the physics.
Extras: pause/resume, reset, a sweeps-per-frame speed slider, and a live ⟨|m|⟩ /
phase readout against T_c. Driven by the engine's `advance()` primitive via
`ising_mc.interactive.LiveSimulation`; needs a display (run it locally, not headless).

---

## Additional presentation outputs

Three extra scripts produce presentation-support material. They build on
the same `ising_mc` engine and do not change the core analysis.

### Cooling animation

```bash
python -m experiments.run_cooling_animation
```

Slowly cools an L=50 lattice from T/J=3.3 down to 1.5 over 120 frames so
the audience can watch ordered domains nucleate and grow.
Output: `outputs/animation/cooling_transition.gif`.
This is a **non-equilibrium annealing trajectory** for visual explanation
only — it is labelled as such and does not replace the fixed-temperature
quantitative plots.

### Full 27-condition grid

```bash
python -m experiments.run_full_grid
```

Runs the full parameter grid L=[10,20,50] × J=[0.75,1.00,1.25] ×
h/J=[0.00,0.15,0.50] (27 conditions). Outputs go to `outputs/full_grid/`:
a results `.npz`, a summary `.csv`, and two 3×3 small-multiple plots
(magnetization and susceptibility). Small multiples are used so the grid
stays readable instead of one plot with 27 lines.

### Where outputs are saved

```
outputs/animation/cooling_transition.gif
outputs/full_grid/                       (npz, csv, 2 small-multiple PNGs)
```

The full grid is a **verification sweep** — it confirms the expected
trends hold jointly across all parameters. The main project analysis
relies on the controlled one-variable-at-a-time comparisons
(`run_lattice_sizes.py`, `run_material_proxy.py`, `run_external_field.py`).
Note that the full grid's J values are generic exchange-coupling **proxies**
(the named-material comparison lives in `run_material_proxy.py`), and
h/J=0.15/0.50 produce **field-biased crossover** behavior, not a clean
shifted critical temperature.

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
    spin_evolution_low_T.gif      spin dynamics in the ordered phase
    spin_evolution_critical.gif   spin dynamics near T_c
    spin_evolution_high_T.gif     spin dynamics in the disordered phase
  lattice_size/
    lattice_size_results.npz
    plot_magnetization_by_L.png   |m| vs T/J for L=10,20,50
    plot_susceptibility_by_L.png  chi vs T/J for L=10,20,50
  material_proxy/
    material_proxy_results.npz
    plot_magnetization_vs_absolute_T_by_J.png
    plot_susceptibility_vs_absolute_T_by_J.png
    plot_overlay_reduced_units.png            curves overlap in T/J (J-independence check)
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
