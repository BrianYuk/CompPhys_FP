"""Pure observable computations for the 2D Ising model.

Everything here operates on plain arrays and scalars — no random numbers, no
lattice kernels, no I/O. New observables and analysis quantities should be added
here so the rest of the codebase stays untouched.

Reduced units throughout: temperatures are T/J, susceptibilities are reported as
chi*J, energies as e/J per site.
"""

import numpy as np


def extract(results, key):
    """Stack one observable across a list of per-temperature result dicts."""
    return np.array([r[key] for r in results])


def mean_and_sem(curve_stack):
    """Per-temperature mean and standard error across repeated, independent runs.

    `curve_stack` has shape (n_runs, n_temperatures): one independent repeat of the
    same simulation (different random seed) per row. Averaging independent repeats
    sharpens the estimate of a noisy observable — e.g. the susceptibility near T_c,
    where critical slowing down makes a single run jumpy — and the standard error
    sem = std / sqrt(n_runs) is the honest uncertainty on each averaged point.

    With a single run there is no spread to measure, so sem is returned as zeros.
    That means "unmeasurable from one run", not "zero uncertainty": a meaningful
    error bar needs n_runs >= 2 (the sample std divides by n_runs - 1).
    """
    stack = np.asarray(curve_stack, dtype=float)
    mean = stack.mean(axis=0)
    if stack.shape[0] < 2:
        return mean, np.zeros_like(mean)
    sem = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
    return mean, sem


def susceptibility_abs(abs_m_mean, m2_mean, n_sites, T_over_J):
    """chi*J from |m| fluctuations: N(<m^2> - <|m|>^2) / (T/J).

    Uses the absolute magnetization, the meaningful order parameter at h = 0
    where the Z2 symmetry makes the signed mean vanish.
    """
    return n_sites * (m2_mean - abs_m_mean ** 2) / T_over_J


def susceptibility_signed(m_mean, m2_mean, n_sites, T_over_J):
    """chi*J from signed-m fluctuations: N(<m^2> - <m>^2) / (T/J).

    The correct susceptibility once a field h breaks the symmetry and selects a
    preferred direction, so fluctuations are measured about the biased mean.
    """
    return n_sites * (m2_mean - m_mean ** 2) / T_over_J


def heat_capacity_per_site(e_mean, e2_mean, n_sites, T_over_J):
    """Heat capacity per site: N(<e^2> - <e>^2) / (T/J)^2 (fluctuation form)."""
    return n_sites * (e2_mean - e_mean ** 2) / T_over_J ** 2


def observables_from_samples(magnetizations, energies, n_sites, T_over_J):
    """Reduce per-sample magnetization and energy series to scalar observables.

    `magnetizations` and `energies` are the sampled per-site time series. Returns
    the equilibrium averages and fluctuation quantities the engine reports for a
    single (T/J, h/J) point.
    """
    m_mean = magnetizations.mean()                 # signed <m>
    abs_m = np.abs(magnetizations).mean()          # <|m|>, the h=0 order parameter
    m2 = (magnetizations ** 2).mean()              # <m^2>, for the susceptibility
    e_mean = energies.mean()                       # <e>/J per site
    e2 = (energies ** 2).mean()                    # <e^2>, for the heat capacity
    return {
        "m_mean": m_mean,
        "abs_m": abs_m,
        "m2": m2,
        "energy": e_mean,
        "chi_abs": susceptibility_abs(abs_m, m2, n_sites, T_over_J),
        "chi_signed": susceptibility_signed(m_mean, m2, n_sites, T_over_J),
        "cv": heat_capacity_per_site(e_mean, e2, n_sites, T_over_J),
    }


def peak_info(T_arr, chi_arr):
    """Return (T_peak, chi_max): temperature and value at the susceptibility peak.

    The susceptibility peak marks the (finite-size rounded) phase transition.
    Raw discrete maximum: T_peak is pinned to the sampled T/J grid and is pulled
    around by single-point noise. Use `peak_parabolic` for a sub-grid estimate.
    """
    idx = int(np.argmax(chi_arr))
    return float(T_arr[idx]), float(chi_arr[idx])


def peak_parabolic(T_arr, chi_arr, half_window=2):
    """Return (T_peak, chi_peak) from a quadratic fit near the susceptibility max.

    The chi(T) maximum locates the finite-size (pseudo-)critical temperature, but
    a raw argmax snaps that estimate to the T/J grid and follows the single
    noisiest point. Near its top the peak is approximately parabolic, so fitting
    chi ≈ a (T - T_peak)^2 + chi_peak over the `2*half_window + 1` points centred
    on the discrete maximum recovers a sub-grid, noise-averaged location. For a
    concave fit (a < 0) the vertex sits at T_peak = -b / (2a).

    Falls back to the discrete `peak_info` point whenever the parabola is not a
    trustworthy description of a bracketed peak: too few points to fit, a
    non-concave fit (a >= 0), or a vertex extrapolating outside the fit window.
    """
    T_arr = np.asarray(T_arr, dtype=float)
    chi_arr = np.asarray(chi_arr, dtype=float)
    order = np.argsort(T_arr)
    T_arr, chi_arr = T_arr[order], chi_arr[order]

    grid_peak = peak_info(T_arr, chi_arr)
    peak_idx = int(np.argmax(chi_arr))

    lo = max(peak_idx - half_window, 0)
    hi = min(peak_idx + half_window + 1, T_arr.size)
    T_win, chi_win = T_arr[lo:hi], chi_arr[lo:hi]
    if T_win.size < 3:
        return grid_peak

    a, b, c = np.polyfit(T_win, chi_win, 2)
    if a >= 0.0:
        return grid_peak                       # not concave: no interior maximum
    T_peak = -b / (2.0 * a)
    if not (T_win[0] <= T_peak <= T_win[-1]):
        return grid_peak                       # vertex extrapolates past the data
    chi_peak = a * T_peak ** 2 + b * T_peak + c
    return float(T_peak), float(chi_peak)


def edge_abs_m(T_arr, m_arr):
    """Return (low-T |m|, high-T |m|) at the first and last sampled temperature."""
    return float(m_arr[0]), float(m_arr[-1])


def estimate_T_half(T_arr, abs_m_arr):
    """Interpolated T/J where <|m|> first drops below 0.5 (descending in m).

    Used as a rough transition locator. Returns NaN if <|m|> never crosses 0.5
    within the sampled range (e.g. strong-field cases where it stays biased high).
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
