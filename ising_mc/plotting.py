"""Reusable matplotlib helpers for the Ising experiment scripts.

Each function builds a figure, writes it to `out_path`, and closes it. The only
side effect is the explicit file write named in each function. Styling defaults
(figure size, dpi, the dashed grey T_c marker) match the project's existing plots.
"""

import matplotlib.pyplot as plt
from matplotlib import animation


def line_plot(curves, *, xlabel, ylabel, title, out_path,
              tc=None, tc_label=None, bands=None,
              figsize=(7.0, 5.0), lw=1.5, ms=4, dpi=140):
    """Plot one or more (x, y) curves vs temperature and save to `out_path`.

    `curves` is a list of (x, y, style, color, label) tuples; pass label=None for
    a curve that should not appear in the legend (e.g. the lone baseline series,
    where only the T_c marker is labelled). If `tc` is given, a dashed vertical
    line is drawn there with `tc_label`.

    `bands` (optional) is a list of (x, lower, upper, color) tuples drawn as a
    shaded fill between lower/upper — e.g. a mean ± standard-error envelope behind
    each curve (a zero-width band, from a single run, simply draws nothing).
    """
    fig, ax = plt.subplots(figsize=figsize)
    for x, lower, upper, color in (bands or []):
        ax.fill_between(x, lower, upper, color=color, alpha=0.2, linewidth=0)
    for x, y, style, color, label in curves:
        ax.plot(x, y, style, color=color, lw=lw, ms=ms, label=label)
    if tc is not None:
        ax.axvline(tc, color="grey", ls="--", lw=1, alpha=0.7, label=tc_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def snapshot(lattice, *, title, out_path, figsize=(4.5, 4.5), dpi=140):
    """Save a single spin-lattice image (blue/red = down/up spins)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(lattice, cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def snapshot_panel(lattices, titles, *, suptitle, out_path,
                   figsize=(11, 4), dpi=140):
    """Save a 1xN row of spin-lattice images under a shared title."""
    fig, axes = plt.subplots(1, len(lattices), figsize=figsize)
    for ax, lattice, title in zip(axes, lattices, titles):
        ax.imshow(lattice, cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def small_multiples_grid(*, row_values, col_values, series_values,
                         series_color, series_label,
                         cell_xy, cell_title, cell_vline,
                         xlabel, ylabel, suptitle, out_path,
                         figsize=(13, 11), dpi=140, marker="o-"):
    """Save a grid of line plots: one subplot per (row, col), one line per series.

    Callbacks decouple data selection from layout:
      cell_xy(series, col, row)  -> (x, y) for one line
      cell_title(col, row)       -> subplot title
      cell_vline(col)            -> x position of the dashed marker in that column
      series_color(series)/series_label(series) -> per-line styling
    Axis labels appear only on the outer edge (left column, bottom row), and the
    legend only on the top-left subplot, matching the project's full-grid figure.
    """
    fig, axes = plt.subplots(len(row_values), len(col_values),
                             figsize=figsize, sharex="col")
    last_row = len(row_values) - 1
    for row, row_value in enumerate(row_values):
        for col, col_value in enumerate(col_values):
            ax = axes[row][col]
            for series in series_values:
                x, y = cell_xy(series, col_value, row_value)
                ax.plot(x, y, marker, color=series_color(series),
                        lw=1.3, ms=3, label=series_label(series))
            ax.axvline(cell_vline(col_value), color="grey", ls="--", lw=1, alpha=0.6)
            ax.grid(True, alpha=0.3)
            ax.set_title(cell_title(col_value, row_value), fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == last_row:
                ax.set_xlabel(xlabel)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_gif(fig, update, n_frames, out_path, *, interval=80, fps=12, dpi=110):
    """Render `update(frame_index)` over `n_frames` into an animated GIF.

    `update` mutates `fig` for the given frame index (e.g. sets the image data and
    title). blit is disabled so the callback need not return artists.
    """
    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=interval, blit=False)
    anim.save(out_path, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
