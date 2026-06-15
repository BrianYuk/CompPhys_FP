"""
Interactive live Ising explorer (Tkinter + embedded matplotlib).

Pick a lattice size L, a coupling proxy J, and a starting temperature; then watch the
spin grid (red = +1, blue = -1) evolve while you drag the temperature and magnetic-
field sliders. The grid is driven by the engine's resumable `advance()` primitive via
`LiveSimulation`, so changing a control mid-run lets the *current* configuration keep
evolving instead of restarting.

Units: the J choice is a real coupling, so the sliders are in ABSOLUTE temperature and
field. The engine works in reduced units, so each frame we pass T/J = T_abs / J and
h/J = h_abs / J. A larger J therefore pushes the transition to higher absolute T
(Onsager: T_c = 2.26919 * J) — matching the material-proxy study.

Run:
    python -m experiments.run_interactive
"""

import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ising_mc import config
from ising_mc.interactive import LiveSimulation

J_OPTIONS = [0.75, 1.00, 1.25]
L_MIN, L_MAX, L_DEFAULT = 4, 150, 50
T_DEFAULT = 2.5
SPEED_MIN, SPEED_MAX, SPEED_DEFAULT = 1, 20, 4
FRAME_INTERVAL_MS = 50          # ~20 fps target for the live loop
SEED = 42


class IsingApp:
    """Tkinter app: a setup form, then a live spin-grid animation with controls."""

    def __init__(self, root):
        self.root = root
        self.root.title("2D Ising — interactive explorer")
        self.sim = None
        self.J = 1.0
        self.running = False
        self._after_id = None
        self._build_setup()

    # ----- setup screen --------------------------------------------------------
    def _build_setup(self):
        # Pre-fill with the last-used values (if any) so returning to the form to
        # tweak one parameter doesn't lose the others.
        L_init = getattr(self, "_last_L", L_DEFAULT)
        J_init = getattr(self, "_last_J", 1.00)
        T_init = getattr(self, "_last_T", T_DEFAULT)

        self.setup = tk.Frame(self.root, padx=16, pady=16)
        self.setup.pack()

        tk.Label(self.setup, text=f"Lattice size L ({L_MIN}–{L_MAX}):").grid(
            row=0, column=0, sticky="w")
        self.L_var = tk.StringVar(value=str(L_init))
        tk.Entry(self.setup, textvariable=self.L_var, width=8).grid(row=0, column=1, sticky="w")

        tk.Label(self.setup, text="Coupling proxy J:").grid(row=1, column=0, sticky="w")
        self.J_var = tk.DoubleVar(value=J_init)
        j_row = tk.Frame(self.setup)
        j_row.grid(row=1, column=1, sticky="w")
        for j in J_OPTIONS:
            tk.Radiobutton(j_row, text=f"{j:.2f}", variable=self.J_var, value=j).pack(side="left")

        tk.Label(self.setup, text="Starting temperature T (absolute):").grid(
            row=2, column=0, sticky="w")
        self.T_var = tk.StringVar(value=str(T_init))
        tk.Entry(self.setup, textvariable=self.T_var, width=8).grid(row=2, column=1, sticky="w")

        self.setup_error = tk.Label(self.setup, text="", fg="red")
        self.setup_error.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        tk.Button(self.setup, text="Start", command=self._on_start).grid(
            row=4, column=0, columnspan=2, pady=(10, 0))

    def _on_start(self):
        try:
            L = int(self.L_var.get())
            T0 = float(self.T_var.get())
        except ValueError:
            self.setup_error.config(text="L must be an integer and T a number.")
            return
        if not (L_MIN <= L <= L_MAX):
            self.setup_error.config(
                text=f"Keep L between {L_MIN} and {L_MAX} for a smooth animation.")
            return
        if T0 <= 0:
            self.setup_error.config(text="Temperature must be positive.")
            return

        self.J = float(self.J_var.get())
        self._last_L, self._last_J, self._last_T = L, self.J, T0
        self.sim = LiveSimulation(L, seed=SEED)
        self.setup.destroy()
        self._build_animation(T0)
        self.running = True
        self._tick()

    # ----- animation screen ----------------------------------------------------
    def _build_animation(self, T0):
        tc_abs = config.T_C * self.J            # Onsager transition in ABSOLUTE units

        self.fig = Figure(figsize=(5.0, 5.0))
        self.ax = self.fig.add_subplot(111)
        # Let the axes fill the figure so the grid uses the whole window when resized.
        self.fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
        # Red = +1 spin, blue = -1, matching the project's snapshot convention.
        self.im = self.ax.imshow(self.sim.lattice, cmap="RdBu", vmin=-1, vmax=1,
                                 interpolation="nearest")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # fill="both"/expand so maximizing the window enlarges the grid; the Tk
        # backend resizes the figure to match the canvas widget.
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.grid_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.controls = tk.Frame(self.root, padx=12, pady=12)
        self.controls.pack(side="left", fill="y")
        controls = self.controls

        # Absolute-temperature slider: T/J in [0.5, 4] brackets the transition.
        self.T_scale = self._add_scale(
            controls, "Temperature T", 0.5 * self.J, 4.0 * self.J, 0.01,
            min(max(T0, 0.5 * self.J), 4.0 * self.J))
        # Absolute field slider: h/J in [-1, 1]; starts unbiased.
        self.h_scale = self._add_scale(
            controls, "Magnetic field h", -1.0 * self.J, 1.0 * self.J, 0.01, 0.0)
        self.speed_scale = self._add_scale(
            controls, "Speed (sweeps/frame)", SPEED_MIN, SPEED_MAX, 1, SPEED_DEFAULT)

        btns = tk.Frame(controls)
        btns.pack(pady=(8, 4))
        self.pause_btn = tk.Button(btns, text="Pause", command=self._toggle_pause)
        self.pause_btn.pack(side="left", padx=4)
        tk.Button(btns, text="Reset", command=self._reset).pack(side="left", padx=4)
        tk.Button(btns, text="Back to setup", command=self._back_to_setup).pack(
            side="left", padx=4)

        self.readout = tk.Label(controls, justify="left", font=("TkFixedFont", 11))
        self.readout.pack(anchor="w", pady=(8, 0))
        tk.Label(controls, fg="grey",
                 text=f"T_c = {tc_abs:.3f}  (J = {self.J:.2f})").pack(anchor="w")

        self.root.geometry("880x600")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _add_scale(self, parent, label, lo, hi, resolution, init):
        tk.Label(parent, text=label).pack(anchor="w")
        scale = tk.Scale(parent, from_=lo, to=hi, resolution=resolution,
                         orient="horizontal", length=240)
        scale.set(init)
        scale.pack(fill="x")
        return scale

    # ----- live loop -----------------------------------------------------------
    def _tick(self):
        if self.running:
            T_abs = self.T_scale.get()
            h_abs = self.h_scale.get()
            sweeps = int(self.speed_scale.get())
            # Engine is in reduced units; convert the absolute slider values.
            _, abs_m = self.sim.step(T_abs / self.J, h_abs / self.J, sweeps)
            self._render(T_abs, h_abs, abs_m)
        self._after_id = self.root.after(FRAME_INTERVAL_MS, self._tick)

    def _render(self, T_abs, h_abs, abs_m):
        self.im.set_data(self.sim.lattice)
        self.canvas.draw_idle()
        phase = "ordered" if T_abs < config.T_C * self.J else "disordered"
        self.readout.config(text=(
            f"T = {T_abs:.3f}  (T/J = {T_abs / self.J:.3f})\n"
            f"h = {h_abs:.3f}  (h/J = {h_abs / self.J:.3f})\n"
            f"<|m|> = {abs_m:.3f}   [{phase}]"))

    def _toggle_pause(self):
        self.running = not self.running
        self.pause_btn.config(text="Pause" if self.running else "Resume")

    def _reset(self):
        self.sim.reset()
        # Show the fresh lattice at once; |m| recomputed from the new configuration.
        self._render(self.T_scale.get(), self.h_scale.get(),
                     float(abs(self.sim.lattice.mean())))

    def _back_to_setup(self):
        # Stop the live loop, tear down the animation view, and re-show the form.
        self.running = False
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self.grid_frame.destroy()
        self.controls.destroy()
        self.sim = None
        self._build_setup()
        self.root.geometry("")        # shrink back to the form's natural size

    def _on_close(self):
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self.root.destroy()


def main():
    root = tk.Tk()
    IsingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
