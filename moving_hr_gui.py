"""
moving_hr_gui.py

The "Moving person (phase 12)" part of the HR tab: settings and a Run button in the
settings column, and a "Moving Person" page among the HR tab's plots with four plots -
the range of the trunk with the motion class of every window, the chest displacement,
the heart band of every window with the tracked heart rate, and the speed of the trunk
with the step frequency.

The host passes get_raw() -> (cube, params, fps, t0_s, name) or None, and run(fn, **kw)
which calls fn(progress_cb=..., **kw) off the Tk thread with the progress bar.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import theme

import numpy as np
from matplotlib.figure import Figure

import moving_hr
from plot_interactivity import create_interactive_canvas

CLASS_COLORS = {1: "#ffe08a", 2: "#f4a3a3"}          # light, strong motion


class MovingHRPanel:
    def __init__(self, ctrl_parent, viz_nb, get_raw, run=None, report_error=None):
        self.get_raw = get_raw
        self.run_fn = run or (lambda fn, **kw: fn(**kw))
        self.report_error = report_error or (lambda title, msg, exc=None:
                                             messagebox.showerror(title, msg))
        self.result: moving_hr.MovingHR | None = None
        self.name = ""

        box = ttk.LabelFrame(ctrl_parent, text="Moving person (phase 12)")
        box.pack(fill="x", padx=5, pady=5)
        self.method = tk.StringVar(value="auto")
        self.win = tk.StringVar(value=f"{moving_hr.HR_WIN_S:g}")
        self.max_change = tk.StringVar(value=f"{moving_hr.MAX_CHANGE_BPM_S:g}")
        self.r_from = tk.StringVar(value=f"{moving_hr.R_MIN_M:g}")
        self.r_to = tk.StringVar(value="")
        self.allow_strong = tk.BooleanVar(value=False)
        ttk.Label(box, text="Method:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(box, textvariable=self.method, state="readonly", width=12,
                     values=["auto", "envelope", "displacement", "both"]).grid(row=0, column=1, sticky="w")
        ttk.Label(box, text="Window [s]:").grid(row=1, column=0, sticky="w")
        tk.Entry(box, textvariable=self.win, width=6).grid(row=1, column=1, sticky="w")
        ttk.Label(box, text="Max change [bpm/s]:").grid(row=2, column=0, sticky="w")
        tk.Entry(box, textvariable=self.max_change, width=6).grid(row=2, column=1, sticky="w")
        ttk.Label(box, text="Trunk from [m]:").grid(row=3, column=0, sticky="w")
        tk.Entry(box, textvariable=self.r_from, width=6).grid(row=3, column=1, sticky="w")
        ttk.Label(box, text="Trunk to [m]:").grid(row=4, column=0, sticky="w")
        tk.Entry(box, textvariable=self.r_to, width=6).grid(row=4, column=1, sticky="w")
        ttk.Checkbutton(box, text="Estimate in strong motion", variable=self.allow_strong).grid(
            row=5, column=0, columnspan=2, sticky="w")
        ttk.Button(box, text="Run Moving HR", command=self.run).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(4, 1))
        ttk.Button(box, text="Export Moving HR (CSV)", command=self.export_csv).grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=1)
        self.summary = tk.Label(box, text="Follows the trunk and its phase through motion; "
                                          "see the Manual, HR.", wraplength=250,
                                justify="left", font=theme.font(8))
        self.summary.grid(row=8, column=0, columnspan=2, sticky="w", pady=(4, 2))

        self.tab = ttk.Frame(viz_nb)
        viz_nb.add(self.tab, text="Moving Person")
        self.viz_nb = viz_nb
        self.fig = Figure(figsize=(8, 9))
        self.canvas, self.toolbar = create_interactive_canvas(self.tab, self.fig)
        self.fig.text(0.5, 0.5, "Press Run Moving HR in the settings column.", ha="center",
                      va="center", color="gray")

    # --- running -------------------------------------------------------------------------
    def _settings(self):
        try:
            win = float(self.win.get())
            change = float(self.max_change.get())
            r_min = float(self.r_from.get() or moving_hr.R_MIN_M)
            r_max = float(self.r_to.get()) if self.r_to.get().strip() else None
        except ValueError as e:
            messagebox.showerror("Moving HR", f"Invalid setting: {e}")
            return None
        if win < 4 or change <= 0 or (r_max is not None and r_max <= r_min):
            messagebox.showerror("Moving HR", "Window at least 4 s, max change above 0, and "
                                              "the trunk range 'to' beyond 'from'.")
            return None
        return dict(method=self.method.get(), win_s=win, max_change_bpm_s=change,
                    r_min=r_min, r_max=r_max, allow_strong=bool(self.allow_strong.get()))

    def run(self):
        s = self._settings()
        if s is None:
            return
        data = self.get_raw()
        if data is None:
            return
        cube, params, fps, t0_s, self.name = data
        try:
            res = self.run_fn(moving_hr.analyse, cube=cube, params=params, fps=fps, t0_s=t0_s, **s)
        except Exception as e:
            self.report_error("Moving HR", f"Moving HR failed: {e}", e)
            return
        if res is None:
            return
        self.result = res
        text = moving_hr.describe(res)
        print(f"[Moving HR] {self.name}: {text}")
        self.summary.configure(text=text)
        self.draw()
        self.viz_nb.select(self.tab)

    # --- drawing ---------------------------------------------------------------------------
    def draw(self):
        res = self.result
        fig = self.fig
        fig.clear()
        axes = fig.subplots(4, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.6, 1]})
        ax_r, ax_d, ax_h, ax_v = axes
        half = 0.5 * float(self.win.get() or moving_hr.HR_WIN_S)
        for ax in axes:
            for i, m in enumerate(res.motion):
                if m in CLASS_COLORS:
                    step = res.win_t[1] - res.win_t[0] if res.win_t.size > 1 else 1.0
                    ax.axvspan(res.win_t[i] - step / 2, res.win_t[i] + step / 2,
                               color=CLASS_COLORS[m], alpha=0.35, lw=0)
        ax_r.plot(res.t_s, res.range_m, lw=0.8)
        ax_r.set_ylabel("Trunk [m]", fontsize=8)
        ax_r.set_title("Trunk range (yellow: light motion, red: strong motion)", fontsize=9)
        d = (res.displacement_m - np.median(res.displacement_m)) * 1e3
        ax_d.plot(res.t_s, d, lw=0.6)
        ax_d.set_ylabel("Chest [mm]", fontsize=8)
        ax_d.set_title("Chest displacement (phase of every chirp, followed through the gaps)",
                       fontsize=9)
        if res.win_t.size:
            g = res.hr_grid_bpm
            ax_h.imshow(res.spectra.T, aspect="auto", origin="lower", cmap="viridis",
                        extent=[res.win_t[0] - 0.5, res.win_t[-1] + 0.5, g[0], g[-1]],
                        interpolation="nearest")
            hr = np.where(res.valid, res.hr_bpm, np.nan)
            ax_h.plot(res.win_t, res.hr_bpm, color="white", lw=0.8, ls=":")
            ax_h.plot(res.win_t, hr, color="white", lw=1.8, label="heart rate (valid)")
            if np.isfinite(res.cadence_hz).any():
                ax_h.plot(res.win_t, res.cadence_hz * 60, color="orange", lw=1.0, ls="--",
                          label="step frequency")
            ax_h.set_ylim(g[0], g[-1])
            ax_h.legend(fontsize=7, loc="upper right", framealpha=0.6)
        ax_h.set_ylabel("Heart [bpm]", fontsize=8)
        ax_h.set_title(f"Heart band per {2 * half:g} s window (harmonic sum) and the tracked "
                       f"heart rate", fontsize=9)
        ax_v.plot(res.win_t, res.speed_ms, lw=1.0, label="trunk speed")
        ax_v.axhline(moving_hr.REST_MS, color="gray", lw=0.6, ls="--")
        ax_v.axhline(moving_hr.STRONG_MS, color="gray", lw=0.6, ls="--")
        ax_v.set_yscale("log")
        ax_v.set_ylabel("Speed [m/s]", fontsize=8)
        ax_v.set_xlabel("Time [s]", fontsize=8)
        ax_v.set_title("Trunk speed per window (dashed: rest / strong motion limits)", fontsize=9)
        for ax in axes:
            ax.tick_params(labelsize=7)
        fig.tight_layout()
        self.canvas.draw_idle()

    def export_csv(self):
        if self.result is None:
            messagebox.showwarning("Moving HR", "Run Moving HR first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            initialfile=f"{self.name}_moving_hr.csv",
                                            filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        arr, header = self.result.window_table()
        try:
            np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt="%.6g")
            messagebox.showinfo("Moving HR", f"Windows saved to {path}")
        except OSError as e:
            messagebox.showerror("Moving HR", f"Could not save: {e}")
