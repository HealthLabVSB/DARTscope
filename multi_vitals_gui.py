"""
multi_vitals_gui.py

The Multi Vitals view of the Analysis tab (multi_vitals.py): its settings and Run in one
strip, a summary line, and two pages -

  Map    the room from above, the radar at the bottom: every person a dot with its
         number, the breathing and heart rate of the window around the frame on the
         frame slider and the last seconds of its path; the detection map of that
         moment underneath. Follows the frame slider and its playback.
  Rates  breathing and heart rate of every person over time and their chest
         displacement, on one time axis with the frame line.

The host passes get_raw() -> (cube, params, fps, t0_s, name) or None, run(fn, **kw) that
calls fn(progress_cb=..., **kw) off the Tk thread with the progress bar, and optionally
report_error, register_run(button), header_extras(group, fig, data_func, extra_items,
update_func) for the ⓘ icon and the Export menu (update_func: the map at a frame, for the
video), and fft_size() -> the Analysis bar's Range FFT size.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.figure import Figure

import multi_vitals
import theme
from flow_layout import FlowFrame
from plot_interactivity import create_interactive_canvas

METHOD_LABELS = {"Motion (tracking)": "motion", "Breathing (still people)": "breathing",
                 "Range only": "range"}
BEAM_LABELS = {"Capon": "capon", "Bartlett": "bartlett"}
TRAIL_S = 5.0
MAP_RANGE_DB = 15.0            # the map under the dots: this far below the strongest motion of the moment
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2",
          "#17becf", "#bcbd22", "#7f7f7f"]


def color(pid: int) -> str:
    return COLORS[(pid - 1) % len(COLORS)]


def rate_text(state: dict | None) -> str:
    """The label of a dot: the rates of the window around the frame, or why there are none."""
    if state is None:
        return ""
    if state["moving"]:
        return "moving"
    br = f"BR {state['br']:.0f}" if np.isfinite(state["br"]) else "BR -"
    hr = f"HR {state['hr']:.0f}" if np.isfinite(state["hr"]) else "HR -"
    return f"{br} · {hr}"


class MultiVitalsTab:
    def __init__(self, parent, get_raw, run=None, report_error=None, register_run=None,
                 header_extras=None, fft_size=None):
        self.get_raw = get_raw
        self.run_fn = run or (lambda fn, **kw: fn(**kw))
        self.report_error = report_error or (lambda title, msg, exc=None:
                                             messagebox.showerror(title, msg))
        self.fft_size = fft_size or (lambda: None)
        self.result: multi_vitals.MultiVitals | None = None
        self.name = ""
        self._frame = 0
        self._artists = None

        controls = FlowFrame(parent)
        controls.pack(side="top", fill="x", padx=8, pady=4)
        run_group = controls.group()
        self.run_button = theme.Button(run_group, text="Run Multi Vitals", command=self.run)
        self.run_button.pack(side="left")
        if register_run is not None:
            register_run(self.run_button)

        self.method = tk.StringVar(value="Motion (tracking)")
        self.beam = tk.StringVar(value="Capon")
        self.max_people = tk.StringVar(value=f"{multi_vitals.MAX_PEOPLE}")
        self.r_max = tk.StringVar(value=f"{multi_vitals.R_MAX_M:g}")
        self.min_sep = tk.StringVar(value=f"{multi_vitals.MIN_SEP_M:g}")
        self.show_echoes = tk.BooleanVar(value=False)
        self.show_map = tk.BooleanVar(value=True)
        g = controls.group()
        ttk.Label(g, text="Find people by:").pack(side="left", padx=(0, 2))
        ttk.Combobox(g, textvariable=self.method, state="readonly", width=22,
                     values=list(METHOD_LABELS)).pack(side="left")
        g = controls.group()
        ttk.Label(g, text="Beam:").pack(side="left", padx=(0, 2))
        ttk.Combobox(g, textvariable=self.beam, state="readonly", width=9,
                     values=list(BEAM_LABELS)).pack(side="left")
        g = controls.group()
        ttk.Label(g, text="Max people:").pack(side="left", padx=(0, 2))
        tk.Spinbox(g, from_=1, to=10, textvariable=self.max_people, width=3).pack(side="left")
        ttk.Label(g, text="Range to [m]:").pack(side="left", padx=(8, 2))
        tk.Entry(g, textvariable=self.r_max, width=5).pack(side="left")
        ttk.Label(g, text="Min separation [m]:").pack(side="left", padx=(8, 2))
        tk.Entry(g, textvariable=self.min_sep, width=5).pack(side="left")
        g = controls.group()
        tk.Checkbutton(g, text="Show echoes", variable=self.show_echoes,
                       command=self.redraw).pack(side="left")
        tk.Checkbutton(g, text="Detection map", variable=self.show_map,
                       command=self.redraw).pack(side="left")

        self.summary = tk.Label(parent, text="(press Run Multi Vitals: finds everyone in the "
                                             "room, follows them and measures each one's "
                                             "breathing and heart rate)",
                                anchor="w", justify="left", font=theme.font(9), wraplength=900)
        self.summary.pack(side="top", fill="x", padx=10)

        self.viz_nb = ttk.Notebook(parent)
        self.viz_nb.pack(side="top", fill="both", expand=True, padx=6, pady=4)
        map_page, rates_page = ttk.Frame(self.viz_nb), ttk.Frame(self.viz_nb)
        self.viz_nb.add(map_page, text="Map")
        self.viz_nb.add(rates_page, text="Rates")
        self.map_fig = Figure(figsize=(7, 6))
        self.map_canvas, _ = create_interactive_canvas(map_page, self.map_fig)
        self.rates_fig = Figure(figsize=(8, 7))
        self.rates_canvas, _ = create_interactive_canvas(rates_page, self.rates_fig)
        for fig in (self.map_fig, self.rates_fig):
            fig.text(0.5, 0.5, "Press Run Multi Vitals.", ha="center", va="center", color="gray")
        if header_extras is not None:
            header_extras(run_group, self.map_fig, self.positions_for_export,
                          [("Rates over time (CSV)", self.export_rates),
                           ("Rates figure (PNG)", self.export_rates_png)],
                          self.show_frame)

    # --- running -------------------------------------------------------------------------
    def settings(self) -> dict | None:
        try:
            r_max = float(self.r_max.get())
            sep = float(self.min_sep.get())
            n = int(float(self.max_people.get()))
        except ValueError as e:
            messagebox.showerror("Multi Vitals", f"Invalid setting: {e}")
            return None
        if r_max <= multi_vitals.R_MIN_M or sep <= 0 or n < 1:
            messagebox.showerror("Multi Vitals", f"Range to above {multi_vitals.R_MIN_M} m, "
                                                 "min separation above 0 and at least one person.")
            return None
        return dict(method=METHOD_LABELS[self.method.get()], beam=BEAM_LABELS[self.beam.get()],
                    r_max=r_max, min_sep=sep, max_people=n)

    def run(self):
        s = self.settings()
        if s is None:
            return
        data = self.get_raw()
        if data is None:
            return
        cube, params, fps, t0_s, self.name = data
        try:
            res = self.run_fn(multi_vitals.analyse, cube=cube, params=params, fps=fps,
                              stem=self.name, t0_s=t0_s, n_fft_range=self.fft_size(), **s)
        except Exception as e:
            self.report_error("Multi Vitals", f"Multi Vitals failed: {e}", e)
            return
        if res is None:
            return
        self.result = res
        text = res.summary()
        print(f"[Multi Vitals] {self.name}: {text}")
        for n in res.notes:
            print(f"[Multi Vitals]   {n}")
        self.summary.configure(text=text + ("" if not res.notes else "  (" + "; ".join(res.notes) + ")"))
        self.redraw()

    def clear(self, text: str):
        """Forget the result (a new recording, or one whose data changed)."""
        if self.result is None:
            return
        self.result = None
        self._artists = None
        self.summary.configure(text=text)
        for fig, canvas in ((self.map_fig, self.map_canvas), (self.rates_fig, self.rates_canvas)):
            fig.clear()
            fig.text(0.5, 0.5, text, ha="center", va="center", color="gray")
            canvas.draw_idle()

    # --- drawing -----------------------------------------------------------------------------
    def shown(self) -> list:
        res = self.result
        if res is None:
            return []
        return [p for p in res.people if self.show_echoes.get() or p.echo_of is None]

    def redraw(self):
        if self.result is None:
            return
        self._draw_map()
        self._draw_rates()
        self.show_frame(self._frame)

    def _block_of(self, frame: int) -> int:
        res = self.result
        nb = max(1, int(round(res.block_s * res.fps)))
        return int(np.clip(frame // nb, 0, res.block_t.size - 1))

    def _draw_map(self):
        res = self.result
        fig = self.map_fig
        fig.clear()
        ax = fig.add_subplot(111)
        r, a = res.range_m, res.angles_deg
        mesh = None
        if self.show_map.get() and a.size > 1 and r.size > 1:
            dr = np.diff(r).mean()
            da = np.diff(a).mean()
            re = np.concatenate([r - dr / 2, [r[-1] + dr / 2]])
            ae = np.deg2rad(np.concatenate([a - da / 2, [a[-1] + da / 2]]))
            X = re[:, None] * np.sin(ae)[None, :]
            Y = re[:, None] * np.cos(ae)[None, :]
            # from the detection threshold to the strongest motion of the recording
            lo = res.map_thresh_db
            hi = max(lo + 10.0, float(np.percentile(np.asarray(res.map_db, float), 99.9)))
            mesh = ax.pcolormesh(X, Y, np.asarray(res.map_db[0], float), cmap="Greys",
                                 vmin=lo, vmax=hi, shading="flat", alpha=0.45, zorder=0)
            fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.02,
                         label="motion above the noise [dB]")
        rmax = float(r[-1]) if r.size else multi_vitals.R_MAX_M
        if a.size > 1:
            for ang in (a[0], a[-1]):
                ax.plot([0, rmax * np.sin(np.deg2rad(ang))], [0, rmax * np.cos(np.deg2rad(ang))],
                        color="0.7", lw=0.8, zorder=1)
        for rr in np.arange(1.0, rmax + 0.01, 1.0):
            th = np.deg2rad(np.linspace(a[0], a[-1], 60) if a.size > 1 else np.linspace(-60, 60, 60))
            ax.plot(rr * np.sin(th), rr * np.cos(th), color="0.85", lw=0.6, zorder=1)
        ax.plot([0], [0], marker="^", color="black", ms=10, zorder=3)
        ax.set_aspect("equal")
        half = rmax * (np.sin(np.deg2rad(np.abs(a).max())) if a.size > 1 else 0.6)
        ax.set_xlim(-half - 0.3, half + 0.3)
        ax.set_ylim(-0.2, rmax + 0.2)
        ax.set_xlabel("Across [m] (angle sign as in the Angle of Arrival views)")
        ax.set_ylabel("Away from the radar [m]")
        dots = {}
        for p in self.shown():
            c = color(p.id) if p.echo_of is None else "0.6"
            trail, = ax.plot([], [], "-", color=c, lw=1.2, alpha=0.7, zorder=4)
            dot, = ax.plot([], [], "o", color=c, ms=11, zorder=5,
                           mfc=c if p.echo_of is None else "none", mew=2)
            label = ax.text(0, 0, "", fontsize=9, color=c, zorder=6, ha="left", va="bottom",
                            fontweight="bold", clip_on=True,
                            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))
            dots[p.id] = (p, trail, dot, label)
        title = ax.set_title("", fontsize=10)
        self._artists = dict(ax=ax, mesh=mesh, dots=dots, title=title)

    def show_frame(self, frame: int):
        """Put the dots, labels and the map at the frame on the frame slider."""
        self._frame = int(frame)
        res = self.result
        if res is None or not self._artists:
            return
        b = self._block_of(self._frame)
        t = self._frame / res.fps
        art = self._artists
        if art["mesh"] is not None:
            m = np.asarray(res.map_db[b], float)
            art["mesh"].set_array(m.ravel())
            # the strongest motion of the moment and MAP_RANGE_DB below it: over the whole
            # range from the threshold a person's sidelobes lit their entire range ring
            top = float(np.nanmax(m)) if m.size else res.map_thresh_db + 10.0
            art["mesh"].set_clim(max(res.map_thresh_db, top - MAP_RANGE_DB), max(top, res.map_thresh_db + 1.0))
        trail_blocks = int(round(TRAIL_S / res.block_s))
        n_here = 0
        # labels of people close together are stacked, not drawn over each other
        x0, x1 = art["ax"].get_xlim()
        y0, y1 = art["ax"].get_ylim()
        box_w, box_h = 0.22 * (x1 - x0), 0.09 * (y1 - y0)
        placed = []

        def free_spot(x, y):
            while any(abs(x - u) < box_w and abs(y - v) < box_h for u, v in placed):
                y += box_h
            placed.append((x, y))
            return x, y

        order = sorted(art["dots"].values(),
                       key=lambda d: -(d[0].at(b, t, res.block_s) or {"y": -1e9})["y"])
        for p, trail, dot, label in order:
            st = p.at(b, t, res.block_s)
            if st is None:
                trail.set_data([], [])
                dot.set_data([], [])
                label.set_text("")
                continue
            n_here += 1
            k = np.flatnonzero((p.blocks <= b) & (p.blocks > b - trail_blocks))
            trail.set_data(p.x_m[k], p.y_m[k])
            dot.set_data([st["x"]], [st["y"]])
            dot.set_alpha(1.0 if st["detected"] else 0.45)
            tag = f"#{p.id}" + (f" (echo of #{p.echo_of})" if p.echo_of is not None else "")
            label.set_position(free_spot(st["x"] + 0.12, st["y"] + 0.08))
            label.set_text(f"{tag}\n{rate_text(st)}")
        art["title"].set_text(f"{res.stem}  t = {res.t0_s + t:.1f} s  -  {n_here} here  "
                              f"({res.method}, {res.beam})")
        self.map_canvas.draw_idle()

    def _draw_rates(self):
        res = self.result
        fig = self.rates_fig
        fig.clear()
        ax_br, ax_hr, ax_d = fig.subplots(3, 1, sharex=True)
        shown = [p for p in self.shown() if p.win_t.size or p.disp_t.size]
        # chest traces one above the other, as far apart as the largest swing
        swing = max([np.ptp(p.disp_mm) for p in shown if p.disp_t.size] or [1.0])
        for k, p in enumerate(shown):
            c = color(p.id) if p.echo_of is None else "0.6"
            lab = f"#{p.id}" + (" (echo)" if p.echo_of is not None else "")
            if p.win_t.size:
                br = np.where(p.br_clear, p.br_rate, np.nan)
                ax_br.plot(p.win_t, br, "-o", color=c, ms=3, lw=1.2, label=lab)
                ax_br.plot(p.win_t[~p.br_clear], p.br_rate[~p.br_clear], "o", mfc="none",
                           color=c, ms=3, alpha=0.5)
                hr = np.where(p.hr_valid, p.hr_rate, np.nan)
                ax_hr.plot(p.win_t, hr, "-o", color=c, ms=3, lw=1.2, label=lab)
            if p.disp_t.size:
                off = 1.1 * swing * k
                ax_d.plot(p.disp_t, p.disp_mm + off, color=c, lw=0.6)
                ax_d.text(p.disp_t[0], off, lab, color=c, fontsize=8, ha="right", va="center")
        ax_br.set_ylabel("Breathing [/min]")
        ax_br.set_title("Breathing (rings: windows without a clear rhythm), heart (where it "
                        "counts) and chest displacement of every person", fontsize=9)
        ax_hr.set_ylabel("Heart [bpm]")
        ax_d.set_ylabel("Chest [mm], offset")
        ax_d.set_xlabel("Time [s]")
        for ax in (ax_br, ax_hr, ax_d):
            ax.grid(alpha=0.3)
        if shown:
            ax_br.legend(fontsize=8, loc="upper right", ncol=min(5, len(shown)))
        else:
            ax_br.text(0.5, 0.5, "Nobody stood still long enough for vital signs.",
                       transform=ax_br.transAxes, ha="center", color="gray")
        if res.frames:
            ax_d.set_xlim(0, res.frames / res.fps)
        fig.tight_layout()
        self.rates_canvas.draw_idle()

    # --- export ------------------------------------------------------------------------------
    def positions_for_export(self):
        """Every person in every block: (array, header) or None."""
        res = self.result
        if res is None:
            return None
        rows = []
        for p in res.people:
            for k, b in enumerate(p.blocks):
                t = float(res.block_t[b])
                st = p.at(int(b), t, res.block_s) or {}
                rows.append([p.id, t, p.x_m[k], p.y_m[k], np.hypot(p.x_m[k], p.y_m[k]),
                             np.rad2deg(np.arctan2(p.x_m[k], p.y_m[k])), float(p.detected[k]),
                             float(p.moving[k]), st.get("br", np.nan), st.get("hr", np.nan),
                             -1 if p.echo_of is None else p.echo_of])
        if not rows:
            return None
        return (np.array(rows, float),
                "id,time_s,x_m,y_m,range_m,angle_deg,detected,moving,br_per_min,hr_bpm,echo_of")

    def export_rates(self):
        res = self.result
        if res is None:
            messagebox.showwarning("Export", "Run Multi Vitals first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")],
                                            title="Save the rates of every person as CSV")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write("id,time_s,br_per_min,br_clear,hr_bpm,hr_valid,echo_of\n")
            for p in res.people:
                for i in range(p.win_t.size):
                    f.write(f"{p.id},{p.win_t[i]:.2f},{p.br_rate[i]:.2f},{int(p.br_clear[i])},"
                            f"{p.hr_rate[i]:.2f},{int(p.hr_valid[i])},"
                            f"{'' if p.echo_of is None else p.echo_of}\n")
        messagebox.showinfo("Export", f"Rates saved to {path}")

    def export_rates_png(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")],
                                            title="Save the Rates figure")
        if path:
            self.rates_fig.savefig(path, dpi=150)
