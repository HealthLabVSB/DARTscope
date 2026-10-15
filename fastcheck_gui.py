"""
fastcheck_gui.py

The Fastcheck tab: one button checks the whole recording (fastcheck.py) and shows

    - a verdict and the key numbers (length, frames, how complete, dropouts),
    - every finding, worst first - the capture health report of the logs, files and
      first frames, and what the pass over every frame found,
    - four plots: the range-time intensity (RTI) with the dropouts marked; the power
      of every receiver per frame with zero-filled, clipped and glitched frames; the
      energy in the breathing and heart bands by range; and at the range with the most
      breathing, both bands over time with the breathing rate.

The tab knows nothing about loading: the host passes get_data() (the decoded capture,
loaded on demand as for the Analysis views) and run(data) (the check in the
background, with the progress bar and Cancel).

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import theme

import numpy as np
from matplotlib.figure import Figure

import fastcheck
from flow_layout import FlowFrame
from plot_interactivity import create_interactive_canvas

VERDICT_FG = {"ok": theme.GREEN, "info": theme.GREEN, "warning": theme.ORANGE, "error": theme.RED}
ROW_BG = {"ok": "#ffffff", "info": theme.TINT_BLUE, "warning": theme.TINT_ORANGE, "error": theme.TINT_RED}
MARK = {"ok": "ok", "info": "info", "warning": "WARNING", "error": "ERROR"}
# receivers in colours that are not the red of the dropouts
RX_COLORS = ("#007AFF", "#FF9500", "#34C759", "#AF52DE", "#A2845E", "#30B0C7", "#8E8E93", "#5856D6")
HINT = ("Select a recording in the Decode tab and press Run Fastcheck. The first run "
        "decodes it; the Analysis views reuse the decoded data.")


class FastcheckTab:
    def __init__(self, parent, get_data, run, report_error=None, open_manual=None,
                 register_run_button=None, cmap=None):
        self.parent = parent
        self.get_data = get_data
        self.run_fn = run
        self.report_error = report_error or (lambda title, msg, exc=None:
                                             messagebox.showerror(title, msg))
        self.open_manual = open_manual or (lambda section=None: None)
        self.cmap = cmap or (lambda default="viridis": default)
        self.result: fastcheck.Fastcheck | None = None
        self._findings = {}
        self._build(register_run_button)
        self.clear()

    # --- layout --------------------------------------------------------------------------
    def _build(self, register_run_button):
        bar = FlowFrame(self.parent)
        bar.pack(side="top", fill="x", padx=8, pady=(6, 2))
        g = bar.group()
        self.run_button = theme.Button(g, text="Run Fastcheck", command=self.run_check)
        self.run_button.pack(side="left")
        if register_run_button is not None:
            register_run_button("Fastcheck", self.run_button)
        self.motion_var = tk.BooleanVar(value=False)
        tk.Checkbutton(bar.group(), text="Motion only (RTI)", variable=self.motion_var,
                       command=self._redraw).pack(side="left")
        g = bar.group()
        info = tk.Label(g, text="ⓘ", font=theme.font(12), fg=theme.ACCENT, cursor="hand2")
        info.pack(side="left", padx=(0, 6))
        info.bind("<Button-1>", lambda _e: self.open_manual("Fastcheck tab"))
        export = ttk.Menubutton(g, text="Export")
        menu = tk.Menu(export, tearoff=0)
        menu.add_command(label="Report (TXT)", command=self.export_report)
        menu.add_command(label="Copy report", command=self.copy_report)
        menu.add_command(label="Frames (CSV)", command=self.export_frames)
        menu.add_command(label="Figure (PNG)", command=self.export_figure)
        export["menu"] = menu
        export.pack(side="left")
        self.status_var = tk.StringVar(value="")
        ttk.Label(bar.group(), textvariable=self.status_var, foreground=theme.SECONDARY).pack(side="left")

        head = ttk.Frame(self.parent)
        head.pack(side="top", fill="x", padx=10)
        self.verdict = tk.Label(head, text="", font=theme.font(12, "bold"), anchor="w",
                                justify="left")
        self.verdict.pack(side="top", fill="x")
        self.numbers = ttk.Label(head, text="", anchor="w", justify="left")
        self.numbers.pack(side="top", fill="x")
        # both lines wrap with the window instead of running off its edge
        head.bind("<Configure>", lambda e: [w.configure(wraplength=max(200, e.width - 10))
                                            for w in (self.verdict, self.numbers)])

        paned = ttk.PanedWindow(self.parent, orient="vertical")
        paned.pack(side="top", fill="both", expand=True, padx=8, pady=6)

        table = ttk.Frame(paned)
        cols = ("status", "check", "result")
        self.tree = ttk.Treeview(table, columns=cols, show="headings", height=7,
                                 selectmode="browse")
        for c, title, width, stretch in (("status", "Status", 80, False),
                                         ("check", "Check", 150, False),
                                         ("result", "Result", 900, True)):
            self.tree.heading(c, text=title, anchor="w")
            self.tree.column(c, width=width, minwidth=60, stretch=stretch, anchor="w")
        for sev, bg in ROW_BG.items():
            self.tree.tag_configure(sev, background=bg)
        ys = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ys.set)
        self.detail = tk.Label(table, text="", anchor="w", justify="left", wraplength=1100,
                               fg=theme.LABEL)
        self.detail.pack(side="bottom", fill="x", pady=(2, 0))
        ys.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        table.bind("<Configure>", lambda e: self.detail.configure(wraplength=max(200, e.width - 20)))
        paned.add(table, weight=1)

        plots = ttk.Frame(paned)
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas, self.toolbar = create_interactive_canvas(plots, self.fig)
        paned.add(plots, weight=3)

    # --- running ---------------------------------------------------------------------------
    def run_check(self):
        data = self.get_data()
        if data is None:
            return
        t0 = time.perf_counter()
        try:
            res = self.run_fn(data)
        except Exception as e:
            self.report_error("Fastcheck", f"Fastcheck failed: {e}", e)
            return
        if res is None:
            return
        self.result = res
        print("[Fastcheck] " + res.report().replace("\n", "\n[Fastcheck] "))
        self.status_var.set(f"{res.stem}: checked in {time.perf_counter() - t0:.1f} s")
        self._show(res)

    def clear(self, reason: str | None = None):
        """Forget the result (a new recording was opened)."""
        self.result = None
        self.verdict.configure(text="", fg=theme.LABEL)
        self.numbers.configure(text="")
        self.tree.delete(*self.tree.get_children(""))
        self.detail.configure(text="")
        self.status_var.set(reason or "")
        self.fig.clear()
        self.fig.text(0.5, 0.5, HINT, ha="center", va="center", wrap=True, color="gray",
                      fontsize=11)
        self.canvas.draw_idle()

    def mark_stale(self, reason: str):
        """Keep the result on screen, but say it no longer matches the settings."""
        if self.result is not None:
            self.status_var.set(f"{reason} - press Run Fastcheck to update")

    # --- showing -----------------------------------------------------------------------------
    def _show(self, res: fastcheck.Fastcheck):
        self.verdict.configure(text=res.verdict(), fg=VERDICT_FG.get(res.worst, theme.LABEL))
        self.numbers.configure(text=res.key_numbers())
        self.tree.delete(*self.tree.get_children(""))
        self._findings.clear()
        order = {s: i for i, s in enumerate(("error", "warning", "info", "ok"))}
        for f in sorted(res.findings, key=lambda f: order.get(f.severity, 9)):
            iid = self.tree.insert("", "end", values=(MARK[f.severity], f.check, f.message),
                                   tags=(f.severity,))
            self._findings[iid] = f
        self.detail.configure(text="Click a row to read all of it here.")
        self._redraw()

    def _on_select(self, _evt=None):
        sel = self.tree.selection()
        f = self._findings.get(sel[0]) if sel else None
        if f is not None:
            self.detail.configure(text=f"{f.check}: {f.message}")

    def _redraw(self):
        res = self.result
        if res is None:
            return
        fig = self.fig
        fig.clear()
        gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.5, wspace=0.3,
                              left=0.07, right=0.93, top=0.93, bottom=0.09)
        ax_rti = fig.add_subplot(gs[0, 0])
        ax_int = fig.add_subplot(gs[0, 1], sharex=ax_rti)
        ax_rng = fig.add_subplot(gs[1, 0])
        sub = gs[1, 1].subgridspec(2, 1, hspace=0.12)
        ax_br = fig.add_subplot(sub[0], sharex=ax_rti)
        ax_en = fig.add_subplot(sub[1], sharex=ax_rti)
        self._plot_rti(ax_rti, res)
        self._plot_integrity(ax_int, res)
        self._plot_range_energy(ax_rng, res)
        self._plot_time_energy(ax_br, ax_en, res)
        self.canvas.draw_idle()

    @staticmethod
    def _spans(ax, res, ymin=0.0, ymax=1.0, label=None):
        for i, (a, b) in enumerate(fastcheck.damaged_spans(res.damaged)):
            ax.axvspan(a / res.fps, max(b, a + 1) / res.fps, ymin=ymin, ymax=ymax, color="red",
                       alpha=0.35, lw=0, label=label if i == 0 else None)

    def _plot_rti(self, ax, res):
        motion = bool(self.motion_var.get())
        img = res.motion_db if motion else res.rti_db
        finite = img[np.isfinite(img)]
        lo, hi = (np.percentile(finite, [5, 99.7]) if finite.size else (0, 1))
        r = res.range_m
        ax.imshow(img.T, aspect="auto", origin="lower", cmap=self.cmap("viridis"),
                  extent=[0, res.duration_s, r[0], r[-1]], vmin=lo, vmax=hi,
                  interpolation="nearest")
        self._spans(ax, res, 0.94, 1.0)
        if res.best_bin is not None:
            ax.axhline(r[res.best_bin], color="white", ls="--", lw=1.0)
            ax.text(res.duration_s, r[res.best_bin], " breathing", color="black", fontsize=7,
                    va="center", ha="left", clip_on=False)
        ax.set_title("RTI - motion only" if motion else "RTI (dropouts in red on top)",
                     fontsize=9)
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.set_ylabel("Range [m]", fontsize=8)
        ax.tick_params(labelsize=7)

    def _plot_integrity(self, ax, res):
        t = res.t_s
        good = ~res.damaged
        rx = np.where(res.damaged[:, None], np.nan, res.rx_db)   # broken at the dropouts
        for r in range(rx.shape[1]):
            ax.plot(t, rx[:, r], lw=0.7, color=RX_COLORS[r % len(RX_COLORS)], label=f"RX{r + 1}")
        vals = res.rx_db[good] if good.any() else res.rx_db
        lo, hi = np.percentile(vals, 1) - 12, np.percentile(vals, 99) + 6
        ax.set_ylim(lo, hi)
        self._spans(ax, res, label="zero-filled")
        top = hi - 1.5
        clip = res.clip_frac >= fastcheck.CLIP_FRAME
        if clip.any():
            ax.plot(t[clip], np.full(int(clip.sum()), top), "v", color="orange", ms=5,
                    label="clipped")
        if res.glitch.any():
            ax.plot(t[res.glitch], np.full(int(res.glitch.sum()), top - 2.5), "x",
                    color="purple", ms=6, label="glitch")
        ax.set_title("Integrity: power of every receiver per frame", fontsize=9)
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.set_ylabel("Power [dBFS]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="lower right", ncol=3, framealpha=0.7)

    def _plot_range_energy(self, ax, res):
        ax.set_title("Vital-band energy by range", fontsize=9)
        if not np.isfinite(res.br_ratio).any():
            ax.text(0.5, 0.5, self._not_judged(res), ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray", wrap=True)
            ax.set_axis_off()
            return
        r = res.range_m
        db = lambda v: 10 * np.log10(np.maximum(v, 1e-12))   # noqa: E731
        ax.plot(r, db(res.br_ratio), lw=1.0,
                label=f"breathing {res.bands.resp_bpm[0]:g}-{res.bands.resp_bpm[1]:g} /min")
        if np.isfinite(res.hr_ratio).any():
            ax.plot(r, db(res.hr_ratio), lw=1.0,
                    label=f"heart {res.bands.heart_bpm[0]:g}-{res.bands.heart_bpm[1]:g} bpm")
        if res.best_bin is not None:
            ax.axvline(r[res.best_bin], color="k", ls="--", lw=0.8,
                       label=f"most breathing: {r[res.best_bin]:.2f} m")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xlabel("Range [m]", fontsize=8)
        ax.set_ylabel("Band power / noise [dB]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

    def _plot_time_energy(self, ax_br, ax_en, res):
        """Top: the breathing rate of every window. Bottom: how far the breathing and the
        heart band stand above the noise at that range, window by window."""
        if res.win_t.size == 0:
            msg = (self._not_judged(res) if not np.isfinite(res.br_ratio).any()
                   else "No range with slow motion to follow.")
            ax_br.set_title("Breathing and heart band over time", fontsize=9)
            ax_br.text(0.5, 0.5, msg, ha="center", va="center", transform=ax_br.transAxes,
                       fontsize=8, color="gray", wrap=True)
            ax_br.set_axis_off()
            ax_en.set_axis_off()
            return
        where = res.range_m[res.best_bin]
        clear = res.clear_rhythm()
        win = res.notes.get("vital_window_s", fastcheck.WINDOW_S)
        for ax in (ax_br, ax_en):
            self._spans(ax, res)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
        # rate: a line through the clear windows, the others as grey rings
        rate_clear = np.where(clear, res.br_rate, np.nan)
        ax_br.plot(res.win_t, rate_clear, "-o", color="black", ms=3, lw=1.0,
                   label="clear rhythm")
        other = ~clear & np.isfinite(res.br_rate)
        if other.any():
            ax_br.plot(res.win_t[other], res.br_rate[other], "o", mfc="none", mec="gray", ms=4,
                       label="unclear")
        # breathing in the lower part of the plot, the heart rate in the upper: on their
        # own ranges the two lines ran across each other
        lo, hi = res.bands.resp_bpm
        ax_br.set_ylim(0, 2.2 * hi)
        ax_br.set_yticks([v for v in np.arange(0, hi + 1, 10) if lo <= v <= hi] or [lo, hi])
        ax_br.set_ylabel("Breathing [/min]", fontsize=8)
        ax_br.tick_params(labelbottom=False)
        rate = (f"breathing {np.nanmedian(res.br_rate[clear]):.0f} /min" if clear.any()
                else "no clear breathing")
        # the heart rate on its own axis, in red: solid where its peak counts
        handles, labels = ax_br.get_legend_handles_labels()
        if res.hr_rate.size and np.isfinite(res.hr_rate).any():
            ax_hr = ax_br.twinx()
            ax_hr.plot(res.win_t, res.hr_rate, ":", color="#d62728", lw=1.0)
            ax_hr.plot(res.win_t, np.where(res.hr_valid, res.hr_rate, np.nan), "-s",
                       color="#d62728", ms=3, lw=1.3, label="heart rate")
            h_lo, h_hi = res.bands.heart_bpm
            ax_hr.set_ylim(h_lo - 1.2 * (h_hi - h_lo), h_hi + 0.25 * (h_hi - h_lo))
            ax_hr.set_yticks([v for v in np.arange(0, h_hi + 1, 30) if h_lo <= v <= h_hi])
            ax_hr.set_ylabel("Heart [bpm]", fontsize=8, color="#d62728")
            ax_hr.tick_params(labelsize=7, colors="#d62728")
            h2, l2 = ax_hr.get_legend_handles_labels()
            handles, labels = handles + h2, labels + l2
            if res.hr_valid.any():
                rate += f", heart {np.median(res.hr_rate[res.hr_valid]):.0f} bpm"
        ax_br.set_title(f"At {where:.2f} m ({win:.0f} s windows): {rate}", fontsize=9)
        ax_br.legend(handles, labels, fontsize=7, loc="upper left", ncol=3, framealpha=0.7)
        # energy: band power over the noise floor above the heart band
        ax_en.plot(res.win_t, res.br_db, "-", color="#1f77b4", lw=1.2, label="breathing band")
        if np.isfinite(res.hr_db).any():
            ax_en.plot(res.win_t, res.hr_db, "-", color="#d62728", lw=1.2, label="heart band")
        ax_en.axhline(0, color="gray", lw=0.6)
        ax_en.set_ylabel("Over noise [dB]", fontsize=8)
        ax_en.set_xlabel("Time [s] (window centres; red: lost frames)", fontsize=8)
        ax_en.legend(fontsize=7, loc="lower right", ncol=2, framealpha=0.7)

    @staticmethod
    def _not_judged(res) -> str:
        return (f"Not judged: {res.duration_s:.1f} s at {res.fps:g} fps is too little for "
                f"the breathing band (at least {fastcheck.MIN_VITAL_S:g} s needed).")

    # --- export ----------------------------------------------------------------------------
    def _need_result(self) -> bool:
        if self.result is None:
            messagebox.showwarning("Fastcheck", "Run Fastcheck first.")
            return False
        return True

    def export_report(self):
        if not self._need_result():
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                            initialfile=f"{self.result.stem}_fastcheck.txt",
                                            filetypes=[("Text files", "*.txt")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.result.report() + "\n")
            messagebox.showinfo("Fastcheck", f"Report saved to {path}")
        except OSError as e:
            messagebox.showerror("Fastcheck", f"Could not save: {e}")

    def copy_report(self):
        if not self._need_result():
            return
        self.parent.clipboard_clear()
        self.parent.clipboard_append(self.result.report())
        self.status_var.set("Report copied to the clipboard")

    def export_frames(self):
        if not self._need_result():
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            initialfile=f"{self.result.stem}_frames.csv",
                                            filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        data, header = self.result.frame_table()
        try:
            np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.6g")
            messagebox.showinfo("Fastcheck", f"Frames saved to {path}")
        except OSError as e:
            messagebox.showerror("Fastcheck", f"Could not save: {e}")

    def export_figure(self):
        if not self._need_result():
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            initialfile=f"{self.result.stem}_fastcheck.png",
                                            filetypes=[("PNG files", "*.png")])
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=150)
            messagebox.showinfo("Fastcheck", f"Figure saved to {path}")
        except OSError as e:
            messagebox.showerror("Fastcheck", f"Could not save: {e}")
