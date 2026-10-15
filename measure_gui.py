"""
measure_gui.py

The Measure tab: record with the IWR6843AOPEVM + DCA1000 without mmWave Studio.

    Devices    the radar's COM port found by its USB identity (acquisition.ports), the
               DCA1000 checked on the network, a warning while mmWave Studio runs (it
               holds the same USB devices); or the simulator, to try everything without
               hardware
    Single     the settings in physical units (or a lab script, a recording's log, a
               preset from the chirp Helper) with what they give (resolutions, maximum
               range and velocity, data rate) and every check. "Prepare" configures the
               radar - the first time it loads the firmware (~20 s), afterwards about a
               second - then the operator takes their place and starts the take
               (the Start button or the space bar), after an optional countdown with
               beeps. A short report follows (acquisition.quality); the recording opens
               in the Decode tab with one click.
    Series     parameters and their values (TX pattern, RX, loops, slope, frame period
               ...), every combination planned and checked, each configuration recorded
               once or more, in order or shuffled; every take started by the operator.
               After each take the ranking for the chosen kind of movement, how much
               each parameter mattered, and what to try next (acquisition.series), also
               as an HTML report.

The hardware work runs in a worker thread; it talks to Tk only through a queue.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import copy
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import theme

from acquisition.config import ConfigError, RadarConfig
from acquisition.params import MeasureParams, checks, derived, format_rx, format_tx, parse_tx
from acquisition.quality import OBJECTIVES, analyze_recording, save as save_quality
from acquisition.series import PARAMS, Series, format_value, parse_values, plan
from acquisition.serialport import port_label

def open_in_file_manager(folder: str) -> bool:
    """Show a folder in the file manager: Explorer on Windows, Finder on macOS, the desktop's
    own elsewhere (xdg-open). os.startfile alone exists only on Windows - on a Mac the
    Open folder button did nothing."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(folder)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", folder],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"[Measure] the folder cannot be shown: {e}")
        return False


FIELDS = (   # attribute, label, unit, kind
    ("start_ghz", "Start frequency", "GHz", float),
    ("slope_mhz_us", "Slope", "MHz/us", float),
    ("samples", "ADC samples", "", int),
    ("fs_ksps", "Sample rate", "ksps", int),
    ("idle_us", "Idle time", "us", float),
    ("adc_start_us", "ADC start time", "us", float),
    ("ramp_end_us", "Ramp end time (0 = auto)", "us", float),
    ("rx_gain_db", "RX gain", "dB", int),
    ("loops", "Loops per frame", "", int),
    ("period_ms", "Frame period", "ms", float),
    ("duration_s", "Duration (0 = until Stop)", "s", float),
)
TEXT_FONT = theme.font(9)
MOVEMENTS = ("nothing (empty scene)", "sitting still, breathing", "standing, swaying",
             "walking towards / away", "walking across", "squats", "arm waving",
             "hand gestures", "other")
IDLE, PREPARING, READY, COUNTDOWN, RECORDING, ANALYSING = (
    "idle", "preparing", "ready", "countdown", "recording", "analysing")
STATE_TEXT = {
    IDLE: ("Not prepared", theme.SECONDARY),
    PREPARING: ("Preparing the radar ...", theme.BLUE),
    READY: ("READY - take your place and press Start (space bar)", theme.GREEN),
    COUNTDOWN: ("", theme.ORANGE),
    RECORDING: ("RECORDING", theme.RED),
    ANALYSING: ("Analysing the take ...", theme.BLUE),
}
LEVEL_FG = dict(theme.LEVEL)
TYPING = ("Entry", "TEntry", "Text", "TCombobox", "Spinbox", "TSpinbox", "Listbox")


def _settings_path() -> Path:
    try:
        from platformdirs import user_config_dir  # type: ignore
        d = Path(user_config_dir("bme_radar_gui", appauthor=False))
    except Exception:
        d = Path.home() / ".bme_radar_gui"
    d.mkdir(parents=True, exist_ok=True)
    return d / "measure_gui.json"


def _beep(pattern):
    """[(Hz, ms), ...] off the Tk thread (winsound blocks)."""
    def run():
        try:
            import winsound
            for f, ms in pattern:
                winsound.Beep(int(f), int(ms))
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()


class MeasureTab:
    def __init__(self, parent, open_recording=None, report_error=None):
        self.parent = parent
        self.open_recording = open_recording
        self.report_error = report_error or (lambda title, msg, exc=None: messagebox.showerror(title, msg))
        self.q: queue.Queue = queue.Queue()
        self.state = IDLE
        self.backend = None
        self.backend_key = None
        self.worker: threading.Thread | None = None
        self.gate: threading.Event | None = None
        self.stop_event = threading.Event()
        self.base_cfg = RadarConfig()
        self.base_source = ""
        self.detection = None
        self.last_stem = None
        self.series: Series | None = None
        self.series_queue: list = []
        self.current_step = None
        self.planned_steps: list = []
        self.job_kind = None                   # "single" or "series"
        self._countdown_left = 0
        self._record_t0 = None
        self._eval_after = None
        self._settings = self._load_settings()
        self._build()
        self._apply_settings()
        self._detect_ports()
        self._evaluate()
        self.frame.after(100, self._poll)
        self.frame.bind("<Destroy>", self._on_destroy, add="+")
        self.frame.winfo_toplevel().bind_all("<KeyPress-space>", self._on_space, add="+")

    # --- layout ------------------------------------------------------------------------------
    def _build(self):
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill="both", expand=True)
        dev = ttk.LabelFrame(self.frame, text="Devices")
        dev.pack(side="top", fill="x", padx=8, pady=(6, 4))
        self.mode_var = tk.StringVar(value="native")
        modes = ttk.Frame(dev)
        modes.grid(row=0, column=0, sticky="w", padx=4)
        ttk.Radiobutton(modes, text="Radar (no mmWave Studio)", variable=self.mode_var, value="native",
                        command=self._on_mode).pack(side="left")
        ttk.Radiobutton(modes, text="Simulator (no hardware)", variable=self.mode_var,
                        value="simulator", command=self._on_mode).pack(side="left", padx=(8, 0))
        ttk.Label(dev, text="Radar COM port:").grid(row=0, column=1, sticky="e", padx=(16, 4))
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(dev, textvariable=self.port_var, width=58, state="readonly")
        self.port_combo.grid(row=0, column=2, sticky="w")
        ttk.Button(dev, text="Detect", command=self._detect_ports).grid(row=0, column=3, padx=4)
        ttk.Button(dev, text="Check DCA1000", command=self._check_dca).grid(row=0, column=4, padx=4)
        self.radar_state = ttk.Label(dev, text="radar: not powered", foreground=theme.SECONDARY)
        self.radar_state.grid(row=1, column=0, sticky="w", padx=4, pady=(2, 2))
        self.dev_status = ttk.Label(dev, text="", foreground=theme.SECONDARY, wraplength=900)
        self.dev_status.grid(row=1, column=1, columnspan=4, sticky="w", padx=(16, 4))

        # packed before the notebook: the notebook expands into what is left, so the
        # Take panel must claim its strip first
        run = ttk.LabelFrame(self.frame, text="Take")
        run.pack(side="bottom", fill="x", padx=8, pady=(0, 6))
        self.nb = ttk.Notebook(self.frame)
        self.nb.pack(side="top", fill="both", expand=True, padx=8, pady=4)
        single = ttk.Frame(self.nb)
        series = ttk.Frame(self.nb)
        self.nb.add(single, text="Single measurement")
        self.nb.add(series, text="Series (parameter sweep)")
        self._build_settings(single)
        self._build_series(series)
        self.state_label = tk.Label(run, text="", font=theme.font(15, "bold"), anchor="w")
        self.state_label.grid(row=0, column=0, columnspan=8, sticky="we", padx=6, pady=(4, 2))
        self.prepare_btn = ttk.Button(run, text="1. Prepare radar", command=self._prepare_single)
        self.prepare_btn.grid(row=1, column=0, padx=4, pady=4)
        self.start_btn = theme.Button(run, text="2. START", width=14, command=self._start,
                                      state="disabled")
        if not isinstance(self.start_btn, ttk.Button):
            self.start_btn.configure(font=theme.font(12, "bold"), fg="white", bg="#9e9e9e")
        self.start_btn.grid(row=1, column=1, padx=4, pady=4)
        self.stop_btn = ttk.Button(run, text="Stop / cancel", command=self._stop, state="disabled")
        self.stop_btn.grid(row=1, column=2, padx=4)
        ttk.Label(run, text="Countdown").grid(row=1, column=3, padx=(16, 2))
        self.countdown_var = tk.IntVar(value=3)
        ttk.Spinbox(run, from_=0, to=30, width=4, textvariable=self.countdown_var).grid(row=1, column=4)
        ttk.Label(run, text="s").grid(row=1, column=5, padx=(2, 8))
        self.beep_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(run, text="beep", variable=self.beep_var).grid(row=1, column=6)
        self.progress = ttk.Progressbar(run, length=320, maximum=100.0)
        self.progress.grid(row=1, column=7, padx=12, sticky="we")
        self.progress_label = ttk.Label(run, text="")
        self.progress_label.grid(row=2, column=0, columnspan=8, sticky="w", padx=6)
        run.columnconfigure(7, weight=1)
        self._set_state(IDLE)

    def _build_settings(self, parent):
        left = ttk.LabelFrame(parent, text="Settings")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=4)
        mid = ttk.LabelFrame(parent, text="What they give")
        mid.grid(row=0, column=1, sticky="nsew", padx=6, pady=4)
        right = ttk.LabelFrame(parent, text="Recording")
        right.grid(row=0, column=2, sticky="nsew", padx=(6, 0), pady=4)
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=1)
        parent.rowconfigure(0, weight=1)

        top = ttk.Frame(left)
        top.grid(row=0, column=0, columnspan=3, sticky="we", pady=(2, 6))
        ttk.Button(top, text="Load script / log ...", command=self._load_file).pack(side="left", padx=2)
        import chirp_design as cd
        self.preset_var = tk.StringVar(value="Preset ...")
        pre = ttk.Combobox(top, textvariable=self.preset_var, values=list(cd.PURPOSES), width=30,
                           state="readonly")
        pre.pack(side="left", padx=4)
        pre.bind("<<ComboboxSelected>>", lambda e: self._apply_preset(self.preset_var.get()))
        self.source_label = ttk.Label(left, text="", foreground=theme.SECONDARY, wraplength=330)
        self.source_label.grid(row=1, column=0, columnspan=3, sticky="w")
        self.vars = {}
        r = 2
        for attr, label, unit, _kind in FIELDS[:8]:
            r = self._field(left, r, attr, label, unit)
        ttk.Label(left, text="TX pattern").grid(row=r, column=0, sticky="w", padx=4, pady=1)
        self.tx_var = tk.StringVar()
        ttk.Entry(left, textvariable=self.tx_var, width=12).grid(row=r, column=1, sticky="w")
        ttk.Label(left, text="1 | 1, 2, 3 | 2, 1+3", foreground=theme.TERTIARY).grid(row=r, column=2, sticky="w")
        self.tx_var.trace_add("write", lambda *a: self._schedule_eval())
        r += 1
        ttk.Label(left, text="RX").grid(row=r, column=0, sticky="w", padx=4, pady=1)
        rxf = ttk.Frame(left)
        rxf.grid(row=r, column=1, columnspan=2, sticky="w")
        self.rx_vars = []
        for i in range(4):
            v = tk.BooleanVar(value=True)
            ttk.Checkbutton(rxf, text=f"RX{i + 1}", variable=v,
                            command=self._schedule_eval).pack(side="left")
            self.rx_vars.append(v)
        r += 1
        for attr, label, unit, _kind in FIELDS[8:]:
            r = self._field(left, r, attr, label, unit)

        self.derived_text = tk.Text(mid, height=16, width=40, wrap="word", relief="flat",
                                    background=self.frame.winfo_toplevel().cget("bg"), font=TEXT_FONT)
        self.derived_text.pack(fill="both", expand=True, padx=4, pady=4)
        for lv, fg in LEVEL_FG.items():
            self.derived_text.tag_configure(lv, foreground=fg)
        self.derived_text.tag_configure("bold", font=theme.font(9, "bold"))

        self.folder_var = tk.StringVar()
        self.name_var = tk.StringVar(value="take")
        self.subject_var = tk.StringVar()
        self.movement_var = tk.StringVar(value=MOVEMENTS[1])
        self.note_var = tk.StringVar()
        self.auto_next_var = tk.BooleanVar(value=True)
        rows = (("Folder", self.folder_var), ("Name", self.name_var), ("Subject", self.subject_var))
        for i, (lab, var) in enumerate(rows):
            ttk.Label(right, text=lab).grid(row=i, column=0, sticky="w", padx=4, pady=2)
            ttk.Entry(right, textvariable=var, width=30).grid(row=i, column=1, sticky="we", pady=2)
        ttk.Button(right, text="...", width=3, command=self._browse_folder).grid(row=0, column=2, padx=2)
        ttk.Label(right, text="Movement").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        ttk.Combobox(right, textvariable=self.movement_var, values=MOVEMENTS, width=28).grid(
            row=3, column=1, sticky="we", pady=2)
        ttk.Label(right, text="Note").grid(row=4, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(right, textvariable=self.note_var, width=30).grid(row=4, column=1, sticky="we", pady=2)
        ttk.Checkbutton(right, text="prepare the next take automatically",
                        variable=self.auto_next_var).grid(row=5, column=0, columnspan=3, sticky="w", padx=4)
        ttk.Label(right, text="Last take:").grid(row=6, column=0, sticky="nw", padx=4, pady=(8, 0))
        self.result_text = tk.Text(right, height=9, width=36, wrap="word", font=TEXT_FONT)
        self.result_text.grid(row=6, column=1, columnspan=2, sticky="nsew", pady=(8, 2))
        for lv, fg in LEVEL_FG.items():
            self.result_text.tag_configure(lv, foreground=fg)
        self.result_text.tag_configure("bold", font=theme.font(9, "bold"))
        btns = ttk.Frame(right)
        btns.grid(row=7, column=1, columnspan=2, sticky="w")
        self.open_btn = ttk.Button(btns, text="Open in DARTscope", command=self._open_last,
                                   state="disabled")
        if self.open_recording is not None:
            self.open_btn.pack(side="left", padx=2)
        self.folder_btn = ttk.Button(btns, text="Open folder", command=self._open_folder, state="disabled")
        self.folder_btn.pack(side="left", padx=2)
        right.columnconfigure(1, weight=1)
        right.rowconfigure(6, weight=1)

    def _field(self, parent, row, attr, label, unit):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=1)
        var = tk.StringVar()
        ttk.Entry(parent, textvariable=var, width=12).grid(row=row, column=1, sticky="w")
        ttk.Label(parent, text=unit, foreground=theme.TERTIARY).grid(row=row, column=2, sticky="w")
        var.trace_add("write", lambda *a: self._schedule_eval())
        self.vars[attr] = var
        return row + 1

    def _build_series(self, parent):
        left = ttk.LabelFrame(parent, text="What to vary")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=4)
        right = ttk.LabelFrame(parent, text="Plan and results")
        right.grid(row=0, column=1, sticky="nsew", pady=4)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        ttk.Label(left, text="Base: the Single measurement settings; the values below replace "
                             "theirs. Numbers: '32, 64, 128' or '50..200/50'. TX / RX: separate "
                             "the choices with '|', e.g. '1 | 1, 2, 3' or '1234 | 12'.",
                  wraplength=330, foreground=theme.SECONDARY).grid(row=0, column=0, columnspan=2, sticky="w",
                                                             padx=4, pady=(2, 6))
        names = [""] + [f"{k} - {p.label}" for k, p in PARAMS.items()]
        self.var_rows = []
        for i in range(4):
            pv, vv = tk.StringVar(), tk.StringVar()
            ttk.Combobox(left, textvariable=pv, values=names, width=30, state="readonly").grid(
                row=1 + i, column=0, sticky="w", padx=4, pady=2)
            ttk.Entry(left, textvariable=vv, width=22).grid(row=1 + i, column=1, sticky="w", padx=4)
            self.var_rows.append((pv, vv))
        f = ttk.Frame(left)
        f.grid(row=5, column=0, columnspan=2, sticky="w", padx=4, pady=(6, 2))
        ttk.Label(f, text="Takes per configuration").pack(side="left")
        self.repeats_var = tk.IntVar(value=1)
        ttk.Spinbox(f, from_=1, to=10, width=4, textvariable=self.repeats_var).pack(side="left", padx=4)
        self.shuffle_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="random order", variable=self.shuffle_var).pack(side="left", padx=6)
        ttk.Label(left, text="Kind of movement (ranking)").grid(row=6, column=0, sticky="w", padx=4, pady=2)
        self.objective_var = tk.StringVar(value=OBJECTIVES["general"][0])
        obj = ttk.Combobox(left, textvariable=self.objective_var, state="readonly", width=30,
                           values=[v[0] for v in OBJECTIVES.values()])
        obj.grid(row=6, column=1, sticky="w", padx=4)
        obj.bind("<<ComboboxSelected>>", lambda e: self._on_objective())
        ttk.Label(left, text="Series name").grid(row=7, column=0, sticky="w", padx=4, pady=2)
        self.series_name_var = tk.StringVar(value="series")
        ttk.Entry(left, textvariable=self.series_name_var, width=24).grid(row=7, column=1, sticky="w", padx=4)
        ttk.Label(left, text="(folder, subject, movement and note: Single tab)", foreground=theme.TERTIARY).grid(
            row=8, column=0, columnspan=2, sticky="w", padx=4)
        b = ttk.Frame(left)
        b.grid(row=9, column=0, columnspan=2, sticky="w", padx=4, pady=(10, 4))
        ttk.Button(b, text="Plan", command=self._plan).pack(side="left", padx=2)
        self.series_start_btn = ttk.Button(b, text="Start series", command=self._series_start)
        self.series_start_btn.pack(side="left", padx=2)
        self.skip_btn = ttk.Button(b, text="Skip take", command=self._series_skip, state="disabled")
        self.skip_btn.pack(side="left", padx=2)
        self.repeat_btn = ttk.Button(b, text="Repeat last", command=self._series_repeat, state="disabled")
        self.repeat_btn.pack(side="left", padx=2)
        self.series_stop_btn = ttk.Button(b, text="End series", command=self._series_end, state="disabled")
        self.series_stop_btn.pack(side="left", padx=2)
        b2 = ttk.Frame(left)
        b2.grid(row=10, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        ttk.Button(b2, text="Open report", command=self._open_report).pack(side="left", padx=2)
        ttk.Button(b2, text="Use the best configuration", command=self._use_best).pack(side="left", padx=2)
        self.plan_label = ttk.Label(left, text="", foreground=theme.SECONDARY, wraplength=330)
        self.plan_label.grid(row=11, column=0, columnspan=2, sticky="w", padx=4, pady=(6, 2))
        self.reco_text = tk.Text(left, height=10, width=48, wrap="word", font=TEXT_FONT)
        self.reco_text.grid(row=12, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)
        left.rowconfigure(12, weight=1)

        cols = ("step", "config", "take", "status", "score", "rres", "vmax", "vres", "fps", "rate", "note")
        heads = ("#", "configuration", "take", "status", "score", "range res. cm", "v max m/s",
                 "v res. m/s", "fps", "MB/s", "report / problems")
        widths = (30, 160, 36, 72, 48, 76, 64, 64, 40, 46, 300)
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=16)
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="w" if c in ("config", "note", "status") else "e",
                             stretch=(c == "note"))
        sb = ttk.Scrollbar(right, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.tag_configure("invalid", foreground=theme.TERTIARY)
        self.tree.tag_configure("current", background=theme.TINT_YELLOW)
        self.tree.tag_configure("done", background=theme.TINT_GREEN)
        self.tree.tag_configure("best", background=theme.TINT_GREEN_STRONG)
        self.tree.tag_configure("skipped", foreground=theme.TERTIARY)
        self.tree.tag_configure("failed", background=theme.TINT_RED)
        self.tree.tag_configure("warn", background=theme.TINT_ORANGE)
        self.tree.tag_configure("bestwarn", background=theme.TINT_GREEN_STRONG, foreground=theme.ORANGE)
        self.tree.bind("<Double-1>", self._tree_open)

    # --- settings: form <-> MeasureParams ---------------------------------------------------
    def _params(self) -> MeasureParams:
        kw = {}
        for attr, label, _unit, kind in FIELDS:
            text = self.vars[attr].get().strip().replace(",", ".")
            try:
                kw[attr] = kind(float(text)) if kind is int else kind(text)
            except ValueError:
                raise ConfigError(f"{label}: {text!r} is not a number") from None
        kw["tx"] = parse_tx(self.tx_var.get())
        kw["rx"] = [i + 1 for i, v in enumerate(self.rx_vars) if v.get()]
        if not kw["rx"]:
            raise ConfigError("enable at least one RX")
        kw["frames"] = 0                    # duration 0: until Stop
        return MeasureParams(**kw)

    def _set_params(self, p: MeasureParams):
        for attr, _label, _unit, kind in FIELDS:
            v = getattr(p, attr)
            self.vars[attr].set(f"{v:g}" if isinstance(v, float) else str(v))
        self.tx_var.set(format_tx(p.tx))
        for i, var in enumerate(self.rx_vars):
            var.set((i + 1) in p.rx)
        self._schedule_eval()

    def _config(self) -> RadarConfig:
        return self._params().to_config(self.base_cfg)

    def _schedule_eval(self):
        if self._eval_after is not None:
            try:
                self.frame.after_cancel(self._eval_after)
            except Exception:
                pass
        self._eval_after = self.frame.after(250, self._evaluate)

    def _evaluate(self):
        self._eval_after = None
        t = self.derived_text
        t.configure(state="normal")
        t.delete("1.0", "end")
        try:
            cfg = self._config()
            d = derived(cfg)
            probs = checks(cfg)
        except Exception as e:
            t.insert("end", f"{e}\n", "error")
            t.configure(state="disabled")
            return None
        lines = [
            ("Range resolution", f"{d['range_res_cm']:.2f} cm"),
            ("Maximum range", f"{d['max_range_m']:.2f} m"),
            ("Maximum velocity", f"+-{d['max_vel_ms']:.2f} m/s"),
            ("Velocity resolution", f"{d['vel_res_ms']:.3f} m/s"),
            ("Frame rate", f"{d['fps']:.1f} fps (chirps {d['duty_pct']:.0f} % of the frame)"),
            ("Chirps per frame", f"{d['chirps_per_frame']} (TX {d['tx_pattern']}, RX {d['rx']}, "
                                 f"{d['virtual_antennas']} virtual antennas)"),
            ("Chirp", f"{d['chirp_time_us']:.2f} us, sweep {d['sweep_ghz']:.3f} GHz"),
            ("Data rate", f"{d['rate_mb_s']:.2f} MB/s"),
            ("Recording", (f"{d['frames']} frames, {d['duration_s']:.1f} s, {d['size_mb']:.0f} MB"
                           if d["frames"] else "until Stop")),
        ]
        for k, v in lines:
            t.insert("end", f"{k}: ", "bold")
            t.insert("end", v + "\n")
        t.insert("end", "\n")
        if not probs:
            t.insert("end", "No problems found.\n", "ok")
        for level, what, msg in probs:
            t.insert("end", f"{level.upper()} {what}: {msg}\n", level)
        t.configure(state="disabled")
        if (self.state == READY and self.job_kind == "single" and self._prepared is not None
                and cfg.to_dict() != self._prepared):
            # the radar holds the old settings: prepare it again with the new ones
            self.progress_label.configure(text="settings changed - preparing the radar again ...")
            self._after_cancel = self._prepare_single
            self.stop_event.set()
        return cfg

    _prepared = None

    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Radar configuration",
            filetypes=[("Lua script, mmWave Studio log, sidecar", "*.lua *.txt *.json"), ("All", "*.*")])
        if path:
            self._load_config_file(path)

    def _load_config_file(self, path: str, quiet: bool = False):
        from acquisition.configfile import load_config
        try:
            cfg, warnings, _text = load_config(path)
        except Exception as e:
            if not quiet:
                self.report_error("Settings", f"{path} could not be read:\n{e}", e)
            return False
        self.base_cfg = cfg
        self.base_source = path
        self.source_label.configure(text=f"from {os.path.basename(path)}" + (
            f" - {warnings[0]}" if warnings else ""))
        self._set_params(MeasureParams.from_config(cfg))
        return True

    def _apply_preset(self, purpose: str):
        import chirp_design as cd
        from acquisition.config import from_design
        try:
            goals = cd.goals_for(purpose)
            cfg = from_design(cd.design(cd.DEVICES["IWR6843AOP"], goals))
        except Exception as e:
            self.report_error("Preset", str(e), e)
            return
        from acquisition.params import fit_duration
        p = MeasureParams.from_config(cfg)
        # the preset's own length (the frame count rounds it up to 30.001 s and the like)
        p.duration_s, shortened = fit_duration(float(goals.duration_s or 30.0), cfg)
        self.base_cfg = cfg
        self.base_source = f"preset: {purpose}"
        note = f", {p.duration_s:g} s to fit one 1 GiB file" if shortened else ""
        self.source_label.configure(text=f"preset '{purpose}' (chirp Helper{note}) - adjust as needed")
        self._set_params(p)

    def _browse_folder(self):
        d = filedialog.askdirectory(initialdir=self.folder_var.get() or os.getcwd())
        if d:
            self.folder_var.set(d)

    # --- devices ---------------------------------------------------------------------------------
    def _detect_ports(self):
        from acquisition.ports import detect, process_running
        try:
            det = detect()
        except Exception as e:
            self.dev_status.configure(text=f"port detection failed: {e}", foreground=LEVEL_FG["error"])
            return
        self.detection = det
        labels = [p.label() for p in det.ports]
        self.port_combo.configure(values=labels)
        want = self._settings.get("com_port")
        chosen = det.radar or next((p for p in det.ports if p.name == want), None)
        if chosen is not None:
            self.port_var.set(chosen.label())
        notes = list(det.notes)
        if process_running("mmWaveStudio.exe"):
            notes.insert(0, "mmWave Studio is running - close it: it holds the radar's COM port and "
                            "the DCA1000's USB")
        if self.mode_var.get() == "native":
            try:
                from acquisition.config import DeviceSetup
                from acquisition.radarlink import native_status
                notes += native_status(firmware=DeviceSetup().firmware_paths())
            except Exception as e:
                notes.append(f"radar path check failed: {e}")
        ok = det.radar is not None and det.dca1000_present and not notes
        self.dev_status.configure(text=det.summary() + ("" if not notes else " - " + "; ".join(notes)),
                                  foreground=LEVEL_FG["ok" if ok else "warning"])

    def _com_port(self):
        """The chosen radar UART: a COM number on Windows, a device path (/dev/cu...) elsewhere."""
        text = self.port_var.get()
        if text.upper().startswith("COM"):
            num = "".join(ch for ch in text.split(" ")[0][3:] if ch.isdigit())
            return int(num) if num else None
        if text.startswith("/dev/"):
            return text.split(" - ")[0].strip()
        return None

    def _check_dca(self):
        from acquisition.status import preflight
        try:
            cfg = self._config()
        except Exception:
            cfg = self.base_cfg
        self.dev_status.configure(text="checking the DCA1000 ...", foreground=theme.SECONDARY)

        def work():
            try:
                rows = preflight(cfg, folder=self.folder_var.get() or None)
                self.q.put(("dca", rows))
            except Exception as e:
                self.q.put(("dca", e))
        threading.Thread(target=work, daemon=True).start()

    def _on_mode(self):
        if self.state not in (IDLE,):
            return
        self._close_backend()
        self._update_radar_state()

    def _get_backend(self):
        mode = self.mode_var.get()
        key = (mode, self._com_port() if mode == "native" else None)
        if self.backend is not None and self.backend_key == key:
            return self.backend
        self._close_backend()
        if mode == "simulator":
            from acquisition.backend import SimulatorBackend
            self.backend = SimulatorBackend()
        else:
            if key[1] is None:
                raise ConfigError("choose the radar's COM port (Detect) - or the simulator")
            from acquisition.backend import HardwareBackend
            self.backend = HardwareBackend(com_port=key[1], on_command=lambda s: None)
        self.backend_key = key
        return self.backend

    def _close_backend(self):
        if self.backend is not None:
            try:
                self.backend.close()
            except Exception:
                pass
        self.backend = None
        self.backend_key = None

    def _update_radar_state(self):
        if self.mode_var.get() == "simulator":
            txt, fg = "simulator: synthetic scene, no hardware", "#1f5fa8"
        elif self.backend is not None and getattr(self.backend, "booted", False):
            txt, fg = "radar: powered, firmware loaded", "#2e7d32"
        else:
            txt, fg = "radar: not powered (first Prepare ~20 s)", "#555555"
        self.radar_state.configure(text=txt, foreground=fg)

    # --- a take ----------------------------------------------------------------------------------
    def _notes(self) -> dict:
        n = {"movement": self.movement_var.get()}
        if self.subject_var.get().strip():
            n["subject"] = self.subject_var.get().strip()
        if self.note_var.get().strip():
            n["note"] = self.note_var.get().strip()
        return n

    def _folder(self) -> str:
        folder = self.folder_var.get().strip()
        if not folder:
            raise ConfigError("choose a folder for the recordings")
        return folder

    def _prepare_single(self):
        if self.state != IDLE:
            return
        try:
            cfg = self._config()
            errs = [p for p in checks(cfg) if p[0] == "error"]
            if errs:
                raise ConfigError("; ".join(f"{w}: {m}" for _l, w, m in errs))
            from acquisition.capture import CaptureRequest
            req = CaptureRequest(config=cfg, folder=self._folder(), name=self.name_var.get().strip() or "take",
                                 auto_increment=True, notes=self._notes(),
                                 source_text=self.base_source or None,
                                 at_start=self._apply_start_fields)
            backend = self._get_backend()
        except Exception as e:
            self.report_error("Prepare", str(e), e)
            return
        self._save_settings()
        self._prepared = cfg.to_dict()
        self._launch(backend, req, kind="single")

    def _launch(self, backend, req, kind: str, step=None):
        self.stop_event.clear()
        self.gate = threading.Event()
        req.start_gate = self.gate
        self.job_kind = kind
        self.current_step = step
        self._set_state(PREPARING)
        first = not getattr(backend, "booted", False)
        self.progress_label.configure(text=("loading the firmware and powering up the radar (~20 s) ..."
                                            if first else "configuring the radar ..."))
        self.progress.configure(mode="indeterminate")
        self.progress.start(15)

        def work():
            try:
                res = backend.run(req, on_event=lambda ph, info: self.q.put(("event", ph, info)),
                                  stop_event=self.stop_event)
                self.q.put(("captured", res))
                qual = None
                if res.status in ("complete", "incomplete") and res.stats.get("bytes_written"):
                    stem = os.path.join(res.measurement.folder, res.measurement.stem)
                    try:
                        qual = analyze_recording(stem, stats=res.stats, verification=res.verification)
                    except Exception as e:      # e.g. a few kB from a radar that dropped out
                        print(f"[Measure] quality check of {res.measurement.stem} failed: {e}")
                        qual = None
                    if qual is not None:
                        try:
                            save_quality(qual, stem)
                        except OSError:
                            pass
                self.q.put(("analysed", res, qual))
            except BaseException as e:
                self.q.put(("error", e, traceback.format_exc()))
        self.worker = threading.Thread(target=work, name="measure-take", daemon=True)
        self.worker.start()

    def _on_space(self, event):
        try:
            w = self.frame.focus_get()
            if w is not None and w.winfo_class() in TYPING:
                return None
            if not self.frame.winfo_viewable():
                return None
        except Exception:
            return None
        if self.state == READY:
            self._start()
            return "break"
        return None

    def _start(self):
        if self.state != READY or self.gate is None:
            return
        n = max(0, int(self.countdown_var.get() or 0))
        if n == 0:
            self._go()
            return
        self._countdown_left = n
        self._set_state(COUNTDOWN)
        self._tick()

    def _tick(self):
        if self.state != COUNTDOWN:
            return
        if self._countdown_left <= 0:
            self._go()
            return
        self.state_label.configure(text=f"Starting in {self._countdown_left} ...")
        if self.beep_var.get():
            _beep([(880, 120)])
        self._countdown_left -= 1
        self.frame.after(1000, self._tick)

    def _go(self):
        if self.job_kind == "single":
            self._start_fields = self._recording_fields()
        if self.beep_var.get():
            _beep([(1320, 350)])
        self._record_t0 = time.monotonic()
        self._set_state(RECORDING)
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self.gate.set()

    # The next take is prepared right after the last one, usually before its name,
    # subject or note have been typed. Name, folder and notes are therefore read when
    # Start is pressed (here, on the Tk thread) and handed to the capture, which applies
    # them through CaptureRequest.at_start before it writes anything.
    _start_fields = None
    _NOTE_KEYS = ("movement", "subject", "note")

    def _recording_fields(self) -> dict:
        fields = {"name": self.name_var.get().strip() or "take", "notes": self._notes()}
        folder = self.folder_var.get().strip()
        if folder:
            fields["folder"] = folder
        return fields

    def _apply_start_fields(self, req):
        """CaptureRequest.at_start - runs on the capture thread, touches no widget."""
        fields = self._start_fields
        if not fields:
            return
        req.name = fields["name"]
        req.folder = fields.get("folder", req.folder)
        notes = {k: v for k, v in req.notes.items() if k not in self._NOTE_KEYS}
        notes.update(fields["notes"])
        req.notes = notes

    def _stop(self):
        if self.state in (PREPARING, READY, COUNTDOWN, RECORDING):
            if self.state == COUNTDOWN:
                self._set_state(READY)
            self.stop_event.set()

    # --- the worker's messages ----------------------------------------------------------------
    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                self._handle(msg)
        except queue.Empty:
            pass
        if self.state == RECORDING and self._record_t0 is not None:
            self.state_label.configure(text=f"RECORDING  {time.monotonic() - self._record_t0:5.1f} s")
        try:
            self.frame.after(100, self._poll)
        except tk.TclError:
            pass

    def _handle(self, msg):
        kind = msg[0]
        if kind == "event":
            _k, phase, info = msg
            if phase == "ready":
                self._set_state(READY)
                self.progress.stop()
                self.progress.configure(mode="determinate", value=0)
                self.progress_label.configure(text=self._ready_text())
                self._update_radar_state()
                if self.beep_var.get():
                    _beep([(660, 90), (880, 90)])
                if self.job_kind == "single":
                    self._schedule_eval()       # settings changed while preparing? prepare again
            elif phase == "recording":
                exp = info.get("expected")
                if exp:
                    self.progress.configure(value=min(100.0, 100.0 * info["bytes"] / exp))
                self.progress_label.configure(
                    text=f"{info['bytes'] / 1e6:.1f} MB, {info['packets']} packets, "
                         f"{info['zero_filled']} lost")
            elif phase == "done" and self.state == RECORDING:
                if self.beep_var.get():
                    _beep([(660, 250)])
                self._set_state(ANALYSING)
        elif kind == "captured":
            res = msg[1]
            if res.status != "cancelled":
                self._set_state(ANALYSING)
        elif kind == "analysed":
            self._finished(msg[1], msg[2])
        elif kind == "error":
            e, tb = msg[1], msg[2]
            print(tb)
            self._set_state(IDLE)
            self.progress.stop()
            self._update_radar_state()
            if self.job_kind == "series":
                self._series_paused(f"error: {e}")
            self.report_error("Measurement", f"{type(e).__name__}: {e}", e)
        elif kind == "dca":
            rows = msg[1]
            if isinstance(rows, Exception):
                self.dev_status.configure(text=f"check failed: {rows}", foreground=LEVEL_FG["error"])
            else:
                bad = [r for r in rows if not r.ok]
                txt = "; ".join(str(r) for r in (bad or rows))
                self.dev_status.configure(text=txt, foreground=LEVEL_FG["warning" if bad else "ok"])

    def _ready_text(self) -> str:
        if self.job_kind == "series" and self.current_step is not None:
            s = self.current_step
            left = len([x for x in self.series_queue]) + 1
            return f"series take {s.index}: {s.label}  ({left} left)"
        return "radar configured - Start begins the recording"

    def _finished(self, res, qual):
        self.progress.stop()
        self._update_radar_state()
        cancelled = res.status == "cancelled"
        if cancelled and self._after_cancel is not None:
            fn, self._after_cancel = self._after_cancel, None
            self._set_state(IDLE)
            self.progress_label.configure(text="")
            self.frame.after(100, fn)
            return
        if not cancelled:
            self.last_stem = os.path.join(res.measurement.folder, res.measurement.stem)
            self.open_btn.configure(state="normal")
            self.folder_btn.configure(state="normal")
            self._show_result(res, qual)
        if self.job_kind == "series":
            self._series_take_done(res, qual)
            return
        self._set_state(IDLE)
        self.progress_label.configure(text=("cancelled" if cancelled else
                                            f"{res.status}: {res.measurement.stem}"))
        if not cancelled and self.auto_next_var.get() and res.status != "failed":
            self.frame.after(300, self._prepare_single)

    def _show_result(self, res, qual):
        t = self.result_text
        t.configure(state="normal")
        t.delete("1.0", "end")
        t.insert("end", f"{res.measurement.stem}: {res.status}\n", "bold")
        if res.error:
            t.insert("end", f"{res.error}\n", "error")
        if qual is not None:
            t.insert("end", qual.headline() + "\n")
            for level, what, message in qual.findings:
                t.insert("end", f"{level.upper()} {what}: {message}\n", level)
            if not qual.findings:
                t.insert("end", "no problems found\n", "ok")
        t.configure(state="disabled")

    def _set_state(self, state):
        self.state = state
        text, fg = STATE_TEXT[state]
        if state != COUNTDOWN:
            self.state_label.configure(text=text, fg=fg)
        else:
            self.state_label.configure(fg=fg)
        self.start_btn.configure(state="normal" if state == READY else "disabled")
        theme.set_prominent(self.start_btn, state == READY, theme.PROMINENT_BG)
        self.stop_btn.configure(state="normal" if state in (PREPARING, READY, COUNTDOWN, RECORDING)
                                else "disabled")
        busy = state != IDLE
        self.prepare_btn.configure(state="disabled" if busy else "normal")
        in_series = self.series is not None and self.job_kind == "series" and busy
        self.series_start_btn.configure(state="disabled" if busy else "normal")
        self.skip_btn.configure(state="normal" if in_series and state in (PREPARING, READY) else "disabled")
        self.series_stop_btn.configure(state="normal" if self.series is not None and
                                       (busy or self.series_queue) else "disabled")
        self.repeat_btn.configure(state="normal" if self.series is not None and self.series.results
                                  and state in (IDLE, READY) else "disabled")

    def _open_last(self):
        if self.last_stem and self.open_recording is not None:
            self.open_recording(self.last_stem + "_Raw_0.bin")

    def _open_folder(self):
        if self.last_stem:
            open_in_file_manager(os.path.dirname(self.last_stem))

    # --- series ------------------------------------------------------------------------------------
    def _objective_key(self) -> str:
        label = self.objective_var.get()
        return next((k for k, v in OBJECTIVES.items() if v[0] == label), "general")

    def _variables(self) -> list:
        out = []
        for pv, vv in self.var_rows:
            name = pv.get().split(" - ")[0].strip()
            if not name:
                continue
            out.append((name, parse_values(name, vv.get())))
        return out

    def _plan(self):
        try:
            base = self._params()
            variables = self._variables()
            steps = plan(base, self.base_cfg, variables, repeats=int(self.repeats_var.get() or 1),
                         shuffle=bool(self.shuffle_var.get()))
        except Exception as e:
            self.report_error("Series plan", str(e), e)
            return None
        self.planned_steps = steps
        self.planned_vars = variables
        self.planned_base = base
        self.tree.delete(*self.tree.get_children())
        for s in steps:
            self._tree_row(s)
        ok = [s for s in steps if s.ok]
        secs = sum((s.derived.get("duration_s") or 0) for s in ok)
        mb = sum((s.derived.get("size_mb") or 0) for s in ok)
        self.plan_label.configure(text=f"{len(steps)} take(s), {len(ok)} can be recorded "
                                       f"({secs / 60:.1f} min of recording, {mb / 1e3:.2f} GB); "
                                       f"{len(steps) - len(ok)} impossible (grey)")
        return steps

    def _tree_row(self, s, result=None, current=False):
        d = s.derived or {}
        if not s.ok:
            status, note, tags = "impossible", "; ".join(f"{p[1]}: {p[2]}" for p in s.problems
                                                          if p[0] == "error"), ("invalid",)
        else:
            status, note, tags = "planned", "", ()
        score = ""
        if result is not None:
            status = result.status
            score = "" if result.score is None else f"{result.score:.1f}"
            q = result.quality or {}
            note = q.get("headline", result.error or "")
            finds = [f for f in q.get("findings", []) if f[0] in ("error", "warning")]
            if finds:
                note += " | " + "; ".join(f"{f[1]}" for f in finds)
            tags = ("done",) if result.status == "complete" else (
                ("skipped",) if result.status in ("skipped", "cancelled") else ("failed",))
            if finds and result.status == "complete":
                tags = ("warn",)
        if current:
            tags = ("current",)
        values = (s.index, s.key, s.take, status, score, _f(d.get("range_res_cm"), 2),
                  _f(d.get("max_vel_ms"), 2), _f(d.get("vel_res_ms"), 3), _f(d.get("fps"), 1),
                  _f(d.get("rate_mb_s"), 1), note)
        iid = f"s{s.index}"
        if self.tree.exists(iid):
            self.tree.item(iid, values=values, tags=tags)
        else:
            self.tree.insert("", "end", iid=iid, values=values, tags=tags)
        return iid

    def _series_start(self):
        if self.job_kind == "single" and self.state in (PREPARING, READY):
            self._after_cancel = self._series_start          # drop the prepared single take first
            self.stop_event.set()
            return
        if self.state != IDLE:
            return
        if self.series is not None and self.series_queue:
            ans = messagebox.askyesnocancel(
                "Series", f"Continue series '{self.series.name}' ({len(self.series_queue)} take(s) "
                          f"left)?\n\nNo starts a new series with the plan as it is now.")
            if ans is None:
                return
            if ans:
                self._series_next()
                return
        steps = self._plan()
        if not steps:
            return
        ok = [s for s in steps if s.ok]
        if not ok:
            messagebox.showwarning("Series", "No combination can be recorded - see the problems column.")
            return
        try:
            folder = self._folder()
            self._get_backend()
        except Exception as e:
            self.report_error("Series", str(e), e)
            return
        notes = self._notes()
        notes["base"] = self.base_source or "Measure tab settings"
        self.series = Series(folder, self.series_name_var.get().strip() or "series", steps,
                             objective=self._objective_key(), notes=notes,
                             base_params=self.planned_base, variables=self.planned_vars)
        os.makedirs(self.series.folder, exist_ok=True)
        self.series.write()
        self.series_queue = list(ok)
        self._save_settings()
        self._series_next()

    def _series_next(self):
        if self.series is None:
            return
        if not self.series_queue:
            self._series_complete()
            return
        step = self.series_queue.pop(0)
        self._tree_row(step, current=True)
        self.tree.see(f"s{step.index}")
        try:
            backend = self._get_backend()
        except Exception as e:
            self.report_error("Series", str(e), e)
            self.series_queue.insert(0, step)
            return
        self._launch(backend, self.series.request(step), kind="series", step=step)

    def _series_take_done(self, res, qual):
        step = self.current_step
        cancelled = res.status == "cancelled"
        if cancelled and (self._ending or self._requeue):
            self._tree_row(step)                           # not recorded; queued again or dropped
        else:
            r = self.series.add(step, None if cancelled else res, qual,
                                status="skipped" if cancelled else None)
            self._tree_row(step, r)
        self._requeue = False
        paths = self.series.write()
        self._refresh_series_view()
        self._set_state(IDLE)
        if self._ending:
            self._ending = False
            self.series_queue = []
            self._set_state(IDLE)
            self.progress_label.configure(text=f"series ended - report: {paths['html']}")
            return
        self.frame.after(200, self._series_next)

    _ending = False
    _requeue = False
    _after_cancel = None

    def _series_skip(self):
        if self.job_kind == "series" and self.state in (PREPARING, READY):
            self.stop_event.set()

    def _series_repeat(self):
        if self.series is None or not self.series.results:
            return
        last = self.series.results[-1]
        base = next((s for s in self.series.steps if s.index == last.step), None)
        if base is None:
            return
        again = copy.copy(base)
        again.index = max(s.index for s in self.series.steps) + 1
        again.take = max(r.take for r in self.series.results if r.key == base.key) + 1
        self.series.steps.append(again)
        self._tree_row(again)
        if self.state == READY and self.job_kind == "series":
            # the prepared take goes back to the queue, the repeat comes first
            self.series_queue.insert(0, self.current_step)
            self.series_queue.insert(0, again)
            self._requeue = True
            self.stop_event.set()
        else:
            self.series_queue.insert(0, again)
            if self.state == IDLE:
                self._series_next()

    def _series_end(self):
        if self.series is None:
            return
        if self.state in (PREPARING, READY, COUNTDOWN, RECORDING) and self.job_kind == "series":
            if self.state == RECORDING and not messagebox.askyesno(
                    "End series", "A take is being recorded. Stop it and end the series?"):
                return
            self._ending = True
            self.stop_event.set()
        else:
            paths = self.series.write()
            self.progress_label.configure(text=f"series ended - report: {paths['html']}")
            self.series_queue = []
            self._set_state(IDLE)

    def _series_paused(self, why: str):
        if self.series is not None and self.current_step is not None:
            self.series_queue.insert(0, self.current_step)
            self._tree_row(self.current_step)
            self.series.write()
        self.progress_label.configure(text=f"series paused ({why}) - Start series continues "
                                           f"with a new plan; the report keeps what was recorded")

    def _series_complete(self):
        paths = self.series.write()
        self._set_state(IDLE)
        self.progress_label.configure(text=f"series complete - report: {paths['html']}")
        if self.beep_var.get():
            _beep([(880, 150), (1100, 150), (1320, 300)])
        self._refresh_series_view()

    def _refresh_series_view(self):
        if self.series is None:
            return
        rank = self.series.ranking()
        best = rank[0]["key"] if rank else None
        for r in self.series.results:
            s = next((x for x in self.series.steps if x.index == r.step), None)
            if s is not None:
                iid = self._tree_row(s, r)
                if best and r.key == best and r.score is not None:
                    warned = "warn" in self.tree.item(iid, "tags")
                    self.tree.item(iid, tags=("bestwarn" if warned else "best",))
        t = self.reco_text
        t.configure(state="normal")
        t.delete("1.0", "end")
        for line in self.series.recommendations():
            t.insert("end", "- " + line + "\n")
        t.configure(state="disabled")

    def _on_objective(self):
        if self.series is not None:
            self.series.rescore(self._objective_key())
            self.series.write()
            self._refresh_series_view()

    def _open_report(self):
        if self.series is None:
            return
        paths = self.series.write()
        webbrowser.open(Path(paths["html"]).as_uri())

    def _use_best(self):
        if self.series is None or not self.series.ranking():
            return
        best = self.series.ranking()[0]
        step = next((s for s in self.series.steps if s.key == best["key"]), None)
        if step is None:
            return
        self._set_params(step.params)
        self.nb.select(0)
        self.source_label.configure(text=f"best of series {self.series.name}: {best['key']}")

    def _tree_open(self, _event):
        sel = self.tree.selection()
        if not sel or self.series is None or self.open_recording is None:
            return
        idx = int(self.tree.item(sel[0], "values")[0])
        r = next((x for x in reversed(self.series.results) if x.step == idx and x.stem), None)
        if r is not None:
            self.open_recording(os.path.join(self.series.folder, r.stem + "_Raw_0.bin"))

    # --- settings file -----------------------------------------------------------------------------
    def _load_settings(self) -> dict:
        try:
            with open(_settings_path(), encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _apply_settings(self):
        s = self._settings
        self.folder_var.set(s.get("folder") or str(Path.home() / "radar_measurements"))
        self.name_var.set(s.get("name") or "take")
        self.subject_var.set(s.get("subject") or "")
        self.movement_var.set(s.get("movement") or MOVEMENTS[1])
        self.countdown_var.set(int(s.get("countdown", 3)))
        self.beep_var.set(bool(s.get("beep", True)))
        self.mode_var.set(s.get("mode") or "native")
        src = s.get("base_source") or ""
        if src and os.path.isfile(src):
            self._load_config_file(src, quiet=True)
        if s.get("params"):
            try:
                self._set_params(MeasureParams.from_dict(s["params"]))
            except Exception:
                self._set_params(MeasureParams())
        elif not src:
            self._set_params(MeasureParams())
        ser = s.get("series") or {}
        for (pv, vv), row in zip(self.var_rows, ser.get("variables", [])):
            pv.set(row[0])
            vv.set(row[1])
        self.repeats_var.set(int(ser.get("repeats", 1)))
        self.shuffle_var.set(bool(ser.get("shuffle", False)))
        if ser.get("objective") in [v[0] for v in OBJECTIVES.values()]:
            self.objective_var.set(ser["objective"])
        self.series_name_var.set(ser.get("name") or "series")

    def _save_settings(self):
        try:
            params = self._params().to_dict()
        except Exception:
            params = self._settings.get("params")
        s = {"folder": self.folder_var.get(), "name": self.name_var.get(),
             "subject": self.subject_var.get(), "movement": self.movement_var.get(),
             "countdown": int(self.countdown_var.get() or 0), "beep": bool(self.beep_var.get()),
             "mode": self.mode_var.get(), "com_port": port_label(self._com_port()) if self._com_port() else "",
             "base_source": self.base_source if os.path.isfile(self.base_source or "") else "",
             "params": params,
             "series": {"variables": [(pv.get(), vv.get()) for pv, vv in self.var_rows],
                        "repeats": int(self.repeats_var.get() or 1),
                        "shuffle": bool(self.shuffle_var.get()),
                        "objective": self.objective_var.get(), "name": self.series_name_var.get()}}
        self._settings = s
        try:
            with open(_settings_path(), "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2)
        except OSError:
            pass

    def _on_destroy(self, event):
        if event.widget is self.frame:
            self.stop_event.set()
            self._close_backend()

    def selftest_take(self, folder: str, done, duration_s: float = 2.0, timeout_s: float = 120.0):
        """
        One short take with the radar, as a person would make it - for checking a build
        (acquisition.diagnostics). The operator's saved settings are left alone.
        done(result dict) is called on the Tk thread.
        """
        self._save_settings = lambda: None
        self.mode_var.set("native")
        self._on_mode()
        self._detect_ports()
        self.folder_var.set(folder)
        self.name_var.set("selftest")
        self.movement_var.set("selftest of the build")
        self.vars["duration_s"].set(f"{duration_s:g}")
        self.countdown_var.set(0)
        self.beep_var.set(False)
        self.auto_next_var.set(False)
        t0 = time.monotonic()
        out = {"com_port": self._com_port()}
        if out["com_port"] is None:                       # no dialog: a check runs unattended
            out.update(ok=False, error="no radar connected (no radar COM port found)")
            self.frame.after(10, lambda: done(out))
            return

        def step():
            if time.monotonic() - t0 > timeout_s:
                out.update(ok=False, error=f"timeout in state {self.state}: "
                                           f"{self.progress_label.cget('text')}")
                done(out)
                return
            if self.state == READY and "ready_s" not in out:
                out["ready_s"] = round(time.monotonic() - t0, 1)
                self._start()
            elif self.state == IDLE and self.last_stem:
                out.update(ok="complete" in self.result_text.get("1.0", "end"),
                           stem=self.last_stem, total_s=round(time.monotonic() - t0, 1),
                           report=self.result_text.get("1.0", "end").strip())
                done(out)
                return
            self.frame.after(200, step)

        self.frame.after(300, lambda: (self._prepare_single(), self.frame.after(200, step)))

    def release_hardware(self) -> str | None:
        """Let go of the radar and the DCA1000 for another tab (the cascade). None when
        done, otherwise why not: a take is being prepared or recorded."""
        if self.state != IDLE:
            return ("The Single-chip radar tab has a take prepared or running - press Stop / cancel "
                    "there first (it holds the DCA1000).")
        self._close_backend()
        self._update_radar_state()
        return None

    def close(self):
        self.stop_event.set()
        self._close_backend()


def _f(v, digits) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def main():
    """python measure_gui.py - the tab in a window of its own."""
    scale = theme.enable_dpi_awareness()
    root = tk.Tk()
    try:
        from version import __version__
    except ImportError:
        __version__ = ""
    root.title(f"DARTscope {__version__} - Measure".replace("  ", " "))
    theme.initial_geometry(root, 1400, 860, scale)
    tab = MeasureTab(root)

    def on_close():
        tab.close()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
