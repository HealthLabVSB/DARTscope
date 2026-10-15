"""
cascade_gui.py

The Measure tab's "Cascade radar" sub-tab: settings, measurement length and recording for
the AWR2243-2X-CAS-EVM (two cascaded AWR2243 + AM2732R) with the DCA1000 - without
mmWave Studio, on macOS or Windows (cascade/).

    Take       a preset or the settings in physical units (chirp profile, TX per chirp,
               loops, frame period, duration) with what they give (range and velocity
               resolution, maximum range, data rate, file size) and every check against
               the AWR2243, the firmware and the DCA1000 -> PC link; folder, name and
               note; Start records (the DCA1000 armed, the firmware loaded over JTAG with
               the settings, the take recorded, the firmware stopped) and Stop ends early.
               If the radar refuses the settings the tab says at which step and why.
    Preview    a take read back (cascade/data.py): range profile, range-time (a click
               picks the range), range-velocity, and the displacement at the picked range
               with its spectrum and a rough breathing and heart rate.

"Open in DARTscope" opens a take in the Decode tab (cascade/decode.py gives it the params a
mmWave Studio log would), from where Fastcheck and the Analysis views work as for the
single-chip radar - except the angle views, which need the cascade's antenna layout.
Also alone: python cascade_gui.py

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import json
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import theme
from cascade import calibration as cc
from cascade import hardware
from cascade.data import CascadeTake
from cascade.params import HPF1_KHZ, HPF2_KHZ, PRESETS, RX_GAINS_DB, TDM6, TX_MODES, CascadeParams, \
    format_tx_chirps, parse_tx_chirps
from cascade.recording import CascadeRecording, check_dca1000

CUSTOM = "(custom)"
CAL_PRESET = "Imaging - best range resolution (3.9 cm)"     # the profile of calibration takes
CUSTOM_TX = "Custom (masks below)"

# (key, label, unit, choices or None)
FIELDS = [
    ("Chirp profile", [
        ("start_freq_ghz", "Start frequency", "GHz", None),
        ("slope_mhz_us", "Slope", "MHz/µs", None),
        ("idle_us", "Idle time", "µs", None),
        ("adc_start_us", "ADC start", "µs", None),
        ("ramp_end_us", "Ramp end", "µs", None),
        ("tx_start_us", "TX start", "µs", None),
        ("num_samples", "Samples per chirp", "", ["64", "128", "256", "384", "512"]),
        ("sample_rate_ksps", "Sampling rate", "ksps",
         ["2500", "5000", "6000", "8000", "10000", "12500", "15000", "20000"]),
        ("rx_gain_db", "RX gain", "dB", [str(g) for g in RX_GAINS_DB]),
        ("hpf1_khz", "HPF1", "kHz", [str(k) for k in HPF1_KHZ]),
        ("hpf2_khz", "HPF2", "kHz", [str(k) for k in HPF2_KHZ]),
        ("tx_backoff_db", "TX backoff", "dB", None),
    ]),
    ("Frame and duration", [
        ("num_loops", "Loops per frame", "", None),
        ("frame_period_ms", "Frame period", "ms", None),
        ("duration_s", "Duration", "s", None),
    ]),
]
INT_KEYS = {"num_samples", "sample_rate_ksps", "rx_gain_db", "hpf1_khz", "hpf2_khz", "tx_backoff_db",
            "num_loops"}


def _settings_path() -> Path:
    try:
        from platformdirs import user_config_dir  # type: ignore
        d = Path(user_config_dir("bme_radar_gui", appauthor=False))
    except Exception:
        d = Path.home() / ".bme_radar_gui"
    d.mkdir(parents=True, exist_ok=True)
    return d / "cascade_gui.json"


class CascadeTab:
    def __init__(self, parent, open_recording=None, release_hardware=None, report_error=None):
        """open_recording(first_bin_path): open a take in DARTscope's Decode tab (the
        buttons are hidden without it); release_hardware() -> str | None: called before a
        take so another tab lets go of the DCA1000; returns why it cannot, or None."""
        self.parent = parent
        self.open_recording = open_recording
        self.release_hardware = release_hardware
        self.report_error = report_error or (lambda title, msg, exc=None: messagebox.showerror(title, msg))
        self.vars: dict[str, tk.StringVar] = {}
        self.q: queue.Queue = queue.Queue()
        self.recording: CascadeRecording | None = None
        self.worker: threading.Thread | None = None
        self._check_job = None
        self._settings = self._load_settings()
        self.calibration: cc.CascadeCalibration | None = cc.load_current()
        self.pending_cal: cc.CascadeCalibration | None = None     # measured, not yet saved
        self._cal_job = None                                       # (kind, settings) of a calibration take
        self._build()
        self._show_calibration()
        self._set_params(self._initial_params())
        self.folder_var.set(self._settings.get("folder") or str(Path.home() / "radar_measurements" / "cascade"))
        self.frame.after(300, self.check_devices)
        self.frame.after(100, self._poll)

    # --- layout ------------------------------------------------------------------------------
    def _build(self):
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill="both", expand=True)
        dev = ttk.LabelFrame(self.frame, text="Cascade devices")
        dev.pack(side="top", fill="x", padx=8, pady=(6, 4))
        self.dca_lbl = ttk.Label(dev, text="DCA1000")
        self.dca_lbl.pack(side="left", padx=(6, 18), pady=4)
        self.jtag_lbl = ttk.Label(dev, text="Cascade JTAG")
        self.jtag_lbl.pack(side="left", padx=(0, 18))
        self.fw_lbl = ttk.Label(dev, text="", foreground=theme.SECONDARY)
        self.fw_lbl.pack(side="left")
        ttk.Button(dev, text="Check devices", command=self.check_devices).pack(side="right", padx=6)
        ttk.Button(dev, text="Test JTAG", command=self.test_jtag).pack(side="right")
        # calibrated or not, in sight before every take
        self.cal_lbl = tk.Label(dev, text="", font=theme.font(10, "bold"), anchor="w")
        self.cal_lbl.pack(side="right", padx=(6, 12))

        self.nb = ttk.Notebook(self.frame)
        self.nb.pack(fill="both", expand=True, padx=8, pady=4)
        take = ttk.Frame(self.nb, padding=6)
        view = ttk.Frame(self.nb, padding=6)
        calf = ttk.Frame(self.nb, padding=6)
        self.nb.add(take, text="Take")
        self.nb.add(view, text="Preview")
        self.nb.add(calf, text="Calibration")
        self.tab_view = view
        self.tab_cal = calf
        self._build_take(take)
        self.viewer = CascadePreview(view, self)
        self._build_calibration(calf)

    def _build_take(self, root):
        left = ttk.Frame(root)
        left.pack(side="left", fill="y", padx=(0, 10))
        right = ttk.Frame(root)
        right.pack(side="left", fill="both", expand=True)

        pf = ttk.LabelFrame(left, text="Preset")
        pf.pack(fill="x", pady=(0, 6))
        self.preset_var = tk.StringVar()
        cb = ttk.Combobox(pf, textvariable=self.preset_var, values=list(PRESETS), state="readonly", width=36)
        cb.grid(row=0, column=0, columnspan=2, sticky="we", padx=4, pady=(4, 2))
        cb.bind("<<ComboboxSelected>>", lambda e: self._set_params(CascadeParams.preset(self.preset_var.get())))
        ttk.Button(pf, text="Load ...", command=self._load_file).grid(row=1, column=0, sticky="we", padx=4, pady=(0, 4))
        ttk.Button(pf, text="Save ...", command=self._save_file).grid(row=1, column=1, sticky="we", padx=4, pady=(0, 4))

        for title, rows in FIELDS:
            lf = ttk.LabelFrame(left, text=title)
            lf.pack(fill="x", pady=(0, 6))
            for r, (key, label, unit, choices) in enumerate(rows):
                ttk.Label(lf, text=label).grid(row=r, column=0, sticky="w", padx=(4, 6), pady=1)
                v = tk.StringVar()
                self.vars[key] = v
                w = (ttk.Combobox(lf, textvariable=v, values=choices, width=10) if choices
                     else ttk.Entry(lf, textvariable=v, width=12))
                w.grid(row=r, column=1, sticky="w", pady=1)
                ttk.Label(lf, text=unit, foreground=theme.SECONDARY).grid(row=r, column=2, sticky="w", padx=(4, 4))
                v.trace_add("write", lambda *a: self._schedule_check())

        tf = ttk.LabelFrame(left, text="Transmitters")
        tf.pack(fill="x", pady=(0, 6))
        self.txmode_var = tk.StringVar()
        txcb = ttk.Combobox(tf, textvariable=self.txmode_var, values=list(TX_MODES) + [CUSTOM_TX],
                            state="readonly", width=36)
        txcb.grid(row=0, column=0, columnspan=2, sticky="we", padx=4, pady=(4, 2))
        txcb.bind("<<ComboboxSelected>>", self._tx_mode_changed)
        ttk.Label(tf, text="TX masks per chirp").grid(row=1, column=0, sticky="w", padx=4)
        v = tk.StringVar()
        self.vars["tx_chirps"] = v
        ttk.Entry(tf, textvariable=v, width=22).grid(row=1, column=1, sticky="w", pady=2)
        v.trace_add("write", lambda *a: (self._sync_tx_mode(), self._schedule_check()))
        ttk.Label(tf, text="primary/secondary; 1 = TX0, 2 = TX1, 4 = TX2; '1/0; 2/0' = two chirps",
                  foreground=theme.SECONDARY, wraplength=330).grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 4))

        df = ttk.LabelFrame(right, text="What they give")
        df.pack(fill="x")
        self.derived_frame = df
        self.derived_labels: dict[str, ttk.Label] = {}

        cf = ttk.LabelFrame(right, text="Checks")
        cf.pack(fill="x", pady=(6, 0))
        self.check_text = tk.Text(cf, height=5, wrap="word", relief="flat", font=theme.font(10),
                                  background=theme.FILL, borderwidth=0)
        self.check_text.pack(fill="x", padx=4, pady=4)
        for tag, colour in (("error", theme.RED), ("warning", theme.ORANGE), ("ok", theme.GREEN)):
            self.check_text.tag_configure(tag, foreground=colour)

        of = ttk.LabelFrame(right, text="Recording")
        of.pack(fill="x", pady=(6, 0))
        self.folder_var = tk.StringVar()
        self.name_var = tk.StringVar(value=self._new_name())
        self.note_var = tk.StringVar()
        ttk.Label(of, text="Folder").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Entry(of, textvariable=self.folder_var, width=60).grid(row=0, column=1, sticky="we")
        ttk.Button(of, text="...", width=3, command=self._choose_folder).grid(row=0, column=2, padx=4)
        ttk.Label(of, text="Name").grid(row=1, column=0, sticky="w", padx=4)
        ttk.Entry(of, textvariable=self.name_var, width=60).grid(row=1, column=1, sticky="we")
        ttk.Button(of, text="New name", command=lambda: self.name_var.set(self._new_name())).grid(row=1, column=2, padx=4)
        ttk.Label(of, text="Note").grid(row=2, column=0, sticky="w", padx=4, pady=(0, 4))
        ttk.Entry(of, textvariable=self.note_var, width=60).grid(row=2, column=1, sticky="we", pady=(0, 4))
        of.columnconfigure(1, weight=1)

        rf = ttk.LabelFrame(right, text="Take")
        rf.pack(fill="both", expand=True, pady=(6, 0))
        btns = ttk.Frame(rf)
        btns.pack(fill="x", padx=4, pady=4)
        self.start_btn = theme.Button(btns, text="START", width=14, command=self.start)
        if not isinstance(self.start_btn, ttk.Button):
            self.start_btn.configure(font=theme.font(12, "bold"))
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        self.open_btn = ttk.Button(btns, text="Open in DARTscope", command=self._open_last, state="disabled")
        if self.open_recording is not None:
            self.open_btn.pack(side="right")
        self.phase_lbl = tk.Label(btns, text="", font=theme.font(13, "bold"), anchor="w")
        self.phase_lbl.pack(side="left", padx=12)
        self.progress = ttk.Progressbar(rf, maximum=1000)
        self.progress.pack(fill="x", padx=4)
        self.progress_lbl = ttk.Label(rf, text="", foreground=theme.SECONDARY)
        self.progress_lbl.pack(anchor="w", padx=4)
        self.cal_take_lbl = tk.Label(rf, text="", anchor="w", justify="left", font=theme.font(9))
        self.cal_take_lbl.pack(fill="x", padx=4)
        self.log = tk.Text(rf, height=8, wrap="word", font=theme.mono(9))
        self.log.pack(fill="both", expand=True, padx=4, pady=4)
        self.log.tag_configure("error", foreground=theme.RED)
        self.log.tag_configure("ok", foreground=theme.GREEN)

    # --- settings ----------------------------------------------------------------------------
    def _load_settings(self) -> dict:
        try:
            return json.loads(_settings_path().read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_settings(self, params: CascadeParams):
        try:
            _settings_path().write_text(json.dumps({"params": params.to_dict(), "folder": self.folder_var.get()},
                                                   indent=2), encoding="utf-8")
        except OSError:
            pass

    def _initial_params(self) -> CascadeParams:
        try:
            return CascadeParams.from_dict(self._settings["params"])
        except Exception:
            return CascadeParams()

    def _set_params(self, p: CascadeParams):
        d = p.to_dict()
        for k, v in self.vars.items():
            val = d[k]
            if isinstance(val, float):
                val = f"{val:.4f}".rstrip("0").rstrip(".")
            v.set(str(val))
        self._sync_tx_mode()
        match = [n for n in PRESETS if CascadeParams.preset(n).to_dict() == d]
        self.preset_var.set(match[0] if match else CUSTOM)
        self._schedule_check()

    def params(self) -> CascadeParams:
        d = {}
        for k, v in self.vars.items():
            s = v.get()
            if k == "tx_chirps":
                d[k] = s
            else:
                s = s.strip().replace(",", ".")
                d[k] = int(float(s)) if k in INT_KEYS else float(s)
        return CascadeParams(**d)

    def _tx_mode_changed(self, *_):
        if self.txmode_var.get() in TX_MODES:
            self.vars["tx_chirps"].set(format_tx_chirps(TX_MODES[self.txmode_var.get()]))

    def _sync_tx_mode(self):
        try:
            ch = parse_tx_chirps(self.vars["tx_chirps"].get())
        except Exception:
            self.txmode_var.set(CUSTOM_TX)
            return
        self.txmode_var.set(next((n for n, c in TX_MODES.items() if c == ch), CUSTOM_TX))

    def _schedule_check(self):
        if self._check_job:
            self.frame.after_cancel(self._check_job)
        self._check_job = self.frame.after(150, self.update_checks)

    def update_checks(self) -> dict | None:
        self._check_job = None
        t = self.check_text
        t.configure(state="normal")
        t.delete("1.0", "end")
        try:
            rep = self.params().check(Path(self.folder_var.get() or Path.home()))
        except Exception as e:
            t.insert("end", f"✗ Not a valid value: {e}\n", "error")
            t.configure(state="disabled")
            self._set_start(False)
            return None
        for k, v in rep["derived"].items():
            if k not in self.derived_labels:
                r, c = divmod(len(self.derived_labels), 2)
                ttk.Label(self.derived_frame, text=k + ":", foreground=theme.SECONDARY).grid(
                    row=r, column=2 * c, sticky="w", padx=(4, 6))
                lbl = ttk.Label(self.derived_frame, text="", font=theme.font(10, "bold"))
                lbl.grid(row=r, column=2 * c + 1, sticky="w", padx=(0, 24))
                self.derived_labels[k] = lbl
            self.derived_labels[k].configure(text=v)
        for e in rep["errors"]:
            t.insert("end", "✗ " + e + "\n", "error")
        for w in rep["warnings"]:
            t.insert("end", "⚠ " + w + "\n", "warning")
        if not rep["errors"]:
            t.insert("end", "✓ The settings can be recorded.\n", "ok")
        t.configure(state="disabled")
        self._set_start(not rep["errors"])
        return rep

    def _busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def _set_start(self, ok: bool):
        on = ok and not self._busy()
        self.start_btn.configure(state="normal" if on else "disabled")
        theme.set_prominent(self.start_btn, on, theme.PROMINENT_BG)

    def _load_file(self):
        f = filedialog.askopenfilename(filetypes=[("Settings or take", "*.json")])
        if f:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            self._set_params(CascadeParams.from_dict(d.get("params", d)))

    def _save_file(self):
        try:
            p = self.params()
        except Exception as e:
            self.report_error("Cascade settings", str(e))
            return
        f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Settings", "*.json")])
        if f:
            Path(f).write_text(json.dumps({"params": p.to_dict()}, indent=2), encoding="utf-8")

    def _choose_folder(self):
        d = filedialog.askdirectory(initialdir=self.folder_var.get() or str(Path.home()))
        if d:
            self.folder_var.set(d)

    @staticmethod
    def _new_name() -> str:
        return time.strftime("cascade_%Y%m%d_%H%M%S")

    # --- devices -----------------------------------------------------------------------------
    def check_devices(self):
        if self._busy():
            return

        def work():
            self.q.put(("devices", check_dca1000(), hardware.xds110_present(), hardware.scripting()))
        threading.Thread(target=work, daemon=True).start()

    def _show_devices(self, dca, xds, ti):
        self.dca_lbl.configure(text=f"DCA1000: ✓ FPGA {dca}" if dca else
                               "DCA1000: ✗ no answer (5 V, Ethernet, 192.168.33.30, or another tab holds it)",
                               foreground=theme.GREEN if dca else theme.RED)
        self.jtag_lbl.configure(text="Cascade JTAG: ✓ XDS110" if xds else "Cascade JTAG: ✗ no XDS110 (cascade USB)",
                                foreground=theme.GREEN if xds else theme.RED)
        fw = hardware.FIRMWARE
        if not ti:
            self.fw_lbl.configure(text="Neither TI UniFlash nor CCS 20+ is installed", foreground=theme.RED)
        elif not fw.exists():
            self.fw_lbl.configure(text=f"{fw.name} is missing", foreground=theme.RED)
        else:
            self.fw_lbl.configure(text=f"{fw.name}, {ti.name}", foreground=theme.SECONDARY)

    def test_jtag(self):
        """Connect to the AM273x over the XDS110 and let go: TI's debugger, the drivers, the
        cascade's USB and 12 V checked without loading the firmware or starting the radar."""
        if self._busy():
            return
        self.jtag_lbl.configure(text="Cascade JTAG: testing ...", foreground=theme.SECONDARY)
        self.logmsg("JTAG test: connecting to the AM273x (the first run of TI's debugger can take a minute) ...")

        def work():
            self.q.put(("jtag", hardware.jtag("probe")))
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()
        self._set_start(False)

    def _show_jtag(self, res):
        self.worker = None
        if res.get("ok"):
            state = "halted" if res.get("halted") else "running"
            self.jtag_lbl.configure(text=f"Cascade JTAG: ✓ AM273x answers ({state})", foreground=theme.GREEN)
            self.logmsg(f"JTAG test: the AM273x answers, its R5F is {state}.")
        else:
            err = str(res.get("error"))
            self.jtag_lbl.configure(text="Cascade JTAG: ✗ " + (err if len(err) < 70 else err[:67] + "..."),
                                    foreground=theme.RED)
            self.logmsg("✗ JTAG test: " + err, "error")
        self.update_checks()

    # --- a take ------------------------------------------------------------------------------
    def logmsg(self, msg, tag=None):
        self.log.insert("end", time.strftime("%H:%M:%S  ") + msg + "\n", tag)
        self.log.see("end")

    def start(self):
        rep = self.update_checks()
        if not rep or rep["errors"] or self._busy():
            return
        p = self.params()
        name = self.name_var.get().strip()
        if not name or any(c in name for c in "/\\:"):
            self.report_error("Cascade take", "Give the take a name without slashes.")
            return
        folder = Path(self.folder_var.get()).expanduser()
        if (folder / f"{name}_Raw_0.bin").exists():
            self.report_error("Cascade take", f"{name} already exists in {folder}.")
            return
        if self.release_hardware is not None:
            why = self.release_hardware()
            if why:
                self.report_error("Cascade take", why)
                return
        self._save_settings(p)
        self.recording = CascadeRecording(p, folder, name, notes=self.note_var.get(),
                                          calibration=self.calibration.to_dict() if self.calibration else None)
        self._cal_job = None
        self.open_btn.configure(state="disabled")
        self.progress["value"] = 0
        self.progress_lbl.configure(text="")
        self.logmsg(f"Start {name}: {p.duration_s:g} s, {p.num_frames} frames, {rep['derived']['File size']}")
        self.stop_btn.configure(state="normal")

        def progress(kind, **kw):
            self.q.put((kind, kw))

        def work():
            try:
                self.q.put(("done", self.recording.run(progress)))
            except Exception as e:
                self.q.put(("failed", str(e)))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()
        self._set_start(False)

    def _open_last(self):
        if self.recording is not None and self.open_recording is not None:
            self.open_recording(str(self.recording.first_part))

    def stop(self):
        if self.recording is not None and self._busy():
            self.recording.stop_event.set()
            self.logmsg("Stopping ...")

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                kind = item[0]
                if kind == "devices":
                    self._show_devices(*item[1:])
                elif kind == "jtag":
                    self._show_jtag(item[1])
                elif kind == "log":
                    self.logmsg(item[1]["msg"])
                elif kind == "phase":
                    self.phase_lbl.configure(text=item[1]["msg"])
                elif kind == "bytes":
                    kw = item[1]
                    frac = kw["done"] / max(kw["total"], 1)
                    self.progress["value"] = 1000 * frac
                    self.progress_lbl.configure(text=f"{kw['done'] / 1e6:.1f} of {kw['total'] / 1e6:.1f} MB "
                                                     f"({frac * 100:.0f} %), lost packets {kw['lost']}")
                elif kind in ("done", "failed"):
                    self._finished(item[1] if kind == "done" else None, item[1] if kind == "failed" else None)
                elif kind in ("cal_done", "cal_failed"):
                    self._cal_finished(item[1] if kind == "cal_done" else None,
                                       item[1] if kind == "cal_failed" else None)
        except queue.Empty:
            pass
        try:
            self.frame.after(100, self._poll)
        except tk.TclError:
            pass

    def _finished(self, meta, error):
        if self._cal_job is not None:
            self._cal_take_finished(meta, error)
            return
        self.worker = None
        self.stop_btn.configure(state="disabled")
        self.phase_lbl.configure(text="")
        if error:
            self.logmsg("✗ " + error, "error")
            self.report_error("Cascade take failed", error)
        else:
            r = meta["record"]
            ok = r["complete"] and not r["zero_filled_packets"]
            self.logmsg(f"{'✓' if ok else '⚠'} {self.recording.prefix}: {r['frames_in_file']} frames, "
                        f"{r['bytes'] / 1e6:.1f} MB, lost packets {r['zero_filled_packets']}, firmware "
                        f"{hardware.describe_status(r['firmware_status'])}", "ok" if ok else "error")
            self.name_var.set(self._new_name())
            self.open_btn.configure(state="normal")
            try:
                self.viewer.open(self.recording.sidecar)
                self.nb.select(self.tab_view)
            except Exception as e:
                self.logmsg(f"Preview failed: {e}", "error")
        self.update_checks()

    # --- calibration -------------------------------------------------------------------------
    def _build_calibration(self, root):
        top = ttk.LabelFrame(root, text="The board's calibration")
        top.pack(fill="x")
        self.cal_status = tk.Label(top, text="", anchor="w", justify="left", font=theme.font(10, "bold"))
        self.cal_status.pack(side="left", fill="x", expand=True, padx=6, pady=4)
        ttk.Button(top, text="Forget", command=self._cal_forget).pack(side="right", padx=4)
        ttk.Button(top, text="Export ...", command=self._cal_export).pack(side="right", padx=4)
        ttk.Button(top, text="Load ...", command=self._cal_load).pack(side="right", padx=4)

        cols = ttk.Frame(root)
        cols.pack(fill="x", pady=6)
        rf = ttk.LabelFrame(cols, text="Corner reflector (absolute angles)")
        rf.pack(side="left", fill="both", expand=True, padx=(0, 6))
        ttk.Label(rf, text="A trihedral corner reflector (5-10 cm edges; three squares of cardboard in "
                           "aluminium foil will do) on a non-metallic stand at the radar's height, "
                           "nothing else within half a metre of it, the room still. Six TX, 5 s.",
                  wraplength=430, justify="left").grid(row=0, column=0, columnspan=4, sticky="w", padx=4, pady=4)
        self.cal_range = tk.StringVar(value="2.5")
        self.cal_az = tk.StringVar(value="0")
        self.ver_range = tk.StringVar(value="2.5")
        self.ver_az = tk.StringVar(value="25")
        ttk.Label(rf, text="Distance [m]").grid(row=1, column=0, sticky="w", padx=4)
        ttk.Entry(rf, textvariable=self.cal_range, width=6).grid(row=1, column=1, sticky="w")
        ttk.Label(rf, text="Angle [deg]").grid(row=1, column=2, sticky="w", padx=4)
        ttk.Entry(rf, textvariable=self.cal_az, width=6).grid(row=1, column=3, sticky="w")
        self.cal_btn = ttk.Button(rf, text="Record and calibrate", command=lambda: self._cal_record("reflector"))
        self.cal_btn.grid(row=2, column=0, columnspan=4, sticky="we", padx=4, pady=4)
        ttk.Label(rf, text="(then move it to another angle and check)", foreground=theme.SECONDARY).grid(
            row=3, column=0, columnspan=4, sticky="w", padx=4)
        ttk.Label(rf, text="Distance [m]").grid(row=4, column=0, sticky="w", padx=4)
        ttk.Entry(rf, textvariable=self.ver_range, width=6).grid(row=4, column=1, sticky="w")
        ttk.Label(rf, text="Angle [deg]").grid(row=4, column=2, sticky="w", padx=4)
        ttk.Entry(rf, textvariable=self.ver_az, width=6).grid(row=4, column=3, sticky="w")
        self.ver_btn = ttk.Button(rf, text="Record and verify", command=lambda: self._cal_record("verify"))
        self.ver_btn.grid(row=5, column=0, columnspan=4, sticky="we", padx=4, pady=(4, 6))

        pf = ttk.LabelFrame(cols, text="Person (no reflector)")
        pf.pack(side="left", fill="both", expand=True, padx=(0, 6))
        ttk.Label(pf, text="Someone sits still, alone, facing the radar 1.5-3 m away and breathes "
                           "quietly; the breathing chest is taken as a point. Angle: where they sit "
                           "(empty: where the raw channels put them - angles relative). Six TX, 20 s.",
                  wraplength=330, justify="left").grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=4)
        self.person_az = tk.StringVar(value="0")
        ttk.Label(pf, text="Angle [deg]").grid(row=1, column=0, sticky="w", padx=4)
        ttk.Entry(pf, textvariable=self.person_az, width=6).grid(row=1, column=1, sticky="w")
        self.person_btn = ttk.Button(pf, text="Record a person and calibrate",
                                     command=lambda: self._cal_record("person"))
        self.person_btn.grid(row=2, column=0, columnspan=2, sticky="we", padx=4, pady=(4, 2))
        ttk.Button(pf, text="Calibrate on a person in a take ...",
                   command=lambda: self._cal_from_take("person")).grid(row=3, column=0, columnspan=2,
                                                                       sticky="we", padx=4, pady=(2, 6))

        af = ttk.LabelFrame(cols, text="Autofocus (no reflector, angles relative)")
        af.pack(side="left", fill="both", expand=True)
        ttk.Label(af, text="The still point-like reflectors of the room, judged by the echoes left out "
                           "of the fit and by the people in it: kept only when they focus better and "
                           "none the raw channels focus well gets worse. A turn of the whole picture "
                           "stays unknown. Six TX, 10 s.",
                  wraplength=430, justify="left").pack(anchor="w", padx=4, pady=4)
        self.af_btn = ttk.Button(af, text="Record the room and autofocus", command=lambda: self._cal_record("autofocus"))
        self.af_btn.pack(fill="x", padx=4, pady=2)
        ttk.Button(af, text="Autofocus a take ...",
                   command=lambda: self._cal_from_take("autofocus")).pack(fill="x", padx=4, pady=(2, 6))

        res = ttk.LabelFrame(root, text="Result")
        res.pack(fill="both", expand=True)
        bar = ttk.Frame(res)
        bar.pack(fill="x", padx=4, pady=4)
        self.cal_save_btn = theme.Button(bar, text="Save as the board's calibration", command=self._cal_save,
                                         state="disabled")
        self.cal_save_btn.pack(side="left")
        self.cal_phase = tk.Label(bar, text="", font=theme.font(11, "bold"))
        self.cal_phase.pack(side="left", padx=10)
        self.cal_text = tk.Text(res, height=12, wrap="word", font=theme.mono(9))
        self.cal_text.pack(fill="both", expand=True, padx=4, pady=(0, 4))

    def _show_calibration(self):
        cal = self.calibration
        if cal is None:
            text, colour = "✗ Not calibrated - angles are not measured (Calibration tab)", theme.ORANGE
        elif cal.method in ("autofocus", "person"):
            text, colour = f"✓ Calibrated: {cal.describe()}", theme.ORANGE
        else:
            text, colour = f"✓ Calibrated: {cal.describe()}", theme.GREEN
        self.cal_lbl.configure(text=text, fg=colour)
        self.cal_status.configure(text=text, fg=colour)
        self.cal_take_lbl.configure(
            text=("Takes keep this calibration in their _cascade.json; angles in DARTscope use it."
                  if cal is not None else
                  "Without a calibration Multi Vitals finds people by range only; the angle views "
                  "need the antenna configuration selected."), fg=theme.SECONDARY)

    def _cal_text(self, text: str, clear: bool = True):
        if clear:
            self.cal_text.delete("1.0", "end")
        self.cal_text.insert("end", text + "\n")
        self.cal_text.see("end")

    def _cal_numbers(self, *pairs):
        try:
            return [float(v.get()) for v in pairs]
        except ValueError:
            self.report_error("Calibration", "Distance and angle must be numbers.")
            return None

    def _person_angle(self):
        """[0, angle or None]: an empty field keeps the raw channels' pointing."""
        s = self.person_az.get().strip()
        if not s:
            return [0.0, None]
        try:
            return [0.0, float(s)]
        except ValueError:
            self.report_error("Calibration", "The person's angle must be a number (or empty).")
            return None

    def _cal_params(self, seconds: float) -> CascadeParams | None:
        """
        The calibration take: CAL_PRESET's chirp profile, all six TX one after another,
        seconds long - not the take settings: with the SDK profile there (2.1 m range
        resolution) the whole room fell into two range cells and no echo was a point.
        The phases are moved to the frequency of every other take (path offsets).
        """
        p = CascadeParams.preset(CAL_PRESET)
        p.tx_chirps = TDM6
        p.num_loops = 8
        p.frame_period_ms = max(float(p.frame_period_ms), 50.0)
        p.duration_s = seconds
        rep = p.check(Path(self.folder_var.get()).expanduser())
        if rep["errors"]:
            self.report_error("Calibration", "The take settings cannot be recorded with six TX: "
                                             + "; ".join(rep["errors"]))
            return None
        return p

    def _cal_record(self, kind: str):
        if self._busy():
            return
        if kind == "verify" and self.pending_cal is None and self.calibration is None:
            self.report_error("Calibration", "There is no calibration to verify yet.")
            return
        if kind == "person":
            nums = self._person_angle()
        else:
            nums = (self._cal_numbers(self.ver_range, self.ver_az) if kind == "verify" else
                    self._cal_numbers(self.cal_range, self.cal_az) if kind == "reflector" else [0.0, 0.0])
        if nums is None:
            return
        p = self._cal_params({"autofocus": 10.0, "person": 20.0}.get(kind, 5.0))
        if p is None:
            return
        if self.release_hardware is not None:
            why = self.release_hardware()
            if why:
                self.report_error("Calibration", why)
                return
        folder = Path(self.folder_var.get()).expanduser() / "calibration"
        folder.mkdir(parents=True, exist_ok=True)
        name = time.strftime(f"cal_{kind}_%Y%m%d_%H%M%S")
        self._cal_job = (kind, nums)
        self.recording = CascadeRecording(p, folder, name, notes=f"calibration take ({kind})")
        self._cal_text(f"Recording {name} in {folder} ({CAL_PRESET} profile, six TX, {p.duration_s:g} s) ...")
        self.cal_phase.configure(text="Recording ...")
        self.stop_btn.configure(state="normal")
        self.progress["value"] = 0

        def progress(k, **kw):
            self.q.put((k, kw))

        def work():
            try:
                self.q.put(("done", self.recording.run(progress)))
            except Exception as e:
                self.q.put(("failed", str(e)))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()
        self._set_start(False)

    def _cal_take_finished(self, meta, error):
        """A calibration take is in: compute from it off the Tk thread."""
        self.worker = None
        self.stop_btn.configure(state="disabled")
        self.phase_lbl.configure(text="")
        if error:
            self._cal_job = None
            self.cal_phase.configure(text="")
            self._cal_text("✗ " + error, clear=False)
            self.report_error("Calibration take failed", error)
            self.update_checks()
            return
        self._cal_compute(self.recording.first_part, *self._cal_job)

    def _cal_compute(self, bin_path, kind, nums):
        self.cal_phase.configure(text="Computing ...")
        cal = self.pending_cal or self.calibration

        def work():
            try:
                from cascade.decode import cascade_params
                from radar_processing import derive_capture_dimensions, load_raw_cube
                import contextlib
                import io
                with contextlib.redirect_stdout(io.StringIO()):
                    params, _ = cascade_params(bin_path)
                    cube = load_raw_cube(str(bin_path), params, derive_capture_dimensions(params))
                src = Path(bin_path).name.split("_Raw_")[0]
                if kind == "reflector":
                    out = cc.from_reflector(cube, params, range_m=nums[0], az_deg=nums[1], source=src)
                elif kind == "autofocus":
                    out = cc.autofocus(cube, params, source=src)
                elif kind == "person":
                    out = cc.from_person(cube, params, az_deg=nums[1], source=src)
                else:
                    out = cc.verify(cube, params, cal, range_m=nums[0], az_deg=nums[1])
                self.q.put(("cal_done", (kind, out)))
            except Exception as e:
                self.q.put(("cal_failed", f"{type(e).__name__}: {e}"))

        threading.Thread(target=work, daemon=True).start()

    def _cal_finished(self, result, error):
        self._cal_job = None
        self.cal_phase.configure(text="")
        self.update_checks()
        if error:
            self._cal_text("✗ " + error, clear=False)
            return
        kind, out = result
        if kind == "verify":
            target = self.pending_cal or self.calibration
            target.verify = out
            ok = out["error_after_deg"] <= 1.0
            self._cal_text(f"{'✓' if ok else '⚠'} reflector at {out['az_deg']:+.1f} deg: {out['az_before']:+.1f} "
                           f"deg without, {out['az_after']:+.1f} deg with the calibration (beam "
                           f"{out['coherence_before']:.2f} -> {out['coherence_after']:.2f})", clear=False)
            if target is self.calibration and self.pending_cal is None:
                cc.save_current(self.calibration)          # the check belongs to it
            self._show_calibration()
            return
        self.pending_cal = out
        self._cal_text(out.summary())
        self._cal_text("\nSave it to make it the board's calibration (and check it at another angle).",
                       clear=False)
        self.cal_save_btn.configure(state="normal")

    def _cal_from_take(self, kind: str = "autofocus"):
        if self._busy():
            return
        nums = self._person_angle() if kind == "person" else [0.0, 0.0]
        if nums is None:
            return
        f = filedialog.askopenfilename(title="A six-TX cascade take",
                                       initialdir=self.folder_var.get() or str(Path.home()),
                                       filetypes=[("Cascade take", "*_Raw_0.bin")])
        if f:
            self._cal_text(f"{'Person calibration' if kind == 'person' else 'Autofocus'} on {Path(f).name} ...")
            self._cal_compute(Path(f), kind, nums)

    def _cal_save(self):
        if self.pending_cal is None:
            return
        cc.save_current(self.pending_cal)
        self.calibration, self.pending_cal = self.pending_cal, None
        self.cal_save_btn.configure(state="disabled")
        self._cal_text(f"Saved: {cc.current_path()}", clear=False)
        self._show_calibration()

    def _cal_forget(self):
        if self.calibration is None:
            return
        if not messagebox.askyesno("Calibration", "Forget the board's calibration? (It stays in the "
                                                  "archive, and takes keep their own copy.)"):
            return
        cc.forget_current()
        self.calibration = None
        self._show_calibration()

    def _cal_load(self):
        f = filedialog.askopenfilename(title="A cascade calibration", initialdir=str(cc.store_dir()),
                                       filetypes=[("Calibration", "*.json")])
        if not f:
            return
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            self.pending_cal = cc.CascadeCalibration.from_dict(d.get("calibration", d))
        except Exception as e:
            self.report_error("Calibration", f"Not a cascade calibration: {e}")
            return
        self._cal_text(self.pending_cal.summary())
        self.cal_save_btn.configure(state="normal")

    def _cal_export(self):
        cal = self.pending_cal or self.calibration
        if cal is None:
            return
        f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Calibration", "*.json")],
                                         initialfile="cascade_calibration.json")
        if f:
            Path(f).write_text(json.dumps(cal.to_dict(), indent=2), encoding="utf-8")

    def busy(self) -> bool:
        return self._busy()

    def close(self):
        if self.recording is not None:
            self.recording.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=60)


class CascadePreview:
    """A cascade take read back: range profile, range-time, range-velocity, displacement."""

    def __init__(self, parent, tab: CascadeTab):
        self.tab = tab
        self.take: CascadeTake | None = None
        top = ttk.Frame(parent)
        top.pack(fill="x")
        ttk.Button(top, text="Open take ...", command=self._choose).pack(side="left")
        self.dart_btn = ttk.Button(top, text="Open in DARTscope", state="disabled",
                                   command=lambda: tab.open_recording(str(self.take.parts[0][0])))
        if tab.open_recording is not None:
            self.dart_btn.pack(side="right")
        self.title = ttk.Label(top, text="No take open", font=theme.font(11, "bold"))
        self.title.pack(side="left", padx=12)
        ttk.Label(top, text="Frame").pack(side="left", padx=(20, 4))
        self.scale = ttk.Scale(top, from_=0, to=1, orient="horizontal", length=220,
                               command=lambda v: self._frame_moved())
        self.scale.pack(side="left")
        self.frame_lbl = ttk.Label(top, text="0", width=6)
        self.frame_lbl.pack(side="left")
        ttk.Label(top, text="RX").pack(side="left", padx=(12, 4))
        self.rx_var = tk.StringVar(value="1")
        rx = ttk.Combobox(top, textvariable=self.rx_var, values=[str(i) for i in range(1, 9)], width=3,
                          state="readonly")
        rx.pack(side="left")
        rx.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        ttk.Label(parent, text="A click in the range-time map picks the range of the displacement.",
                  foreground=theme.SECONDARY).pack(fill="x", pady=(4, 0))
        self.fig = Figure(figsize=(12, 7), dpi=90)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        NavigationToolbar2Tk(self.canvas, parent).update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._click)
        self._job = None

    def _choose(self):
        f = filedialog.askopenfilename(initialdir=self.tab.folder_var.get() or None,
                                       filetypes=[("Cascade take", "*_cascade.json *_Raw_0.bin")])
        if f:
            try:
                self.open(Path(f))
            except Exception as e:
                self.tab.report_error("Cascade preview", str(e))

    def open(self, path: Path):
        take = CascadeTake.open(path)
        if take.num_frames == 0:
            raise ValueError("the take holds no whole frame")
        self.take = take
        self.dart_btn.configure(state="normal")
        self.scale.configure(to=max(take.num_frames - 1, 1))
        self.scale.set(take.num_frames // 2)
        self.title.configure(text=f"{take.prefix} - {take.num_frames} frames, {take.num_frames / take.frame_rate_hz:.1f} s, "
                                  f"range resolution {take.range_res_m * 100:.1f} cm")
        win = np.hanning(take.num_samples).astype(np.float32)
        self.rt = np.fft.fft(take.first_chirps() * win, axis=2)          # [frame, rx, bin]
        prof = np.abs(self.rt).mean(axis=(0, 1))
        skip = max(1, int(round(0.25 / take.range_res_m)))              # leakage below ~25 cm
        self.sel_bin = skip + int(np.argmax(prof[skip:int(take.num_samples * 0.9)]))
        self.redraw()

    def _frame_moved(self):
        if self.take is None:
            return
        self.frame_lbl.configure(text=str(int(float(self.scale.get()))))
        if self._job:
            self.tab.frame.after_cancel(self._job)
        self._job = self.tab.frame.after(120, self.redraw)

    def _click(self, ev):
        if self.take is not None and ev.inaxes is getattr(self, "ax_rt", None) and ev.ydata is not None:
            self.sel_bin = int(round(ev.ydata / self.take.range_res_m))
            self.redraw()

    def redraw(self):
        take = self.take
        if take is None:
            return
        self._job = None
        f = int(float(self.scale.get()))
        rx = int(self.rx_var.get()) - 1
        n, dr, fps = take.num_samples, take.range_res_m, take.frame_rate_hz
        db = lambda z: 20 * np.log10(np.abs(z) + 1e-6)
        fig = self.fig
        fig.clear()
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.4, 1])
        rng = np.arange(n) * dr

        ax = fig.add_subplot(gs[0, 0])
        for r in range(take.num_rx):
            ax.plot(rng, db(self.rt[f, r]), lw=0.8)
        ax.axvline(self.sel_bin * dr, color="k", ls=":", lw=1)
        ax.set(title=f"Range profile, frame {f}, 8 RX", xlabel="range [m]", ylabel="dB")
        ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[0, 1])
        self.ax_rt = ax
        mag = db(np.abs(self.rt).mean(axis=1)).T
        ax.imshow(mag, aspect="auto", origin="lower", extent=[0, len(self.rt) / fps, 0, n * dr],
                  vmin=np.percentile(mag, 20), vmax=mag.max(), cmap="viridis")
        ax.axhline(self.sel_bin * dr, color="w", ls=":", lw=1)
        ax.axvline(f / fps, color="r", lw=0.8)
        ax.set(title="Range-time (click picks the range)", xlabel="time [s]", ylabel="range [m]")

        ax = fig.add_subplot(gs[0, 2])
        cube = take.frames(f, 1)[0][0::take.chirps_in_loop]              # the first TX slot: [loop, rx, n]
        rp = np.fft.fft(cube * np.hanning(n), axis=2)
        rd = np.fft.fftshift(np.fft.fft(rp * np.hanning(len(rp))[:, None, None], axis=0), axes=0)
        rdm = db(np.abs(rd).mean(axis=1))
        vmax = take.wavelength_m / (4 * take.chirp_time_s * take.chirps_in_loop)
        ax.imshow(rdm.T, aspect="auto", origin="lower", extent=[-vmax, vmax, 0, n * dr],
                  vmin=np.percentile(rdm, 30), vmax=rdm.max(), cmap="viridis")
        ax.set(title="Range-velocity", xlabel="velocity [m/s]", ylabel="range [m]")

        ax = fig.add_subplot(gs[1, :2])
        b = min(self.sel_bin, n - 1)
        ph = np.unwrap(np.angle(self.rt[:, rx, b]))
        disp = (ph - ph.mean()) * take.wavelength_m / (4 * np.pi) * 1000
        t = np.arange(len(disp)) / fps
        ax.plot(t, disp, lw=0.8)
        ax.set(title=f"Displacement at {b * dr:.2f} m (bin {b}), RX{rx + 1}", xlabel="time [s]", ylabel="mm")
        ax.grid(alpha=0.3)
        ax2 = fig.add_subplot(gs[1, 2])
        if len(disp) > 4 * fps:
            d = disp - np.polyval(np.polyfit(t, disp, 2), t)
            spec = np.abs(np.fft.rfft(d * np.hanning(len(d))))
            fq = np.fft.rfftfreq(len(d), 1 / fps)
            ax2.plot(fq * 60, spec, lw=0.8)
            ax2.set(xlim=(0, 180), title="Displacement spectrum", xlabel="per minute")
            txt = ""
            for name, lo, hi, colour in (("breathing", 0.1, 0.6, theme.GREEN), ("heart", 0.8, 2.5, theme.RED)):
                m = (fq >= lo) & (fq <= hi)
                if m.any() and fq[-1] > hi:
                    k = np.argmax(spec[m])
                    txt += f"{name} ≈ {fq[m][k] * 60:.0f}/min\n"
                    ax2.axvline(fq[m][k] * 60, color=colour, ls=":", lw=1)
            ax2.text(0.98, 0.95, txt + "(rough)", transform=ax2.transAxes, ha="right", va="top")
            ax2.grid(alpha=0.3)
        fig.tight_layout()
        self.canvas.draw_idle()


def main():
    """python cascade_gui.py - the tab in a window of its own."""
    scale = theme.enable_dpi_awareness()
    root = tk.Tk()
    try:
        from version import __version__
    except ImportError:
        __version__ = ""
    root.title(f"DARTscope {__version__} - Cascade radar".replace("  ", " "))
    theme.initial_geometry(root, 1400, 900, scale)
    tab = CascadeTab(root)

    def on_close():
        if tab.busy() and not messagebox.askyesno("Cascade radar", "A take is running. Stop it and close?"):
            return
        tab.close()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
