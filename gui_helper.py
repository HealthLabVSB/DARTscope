"""
gui_helper.py

The Helper tab: a chirp configuration for mmWave Studio from what is to be measured
(chirp_design.py does the work).

    1. What to measure   the device, a purpose that fills in sensible goals (vital signs,
                         people and angles, fast limb movements, heart rate of a moving
                         person, tremor), and the goals themselves; Design.
    2. Chirp for mmWave Studio   the designed values in mmWave Studio's own terms, which can
                         be edited or read from the decoded recording, and checked again.
    3. Script            COM port, mmWave Studio folder and the output file of the Lua
                         script; Copy Lua and Save Lua on the script page.

On the right: what the configuration gives with every check (ok, warning, error with
what to change), and the Lua script for mmWave Studio's Lua shell.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import theme

import chirp_design as cd
from flow_layout import FlowFrame, ScrollFrame

LEVEL_FG = {k: theme.LEVEL[k] for k in ("ok", "warning", "error")}
MARK = {"ok": "ok     ", "warning": "WARNING", "error": "ERROR  "}

GOAL_FIELDS = (
    ("max_range_m", "Max range [m]"),
    ("range_res_cm", "Range resolution [cm]"),
    ("max_vel_ms", "Max velocity [m/s]"),
    ("vel_res_ms", "Velocity resolution [m/s]"),
    ("loops_per_frame", "Loops per frame"),
    ("fps", "Frame rate [fps]"),
    ("duration_s", "Recording length [s]"),
    ("rx_gain_db", "RX gain [dB]"),
    ("start_ghz", "Start frequency [GHz]"),
    ("idle_us", "Idle time [us]"),
    ("adc_start_us", "ADC start time [us]"),
)
CHIRP_FIELDS = (
    ("start_ghz", "Start Freq [GHz]"),
    ("idle_us", "Idle Time [us]"),
    ("adc_start_us", "ADC Start Time [us]"),
    ("ramp_end_us", "Ramp End Time [us]"),
    ("slope_mhz_us", "Frequency Slope [MHz/us]"),
    ("tx_start_us", "TX Start Time [us]"),
    ("n_samples", "ADC Samples"),
    ("fs_ksps", "Sample Rate [ksps]"),
    ("rx_gain_db", "RX Gain [dB]"),
    ("tx_order", "TX of each chirp"),
    ("loops", "No of Chirp Loops"),
    ("frames", "No of Frames"),
    ("period_ms", "Periodicity [ms]"),
)


class HelperTab:
    def __init__(self, parent, get_params=None):
        self.parent = parent
        self.get_params = get_params
        self.design: cd.Design | None = None
        self.purpose = ""
        self.goal_vars = {}
        self.chirp_vars = {}

        bar = FlowFrame(parent)
        bar.pack(side="top", fill="x", padx=10, pady=5)
        g = bar.group()
        ttk.Label(g, text="Device:", font=theme.font(10, "bold")).pack(side="left", padx=(0, 4))
        self.device_var = tk.StringVar(value=next(iter(cd.DEVICES)))
        dev = ttk.Combobox(g, textvariable=self.device_var, values=list(cd.DEVICES),
                           state="readonly", width=20)
        dev.pack(side="left")
        dev.bind("<<ComboboxSelected>>", lambda _e: self._on_device())
        g = bar.group()
        ttk.Label(g, text="Purpose:", font=theme.font(10, "bold")).pack(side="left", padx=(0, 4))
        self.purpose_var = tk.StringVar(value="Custom")
        pur = ttk.Combobox(g, textvariable=self.purpose_var,
                           values=["Custom"] + list(cd.PURPOSES), state="readonly", width=34)
        pur.pack(side="left")
        pur.bind("<<ComboboxSelected>>", lambda _e: self._on_purpose())
        g = bar.group()
        ttk.Button(g, text="Design", command=self.run_design).pack(side="left", padx=(6, 2))
        ttk.Button(g, text="Check these values", command=self.run_check).pack(side="left", padx=2)
        if get_params is not None:
            ttk.Button(g, text="From the decoded recording",
                       command=self.from_recording).pack(side="left", padx=2)
        self.status = tk.StringVar(value="Choose a device and a purpose, then Design.")
        ttk.Label(bar.group(), textvariable=self.status, foreground=theme.SECONDARY).pack(side="left")

        paned = ttk.PanedWindow(parent, orient="horizontal")
        paned.pack(side="top", fill="both", expand=True, padx=10, pady=5)
        left_scroll = ScrollFrame(paned)
        paned.add(left_scroll, weight=0)
        left = left_scroll.inner
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        # 1. goals
        box = ttk.LabelFrame(left, text="1. What to measure")
        box.pack(fill="x", pady=4)
        for key, label in GOAL_FIELDS:
            self.goal_vars[key] = self._row(box, label)
        row = ttk.Frame(box)
        row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="Transmitters", width=26).pack(side="left")
        self.tdm_var = tk.StringVar(value="one TX")
        ttk.Combobox(row, textvariable=self.tdm_var, values=["one TX", "all TX (TDM-MIMO)"],
                     state="readonly", width=17).pack(side="left")
        row = ttk.Frame(box)
        row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text="TX for one TX", width=26).pack(side="left")
        self.single_tx_var = tk.StringVar(value="TX1")
        self.single_tx_combo = ttk.Combobox(row, textvariable=self.single_tx_var, state="readonly",
                                            width=6)
        self.single_tx_combo.pack(side="left")
        self.back_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, text="Frames back to back (continuous chirping)",
                        variable=self.back_var).pack(anchor="w", padx=5, pady=2)

        # 2. the chirp in mmWave Studio's terms
        box = ttk.LabelFrame(left, text="2. Chirp for mmWave Studio")
        box.pack(fill="x", pady=4)
        for key, label in CHIRP_FIELDS:
            self.chirp_vars[key] = self._row(box, label)

        # 3. the script
        box = ttk.LabelFrame(left, text="3. Script")
        box.pack(fill="x", pady=4)
        self.com_var = self._row(box, "COM port")
        self.studio_var = self._row(box, "mmWave Studio folder", width=34)
        self.out_var = self._row(box, "Output file (.bin)", width=34)

        # right: result and script
        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)
        self.nb = nb
        page = ttk.Frame(nb)
        nb.add(page, text="Result")
        self.result = tk.Text(page, wrap="word", font=theme.mono(9), state="disabled")
        ys = ttk.Scrollbar(page, orient="vertical", command=self.result.yview)
        self.result.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y")
        self.result.pack(side="left", fill="both", expand=True)
        for lvl, fg in LEVEL_FG.items():
            self.result.tag_configure(lvl, foreground=fg)
        self.result.tag_configure("head", font=theme.mono(10, "bold"))
        page = ttk.Frame(nb)
        nb.add(page, text="Lua script")
        tools = FlowFrame(page)
        tools.pack(side="top", fill="x", padx=4, pady=4)
        ttk.Button(tools.group(), text="Copy Lua", command=self.copy_lua).pack(side="left")
        ttk.Button(tools.group(), text="Save Lua...", command=self.save_lua).pack(side="left")
        ttk.Label(tools.group(), text="(paste into mmWave Studio's Lua shell, or run the saved file)",
                  foreground=theme.SECONDARY).pack(side="left")
        self.lua = tk.Text(page, wrap="none", font=theme.mono(9))
        ys = ttk.Scrollbar(page, orient="vertical", command=self.lua.yview)
        xs = ttk.Scrollbar(page, orient="horizontal", command=self.lua.xview)
        self.lua.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        xs.pack(side="bottom", fill="x")
        ys.pack(side="right", fill="y")
        self.lua.pack(side="left", fill="both", expand=True)

        self._on_device()
        self._fill_goals(cd.Goals())

    # --- layout helpers ----------------------------------------------------------------------
    @staticmethod
    def _row(parent, label, width=12):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text=label, width=26).pack(side="left")
        var = tk.StringVar(value="")
        ttk.Entry(row, textvariable=var, width=width).pack(side="left")
        return var

    def _device(self) -> cd.Device:
        return cd.DEVICES[self.device_var.get()]

    def _on_device(self):
        dev = self._device()
        self.single_tx_combo.configure(values=[f"TX{i}" for i in range(1, dev.n_tx + 1)])
        if self.single_tx_var.get() not in self.single_tx_combo.cget("values"):
            self.single_tx_var.set("TX1")
        self.goal_vars["start_ghz"].set(f"{dev.default_start_ghz:g}")
        self.com_var.set(str(dev.com_port))
        self.studio_var.set(dev.studio_dir)
        self.out_var.set(dev.studio_dir + r"\mmWaveStudio\PostProc\adc_data.bin")
        self.status.set(f"{dev.name}: {dev.f_lo:g}-{dev.f_hi:g} GHz, {dev.n_tx} TX x {dev.n_rx} RX"
                        + (f"; {dev.note}" if dev.note else ""))

    def _on_purpose(self):
        name = self.purpose_var.get()
        if name in cd.PURPOSES:
            g = cd.goals_for(name)
            g.start_ghz = self._device().default_start_ghz
            self._fill_goals(g)
            self.run_design()

    def _fill_goals(self, g: cd.Goals):
        for key, _ in GOAL_FIELDS:
            v = getattr(g, key)
            if key == "start_ghz" and v is None:
                v = self._device().default_start_ghz
            self.goal_vars[key].set("" if v is None else f"{v:g}")
        self.tdm_var.set("all TX (TDM-MIMO)" if g.tdm_all_tx else "one TX")
        self.single_tx_var.set(f"TX{g.single_tx}")
        self.back_var.set(bool(g.back_to_back))

    def _goals(self) -> cd.Goals:
        def num(key, cast=float, optional=False):
            s = self.goal_vars[key].get().strip().replace(",", ".")
            if not s:
                if optional:
                    return None
                raise ValueError(f"{dict(GOAL_FIELDS)[key]} is empty")
            return cast(float(s))
        g = cd.Goals(max_range_m=num("max_range_m"), range_res_cm=num("range_res_cm"),
                     max_vel_ms=num("max_vel_ms"), vel_res_ms=num("vel_res_ms", optional=True),
                     loops_per_frame=num("loops_per_frame", int, optional=True), fps=num("fps"),
                     duration_s=num("duration_s"), rx_gain_db=num("rx_gain_db", int),
                     start_ghz=num("start_ghz", optional=True), idle_us=num("idle_us"),
                     adc_start_us=num("adc_start_us"),
                     tdm_all_tx=self.tdm_var.get().startswith("all"),
                     single_tx=int(self.single_tx_var.get()[2:] or 1),
                     back_to_back=bool(self.back_var.get()))
        if min(g.max_range_m, g.range_res_cm, g.max_vel_ms, g.fps, g.duration_s) <= 0:
            raise ValueError("range, resolution, velocity, frame rate and length must be above 0")
        return g

    # --- actions -----------------------------------------------------------------------------
    def run_design(self):
        try:
            g = self._goals()
        except ValueError as e:
            messagebox.showerror("Helper", f"Invalid value: {e}")
            return
        self.purpose = self.purpose_var.get() if self.purpose_var.get() in cd.PURPOSES else ""
        self._show(cd.design(self._device(), g))

    def run_check(self):
        try:
            vals = {}
            for key, label in CHIRP_FIELDS:
                s = self.chirp_vars[key].get().strip().replace(",", ".") if key != "tx_order" \
                    else self.chirp_vars[key].get()
                if not s:
                    raise ValueError(f"{label} is empty - Design first, or type it in")
                if key == "tx_order":
                    vals[key] = [int(t.strip().upper().replace("TX", "")) for t in s.replace(";", ",").split(",")
                                 if t.strip()]
                elif key in ("n_samples", "fs_ksps", "rx_gain_db", "loops", "frames"):
                    vals[key] = int(float(s))
                else:
                    vals[key] = float(s)
        except ValueError as e:
            messagebox.showerror("Helper", f"Invalid value: {e}")
            return
        try:
            goals = self._goals()
        except ValueError:
            goals = None
        self._show(cd.evaluate(self._device(), goals, **vals), checked=True)

    def from_recording(self):
        params = self.get_params() if self.get_params else None
        if not params:
            messagebox.showwarning("Helper", "Decode a recording first (Decode tab).")
            return
        dev = cd.device_for(params)
        self.device_var.set(dev.name)
        self._on_device()
        vals = cd.from_params(params)
        self.purpose = ""
        self.purpose_var.set("Custom")
        self._show(cd.evaluate(dev, None, **vals), checked=True)
        self.status.set("The configuration of the decoded recording - edit it in 2. and "
                        "Check these values")

    def _show(self, d: cd.Design, checked: bool = False):
        self.design = d
        for key, _ in CHIRP_FIELDS:
            v = getattr(d, key)
            if key == "tx_order":
                v = ", ".join(f"TX{t}" for t in v)
            elif isinstance(v, float):
                v = cd._f(v, 3)
            self.chirp_vars[key].set(str(v))
        self._write_result(d, checked)
        self._write_lua(d)
        n_err = sum(c.level == "error" for c in d.checks)
        n_warn = sum(c.level == "warning" for c in d.checks)
        self.status.set(("Checked" if checked else "Designed") + f": {n_err} error(s), "
                        f"{n_warn} warning(s)" + (" - see Result" if n_err or n_warn else ""))

    def _write_result(self, d: cd.Design, checked: bool):
        t = self.result
        t.configure(state="normal")
        t.delete("1.0", "end")
        dev, p = d.device, d.perf
        t.insert("end", ("The configuration as entered\n" if checked else
                         f"Designed for: {self.purpose or 'custom goals'}\n"), "head")
        for s in cd.summary_lines(d):
            t.insert("end", f"  {s}\n")
        t.insert("end", "\nProfile (mmWave Studio: SensorConfig > Profile, or ar1.ProfileConfig)\n", "head")
        rows = [("Start Freq", f"{cd._f(d.start_ghz)} GHz"), ("Idle Time", f"{cd._f(d.idle_us, 2)} us"),
                ("ADC Start Time", f"{cd._f(d.adc_start_us, 2)} us"),
                ("Ramp End Time", f"{cd._f(d.ramp_end_us, 2)} us"),
                ("Frequency Slope", f"{cd._f(d.slope_mhz_us)} MHz/us"),
                ("TX Start Time", f"{cd._f(d.tx_start_us, 2)} us"), ("ADC Samples", f"{d.n_samples}"),
                ("Sample Rate", f"{d.fs_ksps} ksps"), ("RX Gain", f"{d.rx_gain_db} dB")]
        for k, v in rows:
            t.insert("end", f"  {k:<18}{v}\n")
        t.insert("end", "\nChirps (SensorConfig > Chirp, or ar1.ChirpConfig)\n", "head")
        for i, tx in enumerate(d.tx_order):
            t.insert("end", f"  chirp {i}: profile 0, TX{tx} on\n")
        t.insert("end", "\nFrame (SensorConfig > Frame, or ar1.FrameConfig)\n", "head")
        t.insert("end", f"  Start Chirp {0}, End Chirp {len(d.tx_order) - 1}, No of Frames {d.frames}, "
                        f"No of Chirp Loops {d.loops}, Periodicity {cd._f(d.period_ms, 2)} ms\n")
        t.insert("end", "\nWhat it gives\n", "head")
        rows = [("range resolution", f"{p['range_res_cm']:.2f} cm (effective bandwidth "
                                     f"{p['bandwidth_eff_ghz']:.2f} GHz of {p['sweep_ghz']:.2f} swept)"),
                ("maximum range", f"{p['max_range_m']:.2f} m"),
                ("maximum velocity", f"+-{p['max_vel_ms']:.2f} m/s"),
                ("velocity resolution", f"{p['vel_res_ms']:.3f} m/s"),
                ("chirp", f"{p['chirp_time_us']:.2f} us, sampling {p['adc_time_us']:.2f} us"),
                ("frame", f"{p['chirps_per_frame']} chirps, {p['active_ms']:.2f} ms of "
                          f"{cd._f(d.period_ms, 2)} ms, {p['fps']:.1f} fps"),
                ("data", f"{p['rate_mb_s']:.1f} MB/s on average, {p['burst_mb_s']:.0f} MB/s while "
                         f"chirping, {p['rate_mb_s'] * d.frames * d.period_ms / 1e6:.2f} GB in all"),
                ("wavelength", f"{p['wavelength_mm']:.3f} mm")]
        for k, v in rows:
            t.insert("end", f"  {k:<20}{v}\n")
        t.insert("end", "\nChecks\n", "head")
        order = {"error": 0, "warning": 1, "ok": 2}
        for c in sorted(d.checks, key=lambda c: order[c.level]):
            t.insert("end", f"  {MARK[c.level]} {c.what}: {c.message}\n", c.level)
        t.insert("end", f"\nDevice limits used ({dev.name}): {dev.f_lo:g}-{dev.f_hi:g} GHz, "
                        f"{dev.max_bw_ghz:g} GHz sweep, slope <= {dev.max_slope:g} MHz/us, "
                        f"sampling <= {dev.max_fs_ksps} ksps "
                        f"({'complex' if dev.complex_samples else 'real'})"
                        + (f", IF <= {dev.max_if_mhz:g} MHz" if dev.max_if_mhz else "")
                        + (f". {dev.note}." if dev.note else ".") + "\n")
        t.configure(state="disabled")

    def _write_lua(self, d: cd.Design):
        try:
            com = int(self.com_var.get())
        except ValueError:
            com = None
        script = cd.lua_script(d, com_port=com, studio_dir=self.studio_var.get().strip() or None,
                               out_bin=self.out_var.get().strip() or None, purpose=self.purpose)
        self.lua.delete("1.0", "end")
        self.lua.insert("1.0", script)

    def copy_lua(self):
        if self.design is None:
            messagebox.showwarning("Helper", "Design first.")
            return
        self._write_lua(self.design)             # the script fields may have changed
        self.parent.clipboard_clear()
        self.parent.clipboard_append(self.lua.get("1.0", "end"))
        self.status.set("Lua script copied - paste it into mmWave Studio's Lua shell")

    def save_lua(self):
        if self.design is None:
            messagebox.showwarning("Helper", "Design first.")
            return
        self._write_lua(self.design)
        path = filedialog.asksaveasfilename(defaultextension=".lua", initialfile="chirp_config.lua",
                                            filetypes=[("Lua scripts", "*.lua")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.lua.get("1.0", "end"))
        except OSError as e:
            messagebox.showerror("Helper", f"Could not save: {e}")
            return
        self.status.set(f"Saved {path} - run it from mmWave Studio's Lua shell (dofile)")


helper_tab = None


def init_helper_tab(parent, get_params=None):
    """Builds the Helper tab; get_params() returns the decoded recording's parameters."""
    global helper_tab
    helper_tab = HelperTab(parent, get_params)
    return helper_tab
