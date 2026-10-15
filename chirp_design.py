"""
chirp_design.py

A chirp configuration for mmWave Studio from what is to be measured: the Helper tab
without Tk.

    design(device, goals)   the profile, chirps and frame for the goals (maximum range,
                            range resolution, maximum velocity, velocity resolution or
                            chirps per frame, frame rate, transmitters), in the units
                            mmWave Studio takes, quantised as the chip quantises them;
                            what they give; and every check with what to change
    performance(...)        what a given profile, chirp pattern and frame give
    lua_script(...)         a Lua script for mmWave Studio's Lua shell, in the shape of the
                            lab's working scripts (C:/Data/Radar/config): device set-up,
                            ProfileConfig, one ChirpConfig per transmitter, DCA1000,
                            FrameConfig and the recording

The rules, from the FMCW relations:
    range resolution   c / (2 B)      B = slope x ADC sampling time (N / fs)
    maximum range      IF x c / (2 slope)   IF = the usable IF band: 0.9 fs for complex
                                      samples, 0.45 fs for real ones, and not above the
                                      device's IF bandwidth
    maximum velocity   lambda / (4 T)       T = the time between two chirps of the same
                                      transmitter (all transmitters' chirps with TDM-MIMO)
    velocity resolution  lambda / (2 L T)   L = loops per frame
From range resolution and maximum range follows the number of samples, N >= R_max /
(IF fraction x resolution); from the maximum velocity the chirp time, and so the ADC
sampling time and the sample rate; the slope then gives the resolution.

Device limits: from the TI data sheets where known, otherwise the highest values the
lab's recordings were made with (IWR6843: 200 MHz/us, 12 Msps) - a data sheet may allow
more. They are in DEVICES.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

C0 = 299_792_458.0

NICE_SAMPLES = (32, 48, 64, 80, 96, 128, 150, 160, 192, 200, 256, 320, 384, 400, 512, 640,
                768, 1024)
ADC_BUFFER_BYTES = 16384        # one ping of the ADC buffer: N x RX x bytes per sample
MAX_LOOPS = 255
# Between frames the radar needs at least this (TI: PERIOD >= chirps + InterFrameBlankTime,
# 300 us; FrameConfig answers 80 'Frame ON time > periodicity' below it, 250 us refused and 300 us
# accepted on the IWR6843AOPEVM, 2026-10-06).
FRAME_BLANK_MS = 0.3
PROVEN_RATE_MB_S = 30.6         # the highest average rate the lab recorded without loss
ETHERNET_MB_S = 100.0           # what Gigabit Ethernet from the DCA1000 can carry at best
RAMP_EXCESS_US = 1.0            # ramp beyond the end of sampling
COMFORT_SLOPE = 100.0           # MHz/us: a slope this low when the chirp time allows it
MIN_FS_KSPS = 2000
SLOPE_LSB = {"60G": 0.036209821701, "77G": 0.048279762268}      # MHz/us per code


@dataclass(frozen=True)
class Device:
    name: str
    chip: str                    # SelectChipVersion
    variant: str                 # deviceVariantSelection
    band: str                    # frequencyBandSelection: "60G" or "77G"
    f_lo: float                  # GHz
    f_hi: float
    max_bw_ghz: float
    n_tx: int
    n_rx: int
    max_fs_ksps: int
    default_fs_ksps: int
    max_if_mhz: float | None     # IF bandwidth of the receiver chain
    max_slope: float             # MHz/us
    complex_samples: bool
    lua: str                     # "xwr68" or "xwr25": which argument lists
    com_port: int
    studio_dir: str
    lanes: int
    lane_mbps: float
    default_start_ghz: float
    note: str = ""


DEVICES = {
    "IWR6843AOP": Device(
        "IWR6843AOP", "IWR6843", "IWR6843", "60G", 60.0, 64.0, 4.0, 3, 4, 12500, 10000, 10.0,
        200.0, True, "xwr68", 12, r"C:\ti\mmwave_studio_02_01_01_00", 2, 600.0, 60.25,
        "slope up to 200 MHz/us and 12 Msps are what the lab recorded with"),
    "IWR6843 (ISK, ODS)": Device(
        "IWR6843 (ISK, ODS)", "IWR6843", "IWR6843", "60G", 60.0, 64.0, 4.0, 3, 4, 12500, 10000,
        10.0, 200.0, True, "xwr68", 12, r"C:\ti\mmwave_studio_02_01_01_00", 2, 600.0, 60.25,
        "as the AOP; a different antenna layout, the same chip"),
    "AWR2544": Device(
        "AWR2544", "AWR2544", "XWR2544", "77G", 76.0, 81.0, 5.0, 4, 4, 20000, 10000, None, 100.0,
        False, "xwr25", 50, r"C:\ti\mmwave_studio_03_01_04_04", 1, 600.0, 77.0,
        "real samples as in the lab's script; slope up to 100 MHz/us as recorded - the data "
        "sheet may allow more"),
}


@dataclass
class Goals:
    max_range_m: float = 5.0
    range_res_cm: float = 5.0
    max_vel_ms: float = 2.0
    vel_res_ms: float | None = 0.1          # None: loops_per_frame decides
    loops_per_frame: int | None = None      # chirps of each transmitter per frame
    fps: float = 20.0
    tdm_all_tx: bool = False                # True: every transmitter in turn (TDM-MIMO)
    single_tx: int = 1                      # which transmitter otherwise (1-based)
    back_to_back: bool = False              # frames without a gap (continuous chirping)
    duration_s: float = 60.0
    rx_gain_db: int = 30
    start_ghz: float | None = None          # None: the device default
    idle_us: float = 10.0
    adc_start_us: float = 6.0
    tx_start_us: float = 0.0
    fs_ksps: int | None = None              # None: the device default, raised if needed


PURPOSES = {
    "Vital signs (seated person)": dict(
        max_range_m=3.0, range_res_cm=5.0, max_vel_ms=1.0, vel_res_ms=None, loops_per_frame=4,
        fps=100.0, tdm_all_tx=False, back_to_back=False, duration_s=120.0),
    "People and angles (AoA, tracking)": dict(
        max_range_m=8.0, range_res_cm=5.0, max_vel_ms=2.5, vel_res_ms=0.12, loops_per_frame=None,
        fps=20.0, tdm_all_tx=True, back_to_back=False, duration_s=60.0),
    "Fast limb movements (micro-Doppler)": dict(
        max_range_m=4.0, range_res_cm=5.0, max_vel_ms=15.0, vel_res_ms=None,
        loops_per_frame=255, fps=50.0, tdm_all_tx=False, back_to_back=True, duration_s=30.0,
        idle_us=14.0),
    "Heart rate of a moving person": dict(
        max_range_m=4.0, range_res_cm=5.0, max_vel_ms=2.0, vel_res_ms=None, loops_per_frame=None,
        fps=100.0, tdm_all_tx=True, back_to_back=True, duration_s=120.0, idle_us=14.0),
    "Tremor (hand)": dict(
        max_range_m=2.0, range_res_cm=5.0, max_vel_ms=1.0, vel_res_ms=None, loops_per_frame=4,
        fps=200.0, tdm_all_tx=False, back_to_back=False, duration_s=60.0),
}


@dataclass
class Check:
    level: str                   # "ok", "warning", "error"
    what: str
    message: str


@dataclass
class Design:
    device: Device
    goals: Goals
    start_ghz: float
    idle_us: float
    adc_start_us: float
    ramp_end_us: float
    slope_mhz_us: float
    tx_start_us: float
    n_samples: int
    fs_ksps: int
    rx_gain_db: int
    tx_order: list               # 1-based transmitter of each chirp of the pattern
    loops: int
    frames: int
    period_ms: float
    perf: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)

    @property
    def worst(self) -> str:
        order = ("ok", "warning", "error")
        return max((c.level for c in self.checks), key=order.index, default="ok")


# --- helpers ---------------------------------------------------------------------------------

def _q(x: float, step: float) -> float:
    """x rounded to the grid of step."""
    return round(round(x / step) * step, 10)


def _ceil_to(x: float, step: float) -> float:
    return round(math.ceil(x / step - 1e-9) * step, 10)


def quantise_slope(slope: float, band: str) -> float:
    lsb = SLOPE_LSB.get(band, SLOPE_LSB["77G"])
    return round(max(1, round(slope / lsb)) * lsb, 3)


def _nice_samples(n: float) -> int:
    for v in NICE_SAMPLES:
        if v >= n:
            return v
    return NICE_SAMPLES[-1]


def bytes_per_sample(device: Device) -> int:
    return 4 if device.complex_samples else 2           # 16-bit I and Q, or I alone


def if_fraction(device: Device) -> float:
    return 0.9 if device.complex_samples else 0.45


def usable_if_hz(device: Device, fs_ksps: float) -> float:
    f = if_fraction(device) * fs_ksps * 1e3
    if device.max_if_mhz:
        f = min(f, device.max_if_mhz * 1e6)
    return f


# --- what a configuration gives ----------------------------------------------------------------

def performance(device: Device, start_ghz: float, idle_us: float, adc_start_us: float,
                ramp_end_us: float, slope_mhz_us: float, n_samples: int, fs_ksps: float,
                n_tx_pattern: int, loops: int, period_ms: float) -> dict:
    t_adc_us = n_samples / (fs_ksps / 1e3)
    bw_eff = slope_mhz_us * 1e12 * t_adc_us * 1e-6
    f_mid = (start_ghz * 1e9 + slope_mhz_us * 1e12 * (adc_start_us + t_adc_us / 2) * 1e-6)
    lam = C0 / f_mid
    tc_us = idle_us + ramp_end_us
    t_same = n_tx_pattern * tc_us * 1e-6
    active_ms = loops * n_tx_pattern * tc_us / 1e3
    chirp_bytes = n_samples * device.n_rx * bytes_per_sample(device)
    frame_bytes = chirp_bytes * loops * n_tx_pattern
    fps = 1e3 / period_ms if period_ms > 0 else float("nan")
    return {
        "adc_time_us": t_adc_us,
        "chirp_time_us": tc_us,
        "sweep_ghz": slope_mhz_us * ramp_end_us / 1e3,
        "stop_ghz": start_ghz + slope_mhz_us * ramp_end_us / 1e3,
        "bandwidth_eff_ghz": bw_eff / 1e9,
        "range_res_cm": C0 / (2 * bw_eff) * 100 if bw_eff > 0 else float("nan"),
        "max_range_m": usable_if_hz(device, fs_ksps) * C0 / (2 * slope_mhz_us * 1e12),
        "wavelength_mm": lam * 1e3,
        "max_vel_ms": lam / (4 * t_same),
        "vel_res_ms": lam / (2 * loops * t_same) if loops else float("nan"),
        "chirps_per_frame": loops * n_tx_pattern,
        "active_ms": active_ms,
        "fps": fps,
        "duty_pct": 100.0 * active_ms / period_ms if period_ms > 0 else float("nan"),
        "rate_mb_s": frame_bytes * fps / 1e6,
        "burst_mb_s": chirp_bytes / (tc_us * 1e-6) / 1e6,
        "chirp_bytes": chirp_bytes,
        "lvds_mb_s": device.lanes * device.lane_mbps / 8.0,
    }


# --- the design ----------------------------------------------------------------------------------

def design(device: Device, goals: Goals) -> Design:
    """The configuration for the goals, with its checks (see the module notes)."""
    g = goals
    checks: list[Check] = []
    n_tx_pattern = device.n_tx if g.tdm_all_tx else 1
    tx_order = list(range(1, device.n_tx + 1)) if g.tdm_all_tx else [int(g.single_tx)]
    start = float(g.start_ghz if g.start_ghz is not None else device.default_start_ghz)
    idle, adc_start = _q(g.idle_us, 0.01), _q(g.adc_start_us, 0.01)
    lam = C0 / ((start + 1.0) * 1e9)                     # refined in performance()
    res_m = g.range_res_cm / 100.0
    frac = if_fraction(device)

    # samples: range bins enough to reach the maximum range at the resolution
    n = _nice_samples(max(32.0, g.max_range_m / (frac * res_m)))
    # chirp time budget from the maximum velocity
    tc_max_us = lam / (4.0 * g.max_vel_ms * n_tx_pattern) * 1e6
    adc_budget_us = tc_max_us - idle - adc_start - RAMP_EXCESS_US
    if adc_budget_us <= 1.0:
        checks.append(Check("error", "chirp time",
                            f"a maximum velocity of {g.max_vel_ms:g} m/s with {n_tx_pattern} TX "
                            f"allows {tc_max_us:.1f} us per chirp, less than the idle and ADC "
                            f"start time alone - lower the maximum velocity, use one TX, or "
                            f"shorten the idle time"))
        adc_budget_us = max(1.0, adc_budget_us)
    # the ADC window: long enough for the resolution at a comfortable slope (and at the
    # slope limit at least), for the IF band, and short enough for the chirp time
    bw = C0 / (2 * res_m)
    t_min_us = bw / (device.max_slope * 1e12) * 1e6
    if device.max_if_mhz:
        t_min_us = max(t_min_us, 2 * bw * g.max_range_m / (C0 * device.max_if_mhz * 1e6) * 1e6)
    fs0 = int(g.fs_ksps or device.default_fs_ksps)
    if g.fs_ksps:
        t_adc_us = n / (fs0 / 1e3)
    else:
        t_adc_us = max(n / (fs0 / 1e3), bw / (min(COMFORT_SLOPE, device.max_slope) * 1e12) * 1e6,
                       t_min_us)
        t_adc_us = min(t_adc_us, max(adc_budget_us, t_min_us))
    fs = int(math.floor(n / t_adc_us * 10) * 100)          # whole 100 ksps
    if fs < MIN_FS_KSPS:                     # a long window: more samples, not a slower ADC
        fs = MIN_FS_KSPS
        n = _nice_samples(fs / 1e3 * t_adc_us)
    if fs > device.max_fs_ksps:
        checks.append(Check("error", "sample rate",
                            f"{n} samples fit in the {adc_budget_us:.1f} us the chirp may take only "
                            f"at {fs} ksps, above the device's {device.max_fs_ksps} ksps - coarser "
                            f"range resolution or shorter maximum range (fewer samples), a lower "
                            f"maximum velocity, or fewer TX"))
        fs = device.max_fs_ksps
    t_adc_us = n / (fs / 1e3)
    if t_adc_us > adc_budget_us + 1e-6:
        checks.append(Check("error", "chirp time",
                            f"the {res_m * 100:g} cm resolution needs a sampling window of at least "
                            f"{t_min_us:.1f} us, the maximum velocity leaves {adc_budget_us:.1f} us - "
                            f"coarser resolution, a lower maximum velocity or fewer TX"))
    # slope: the resolution asked for (the lowest slope that gives it - the most range)
    slope = quantise_slope(C0 / (2 * res_m) / (t_adc_us * 1e-6) / 1e12, device.band)
    if slope > device.max_slope:
        checks.append(Check("error", "slope",
                            f"{res_m * 100:g} cm needs {slope:.1f} MHz/us in a {t_adc_us:.1f} us "
                            f"sampling window, above the {device.max_slope:g} MHz/us allowed - "
                            f"coarser range resolution, a lower maximum velocity (a longer "
                            f"chirp), or fewer TX"))
        slope = quantise_slope(device.max_slope, device.band)
    ramp_end = _ceil_to(adc_start + t_adc_us + RAMP_EXCESS_US, 0.1)
    tc = idle + ramp_end

    # loops and frame period
    period_goal_ms = 1e3 / g.fps
    if g.back_to_back:
        loops = int(g.loops_per_frame or MAX_LOOPS)
        loops = max(1, min(loops, MAX_LOOPS,
                           int((period_goal_ms - FRAME_BLANK_MS) * 1e3 // (n_tx_pattern * tc))))
        period = _ceil_to(loops * n_tx_pattern * tc / 1e3 + FRAME_BLANK_MS, 0.01)
    else:
        if g.loops_per_frame:
            loops = int(g.loops_per_frame)
        else:
            vr = g.vel_res_ms or 0.1
            loops = int(math.ceil(lam / (2 * vr * n_tx_pattern * tc * 1e-6)))
        loops = max(1, min(loops, MAX_LOOPS))
        period = _q(period_goal_ms, 0.01)
        fit = int((period - FRAME_BLANK_MS) * 1e3 // (n_tx_pattern * tc))
        if loops > fit:
            checks.append(Check("warning", "frame",
                                f"{loops} loops do not fit in a {period:g} ms frame; {fit} do "
                                f"(the velocity resolution is coarser than asked) - or lower the "
                                f"frame rate"))
            loops = max(1, fit)
    frames = max(1, int(math.ceil(g.duration_s * 1e3 / period)))

    d = Design(device=device, goals=g, start_ghz=round(start, 3), idle_us=idle,
               adc_start_us=adc_start, ramp_end_us=ramp_end, slope_mhz_us=slope,
               tx_start_us=_q(g.tx_start_us, 0.01), n_samples=n, fs_ksps=fs,
               rx_gain_db=int(g.rx_gain_db), tx_order=tx_order, loops=loops, frames=frames,
               period_ms=period)
    d.perf = performance(device, d.start_ghz, idle, adc_start, ramp_end, slope, n, fs,
                         len(tx_order), loops, period)
    d.checks = checks + _checks(d)
    return d


def _checks(d: Design) -> list:
    dev, g, p = d.device, d.goals, d.perf
    out: list[Check] = []

    def add(level, what, msg):
        out.append(Check(level, what, msg))

    # the goals (a configuration read from a recording or typed in has none)
    if g is None:
        add("ok", "range", f"resolution {p['range_res_cm']:.2f} cm, maximum range "
                           f"{p['max_range_m']:.2f} m")
        add("ok", "velocity", f"maximum +-{p['max_vel_ms']:.2f} m/s, resolution "
                              f"{p['vel_res_ms']:.3f} m/s")
        add("ok", "frame rate", f"{p['fps']:.1f} fps")
    elif p["range_res_cm"] <= g.range_res_cm * 1.02:
        add("ok", "range resolution", f"{p['range_res_cm']:.2f} cm (asked {g.range_res_cm:g})")
    else:
        add("warning", "range resolution", f"{p['range_res_cm']:.2f} cm, asked {g.range_res_cm:g}")
    if g is None:
        pass
    elif p["max_range_m"] >= g.max_range_m * 0.98:
        add("ok", "maximum range", f"{p['max_range_m']:.2f} m (asked {g.max_range_m:g})")
    else:
        add("error", "maximum range",
            f"only {p['max_range_m']:.2f} m (asked {g.max_range_m:g}): the IF band ends there - "
            f"a coarser range resolution or a higher sample rate")
    if g is None:
        pass
    elif p["max_vel_ms"] >= g.max_vel_ms * 0.98:
        add("ok", "maximum velocity", f"+-{p['max_vel_ms']:.2f} m/s (asked {g.max_vel_ms:g})")
    else:
        add("warning", "maximum velocity",
            f"+-{p['max_vel_ms']:.2f} m/s, asked {g.max_vel_ms:g}: faster motion folds over")
    if g is None:
        pass
    elif g.vel_res_ms and not g.loops_per_frame and not g.back_to_back:
        lvl = "ok" if p["vel_res_ms"] <= g.vel_res_ms * 1.02 else "warning"
        add(lvl, "velocity resolution", f"{p['vel_res_ms']:.3f} m/s (asked {g.vel_res_ms:g})")
    else:
        add("ok", "velocity resolution", f"{p['vel_res_ms']:.3f} m/s")
    if g is None:
        pass
    elif abs(p["fps"] - g.fps) <= 0.05 * g.fps:
        add("ok", "frame rate", f"{p['fps']:.1f} fps (asked {g.fps:g})")
    elif g.back_to_back:
        add("ok", "frame rate", f"{p['fps']:.1f} fps: frames back to back last as long as their "
                                f"{d.loops} loops (at most {MAX_LOOPS})")
    else:
        add("warning", "frame rate", f"{p['fps']:.1f} fps, asked {g.fps:g}")

    # the device
    if d.slope_mhz_us > dev.max_slope + 1e-6:
        add("error", "slope", f"{d.slope_mhz_us:.3f} MHz/us above {dev.max_slope:g}")
    if p["sweep_ghz"] > dev.max_bw_ghz or p["stop_ghz"] > dev.f_hi or d.start_ghz < dev.f_lo:
        add("error", "band", f"the ramp sweeps {d.start_ghz:g}-{p['stop_ghz']:.3f} GHz "
                             f"({p['sweep_ghz']:.2f} GHz); the device covers {dev.f_lo:g}-"
                             f"{dev.f_hi:g} GHz, {dev.max_bw_ghz:g} GHz at most - lower the start "
                             f"frequency or the resolution")
    else:
        add("ok", "band", f"{d.start_ghz:g}-{p['stop_ghz']:.3f} GHz within {dev.f_lo:g}-{dev.f_hi:g}")
    if d.fs_ksps > dev.max_fs_ksps:
        add("error", "sample rate", f"{d.fs_ksps} ksps above {dev.max_fs_ksps}")
    beat = 2 * d.slope_mhz_us * 1e12 * (g.max_range_m if g else 0.0) / C0
    if beat > usable_if_hz(dev, d.fs_ksps) * 1.001:
        add("error", "IF", f"a target at {g.max_range_m:g} m beats at {beat / 1e6:.2f} MHz, beyond "
                           f"the usable {usable_if_hz(dev, d.fs_ksps) / 1e6:.2f} MHz")
    if p["chirp_bytes"] > ADC_BUFFER_BYTES:
        add("error", "ADC buffer", f"{p['chirp_bytes']} bytes a chirp, the buffer holds "
                                   f"{ADC_BUFFER_BYTES} - fewer samples")
    if d.loops > MAX_LOOPS:
        add("error", "loops", f"{d.loops} loops, at most {MAX_LOOPS}")

    # frame timing
    if p["active_ms"] + FRAME_BLANK_MS > d.period_ms + 1e-9:
        add("error", "frame", f"the chirps take {p['active_ms']:.2f} ms of a {d.period_ms:g} ms frame; "
                              f"the radar needs {FRAME_BLANK_MS * 1e3:.0f} us between frames (a period "
                              f"of at least {p['active_ms'] + FRAME_BLANK_MS:.2f} ms, or fewer loops)")
    elif g is not None and g.back_to_back:
        add("ok", "frame", f"frames back to back: chirps {p['active_ms']:.2f} ms of {d.period_ms:g} ms, "
                           f"{(d.period_ms - p['active_ms']) * 1e3:.0f} us between frames")
    else:
        add("ok", "frame", f"chirps {p['active_ms']:.2f} ms of {d.period_ms:g} ms "
                           f"({p['duty_pct']:.0f} %)")

    # data
    rate, burst = p["rate_mb_s"], p["burst_mb_s"]
    if rate > ETHERNET_MB_S:
        add("error", "data rate", f"{rate:.1f} MB/s is more than the DCA1000's Gigabit Ethernet "
                                  f"carries - fewer samples, chirps or RX, or a lower frame rate")
    elif rate > PROVEN_RATE_MB_S:
        add("warning", "data rate",
            f"{rate:.1f} MB/s, more than the {PROVEN_RATE_MB_S:g} MB/s recorded here without loss - "
            f"watch the capture health report for lost packets (Fastcheck)")
    else:
        secs = d.frames * d.period_ms / 1e3
        add("ok", "data rate", f"{rate:.1f} MB/s ({rate * secs / 1e3:.2f} GB for {secs:.0f} s)")
    if burst > p["lvds_mb_s"]:
        add("error", "LVDS", f"{burst:.0f} MB/s while chirping, above the {p['lvds_mb_s']:.0f} MB/s "
                             f"of {dev.lanes} LVDS lane(s) - a longer idle time or fewer samples")
    elif burst > 0.8 * p["lvds_mb_s"]:
        add("warning", "LVDS", f"{burst:.0f} MB/s while chirping, near the {p['lvds_mb_s']:.0f} MB/s "
                               f"of the LVDS lanes")

    # what DARTscope needs for its analyses
    hr_hi = 150.0 / 60.0
    if p["fps"] < 2.2 * hr_hi:
        add("warning", "vital signs", f"{p['fps']:.1f} fps is too few for the heart band (the "
                                      f"Vitals view needs about 20 fps or more)")
    if g is not None and g.tdm_all_tx and len(d.tx_order) < 2:
        add("warning", "angles", "one TX: angles only from the RX array")
    return out


# --- mmWave Studio ---------------------------------------------------------------------------------

def _f(x, digits=3) -> str:
    """A number as Lua takes it: a decimal point, no trailing zeros."""
    s = f"{x:.{digits}f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def summary_lines(d: Design) -> list:
    p = d.perf
    return [
        f"{d.device.name}, {'TDM-MIMO ' + str(len(d.tx_order)) + ' TX' if len(d.tx_order) > 1 else 'TX' + str(d.tx_order[0])}"
        f" x {d.device.n_rx} RX",
        f"range resolution {p['range_res_cm']:.2f} cm, maximum range {p['max_range_m']:.2f} m",
        f"maximum velocity +-{p['max_vel_ms']:.2f} m/s, velocity resolution {p['vel_res_ms']:.3f} m/s",
        f"{p['fps']:.1f} frames/s, {p['chirps_per_frame']} chirps per frame, "
        f"{d.frames} frames = {d.frames * d.period_ms / 1e3:.1f} s",
        f"data rate {p['rate_mb_s']:.1f} MB/s, file {p['rate_mb_s'] * d.frames * d.period_ms / 1e6:.2f} GB",
    ]


def lua_script(d: Design, com_port: int | None = None, studio_dir: str | None = None,
               out_bin: str | None = None, purpose: str = "") -> str:
    """The Lua script for mmWave Studio's Lua shell (Lua does not take decimal commas)."""
    dev = d.device
    com = int(com_port if com_port is not None else dev.com_port)
    studio = (studio_dir or dev.studio_dir).rstrip("\\/")
    out = out_bin or (studio + r"\mmWaveStudio\PostProc\adc_data.bin")
    lua_path = lambda s: s.replace("\\", "\\\\")            # noqa: E731
    wait_ms = int(math.ceil(d.frames * d.period_ms)) + 2000
    ntx = dev.n_tx
    lines = [f"-- DARTscope chirp helper{': ' + purpose if purpose else ''}"]
    lines += [f"-- {s}" for s in summary_lines(d)]
    lines += [
        "-- After the recording save the mmWave Studio log next to the .bin as",
        "-- <name>_LogFile.txt: DARTscope reads the configuration from it.",
        "",
        "ar1.FullReset()",
        "ar1.SOPControl(2)",
        f"ar1.Connect({com}, 921600, 1000)",
        "ar1.Calling_IsConnected()",
        "",
        f'ar1.frequencyBandSelection("{dev.band}")',
        f'ar1.SelectChipVersion("{dev.chip}")',
        f'ar1.deviceVariantSelection("{dev.variant}")',
        "",
    ]
    tx_on = [1 if (t + 1) in d.tx_order else 0 for t in range(ntx)]
    if dev.lua == "xwr25":
        lines += [
            f'BSS_FW = "{lua_path(studio)}\\\\rf_eval_firmware\\\\radarss\\\\xwr25xx_radarss_rprc.bin"',
            f'MSS_FW = "{lua_path(studio)}\\\\rf_eval_firmware\\\\masterss\\\\awr2xxx_mmwave_full_mss_rprc.bin"',
            "", "ar1.DownloadBSSFw(BSS_FW)", "ar1.DownloadMSSFw(MSS_FW)",
            "ar1.PowerOn(0, 1000, 0, 0)", "ar1.RfEnable()", "",
            f"ar1.ChanNAdcConfig({', '.join(map(str, tx_on))}, 1, 1, 1, 1, 2, "
            f"{1 if dev.complex_samples else 0}, 0)",
            "ar1.LPModConfig(0, 0)", "ar1.RfInit()", "RSTD.Sleep(1000)", "",
            "ar1.DataPathConfig(513, 1216644097, 0)",
            "ar1.LVDSLaneConfig(0, 1, 0, 0, 0, 1, 0, 0)", "",
            f"ar1.ProfileConfig(0, {_f(d.start_ghz)}, {_f(d.idle_us, 2)}, {_f(d.adc_start_us, 2)}, "
            f"{_f(d.ramp_end_us, 2)}, 0, 0, 0, 0, 0, 0, 0, 0, {_f(d.slope_mhz_us)}, "
            f"{_f(d.tx_start_us, 2)}, {d.n_samples}, {d.fs_ksps}, 2216755200, 0, {d.rx_gain_db}, "
            f"0, 0, 0)",
        ]
        for i, tx in enumerate(d.tx_order):
            en = ", ".join("1" if t + 1 == tx else "0" for t in range(ntx))
            lines.append(f"ar1.ChirpConfig({i}, {i}, 0, 0, 0, 0, 0, {en})")
        frame = (f"ar1.FrameConfig(0, {len(d.tx_order) - 1}, {d.frames}, {d.loops}, "
                 f"{_f(d.period_ms, 2)}, 0, 1)")
    else:
        lines += [
            f'BSS_FW = "{lua_path(studio)}\\\\rf_eval_firmware\\\\radarss\\\\xwr68xx_radarss.bin"',
            f'MSS_FW = "{lua_path(studio)}\\\\rf_eval_firmware\\\\masterss\\\\xwr68xx_masterss.bin"',
            "", "ar1.DownloadBSSFw(BSS_FW)", "ar1.DownloadMSSFw(MSS_FW)",
            "ar1.PowerOn(1, 1000, 0, 0)", "ar1.RfEnable()", "",
            f"ar1.ChanNAdcConfig({', '.join(map(str, tx_on))}, 1, 1, 1, 1, 2, "
            f"{1 if dev.complex_samples else 0}, 0)",
            "ar1.LPModConfig(0, 0)", "ar1.RfInit()", "RSTD.Sleep(1000)",
            "ar1.DataPathConfig(1, 1, 0)", "ar1.LvdsClkConfig(1, 1)",
            "ar1.LVDSLaneConfig(0, 1, 1, 0, 0, 1, 0, 0)", "",
            f"ar1.ProfileConfig(0, {_f(d.start_ghz)}, {_f(d.idle_us, 2)}, {_f(d.adc_start_us, 2)}, "
            f"{_f(d.ramp_end_us, 2)}, 0, 0, 0, 0, 0, 0, {_f(d.slope_mhz_us)}, {_f(d.tx_start_us, 2)}, "
            f"{d.n_samples}, {d.fs_ksps}, 0, 131072, {d.rx_gain_db})",
        ]
        for i, tx in enumerate(d.tx_order):
            en = ", ".join("1" if t + 1 == tx else "0" for t in range(ntx))
            lines.append(f"ar1.ChirpConfig({i}, {i}, 0, 0, 0, 0, 0, {en})")
        frame = (f"ar1.FrameConfig(0, {len(d.tx_order) - 1}, {d.frames}, {d.loops}, "
                 f"{_f(d.period_ms, 2)}, 0, 0, 1)")
    lines += [
        "",
        'ar1.SelectCaptureDevice("DCA1000")',
        'ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098)',
        "ar1.CaptureCardConfig_Mode(1, 2, 1, 2, 3, 30)",
        "ar1.CaptureCardConfig_PacketDelay(25)",
    ]
    if dev.lua == "xwr25":
        lines.append("ar1.DisableTestSource(0)")
    lines += [
        "",
        f"-- chirps 0..{len(d.tx_order) - 1}, {d.frames} frames, {d.loops} loops, "
        f"{_f(d.period_ms, 2)} ms a frame",
        frame,
        "",
        f'ar1.CaptureCardConfig_StartRecord("{lua_path(out)}", 1)',
        "RSTD.Sleep(1000)",
        "ar1.StartFrame()",
        f"RSTD.Sleep({wait_ms}) -- the whole recording and 2 s more",
        "ar1.StopFrame()",
        "ar1.CaptureCardConfig_StopRecord()",
    ]
    return "\n".join(lines) + "\n"


def goals_for(purpose: str, base: Goals | None = None) -> Goals:
    """The goals of a purpose (PURPOSES), the rest from base."""
    g = base or Goals()
    return replace(g, **PURPOSES.get(purpose, {}))


# --- a configuration as it is ------------------------------------------------------------------

def evaluate(device: Device, goals: Goals | None, *, start_ghz: float, idle_us: float,
             adc_start_us: float, ramp_end_us: float, slope_mhz_us: float, tx_start_us: float,
             n_samples: int, fs_ksps: int, rx_gain_db: int, tx_order, loops: int, frames: int,
             period_ms: float) -> Design:
    """What a configuration typed or read from a recording gives, with its checks."""
    d = Design(device=device, goals=goals, start_ghz=float(start_ghz), idle_us=float(idle_us),
               adc_start_us=float(adc_start_us), ramp_end_us=float(ramp_end_us),
               slope_mhz_us=float(slope_mhz_us), tx_start_us=float(tx_start_us),
               n_samples=int(n_samples), fs_ksps=int(fs_ksps), rx_gain_db=int(rx_gain_db),
               tx_order=[int(t) for t in tx_order], loops=int(loops), frames=int(frames),
               period_ms=float(period_ms))
    d.perf = performance(device, d.start_ghz, d.idle_us, d.adc_start_us, d.ramp_end_us,
                         d.slope_mhz_us, d.n_samples, d.fs_ksps, len(d.tx_order), d.loops,
                         d.period_ms)
    checks = []
    if d.adc_start_us + d.perf["adc_time_us"] > d.ramp_end_us:
        checks.append(Check("error", "ADC window",
                            f"sampling ends at {d.adc_start_us + d.perf['adc_time_us']:.2f} us, "
                            f"after the ramp ({d.ramp_end_us:g} us): lengthen the ramp end time "
                            f"or take fewer samples"))
    bad_tx = [t for t in d.tx_order if not 1 <= t <= device.n_tx]
    if bad_tx or not d.tx_order:
        checks.append(Check("error", "chirps", f"the {device.name} has TX1-TX{device.n_tx}"))
    d.checks = checks + _checks(d)
    return d


def device_for(params: dict) -> Device:
    """The device of a decoded recording, by its chip; the IWR6843AOP when unknown."""
    hint = str((params or {}).get("chip_hint") or "").upper()
    if "2544" in hint:
        return DEVICES["AWR2544"]
    return DEVICES["IWR6843AOP"]


def from_params(params: dict) -> dict:
    """The configuration of a decoded recording, as evaluate() takes it."""
    p = params or {}
    fc = p.get("frameConfig") or {}
    masks = p.get("tx_order_masks") or [1]
    order = []
    for m in masks:
        m = int(m)
        bits = [b + 1 for b in range(4) if m >> b & 1]
        order.append(bits[0] if bits else 1)
    return dict(
        start_ghz=round(float(p.get("startFreqGHz") or 60.25), 3),
        idle_us=float(p.get("idleTime_us") or 0.0),
        adc_start_us=float(p.get("adcStartTime_us") or 0.0),
        ramp_end_us=float(p.get("rampEndTime_us") or 0.0),
        slope_mhz_us=round(float(p.get("freqSlopeMHz_us") or 0.0), 3),
        tx_start_us=float(p.get("txStartTime_us") or 0.0),
        n_samples=int(p.get("numAdcSamples") or 0),
        fs_ksps=int(round(float(p.get("digOutSampleRate_MHz") or 0.0) * 1e3)),
        rx_gain_db=int(p.get("rxGain") or 30) & 0x3F,        # the gain in dB, flags above
        tx_order=order,
        loops=int(fc.get("numLoops") or 1),
        frames=int(fc.get("numFrames") or 1),
        period_ms=float(p.get("framePeriodicity_ms_selected") or fc.get("framePeriodicity_ms") or 0.0),
    )
