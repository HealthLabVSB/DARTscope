"""
acquisition/params.py

The parameters a person sets for a measurement, in physical units - what the Measure tab
shows - and the radar configuration they make.

    MeasureParams.from_config(cfg)    a loaded Lua script, log or sidecar -> the form
    params.to_config(base)            the form -> a RadarConfig: everything the form does
                                      not show (HPF codes and flags, LVDS, DCA1000, COM
                                      port, firmware) is kept from `base`
    derived(cfg)                      what a configuration gives: range resolution and
                                      maximum range, maximum velocity and its resolution,
                                      frame rate, duty cycle, data rate, file size
    checks(cfg)                       every problem, worst first: the configuration's own
                                      checks, the device limits of the chirp Helper
                                      (chirp_design) and what the loader would misread

The TX pattern is the transmitters of each chirp of a loop, in order: "1, 2, 3" is TDM-MIMO
over three transmitters, "1" one transmitter, "2, 1+3" what config/Dataset_motion.lua does
(TX2, then TX1 and TX3 together). Stored as TI's masks (bit 0 = TX1).

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import asdict, dataclass, field

from .config import Chirp, ConfigError, Frame, Profile, RadarConfig, mask_bits

RAMP_EXCESS_US = 1.0            # ramp beyond the end of sampling (chirp_design's rule)
MIN_RANGE_M = 0.3               # nearer than this is antenna coupling
ONE_FILE_BYTES = 1024 * 1024 * 1024     # where mmWave Studio's recorder starts a second file


def fit_duration(duration_s: float, cfg: RadarConfig,
                 limit_bytes: int = ONE_FILE_BYTES) -> tuple[float, bool]:
    """
    (seconds, shortened): a preset's own recording length, cut to whole seconds only when
    the recording would not fit in one file. Vital signs keep their 2 minutes (61 MB) -
    breathing is analysed in 30 s windows - while 'Heart rate of a moving person' (30 MB/s)
    becomes about 35 s instead of 3.6 GB.
    """
    rate = cfg.bytes_per_frame / (cfg.frame.period_ms * 1e-3) if cfg.frame.periodicity else 0.0
    if not duration_s or duration_s <= 0 or rate <= 0 or duration_s * rate <= limit_bytes:
        return duration_s, False
    return float(max(1, math.floor(limit_bytes / rate))), True


def parse_tx(text: str) -> list[int]:
    """'1, 2, 3' -> [1, 2, 4]; '2, 1+3' -> [2, 5]; 'TX1,TX3' -> [1, 4]."""
    out = []
    for item in re.split(r"[,;\s]+", str(text).upper().replace("TX", "")):
        if not item:
            continue
        mask = 0
        for part in item.split("+"):
            if not part.isdigit() or not 1 <= int(part) <= 3:
                raise ConfigError(f"TX pattern {text!r}: {part!r} is not TX1-TX3")
            mask |= 1 << (int(part) - 1)
        out.append(mask)
    if not out:
        raise ConfigError("TX pattern is empty: e.g. '1' or '1, 2, 3'")
    if len(out) > 32:
        raise ConfigError("TX pattern: at most 32 chirps a loop")
    return out


def format_tx(masks) -> str:
    return ", ".join("+".join(str(b) for b in mask_bits(int(m))) for m in masks)


def parse_rx(text: str) -> list[int]:
    """'1,2,3,4', '1234' or 'RX1 RX2' -> [1, 2, 3, 4]."""
    s = str(text).upper().replace("RX", "")
    digits = re.findall(r"\d", s)
    rx = sorted({int(d) for d in digits})
    if not rx or any(r < 1 or r > 4 for r in rx):
        raise ConfigError(f"RX {text!r}: receivers are RX1-RX4, at least one")
    return rx


def format_rx(rx) -> str:
    return ",".join(str(r) for r in rx)


@dataclass
class MeasureParams:
    start_ghz: float = 60.25
    slope_mhz_us: float = 100.0
    samples: int = 256
    fs_ksps: int = 10000
    idle_us: float = 10.0
    adc_start_us: float = 6.0
    ramp_end_us: float = 0.0            # 0: ADC start + sampling time + 1 us
    tx_start_us: float = 0.0
    rx_gain_db: int = 30
    tx: list = field(default_factory=lambda: [1])          # TI masks per chirp of a loop
    rx: list = field(default_factory=lambda: [1, 2, 3, 4])
    loops: int = 64
    period_ms: float = 50.0
    duration_s: float = 10.0            # 0: `frames` decides (0 frames = until stopped)
    frames: int = 200

    @property
    def ramp_end_effective(self) -> float:
        if self.ramp_end_us and self.ramp_end_us > 0:
            return float(self.ramp_end_us)
        t = self.adc_start_us + self.samples / (self.fs_ksps / 1e3) + RAMP_EXCESS_US
        return math.ceil(t * 100 - 1e-6) / 100.0

    @property
    def frame_count(self) -> int:
        if self.duration_s and self.duration_s > 0:
            return max(1, int(round(self.duration_s * 1e3 / self.period_ms)))
        return int(self.frames)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MeasureParams":
        names = cls.__dataclass_fields__
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    @classmethod
    def from_config(cls, cfg: RadarConfig) -> "MeasureParams":
        ph = cfg.physical()
        fr = cfg.frame
        period = fr.period_ms
        return cls(
            start_ghz=round(ph["start_ghz"], 4),
            slope_mhz_us=round(ph["slope_mhz_us"], 3),
            samples=int(cfg.profile.num_adc_samples),
            fs_ksps=int(cfg.profile.dig_out_sample_rate),
            idle_us=round(ph["idle_us"], 2),
            adc_start_us=round(ph["adc_start_us"], 2),
            ramp_end_us=round(ph["ramp_end_us"], 2),
            tx_start_us=round(ph["tx_start_us"], 2),
            rx_gain_db=int(cfg.profile.rx_gain) & 0x3F,
            tx=[int(m) for m in cfg.tx_pattern] or [1],
            rx=mask_bits(int(cfg.channel.rx_mask)),
            loops=int(fr.num_loops),
            period_ms=round(period, 4),
            duration_s=round(int(fr.num_frames) * period / 1e3, 3) if fr.num_frames else 0.0,
            frames=int(fr.num_frames),
        )

    def to_config(self, base: RadarConfig | None = None) -> RadarConfig:
        """The configuration; what the form does not set comes from `base` (a lab script)."""
        cfg = copy.deepcopy(base) if base is not None else RadarConfig()
        old = cfg.profile
        prof = Profile.from_physical(
            cfg.band, start_ghz=float(self.start_ghz), idle_us=float(self.idle_us),
            adc_start_us=float(self.adc_start_us), ramp_end_us=self.ramp_end_effective,
            slope_mhz_us=float(self.slope_mhz_us), tx_start_us=float(self.tx_start_us),
            num_adc_samples=int(self.samples), fs_ksps=float(self.fs_ksps),
            rx_gain=(int(old.rx_gain) & ~0x3F) | (int(self.rx_gain_db) & 0x3F),
            hpf1=old.hpf_corner_freq1, hpf2=old.hpf_corner_freq2, profile_id=old.profile_id,
            hpf1_lua_flags=old.hpf1_lua_flags, hpf2_lua_flags=old.hpf2_lua_flags)
        prof.tx_out_power_backoff = old.tx_out_power_backoff
        prof.tx_phase_shifter = old.tx_phase_shifter
        cfg.profile = prof
        masks = [int(m) for m in self.tx]
        used = 0
        for m in masks:
            used |= m
        cfg.channel.tx_mask = int(cfg.channel.tx_mask) | used
        rx_mask = 0
        for r in self.rx:
            rx_mask |= 1 << (int(r) - 1)
        cfg.channel.rx_mask = rx_mask
        cfg.chirps = [Chirp(start_idx=i, end_idx=i, profile_id=prof.profile_id, tx_mask=m)
                      for i, m in enumerate(masks)]
        cfg.frame.chirp_start, cfg.frame.chirp_end = 0, len(masks) - 1
        cfg.frame.num_loops = int(self.loops)
        cfg.frame.periodicity = Frame.ticks(float(self.period_ms))
        cfg.frame.num_frames = self.frame_count
        cfg.source = (cfg.source + " + " if cfg.source else "") + "Measure tab"
        return cfg


def _device(cfg: RadarConfig):
    import chirp_design as cd
    name = cfg.device.board if cfg.device.board in cd.DEVICES else "IWR6843AOP"
    if "2544" in (cfg.device.chip or ""):
        name = "AWR2544"
    return cd.DEVICES[name]


def virtual_antennas(cfg: RadarConfig) -> int:
    """Chirps of a loop that transmit on a single TX, each on a different one, x RX."""
    single = {m for m in cfg.tx_pattern if bin(int(m)).count("1") == 1}
    return max(1, len(single)) * cfg.channel.n_rx


def derived(cfg: RadarConfig) -> dict:
    """What a configuration gives, in the units people think in."""
    import chirp_design as cd
    dev = _device(cfg)
    ph = cfg.physical()
    perf = cd.performance(dev, ph["start_ghz"], ph["idle_us"], ph["adc_start_us"],
                          ph["ramp_end_us"], ph["slope_mhz_us"], int(cfg.profile.num_adc_samples),
                          float(cfg.profile.dig_out_sample_rate), max(1, cfg.unique_chirps),
                          int(cfg.frame.num_loops), cfg.frame.period_ms)
    total = cfg.total_bytes
    return {
        "range_res_cm": perf["range_res_cm"],
        "max_range_m": perf["max_range_m"],
        "max_vel_ms": perf["max_vel_ms"],
        "vel_res_ms": perf["vel_res_ms"],
        "fps": perf["fps"],
        "duty_pct": perf["duty_pct"],
        "chirp_time_us": perf["chirp_time_us"],
        "bandwidth_eff_ghz": perf["bandwidth_eff_ghz"],
        "sweep_ghz": perf["sweep_ghz"],
        "wavelength_mm": perf["wavelength_mm"],
        "rate_mb_s": cfg.data_rate_mb_s,
        "size_mb": total / 1e6 if total else None,
        "duration_s": cfg.duration_s,
        "frames": int(cfg.frame.num_frames),
        "chirps_per_frame": cfg.chirps_per_frame,
        "tx_pattern": format_tx(cfg.tx_pattern),
        "rx": format_rx(mask_bits(int(cfg.channel.rx_mask))),
        "virtual_antennas": virtual_antennas(cfg),
    }


def checks(cfg: RadarConfig) -> list[tuple[str, str, str]]:
    """(level, what, message), errors first: config, device limits, loader."""
    import chirp_design as cd
    from .verify import loader_mismatches
    out: list[tuple[str, str, str]] = []
    seen = set()

    def add(level, what, msg):
        key = (level, what.lower())
        if key not in seen:
            seen.add(key)
            out.append((level, what, msg))

    for p in cfg.problems():
        add(p.level, p.what, p.message)
    for m in loader_mismatches(cfg):
        add("error", "loader", m)
    try:
        ph = cfg.physical()
        order = []
        for m in cfg.tx_pattern or [1]:
            bits = mask_bits(int(m))
            order.append(bits[0] if bits else 1)
        d = cd.evaluate(_device(cfg), None, start_ghz=ph["start_ghz"], idle_us=ph["idle_us"],
                        adc_start_us=ph["adc_start_us"], ramp_end_us=ph["ramp_end_us"],
                        slope_mhz_us=ph["slope_mhz_us"], tx_start_us=ph["tx_start_us"],
                        n_samples=int(cfg.profile.num_adc_samples),
                        fs_ksps=int(cfg.profile.dig_out_sample_rate),
                        rx_gain_db=int(cfg.profile.rx_gain) & 0x3F, tx_order=order,
                        loops=int(cfg.frame.num_loops), frames=int(cfg.frame.num_frames),
                        period_ms=cfg.frame.period_ms)
        for c in d.checks:
            if c.level in ("error", "warning"):
                add(c.level, c.what, c.message)
    except Exception as e:                       # a broken form must not break the checks
        add("error", "configuration", str(e))
    loops = int(cfg.frame.num_loops)
    if loops < 8:
        add("info", "Doppler", f"{loops} chirp(s) per transmitter per frame: the Range-Doppler view "
                               f"gets {loops} velocity bins - settings for slow motion over the "
                               f"frames (Vitals, HR, Fastcheck), not for Doppler or micro-Doppler")
    order = {"error": 0, "warning": 1}
    return sorted(out, key=lambda t: order.get(t[0], 2))
