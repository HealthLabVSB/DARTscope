"""
cascade/params.py

The settings of a cascade take in physical units (CascadeParams), what they give (range
and velocity resolution, data rate, ...), the checks against the AWR2243, the firmware and
the DCA1000 -> Mac link, and the gUserCfg words the firmware reads
(firmware/src/cascade_user_cfg.h).

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, fields
from pathlib import Path

C = 299_792_458.0
FREQ_LSB_HZ = 3.6e9 / 2 ** 26                    # rlProfileCfg_t.startFreqConst
SLOPE_LSB_MHZ_US = 3.6e9 * 900 / 2 ** 26 / 1e6   # freqSlopeConst: 48.279 kHz/us
HPF1_KHZ = {175: 0, 235: 1, 350: 2, 700: 3}
HPF2_KHZ = {350: 0, 700: 1, 1400: 2, 2800: 3}
RX_GAINS_DB = list(range(24, 54, 2))
MAX_SAMPLES, MAX_CHIRPS = 512, 12                # cascade_user_cfg.h
CSI_BYTES_PER_S = 4 * 600e6 / 8                  # AWR2243 CSI-2: 4 lanes x 600 Mbps
LVDS_BYTES_PER_S = 4 * 600e6 / 8                 # AM273x CBUFF -> DCA1000: 4 lanes x 600 Mbps
USER_CFG_MAGIC, USER_CFG_VERSION = 0xCA5CAD01, 2
CFG_WORDS = 42                                   # magic .. txMaskDev1[11]
NUM_RX = 8                                       # 2 devices x 4 RX
DCA_PAYLOAD = 1456                               # data bytes per DCA1000 datagram
MIN_PACKET_DELAY_US = 36                         # faster bursts lost packets on the lab's Mac
MAX_RATE = DCA_PAYLOAD / MIN_PACKET_DELAY_US * 1e6 / 1.15
LVDS_MARGIN = 1.2                                # chirp time / LVDS transfer time, at least

# TX enable masks per chirp: (primary AWR2243 TX0..2, secondary AWR2243 TX0..2)
TX_MODES = {
    "All 6 TX at once (beamforming)": [(7, 7)],
    "Primary device, 3 TX at once": [(7, 0)],
    "One TX (primary TX0)": [(1, 0)],
    "TDM-MIMO 6 TX (48 virtual channels)": [(1, 0), (2, 0), (4, 0), (0, 1), (0, 2), (0, 4)],
    "TDM-MIMO 3 TX primary (24 virtual)": [(1, 0), (2, 0), (4, 0)],
}


def parse_tx_chirps(text: str) -> list[tuple[int, int]]:
    """'7/7; 1/0' -> [(7, 7), (1, 0)]"""
    out = []
    for part in text.replace(",", ";").split(";"):
        part = part.strip()
        if part:
            a, b = part.split("/")
            out.append((int(a, 0), int(b, 0)))
    return out


def format_tx_chirps(chirps) -> str:
    return "; ".join(f"{a}/{b}" for a, b in chirps)


def packet_delay_us(rate_bytes_s: float) -> int:
    """DCA1000 inter-packet gap: drain its buffer at ~1.5x the average data rate, never
    faster than MIN_PACKET_DELAY_US. On the lab's Mac (USB Ethernet adapter, 2026-10-09):
    25 us (59 MB/s bursts) lost 371 packets in 20 s, 30 us lost 4, 50 us none."""
    if rate_bytes_s <= 0:
        return 50
    return int(min(100, max(MIN_PACKET_DELAY_US, DCA_PAYLOAD / (1.5 * rate_bytes_s) * 1e6)))


TDM6 = "1/0; 2/0; 4/0; 0/1; 0/2; 0/4"    # TX1..TX6 one after another: 48 virtual channels

# One chirp profile per preset, written out in full (protocol: docs/CASCADE_PROTOCOL.md).
# The capability envelope behind them was measured on 2026-10-09: slopes 100-200 MHz/us, a
# 76.0 GHz start with a 4.95 GHz sweep, 22.5 Msps, 3 us idle and 512 samples at 10-15 Msps
# all ran without a lost packet.
_PROFILE_58MM = dict(start_freq_ghz=77.0, slope_mhz_us=100.0, idle_us=7.0, adc_start_us=6.0,
                     ramp_end_us=33.0, tx_start_us=0.0, num_samples=128, sample_rate_ksps=5000,
                     rx_gain_db=30, hpf1_khz=175, hpf2_khz=350, tx_backoff_db=0)
PRESETS = {
    # one person: one TX (a predictable beam), 32 chirps averaged per frame, 100 frames/s
    "Vital signs - 1 person, 100 fps": dict(
        _PROFILE_58MM, tx_chirps="1/0", num_loops=32, frame_period_ms=10.0, duration_s=60.0),
    # several people: all six TX in turn, 48 virtual channels to tell them apart by angle
    "Vital signs - multi-person (TDM 48 ch)": dict(
        _PROFILE_58MM, tx_chirps=TDM6, num_loops=8, frame_period_ms=20.0, duration_s=60.0),
    # walking people: 6 cm, +-5.1 m/s, 0.32 m/s velocity bins, 48 virtual channels
    "People - multi-person tracking (TDM 48 ch)": dict(
        start_freq_ghz=77.0, slope_mhz_us=130.0, idle_us=5.0, adc_start_us=6.0, ramp_end_us=27.0,
        tx_start_us=0.0, num_samples=192, sample_rate_ksps=10000, rx_gain_db=30, hpf1_khz=175,
        hpf2_khz=350, tx_backoff_db=0, tx_chirps=TDM6, num_loops=32, frame_period_ms=66.67,
        duration_s=60.0),
    # people, finest within the link: the 3.9 cm of the imaging profile (192 samples at 7.5 Msps,
    # 7.5 m), 32 loops of all six TX - 0.23 m/s velocity bins, +-3.7 m/s - at 20 frames/s,
    # 23.6 MB/s (the safe ~25 MB/s of the DCA1000 -> Mac link fixes loops x frame rate)
    "People - high resolution (3.9 cm, 0.23 m/s, TDM 48 ch)": dict(
        start_freq_ghz=76.0, slope_mhz_us=150.0, idle_us=10.0, adc_start_us=6.0, ramp_end_us=33.0,
        tx_start_us=0.0, num_samples=192, sample_rate_ksps=7500, rx_gain_db=30, hpf1_khz=175,
        hpf2_khz=350, tx_backoff_db=0, tx_chirps=TDM6, num_loops=32, frame_period_ms=50.0,
        duration_s=60.0),
    # the finest range resolution the device ran: 76.0 GHz + 4.95 GHz, 3.84 GHz sampled
    "Imaging - best range resolution (3.9 cm)": dict(
        start_freq_ghz=76.0, slope_mhz_us=150.0, idle_us=10.0, adc_start_us=6.0, ramp_end_us=33.0,
        tx_start_us=0.0, num_samples=256, sample_rate_ksps=10000, rx_gain_db=30, hpf1_khz=175,
        hpf2_khz=350, tx_backoff_db=0, tx_chirps=TDM6, num_loops=8, frame_period_ms=50.0,
        duration_s=30.0),
    "Motion / micro-Doppler (fast)": dict(
        start_freq_ghz=77.0, slope_mhz_us=60.0, idle_us=5.0, adc_start_us=5.0, ramp_end_us=20.0,
        tx_start_us=0.0, num_samples=128, sample_rate_ksps=10000, rx_gain_db=30, hpf1_khz=175,
        hpf2_khz=350, tx_backoff_db=0, tx_chirps="1/0", num_loops=128, frame_period_ms=33.33,
        duration_s=30.0),
    "Long range (to ~40 m)": dict(
        start_freq_ghz=77.0, slope_mhz_us=20.0, idle_us=10.0, adc_start_us=6.0, ramp_end_us=50.0,
        tx_start_us=0.0, num_samples=256, sample_rate_ksps=6000, rx_gain_db=40, hpf1_khz=175,
        hpf2_khz=350, tx_backoff_db=0, tx_chirps="1/0", num_loops=64, frame_period_ms=50.0,
        duration_s=30.0),
    "SDK test profile (what the repo ran)": dict(
        start_freq_ghz=77.0, slope_mhz_us=229 * SLOPE_LSB_MHZ_US, idle_us=7.0, adc_start_us=7.0,
        ramp_end_us=20.83, tx_start_us=0.0, num_samples=128, sample_rate_ksps=20000, rx_gain_db=42,
        hpf1_khz=175, hpf2_khz=350, tx_backoff_db=0, tx_chirps="7/7", num_loops=128,
        frame_period_ms=30.0, duration_s=30.0),
}


@dataclass
class CascadeParams:
    # chirp profile (one profile for both devices); the defaults are the first preset
    start_freq_ghz: float = 77.0
    slope_mhz_us: float = 100.0
    idle_us: float = 7.0
    adc_start_us: float = 6.0
    ramp_end_us: float = 33.0
    tx_start_us: float = 0.0
    num_samples: int = 128          # 25.6 us at 5 Msps: 2.56 GHz sampled, 5.9 cm, 7.5 m range
    sample_rate_ksps: int = 5000
    rx_gain_db: int = 30
    hpf1_khz: int = 175
    hpf2_khz: int = 350
    tx_backoff_db: int = 0
    # chirps of a loop and the frame
    tx_chirps: str = "1/0"
    num_loops: int = 32
    frame_period_ms: float = 10.0
    duration_s: float = 60.0

    @classmethod
    def preset(cls, name: str) -> "CascadeParams":
        return cls(**PRESETS[name])

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "CascadeParams":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @property
    def chirps(self) -> list[tuple[int, int]]:
        return parse_tx_chirps(self.tx_chirps)

    @property
    def num_frames(self) -> int:
        return max(1, int(round(self.duration_s * 1000.0 / self.frame_period_ms)))

    def slope_code(self) -> int:
        return int(round(self.slope_mhz_us / SLOPE_LSB_MHZ_US))

    def words(self) -> list[int]:
        """gUserCfg from magic to txMaskDev1[11], in the order of cascade_user_cfg.h."""
        ch = self.chirps
        m0 = [a for a, _ in ch] + [0] * (MAX_CHIRPS - len(ch))
        m1 = [b for _, b in ch] + [0] * (MAX_CHIRPS - len(ch))
        backoff = int(self.tx_backoff_db) & 0xFF
        w = [USER_CFG_MAGIC, USER_CFG_VERSION,
             int(round(self.start_freq_ghz * 1e9 / FREQ_LSB_HZ)),
             self.slope_code() & 0xFFFF,
             int(round(self.idle_us * 100)),
             int(round(self.adc_start_us * 100)),
             int(round(self.ramp_end_us * 100)),
             int(round(self.tx_start_us * 100)) & 0xFFFF,
             int(self.num_samples), int(self.sample_rate_ksps), int(self.rx_gain_db),
             HPF1_KHZ[int(self.hpf1_khz)], HPF2_KHZ[int(self.hpf2_khz)],
             backoff | (backoff << 8) | (backoff << 16),
             len(ch), int(self.num_loops),
             int(round(self.frame_period_ms * 1e6 / 5)),
             self.num_frames] + m0[:MAX_CHIRPS] + m1[:MAX_CHIRPS]
        return [int(x) & 0xFFFFFFFF for x in w]

    def check(self, disk: Path | None = None) -> dict:
        """derived values (label -> text), errors (cannot record) and warnings, and the
        numbers the recording needs."""
        errors, warnings, d = [], [], {}
        try:
            ch = self.chirps
        except Exception:
            ch = []
            errors.append("TX chirps: write 'mask0/mask1; ...', e.g. 7/7 or 1/0; 2/0")
        n, fs = int(self.num_samples), float(self.sample_rate_ksps) * 1e3
        slope = self.slope_code() * SLOPE_LSB_MHZ_US                   # quantized MHz/us
        adc_us = n / fs * 1e6 if fs > 0 else float("inf")
        chirp_us = self.idle_us + self.ramp_end_us
        n_ch = max(len(ch), 1)
        chirps_per_frame = n_ch * int(self.num_loops)
        active_ms = chirps_per_frame * chirp_us / 1000.0
        bw = abs(slope) * 1e12 * adc_us * 1e-6                          # sampled bandwidth, Hz
        f_end = self.start_freq_ghz + slope * self.ramp_end_us / 1000.0
        f_center = self.start_freq_ghz * 1e9 + slope * 1e12 * (self.adc_start_us + adc_us / 2) * 1e-6
        lam = C / f_center
        tc_s = n_ch * chirp_us * 1e-6                                   # same-TX chirp spacing
        bytes_chirp = NUM_RX * n * 4                                    # 8 RX x N x I/Q int16
        bytes_frame = chirps_per_frame * bytes_chirp
        rate = bytes_frame / (self.frame_period_ms * 1e-3)
        size = bytes_frame * self.num_frames
        delay = packet_delay_us(rate)
        tx_used = {(dev, b) for a, bb in ch for dev, m in ((0, a), (1, bb)) for b in range(3) if m >> b & 1}
        csi_us = n * 16 / CSI_BYTES_PER_S * 1e6
        lvds_us = bytes_chirp / LVDS_BYTES_PER_S * 1e6
        d.update({
            "Slope (quantized)": f"{slope:.3f} MHz/µs",
            "ADC sampling": f"{adc_us:.2f} µs (ends at {self.adc_start_us + adc_us:.2f} µs)",
            "Sampled bandwidth": f"{bw / 1e6:.0f} MHz",
            "Ramp end frequency": f"{f_end:.3f} GHz",
            "Range resolution": f"{C / (2 * bw) * 100:.2f} cm" if bw else "-",
            "Max. range (IF = fs)": f"{fs * C / (2 * abs(slope) * 1e12):.2f} m" if slope else "-",
            "Chirp (idle + ramp)": f"{chirp_us:.2f} µs",
            "Chirps per frame": f"{chirps_per_frame} ({n_ch} x {self.num_loops} loops)",
            "Active part of frame": f"{active_ms:.2f} of {self.frame_period_ms:g} ms ({active_ms / self.frame_period_ms * 100:.0f} %)",
            "Max. velocity": f"±{lam / (4 * tc_s):.2f} m/s",
            "Velocity resolution": f"{lam / (2 * int(self.num_loops) * tc_s) * 100:.1f} cm/s",
            "Frame rate": f"{1000 / self.frame_period_ms:.2f} Hz",
            "Frames": f"{self.num_frames}",
            "Virtual channels": (f"{len(tx_used) * 8} ({len(tx_used)} TX x 8 RX, TDM)" if len(ch) > 1
                                 else f"8 RX ({len(tx_used)} TX at once)"),
            "Data rate": f"{rate / 1e6:.1f} MB/s",
            "File size": f"{size / 1e9:.2f} GB",
            "Chirp transfer CSI / LVDS": f"{csi_us:.1f} / {lvds_us:.1f} µs",
            "DCA1000 packet gap": f"{delay} µs (sends ≤ {DCA_PAYLOAD / delay:.0f} MB/s)",
        })
        # errors: the radar, the firmware or the data path would refuse or break
        if not 76.0 <= self.start_freq_ghz <= 81.0:
            errors.append("Start frequency must be 76-81 GHz.")
        if not 76.0 <= f_end <= 81.0:
            errors.append(f"The ramp ends at {f_end:.2f} GHz, outside 76-81 GHz (shorter ramp or lower slope).")
        if not 16 <= n <= MAX_SAMPLES or n % 4:
            errors.append(f"Samples per chirp: 16-{MAX_SAMPLES}, a multiple of 4.")
        if not 2000 <= self.sample_rate_ksps <= 22500:
            errors.append("Sampling rate 2000-22500 ksps (complex, AWR2243).")
        if self.adc_start_us + adc_us > self.ramp_end_us:
            errors.append(f"Sampling ends at {self.adc_start_us + adc_us:.2f} µs but the ramp at "
                          f"{self.ramp_end_us:g} µs - make the ramp ≥ {math.ceil(self.adc_start_us + adc_us + 1)} µs.")
        if not 0 < abs(slope) <= 266 or not -32768 <= self.slope_code() <= 32767:
            errors.append("Slope must be non-zero and at most ~266 MHz/µs.")
        if not 1 <= int(self.num_loops) <= 255:
            errors.append("Loops per frame: 1-255.")
        if not 1 <= len(ch) <= MAX_CHIRPS:
            errors.append(f"1-{MAX_CHIRPS} chirps per loop.")
        if any(not (0 <= a <= 7 and 0 <= b <= 7) for a, b in ch):
            errors.append("A TX mask is 0-7 (bit 0..2 = TX0..TX2) for each device.")
        if active_ms > self.frame_period_ms:
            errors.append(f"The chirps take {active_ms:.2f} ms, longer than the frame period {self.frame_period_ms:g} ms.")
        if not 1 <= self.num_frames <= 65535:
            errors.append("Frames: 1-65535 (shorter take or longer frame period).")
        # measured 2026-10-09: 512 samples at 20 Msps stalled after one chirp with a 63 us chirp
        # (LVDS 54.6 us, x1.15) and ran with 72 us (x1.32); 448 at 20 Msps ran at x1.15
        if LVDS_MARGIN * lvds_us > chirp_us or csi_us > chirp_us:
            errors.append(f"A chirp's data cannot be passed on in time: the LVDS transfer takes {lvds_us:.1f} µs "
                          f"and the chirp must last {LVDS_MARGIN:g}x that (≥ {LVDS_MARGIN * lvds_us:.1f} µs, "
                          f"now {chirp_us:.1f} µs) - fewer samples or longer idle/ramp.")
        if rate > MAX_RATE:
            errors.append(f"Data rate {rate / 1e6:.0f} MB/s is more than the DCA1000 -> PC link carries "
                          f"reliably (~{MAX_RATE / 1e6:.0f} MB/s): fewer loops/samples or a longer frame period.")
        elif rate > 25e6:
            warnings.append(f"Data rate {rate / 1e6:.0f} MB/s is near the limit (~{MAX_RATE / 1e6:.0f} MB/s); "
                            "packets may be lost. Safe is up to ~25 MB/s.")
        if int(self.rx_gain_db) not in RX_GAINS_DB:
            errors.append("RX gain 24-52 dB in 2 dB steps.")
        if int(self.hpf1_khz) not in HPF1_KHZ or int(self.hpf2_khz) not in HPF2_KHZ:
            errors.append("Unknown HPF corner.")
        if not 0 <= int(self.tx_backoff_db) <= 26:
            errors.append("TX backoff 0-26 dB.")
        if self.idle_us < 2.0:
            errors.append("Idle time at least 2 µs.")
        # warnings: allowed, but probably not what is wanted
        if self.idle_us < 5.0:
            warnings.append("Idle < 5 µs: at higher sampling rates the radar may refuse the chirp.")
        if self.adc_start_us < 3.0:
            warnings.append("ADC start < 3 µs: the start of the ramp has not settled.")
        if active_ms / self.frame_period_ms > 0.8:
            warnings.append("The chirps fill > 80 % of the frame - the radar needs time between frames.")
        if any(a == 0 and b == 0 for a, b in ch):
            warnings.append("A chirp has no TX switched on.")
        if len(ch) == 1 and sum(bin(a).count("1") + bin(b).count("1") for a, b in ch) > 1:
            warnings.append("Several TX at once without phase calibration: the beam need not point ahead.")
        if disk is not None:
            try:
                free = shutil.disk_usage(disk if disk.exists() else disk.parent).free
                if size > free * 0.9:
                    errors.append(f"Only {free / 1e9:.1f} GB free on the disk, the take needs {size / 1e9:.2f} GB.")
            except OSError:
                pass
        return {"derived": d, "errors": errors, "warnings": warnings,
                "bytes_per_frame": bytes_frame, "bytes_total": size, "chirps_per_frame": chirps_per_frame,
                "range_res_m": C / (2 * bw) if bw else None, "slope_mhz_us": slope,
                "adc_time_us": adc_us, "wavelength_m": lam, "chirp_time_us": chirp_us,
                "frame_rate_hz": 1000 / self.frame_period_ms, "packet_delay_us": delay, "rate_bytes_s": rate}
