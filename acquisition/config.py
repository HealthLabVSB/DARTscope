"""
acquisition/config.py

The configuration of one recording - radar and DCA1000 - held in the units mmWave Studio
writes to its API log (startFreqConst, 10 ns time codes, slope codes, 5 ns frame ticks).

Those units are what the chip is actually given, so they are the only representation
that cannot drift: a Lua script says 57.14 us and mmWave Studio turns it into 5714, a
log says 5714 and the parser turns it back into 57.14. Keeping the integers means the
log this layer writes carries exactly the numbers mmWave Studio would have written, and
dca1000_decode.parse_logfile_full derives exactly the same parameters from it.

Physical values (GHz, us, MHz/us, ms) are available through the properties and the
from_physical() constructors, which round the way mmWave Studio rounds (to the nearest
code: 60 GHz -> 1491308089, 57.14 us -> 5714, 69.994 MHz/us -> 1933, all checked against
the lab's logs).

What a configuration produces on disk - chirps per frame, bytes per frame, TX sequence,
number of packets - is computed here from the same rules radar_processing uses
(_effective_unique_chirps), so the expectation and the decoder cannot disagree.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import glob
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass

C0 = 299_792_458.0

# startFreqConst: GHz per code. 60 GHz devices use 2.7 GHz / 2^26, 77 GHz ones 3.6 GHz / 2^26.
FREQ_LSB_GHZ = {"60G": 2.7 / 2 ** 26, "77G": 3.6 / 2 ** 26}
# freqSlopeConst: MHz/us per code (2.7 GHz * 900 / 2^26 and 3.6 GHz * 900 / 2^26).
SLOPE_LSB_MHZ_US = {"60G": 2.7e3 * 900 / 2 ** 26, "77G": 3.6e3 * 900 / 2 ** 26}
FRAME_BLANK_MS = 0.3               # TI's InterFrameBlankTime: less is refused (code 80)
TIME_LSB_US = 0.01                 # idle, ADC start, ramp end, TX start: 10 ns codes
FRAME_TICK_S = 5e-9                # framePeriodicity: 5 ns ticks

PAYLOAD_BYTES_PER_PACKET = 1456    # DCA1000 data packet payload (TI rf_api_internal.h)
ADC_BUFFER_BYTES = 16384           # one ping of the ADC buffer, all RX of one chirp
GIGABIT_MB_S = 100.0               # what the DCA1000's Ethernet carries at best
ADC_BITS = {0: 12, 1: 14, 2: 16}
ADC_FORMATS = {0: "Real", 1: "Complex1X", 2: "Complex2X"}
DCA_LVDS_LANES = {1: 4, 2: 2}      # DCA1000 lvdsMode -> lanes (TI: 1 = 4 lane, 2 = 2 lane)
DCA_DATA_FORMAT_BITS = {1: 12, 2: 14, 3: 16}


class ConfigError(ValueError):
    """A configuration that cannot be recorded or would not be read back correctly."""


def _round_code(x: float) -> int:
    """Nearest code, halves up - how mmWave Studio turns its Lua arguments into codes."""
    return int(math.floor(x + 0.5))


def popcount(mask: int) -> int:
    return bin(int(mask)).count("1")


def mask_bits(mask: int) -> list[int]:
    """1-based numbers of the bits set in a mask: 5 -> [1, 3] (TX1 and TX3)."""
    return [b + 1 for b in range(int(mask).bit_length()) if (int(mask) >> b) & 1]


# --- the parts ---------------------------------------------------------------------------

@dataclass
class DeviceSetup:
    """What mmWave Studio needs before the first radar command: chip, firmware, COM port."""
    board: str = "IWR6843AOPEVM"
    chip: str = "IWR6843"                  # ar1.SelectChipVersion - also what the log names
    variant: str = "IWR6843"               # ar1.deviceVariantSelection
    band: str = "60G"                      # ar1.frequencyBandSelection
    lua_family: str = "xwr68"              # argument lists of the Lua API: "xwr68" or "xwr25"
    com_port: int = 12
    baud_rate: int = 921600
    studio_dir: str = r"C:\ti\mmwave_studio_02_01_01_00"
    bss_firmware: str = ""                 # "" = the family's default inside studio_dir
    mss_firmware: str = ""
    power_on_arg: int = 1                  # ar1.PowerOn(x, 1000, 0, 0); the lab's scripts use 1

    def firmware_paths(self) -> tuple[str, str]:
        base = self.studio_dir.rstrip("\\/") + r"\rf_eval_firmware"
        if self.lua_family == "xwr25":
            bss = base + r"\radarss\xwr25xx_radarss_rprc.bin"
            mss = base + r"\masterss\awr2xxx_mmwave_full_mss_rprc.bin"
        else:
            bss = base + r"\radarss\xwr68xx_radarss.bin"
            mss = base + r"\masterss\xwr68xx_masterss.bin"
        return (self.bss_firmware or bss, self.mss_firmware or mss)

    def local_firmware_paths(self) -> tuple[str, str]:
        """firmware_paths() as files on this computer: elsewhere than on Windows a path such
        as C:\\ti\\mmwave_studio_02_01_01_00\\... is looked up in the Wine bottles that hold
        an mmWave Studio installation (Whisky, CrossOver, ~/.wine) or $DARTSCOPE_STUDIO_DIR."""
        return tuple(local_path(p) for p in self.firmware_paths())


_WIN_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def wine_drive_roots() -> list[str]:
    """The C: drives of the Wine bottles on this Mac/PC (Whisky, CrossOver, plain Wine)."""
    home = os.path.expanduser("~")
    pats = [os.path.join(home, "Library", "Containers", "com.isaacmarovitz.Whisky", "Bottles", "*", "drive_c"),
            os.path.join(home, "Library", "Application Support", "CrossOver", "Bottles", "*", "drive_c"),
            os.path.join(home, ".wine", "drive_c")]
    out = []
    for pat in pats:
        out += sorted(glob.glob(pat))
    return out


def local_path(path: str) -> str:
    """A Windows path from a Lua script or the defaults, as a file on this computer."""
    if sys.platform.startswith("win") or not path or not _WIN_PATH.match(path):
        return path
    rel = path[3:].replace("\\", "/")
    env = os.environ.get("DARTSCOPE_STUDIO_DIR")
    if env:
        m = re.match(r"(?i)ti/(mmwave_studio[^/]*)/(.*)", rel)
        if m:
            cand = os.path.join(env, m.group(2))
            if os.path.exists(cand):
                return cand
    home_ti = os.path.join(os.path.expanduser("~"), rel)          # ~/ti/mmwave_studio_.../...
    for root in wine_drive_roots():
        cand = os.path.join(root, rel)
        if os.path.exists(cand):
            return cand
    if os.path.exists(home_ti):
        return home_ti
    return path


@dataclass
class ChannelConfig:
    """ar1.ChanNAdcConfig and ar1.LPModConfig -> API:ChannelConfig, AdcOutConfig, DataFmtConfig."""
    tx_mask: int = 0b111
    rx_mask: int = 0b1111
    adc_bits: int = 2                      # 0 = 12, 1 = 14, 2 = 16 bit
    adc_fmt: int = 1                       # 0 = real, 1 = complex 1x, 2 = complex 2x
    iq_swap: int = 0                       # 0 = I first, 1 = Q first
    lp_adc_mode: int = 0                   # ar1.LPModConfig(0, x)

    @property
    def is_complex(self) -> bool:
        return self.adc_fmt != 0

    @property
    def n_rx(self) -> int:
        return popcount(self.rx_mask)


@dataclass
class LvdsConfig:
    """ar1.DataPathConfig, LvdsClkConfig, LVDSLaneConfig -> API:DataPathConfig ... LvdsLaneConfig."""
    lane_mask: int = 0b0011                # lanes 1..4 enabled; the IWR6843 drives two
    lane_fmt_map: int = 0
    msb_first: int = 1
    packet_header: int = 0
    crc: int = 0
    lane_clk: int = 1
    data_rate: int = 1
    intf_sel: int = 1                      # 1 = LVDS
    transfer_fmt_pkt0: int = 1             # 1 = ADC data only
    transfer_fmt_pkt1: int = 0

    @property
    def lane_param(self) -> int:
        # Packed as mmWave Studio logs it. Only (1, 0, 0) -> 1 occurs in the lab's logs;
        # the positions of the other two bits are inferred.
        return (self.msb_first & 1) | ((self.packet_header & 1) << 1) | ((self.crc & 1) << 2)

    @property
    def n_lanes(self) -> int:
        """Lanes the radar drives; three are carried in a four-lane container, as the parser does."""
        n = popcount(self.lane_mask)
        return 4 if n == 3 else n


@dataclass
class Profile:
    """ar1.ProfileConfig in API-log units (one profile: the loader reads only one)."""
    profile_id: int = 0
    start_freq_const: int = 1491308089    # 60 GHz on a 60 GHz device
    idle_time: int = 10000                # 10 ns codes
    adc_start_time: int = 600
    ramp_end_time: int = 6000
    tx_out_power_backoff: int = 0         # packed per TX as logged
    tx_phase_shifter: int = 0
    freq_slope_const: int = 828
    tx_start_time: int = 0
    num_adc_samples: int = 256
    dig_out_sample_rate: int = 10000      # ksps
    hpf_corner_freq1: int = 0
    hpf_corner_freq2: int = 0
    rx_gain: int = 30
    # Bits the lab's Lua scripts pass above the HPF codes (xwr68: 131072 in the hpf2
    # argument, xwr25: 2216755200 in hpf1). mmWave Studio does not log them; they are
    # kept so a script can be written back as it was read.
    hpf1_lua_flags: int = 0
    hpf2_lua_flags: int = 0x20000

    # physical values ---------------------------------------------------------------
    def start_ghz(self, band: str) -> float:
        return self.start_freq_const * FREQ_LSB_GHZ[band]

    def slope_mhz_us(self, band: str) -> float:
        return self.freq_slope_const * SLOPE_LSB_MHZ_US[band]

    @property
    def idle_us(self) -> float:
        return self.idle_time * TIME_LSB_US

    @property
    def adc_start_us(self) -> float:
        return self.adc_start_time * TIME_LSB_US

    @property
    def ramp_end_us(self) -> float:
        return self.ramp_end_time * TIME_LSB_US

    @property
    def tx_start_us(self) -> float:
        return self.tx_start_time * TIME_LSB_US

    @property
    def fs_msps(self) -> float:
        return self.dig_out_sample_rate / 1000.0

    @property
    def adc_time_us(self) -> float:
        return self.num_adc_samples / self.fs_msps if self.dig_out_sample_rate else 0.0

    @classmethod
    def from_physical(cls, band: str, *, start_ghz: float, idle_us: float, adc_start_us: float,
                      ramp_end_us: float, slope_mhz_us: float, tx_start_us: float,
                      num_adc_samples: int, fs_ksps: float, rx_gain: int, hpf1: int = 0,
                      hpf2: int = 0, profile_id: int = 0, tx_backoff=(0, 0, 0),
                      tx_phase=(0, 0, 0), hpf1_lua_flags: int = 0,
                      hpf2_lua_flags: int = 0x20000) -> "Profile":
        if band not in FREQ_LSB_GHZ:
            raise ConfigError(f"unknown band {band!r}; expected one of {sorted(FREQ_LSB_GHZ)}")
        backoff = 0
        for i, v in enumerate(tx_backoff):
            backoff |= (int(v) & 0xFF) << (8 * i)
        phase = 0
        for i, v in enumerate(tx_phase):
            phase |= (int(v) & 0xFF) << (8 * i)
        return cls(
            profile_id=int(profile_id),
            start_freq_const=_round_code(start_ghz / FREQ_LSB_GHZ[band]),
            idle_time=_round_code(idle_us / TIME_LSB_US),
            adc_start_time=_round_code(adc_start_us / TIME_LSB_US),
            ramp_end_time=_round_code(ramp_end_us / TIME_LSB_US),
            tx_out_power_backoff=backoff,
            tx_phase_shifter=phase,
            freq_slope_const=_round_code(slope_mhz_us / SLOPE_LSB_MHZ_US[band]),
            tx_start_time=_round_code(tx_start_us / TIME_LSB_US),
            num_adc_samples=int(num_adc_samples),
            dig_out_sample_rate=_round_code(fs_ksps),
            hpf_corner_freq1=int(hpf1),
            hpf_corner_freq2=int(hpf2),
            rx_gain=int(rx_gain),
            hpf1_lua_flags=int(hpf1_lua_flags),
            hpf2_lua_flags=int(hpf2_lua_flags),
        )


@dataclass
class Chirp:
    """ar1.ChirpConfig for chirps start_idx..end_idx (API-log units)."""
    start_idx: int = 0
    end_idx: int = 0
    profile_id: int = 0
    start_freq_var: int = 0
    freq_slope_var: int = 0
    idle_time_var: int = 0
    adc_start_time_var: int = 0
    tx_mask: int = 1                       # bit 0 = TX1, as TI numbers them


@dataclass
class Frame:
    """ar1.FrameConfig (API-log units)."""
    chirp_start: int = 0
    chirp_end: int = 0
    num_frames: int = 100                  # 0 = until stopped
    num_loops: int = 128
    periodicity: int = 8_000_000           # 5 ns ticks: 40 ms
    trigger_delay: int = 0
    trigger_select: int = 1                # 1 = software (ar1.StartFrame)

    @property
    def period_ms(self) -> float:
        return self.periodicity * FRAME_TICK_S * 1e3

    @staticmethod
    def ticks(period_ms: float) -> int:
        return _round_code(period_ms * 1e-3 / FRAME_TICK_S)


@dataclass
class Dca1000Config:
    """The DCA1000 side: network and ar1.CaptureCardConfig_Mode / _PacketDelay."""
    dca_ip: str = "192.168.33.180"
    host_ip: str = "192.168.33.30"
    mac: str = "12:34:56:78:90:12"
    config_port: int = 4096
    data_port: int = 4098
    log_mode: int = 1                      # 1 = raw, 2 = multi
    lvds_mode: int = 2                     # 1 = four lanes, 2 = two lanes (the container width)
    transfer_mode: int = 1                 # 1 = LVDS capture, 2 = playback
    capture_mode: int = 2                  # 1 = SD card, 2 = Ethernet stream
    data_format: int = 3                   # 1 = 12, 2 = 14, 3 = 16 bit
    timer_s: int = 30                      # LVDS timeout the FPGA uses to end a record
    packet_delay_us: int = 25
    max_file_size_mb: int = 1024

    @property
    def container_lanes(self) -> int:
        return DCA_LVDS_LANES.get(int(self.lvds_mode), 0)


# --- the whole configuration ------------------------------------------------------------------

@dataclass
class Problem:
    level: str                             # "error" or "warning"
    what: str
    message: str

    def __str__(self) -> str:
        return f"{self.level}: {self.what}: {self.message}"


@dataclass
class RadarConfig:
    device: DeviceSetup = field(default_factory=DeviceSetup)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    lvds: LvdsConfig = field(default_factory=LvdsConfig)
    profile: Profile = field(default_factory=Profile)
    chirps: list[Chirp] = field(default_factory=lambda: [Chirp()])
    frame: Frame = field(default_factory=Frame)
    dca: Dca1000Config = field(default_factory=Dca1000Config)
    source: str = ""                       # where it came from: a Lua script, a log, a design

    # --- what it records ------------------------------------------------------------------
    @property
    def band(self) -> str:
        return self.device.band

    def defined_chirp_indices(self) -> set[int]:
        out: set[int] = set()
        for ch in self.chirps:
            if ch.end_idx >= ch.start_idx:
                out.update(range(ch.start_idx, ch.end_idx + 1))
        return out

    def chirp_at(self, idx: int) -> Chirp | None:
        """The ChirpConfig that defines chirp idx; a later definition overrides an earlier one."""
        found = None
        for ch in self.chirps:
            if ch.start_idx <= idx <= ch.end_idx:
                found = ch
        return found

    @property
    def unique_chirps(self) -> int:
        """Chirps of one loop: indices in the frame range that a ChirpConfig defines.

        The same rule as radar_processing._effective_unique_chirps, so the expectation
        here and the dimensions the loader derives cannot differ.
        """
        rng = set(range(self.frame.chirp_start, self.frame.chirp_end + 1))
        used = self.defined_chirp_indices() & rng
        return len(used) if used else max(1, self.frame.chirp_end - self.frame.chirp_start + 1)

    @property
    def tx_pattern(self) -> list[int]:
        """TX mask of every chirp of a loop, in transmit order (bit 0 = TX1)."""
        out = []
        for idx in range(self.frame.chirp_start, self.frame.chirp_end + 1):
            ch = self.chirp_at(idx)
            if ch is not None:
                out.append(int(ch.tx_mask))
        return out

    @property
    def chirps_per_frame(self) -> int:
        return self.unique_chirps * int(self.frame.num_loops)

    @property
    def samples_per_chirp_int16(self) -> int:
        return self.profile.num_adc_samples * self.channel.n_rx * (2 if self.channel.is_complex else 1)

    @property
    def bytes_per_chirp(self) -> int:
        return 2 * self.samples_per_chirp_int16

    @property
    def bytes_per_frame(self) -> int:
        return self.bytes_per_chirp * self.chirps_per_frame

    @property
    def total_bytes(self) -> int | None:
        """Bytes of the whole recording, None when it runs until stopped."""
        n = int(self.frame.num_frames)
        return self.bytes_per_frame * n if n > 0 else None

    @property
    def expected_packets(self) -> int | None:
        t = self.total_bytes
        return -(-t // PAYLOAD_BYTES_PER_PACKET) if t else None

    @property
    def chirp_time_us(self) -> float:
        return self.profile.idle_us + self.profile.ramp_end_us

    @property
    def active_ms(self) -> float:
        return self.chirps_per_frame * self.chirp_time_us / 1e3

    @property
    def duration_s(self) -> float | None:
        n = int(self.frame.num_frames)
        return n * self.frame.period_ms / 1e3 if n > 0 else None

    @property
    def data_rate_mb_s(self) -> float:
        return self.bytes_per_frame / (self.frame.period_ms * 1e-3) / 1e6 if self.frame.periodicity else 0.0

    def expectation(self) -> dict:
        """What the recording must look like on disk and to the loader."""
        return {
            "frames": int(self.frame.num_frames),
            "chirps_per_frame": self.chirps_per_frame,
            "unique_chirps": self.unique_chirps,
            "loops": int(self.frame.num_loops),
            "rx": self.channel.n_rx,
            "rx_mask": int(self.channel.rx_mask),
            "samples": int(self.profile.num_adc_samples),
            "adc_format": ADC_FORMATS.get(self.channel.adc_fmt, f"UNK({self.channel.adc_fmt})"),
            "adc_bits": ADC_BITS.get(self.channel.adc_bits, 16),
            "iq_order": "QI" if self.channel.iq_swap else "IQ",
            "dtype": "int16 little-endian",
            "rx_layout": "non-interleaved (per chirp: all samples of RX0, then RX1, ...)",
            "tx_pattern": self.tx_pattern,
            "radar_lanes": self.lvds.n_lanes,
            "container_lanes": self.dca.container_lanes,
            "bytes_per_chirp": self.bytes_per_chirp,
            "bytes_per_frame": self.bytes_per_frame,
            "total_bytes": self.total_bytes,
            "packets": self.expected_packets,
            "frame_period_ms": self.frame.period_ms,
            "duration_s": self.duration_s,
            "data_rate_mb_s": self.data_rate_mb_s,
        }

    def physical(self) -> dict:
        """The profile in the units of a Lua script, for people and for the sidecar."""
        p, b = self.profile, self.band
        slope = p.slope_mhz_us(b)
        bw_eff_ghz = slope * p.adc_time_us / 1e3
        lam_m = C0 / (p.start_ghz(b) * 1e9) if p.start_freq_const else 0.0
        tc_s = self.chirp_time_us * 1e-6 * max(1, self.unique_chirps)
        return {
            "start_ghz": p.start_ghz(b),
            "idle_us": p.idle_us,
            "adc_start_us": p.adc_start_us,
            "ramp_end_us": p.ramp_end_us,
            "slope_mhz_us": slope,
            "tx_start_us": p.tx_start_us,
            "fs_ksps": p.dig_out_sample_rate,
            "adc_time_us": p.adc_time_us,
            "bandwidth_eff_ghz": bw_eff_ghz,
            "range_res_m": C0 / (2 * bw_eff_ghz * 1e9) if bw_eff_ghz > 0 else None,
            "max_velocity_m_s": lam_m / (4 * tc_s) if tc_s > 0 else None,
            "frame_period_ms": self.frame.period_ms,
        }

    # --- checks ---------------------------------------------------------------------------
    def problems(self) -> list[Problem]:
        out: list[Problem] = []

        def err(what, msg):
            out.append(Problem("error", what, msg))

        def warn(what, msg):
            out.append(Problem("warning", what, msg))

        p, ch, fr, dca = self.profile, self.channel, self.frame, self.dca
        if self.band not in FREQ_LSB_GHZ:
            err("band", f"{self.band!r} is not 60G or 77G")
        if ch.rx_mask & 0xF == 0:
            err("rx", "no receiver enabled")
        if p.num_adc_samples <= 0:
            err("samples", "numAdcSamples must be positive")
        if p.dig_out_sample_rate <= 0:
            err("sample rate", "digOutSampleRate must be positive")
        elif p.adc_start_us + p.adc_time_us > p.ramp_end_us + 1e-9:
            err("ADC window", f"sampling ends at {p.adc_start_us + p.adc_time_us:.2f} us, after the "
                              f"ramp ({p.ramp_end_us:g} us)")
        if ch.adc_bits != 2 or dca.data_format != 3:
            err("ADC bits", "only 16-bit samples are read by the loader (adcBits 2, DCA1000 "
                            "dataFormat 3)")
        if ch.adc_fmt not in ADC_FORMATS:
            err("ADC format", f"adcFmt {ch.adc_fmt} unknown")
        if ch.iq_swap:
            warn("IQ swap", "no lab recording used IQ swap; the log marks it (QI) and the loader "
                            "swaps, but this has not been checked on hardware")
        if not (1 <= fr.num_loops <= 255):
            err("loops", f"numLoops {fr.num_loops} outside 1..255")
        if fr.chirp_end < fr.chirp_start:
            err("frame", "chirpEndIdx before chirpStartIdx")
        missing = [i for i in range(fr.chirp_start, fr.chirp_end + 1)
                   if i not in self.defined_chirp_indices()]
        if missing:
            err("chirps", f"the frame uses chirps {missing} that no ChirpConfig defines")
        for c in self.chirps:
            if c.profile_id != p.profile_id:
                err("chirps", f"chirps {c.start_idx}..{c.end_idx} use profile {c.profile_id}, "
                              f"only profile {p.profile_id} is configured")
            if c.tx_mask & ~ch.tx_mask:
                err("chirps", f"chirps {c.start_idx}..{c.end_idx} transmit on TX mask "
                              f"{c.tx_mask}, ChanNAdcConfig enables only {ch.tx_mask}")
        if fr.periodicity <= 0:
            err("frame", "framePeriodicity must be positive")
        elif self.active_ms + FRAME_BLANK_MS > fr.period_ms + 1e-9:
            err("frame", f"the chirps take {self.active_ms:.3f} ms of a {fr.period_ms:g} ms frame; the "
                         f"radar needs {FRAME_BLANK_MS * 1e3:.0f} us between frames (FrameConfig "
                         f"answers 80) - a period of at least "
                         f"{self.active_ms + FRAME_BLANK_MS:.3f} ms, or fewer loops")
        if self.bytes_per_chirp > ADC_BUFFER_BYTES:
            err("ADC buffer", f"{self.bytes_per_chirp} bytes a chirp, the buffer holds "
                              f"{ADC_BUFFER_BYTES}")
        if dca.log_mode != 1:
            err("DCA1000", "only raw log mode (1) writes the file the loader reads")
        if dca.transfer_mode != 1 or dca.capture_mode != 2:
            err("DCA1000", "recording needs LVDS capture (transferMode 1) over Ethernet "
                           "(captureMode 2)")
        if dca.container_lanes == 0:
            err("DCA1000", f"lvdsMode {dca.lvds_mode} unknown (1 = four lanes, 2 = two)")
        elif self.lvds.n_lanes and dca.container_lanes != self.lvds.n_lanes:
            warn("lanes", f"the radar drives {self.lvds.n_lanes} lane(s), the DCA1000 records "
                          f"{dca.container_lanes}; the loader follows the recorder log, as for "
                          f"the lane-sweep recordings")
        if not (1 <= dca.max_file_size_mb <= 1024):
            err("DCA1000", "maxRecFileSize must be 1..1024 MB")
        rate = self.data_rate_mb_s
        if rate > GIGABIT_MB_S:
            err("data rate", f"{rate:.1f} MB/s is more than Gigabit Ethernet carries")
        elif rate > 30.6:
            warn("data rate", f"{rate:.1f} MB/s is above the 30.6 MB/s the lab has recorded "
                              f"without loss")
        return out

    def errors(self) -> list[Problem]:
        return [p for p in self.problems() if p.level == "error"]

    def check(self) -> "RadarConfig":
        """Raise ConfigError listing every error; returns self for chaining."""
        errs = self.errors()
        if errs:
            raise ConfigError("; ".join(f"{e.what}: {e.message}" for e in errs))
        return self

    # --- persistence ------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RadarConfig":
        return _from_dict(cls, d)


def _from_dict(cls, d):
    if not is_dataclass(cls):
        return d
    kwargs = {}
    names = {f.name: f for f in fields(cls)}
    types = {
        "device": DeviceSetup, "channel": ChannelConfig, "lvds": LvdsConfig, "profile": Profile,
        "frame": Frame, "dca": Dca1000Config,
    }
    for k, v in (d or {}).items():
        if k not in names:
            continue                       # a field from a newer version: ignore, do not fail
        if k == "chirps":
            kwargs[k] = [Chirp(**{kk: vv for kk, vv in c.items() if kk in Chirp.__dataclass_fields__})
                         for c in v]
        elif k in types and isinstance(v, dict):
            sub = types[k]
            kwargs[k] = sub(**{kk: vv for kk, vv in v.items() if kk in sub.__dataclass_fields__})
        else:
            kwargs[k] = v
    return cls(**kwargs)


# --- from the chirp Helper ------------------------------------------------------------------

def from_design(design, dca: Dca1000Config | None = None) -> RadarConfig:
    """
    The configuration of a chirp_design.Design (the Helper tab), as its Lua script sets it.

    Mirrors chirp_design.lua_script: one ChirpConfig per transmitter of the pattern, the
    transmitters used enabled in ChanNAdcConfig, HPF codes 0, the lab's LVDS set-up.
    """
    dev = design.device
    if dev.lua != "xwr68":
        raise ConfigError(f"recording the {dev.name} through this layer is not supported yet "
                          f"(only the xwr68 family: IWR6843, IWR6843AOP)")
    band = dev.band
    profile = Profile.from_physical(
        band, start_ghz=design.start_ghz, idle_us=design.idle_us, adc_start_us=design.adc_start_us,
        ramp_end_us=design.ramp_end_us, slope_mhz_us=design.slope_mhz_us,
        tx_start_us=design.tx_start_us, num_adc_samples=design.n_samples, fs_ksps=design.fs_ksps,
        rx_gain=design.rx_gain_db)
    chirps = [Chirp(start_idx=i, end_idx=i, tx_mask=1 << (tx - 1))
              for i, tx in enumerate(design.tx_order)]
    tx_mask = 0
    for tx in design.tx_order:
        tx_mask |= 1 << (tx - 1)
    return RadarConfig(
        device=DeviceSetup(board=dev.name, chip=dev.chip, variant=dev.variant, band=band,
                           lua_family=dev.lua,
                           com_port=dev.com_port, studio_dir=dev.studio_dir),
        channel=ChannelConfig(tx_mask=tx_mask, rx_mask=(1 << dev.n_rx) - 1,
                              adc_fmt=1 if dev.complex_samples else 0),
        lvds=LvdsConfig(lane_mask=(1 << dev.lanes) - 1),
        profile=profile,
        chirps=chirps,
        frame=Frame(chirp_start=0, chirp_end=len(design.tx_order) - 1,
                    num_frames=int(design.frames), num_loops=int(design.loops),
                    periodicity=Frame.ticks(design.period_ms)),
        dca=dca or Dca1000Config(lvds_mode=2 if dev.lanes <= 2 else 1),
        source=f"chirp_design: {dev.name}",
    )
