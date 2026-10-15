"""
acquisition/studio_log.py

The <stem>_LogFile.txt that dca1000_decode.parse_logfile_full reads, written the way
mmWave Studio writes it, and read back into a RadarConfig.

mmWave Studio logs every radar command as

    02-Apr-2026 16:23:52: API:ProfileConfig,0,1509949440,6000,100,1500,0,0,5523,100,150,10785,1,0,36,0,

- the arguments in the chip's units and, last, the command's return status. The lines
written here are those lines, character for character: the same commands, the same
order, the same constant fields mmWave Studio adds (DataPathConfig's ",2", SetHsiClock 9,
EnableTestSource), the same 55-field AdvancedFrameConfig. That is checked in the tests by
reading every real log in the data folders into a RadarConfig and writing it back.

Two differences from mmWave Studio, both on purpose:
  - one log per recording, holding only that recording's configuration. mmWave Studio
    appends a whole session to one file, which is why the parser takes the last line of
    each kind and needs work-arounds for chirps that an older configuration left behind.
  - a first line naming this program. It carries no API: tag, so the parser ignores it.

The chip is logged as the Lua script's SelectChipVersion gives it ("IWR6843", never
"IWR6843AOP"), as mmWave Studio does: AOA.py then resolves the antenna layout exactly as
for every recording made so far. The board goes into the JSON sidecar instead.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import os
import re
from datetime import datetime

from .config import (Chirp, ChannelConfig, ConfigError, DeviceSetup, Frame, LvdsConfig,
                     Profile, RadarConfig)
from .lua import family_for_chip

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_TS_RE = re.compile(r"^(\d{2})-([A-Za-z]{3})-(\d{4}) (\d{2}):(\d{2}):(\d{2}): ")
ADV_FRAME_FIELDS = 55                       # numeric fields of API:AdvancedFrameConfig


def timestamp(when: datetime) -> str:
    """'29-Sep-2026 14:03:07' - English month names whatever the locale, as mmWave Studio."""
    return (f"{when.day:02d}-{_MONTHS[when.month - 1]}-{when.year:04d} "
            f"{when.hour:02d}:{when.minute:02d}:{when.second:02d}")


def _line(verb: str, *values) -> str:
    return f"API:{verb}," + "".join(f"{v}," for v in values)


def config_lines(cfg: RadarConfig) -> list[str]:
    """The API lines of a configuration, in mmWave Studio's order, without timestamps."""
    d, ch, lv, p, fr = cfg.device, cfg.channel, cfg.lvds, cfg.profile, cfg.frame
    out = [_line("select_capture_device", "DCA1000", 0)]
    if d.chip:
        out.append(_line("select_chip_version", d.chip, 0))
    out += [
        _line("ChannelConfig", ch.tx_mask, ch.rx_mask, 0),
        _line("AdcOutConfig", ch.adc_bits, ch.adc_fmt, 0),
        _line("DataFmtConfig", ch.rx_mask, ch.adc_bits, 1 if ch.is_complex else 0, ch.iq_swap, 1, 0),
        _line("LowPowerConfig", 0, ch.lp_adc_mode, 0),
        _line("DataPathConfig", lv.intf_sel, lv.transfer_fmt_pkt0, lv.transfer_fmt_pkt1, 2, 0),
        _line("LvdsClkConfig", lv.lane_clk, lv.data_rate, 0),
    ]
    if d.lua_family == "xwr68" and lv.data_rate == 1:
        out.append(_line("SetHsiClock", 9, 0))
    out += [
        _line("LaneConfig", lv.lane_mask, 0),
        _line("LvdsLaneConfig", lv.lane_fmt_map, lv.lane_param, 0),
        _line("ProfileConfig", p.profile_id, p.start_freq_const, p.idle_time, p.adc_start_time,
              p.ramp_end_time, p.tx_out_power_backoff, p.tx_phase_shifter, p.freq_slope_const,
              p.tx_start_time, p.num_adc_samples, p.dig_out_sample_rate, p.hpf_corner_freq1,
              p.hpf_corner_freq2, p.rx_gain, 0),
    ]
    for c in cfg.chirps:
        out.append(_line("ChirpConfig", c.start_idx, c.end_idx, c.profile_id, c.start_freq_var,
                         c.freq_slope_var, c.idle_time_var, c.adc_start_time_var, c.tx_mask, 0))
    samples_field = p.num_adc_samples * (2 if ch.is_complex else 1)
    out += [
        _line("EnableTestSource", 0, 1, 0),
        _line("FrameConfig", fr.chirp_start, fr.chirp_end, fr.num_frames, fr.num_loops,
              fr.periodicity, fr.trigger_delay, samples_field, 0),
        advanced_frame_line(cfg),
    ]
    return out


def advanced_frame_line(cfg: RadarConfig) -> str:
    """
    API:AdvancedFrameConfig as mmWave Studio logs it for a plain FrameConfig.

    Only these positions are ever non-zero in the lab's 300 logs: [0] one sub-frame,
    [4] chirps in the frame range, [5] loops, [6] and [10] the period, [8] and [9] one
    burst, [38] frames. [3] - the first chirp - is always 0 there and inferred.
    """
    fr = cfg.frame
    v = [0] * ADV_FRAME_FIELDS
    v[0] = 1
    v[3] = fr.chirp_start
    v[4] = fr.chirp_end - fr.chirp_start + 1
    v[5] = fr.num_loops
    v[6] = fr.periodicity
    v[8] = 1
    v[9] = 1
    v[10] = fr.periodicity
    v[38] = fr.num_frames
    return _line("AdvancedFrameConfig", *v)


def header_line(program: str) -> str:
    return f"{program}: configuration log written without mmWave Studio,0,"


def write_log(path: str, cfg: RadarConfig, when: datetime | None = None,
              program: str = "DARTscope acquisition") -> str:
    """Write <stem>_LogFile.txt for cfg. Refuses to overwrite an existing file."""
    when = when or datetime.now()
    ts = timestamp(when)
    body = [header_line(program)] + config_lines(cfg)
    with open(path, "x", encoding="ascii", newline="\n") as f:
        for line in body:
            f.write(f"{ts}: {line}\n")
    return path


def append_event(path: str, line: str, when: datetime | None = None) -> None:
    """Append one line (e.g. API:SensorStart,0,) the way the session continues in mmWave Studio."""
    with open(path, "a", encoding="ascii", newline="\n") as f:
        f.write(f"{timestamp(when or datetime.now())}: {line}\n")


def sensor_start_line() -> str:
    return _line("SensorStart", 0)


def sensor_stop_line() -> str:
    return _line("SensorStop", 0)


def files_and_frames_line(n_files: int, n_frames: int) -> str:
    return _line("update_num_adc_files_and_frames", n_files, n_frames, 1, 0)


# --- reading ------------------------------------------------------------------------------------

_ENTRY_SPLIT = re.compile(r"(?=\d{2}-[A-Za-z]{3}-\d{4} \d{2}:\d{2}:\d{2}: )")


def api_lines(text: str) -> list[tuple[str, list[str], str]]:
    """(verb, fields, entry without timestamp) for every API: entry, in file order."""
    out = []
    for raw in text.splitlines():
        # mmWave Studio sometimes runs two entries into one line ("... name is ,C:02-Apr-...")
        for part in _ENTRY_SPLIT.split(raw.strip()):
            s = _TS_RE.sub("", part, count=1).strip()
            m = re.match(r"API:([A-Za-z_]+),(.*)$", s)
            if not m:
                continue
            fields_ = m.group(2).split(",")
            if fields_ and fields_[-1] == "":
                fields_ = fields_[:-1]
            out.append((m.group(1), fields_, s))
    return out


def _ints(fields_: list[str]) -> list[int] | None:
    try:
        return [int(x) for x in fields_]
    except ValueError:
        return None


def read_config(text_or_path: str) -> RadarConfig:
    """
    The configuration a log describes, by the parser's rule: the last line of each kind
    wins, chirps are kept per (start, end, profile) with their last definition.

    Only lines that succeeded (status 0) count, since a rejected command left the radar
    as it was.
    """
    text = text_or_path
    source = ""
    if "\n" not in text_or_path and os.path.isfile(text_or_path):
        with open(text_or_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        source = text_or_path
    cfg = RadarConfig(chirps=[], source=source, device=DeviceSetup(), channel=ChannelConfig(),
                      lvds=LvdsConfig(), profile=Profile(), frame=Frame())
    chirps: dict = {}
    seen: set[str] = set()
    chip = None
    for verb, flds, _ in api_lines(text):
        if verb == "select_chip_version" and flds:
            chip = chip or flds[0]
            continue
        vals = _ints(flds)
        if vals is None or not vals or vals[-1] != 0:
            continue
        vals = vals[:-1]                                      # the status
        seen.add(verb)
        if verb == "ChannelConfig" and len(vals) >= 2:
            cfg.channel.tx_mask, cfg.channel.rx_mask = vals[0], vals[1]
        elif verb == "AdcOutConfig" and len(vals) >= 2:
            cfg.channel.adc_bits, cfg.channel.adc_fmt = vals[0], vals[1]
        elif verb == "DataFmtConfig" and len(vals) >= 4:
            cfg.channel.rx_mask = vals[0]
            cfg.channel.iq_swap = vals[3]
            if "AdcOutConfig" not in seen:
                cfg.channel.adc_bits = vals[1]
                cfg.channel.adc_fmt = vals[2]
        elif verb == "LowPowerConfig" and len(vals) >= 2:
            cfg.channel.lp_adc_mode = vals[1]
        elif verb == "DataPathConfig" and len(vals) >= 3:
            cfg.lvds.intf_sel, cfg.lvds.transfer_fmt_pkt0, cfg.lvds.transfer_fmt_pkt1 = vals[:3]
        elif verb == "LvdsClkConfig" and len(vals) >= 2:
            cfg.lvds.lane_clk, cfg.lvds.data_rate = vals[0], vals[1]
        elif verb == "LaneConfig" and vals:
            cfg.lvds.lane_mask = vals[0]
        elif verb == "LvdsLaneConfig" and len(vals) >= 2:
            cfg.lvds.lane_fmt_map = vals[0]
            cfg.lvds.msb_first = vals[1] & 1
            cfg.lvds.packet_header = (vals[1] >> 1) & 1
            cfg.lvds.crc = (vals[1] >> 2) & 1
        elif verb == "ProfileConfig" and len(vals) >= 14:
            p = cfg.profile
            (p.profile_id, p.start_freq_const, p.idle_time, p.adc_start_time, p.ramp_end_time,
             p.tx_out_power_backoff, p.tx_phase_shifter, p.freq_slope_const, p.tx_start_time,
             p.num_adc_samples, p.dig_out_sample_rate, p.hpf_corner_freq1, p.hpf_corner_freq2,
             p.rx_gain) = vals[:14]
        elif verb == "ChirpConfig" and len(vals) >= 8:
            c = Chirp(*vals[:8])
            chirps[(c.start_idx, c.end_idx, c.profile_id)] = c
        elif verb == "FrameConfig" and len(vals) >= 6:
            cfg.frame = Frame(chirp_start=vals[0], chirp_end=vals[1], num_frames=vals[2],
                              num_loops=vals[3], periodicity=vals[4], trigger_delay=vals[5])
    if "ProfileConfig" not in seen:
        raise ConfigError(f"{source or 'the log'} holds no successful API:ProfileConfig - not an "
                          f"mmWave Studio configuration log")
    if chip:
        cfg.device.chip = chip
        cfg.device.variant = chip
        cfg.device.lua_family = family_for_chip(chip) or cfg.device.lua_family
    else:
        cfg.device.chip = ""
    cfg.device.band = _band(cfg.device.chip, cfg.profile.start_freq_const)
    cfg.chirps = sorted(chirps.values(), key=lambda c: (c.start_idx, c.end_idx))
    if "ChannelConfig" not in seen:
        m = 0
        for c in cfg.chirps:
            m |= c.tx_mask
        cfg.channel.tx_mask = m or 1
    return cfg


def _band(chip: str, start_const: int) -> str:
    """The parser's rule: the chip decides, otherwise the start frequency that makes sense."""
    c = (chip or "").upper()
    if any(k in c for k in ("6843", "68XX")):
        return "60G"
    if any(k in c for k in ("1642", "1843", "1443", "16XX", "18XX", "2544")):
        return "77G"
    f77 = start_const * 3.6 / 2 ** 26
    f60 = start_const * 2.7 / 2 ** 26
    if 74 <= f77 <= 82:
        return "77G"
    if 57 <= f60 <= 64:
        return "60G"
    return "77G"
