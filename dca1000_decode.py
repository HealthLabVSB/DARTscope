"""
dca1000_decode.py

Parses mmWave Studio log files and derives processing parameters used by the GUI and signal pipeline.
Extracts Profile/Chirp/Frame/AdvancedFrame, DataFmt/AdcOut, and LaneConfig; derives start/stop/effective
frequencies, slope, bandwidth, sampling, effective capture window and range resolution, and builds summaries.

Includes:
- Robust chip/band inference (60/77 GHz), numeric LSB conversions.
- DataFmt/AdcOut reconciliation (adcBits, Complex/Real, IQ order, interleave).
- Effective bandwidth and range-bin spacing (with zero‑pad guidance).
- Lane count from LaneConfig for LVDS demux.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
import os
import re
from datetime import datetime
import sys

# NEW: regex helper for LvdsLaneConfig (mmWave Studio specific)
LVDS_LANE_RE = re.compile(r'API:LvdsLaneConfig,([\d,]+)', re.IGNORECASE)


C_LIGHT = 299792458.0   # exact, not 3e8


def compute_start_freqs(start_freq_const: int):
    """Return (freq_60_GHz, freq_77_GHz) candidates from raw startFreqConst."""
    freq_60 = (start_freq_const / (2**26)) * 2.7
    freq_77 = (start_freq_const / (2**26)) * 3.6
    return freq_60, freq_77

# NEW: helper for next power-of-two (for zero-padding bin spacing)
def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p

def find_log_file(bin_path):
    if not bin_path:
        return None
    folder, bin_name = os.path.split(bin_path)
    
    # Robustly find prefix: match ..._Raw_0.bin or ..._0.bin
    match = re.search(r'^(.*?)(?:_Raw)?_(\d+)\.bin$', bin_name)
    if match:
        base = match.group(1)
    else:
        base = os.path.splitext(bin_name)[0]
        # remove _Raw if present at the end of single file
        if base.endswith('_Raw'):
            base = base[:-4]

    log_name = f"{base}_LogFile.txt"
    log_path = os.path.join(folder, log_name)
    if os.path.isfile(log_path):
        return log_path
    
    # Fallback to older logic if re search fails or file not found
    if '_Raw_0.bin' in bin_name:
        base_fallback = bin_name.replace('_Raw_0.bin', '')
        fallback_path = os.path.join(folder, f"{base_fallback}_LogFile.txt")
        if os.path.isfile(fallback_path):
            return fallback_path
            
    return None

def find_raw_log_csv(log_path):
    """The recorder's own _Raw_LogFile.csv that sits next to the API log."""
    if not log_path:
        return None
    for cand in (log_path.replace("_LogFile.txt", "_Raw_LogFile.csv"),
                 log_path.replace("_LogFile.txt", "_Raw_LogFile.CSV")):
        if cand != log_path and os.path.isfile(cand):
            return cand
    return None


_RECORDER_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}

_RECORDER_TIME_RE = re.compile(
    r"[A-Za-z]{3}\s+([A-Za-z]{3})\s+(\d{1,2})\s+(\d{2}):(\d{2}):(\d{2})\s+(\d{4})")


def parse_recorder_time(value):
    """
    Read 'Wed Oct 22 15:06:12 2025' the way the recorder writes it.

    Deliberately not strptime with %a/%b: those follow LC_TIME, and any library that
    calls setlocale would then make this fail on a file that is always written with
    English abbreviations.
    """
    if not value:
        return None
    m = _RECORDER_TIME_RE.search(value)
    if not m:
        return None
    month = _RECORDER_MONTHS.get(m.group(1).lower())
    if not month:
        return None
    try:
        return datetime(int(m.group(6)), month, int(m.group(2)),
                        int(m.group(3)), int(m.group(4)), int(m.group(5)))
    except ValueError:
        return None


def parse_raw_log_csv(csv_path):
    """
    Read the recorder's own log: lane mode, packet counts, capture duration.

    This file was never read, and it is the only place that says what the DCA1000
    itself was doing. Its lane mode describes the width of the container the samples
    are written into, which is not the same thing as the radar's LaneConfig mask - on an
    AWR1642 capture recorded with one active lane into a recorder running in 2-lane
    mode, every second int16 in the file is an empty slot. It also reports lost packets
    first-hand, instead of us inferring them from zeros in the data.
    """
    out = {}
    if not csv_path or not os.path.isfile(csv_path):
        return out
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception as exc:
        print(f"[parse_raw_log_csv] Could not read {csv_path}: {exc}")
        return out

    patterns = {
        "dca_lane_mode": (r"LVDS lane mode\s*:\s*(\d+)\s*lane", int),
        "dca_log_mode": (r"Log mode\s*:\s*([A-Za-z]+)", str),
        "dca_stop_mode": (r"Record stop mode\s*:\s*([A-Za-z]+)", str),
        "dca_packets_received": (r"Number of received packets\s*-\s*(\d+)", int),
        "dca_packets_zero_filled": (r"Number of zero filled packets\s*-\s*(\d+)", int),
        "dca_bytes_zero_filled": (r"Number of zero filled bytes\s*-\s*(\d+)", int),
        "dca_out_of_sequence": (r"Out of sequence count\s*-\s*(\d+)", int),
        "dca_first_packet_id": (r"First Packet ID\s*-\s*(\d+)", int),
        "dca_last_packet_id": (r"Last Packet ID\s*-\s*(\d+)", int),
        "dca_duration_s": (r"Duration\(sec\)\s*-\s*(\d+)", int),
    }
    # When the capture actually began, which is the only wall clock in the whole
    # capture that a reference recording can be lined up against.
    for key, pat in (("dca_capture_start", r"Capture start time\s*-\s*(.+)"),
                     ("dca_capture_end", r"Capture end time\s*-\s*(.+)")):
        m = re.search(pat, text, re.IGNORECASE)
        dt = parse_recorder_time(m.group(1)) if m else None
        if dt:
            out[key] = dt.strftime("%d-%m-%Y %H:%M:%S")
            out[key + "_timestamp"] = dt.timestamp()

    for key, (pat, cast) in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                out[key] = cast(m.group(1))
            except Exception:
                pass

    exp = out.get("dca_last_packet_id")
    got = out.get("dca_packets_received")
    if exp and got is not None and exp > 0:
        out["dca_packet_loss_pct"] = 100.0 * (exp - got) / float(exp)
    out["dca_csv_path"] = csv_path
    return out


# NEW: chip → band mapper
def map_chip_to_band(chip_name: str):
    if not chip_name:
        return None
    c = chip_name.upper()
    if '6843' in c or '68XX' in c or 'IWR6843' in c or 'AWR6843' in c:
        return '60'
    if '1642' in c or '1843' in c or '1443' in c or '16XX' in c or '18XX' in c:
        return '77'
    return None

# NEW: helper for DataFmtConfig parsing
def _decode_data_fmt(values):
    """
    Updated layout per user examples:
      API:DataFmtConfig,rxMask,adcBits,isComplex,iqSwap,_,interleave_flag
    Index meaning now:
      0: rxChanEn mask
      1: adcBits enum (0=12,1=14,2=16)
      2: outFmt (0=Real,1=Complex)
      3: iqSwapSel (0=IQ, 1=QI)
      4: chInterleave (legacy, ignored)
      5: interleave flag (0=non-interleaved, 1=interleaved)
    """
    if len(values) < 6:
        return {}
    rx_mask = values[0]
    adc_bits_enum = values[1]
    out_fmt_simple = values[2]
    iq_swap = values[3]
    interleave_flag = values[5]  # 6th parameter

    adc_bits_map = {0: 12, 1: 14, 2: 16}
    fmt_str = "Complex1X" if out_fmt_simple == 1 else "Real"

    rx_list = [(rx_mask >> i) & 1 for i in range(4)]

    return {
        "dataFmt_rxChanEnMask": rx_mask,
        "dataFmt_rxChannels": " ".join(str(b) for b in rx_list),
        "dataFmt_adcBitsEnum": adc_bits_enum,
        "dataFmt_adcBits": adc_bits_map.get(adc_bits_enum, 16),
        "dataFmt_adcFmtEnum": out_fmt_simple,
        "dataFmt_adcFmt": fmt_str,
        "dataFmt_iqSwapSel": iq_swap,
        "dataFmt_iqOrder": "QI" if iq_swap == 1 else "IQ",
        "dataFmt_chInterleave": interleave_flag,                 # raw flag
        "dataFmt_isInterleaved": (interleave_flag == 1),         # 1=interleaved, 0=non-interleaved
        "dataFmt_reserved": values[4],
    }

# NEW: TX enable decoder
def decode_tx_enable(mask: int):
    """
    Human-readable transmitter list for a ChirpConfig txEnable mask.

    Two things were wrong. Bit 0 was named TX0, but TI numbers transmitters from one, so
    every label was one below the name on the datasheet. And only three bits were read,
    so a four-transmitter mask silently lost one: mask 12 (bits 2 and 3) came out as a
    single "TX2". These names drive the chirp selector, where two chirps differing only
    in the dropped bit would have been indistinguishable.
    """
    if mask is None:
        return ""
    mask = int(mask)
    if mask <= 0:
        return "None"
    txs = [f"TX{i + 1}" for i in range(mask.bit_length()) if (mask >> i) & 1]
    return ",".join(txs) if txs else "None"

def extract_tokens(line, api_prefix, lua_prefix):
    # Try API format first
    if api_prefix in line:
        try:
            payload = line.split(api_prefix, 1)[1]
            payload = payload.strip().strip(',')
            return [tok.strip() for tok in payload.split(',') if tok.strip() != '']
        except Exception:
            pass
    # Try Lua format
    if lua_prefix in line:
        try:
            start = line.find(lua_prefix)
            p_start = line.find('(', start)
            p_end = line.rfind(')')
            if p_start != -1 and p_end != -1 and p_end > p_start:
                payload = line[p_start+1:p_end]
                return [tok.strip() for tok in payload.split(',') if tok.strip() != '']
        except Exception:
            pass
    return None

# NEW regex for Frame / AdvancedFrame
FRAME_RE = re.compile(r'API:FrameConfig,([\d,]+)')
ADV_FRAME_RE = re.compile(r'API:AdvancedFrameConfig,([\d,]+)')

def _score_frame_layout(start_idx, end_idx, chirp_ranges):
    """
    How well a candidate chirp range matches the chirps the log actually defines.

    Returns (covered_fraction, unique_chirps). A frame can only transmit chirps that
    ChirpConfig defined, so a candidate claiming chirps 2..1200 when only 0..2 exist
    is wrong no matter how plausible the field order looks.
    """
    if end_idx < start_idx:
        return -1.0, 0
    n = end_idx - start_idx + 1
    if not chirp_ranges:
        return 0.0, n
    covered = 0
    for c_idx in range(start_idx, end_idx + 1):
        if any(s <= c_idx <= e for s, e in chirp_ranges):
            covered += 1
    return covered / float(n), n


def parse_frame_config_line(line: str, chirp_ranges=None):
    """
    Parse an API:FrameConfig line.

    mmWave Studio logs this line in two shapes that both carry 8 integers:
      A) frameCfgIdx, chirpStartIdx, chirpEndIdx, numLoops, periodicity,
         triggerSelect, numFrames, triggerDelay
      B) chirpStartIdx, chirpEndIdx, numFrames, numLoops, periodicity,
         triggerSelect, <reserved>, triggerDelay

    Assuming A unconditionally silently mangled shape-B logs: a real capture with
    chirps 0..2 was read as chirps 2..1200, which then produced a rotated
    tx_order_masks pattern and therefore a scrambled MIMO virtual array. The layout is
    now chosen by checking the candidate chirp range against the chirps ChirpConfig
    actually defines (chirp_ranges), falling back to the smaller, plausible range.

    Note: framePeriodicity is in 5 ns ticks.
    """
    parts = extract_tokens(line, 'API:FrameConfig,', 'ar1.FrameConfig')
    if not parts or len(parts) < 8:
        # Fallback to old regex
        m = FRAME_RE.search(line)
        if not m:
            return None
        parts = [p for p in m.group(1).split(',') if p != '']
        if len(parts) < 8:
            return None
    try:
        vals = list(map(int, parts[:8]))
    except ValueError:
        return None

    cand_a = {'frameCfgIdx': vals[0], 'chirpStartIdx': vals[1], 'chirpEndIdx': vals[2],
              'numLoops': vals[3], 'numFrames': vals[6]}
    cand_b = {'frameCfgIdx': 0, 'chirpStartIdx': vals[0], 'chirpEndIdx': vals[1],
              'numFrames': vals[2], 'numLoops': vals[3]}

    score_a, n_a = _score_frame_layout(cand_a['chirpStartIdx'], cand_a['chirpEndIdx'], chirp_ranges)
    score_b, n_b = _score_frame_layout(cand_b['chirpStartIdx'], cand_b['chirpEndIdx'], chirp_ranges)

    if chirp_ranges:
        chosen = cand_a if score_a >= score_b else cand_b
        if score_a < 1.0 and score_b < 1.0:
            print(f"[parse_frame_config_line] WARNING: neither FrameConfig layout fully "
                  f"matches the defined chirps (coverage {score_a:.2f} vs {score_b:.2f}); "
                  f"using chirps {chosen['chirpStartIdx']}..{chosen['chirpEndIdx']}.")
    else:
        # No ChirpConfig to check against: prefer the layout with a sane chirp count.
        a_ok = 1 <= n_a <= 64
        b_ok = 1 <= n_b <= 64
        chosen = cand_a if (a_ok or not b_ok) else cand_b

    frameCfgIdx = chosen['frameCfgIdx']
    chirpStartIdx = chosen['chirpStartIdx']
    chirpEndIdx = chosen['chirpEndIdx']
    numLoops = chosen['numLoops']
    numFrames = chosen['numFrames']
    framePeriodicity_ticks = vals[4]
    triggerSelect = vals[5]
    triggerDelay = vals[7]

    # Convert from 5ns ticks to microseconds
    framePeriodicity_us = framePeriodicity_ticks * 5.0 / 1000.0
    framePeriodicity_ms = framePeriodicity_ticks * 5.0 / 1000000.0

    uniqueChirps = (chirpEndIdx - chirpStartIdx + 1) if chirpEndIdx >= chirpStartIdx else None
    chirpsPerFrame = (uniqueChirps * numLoops) if (uniqueChirps is not None) else None
    return {
        "frameCfgIdx": frameCfgIdx,
        "chirpStartIdx": chirpStartIdx,
        "chirpEndIdx": chirpEndIdx,
        "numLoops": numLoops,
        "framePeriodicity_ticks": framePeriodicity_ticks, # Store raw value
        "framePeriodicity_us": framePeriodicity_us,
        "framePeriodicity_ms": framePeriodicity_us / 1000.0,
        "triggerSelect": triggerSelect,
        "numFrames": numFrames,
        "triggerDelay": triggerDelay,
        "uniqueChirps": uniqueChirps,
        "chirpsPerFrame": chirpsPerFrame
    }

def parse_advanced_frame_config_line(line: str):
    """
    Stores raw advanced frame config. We extract a few heuristic fields:
      adv_frameHeader (first int),
      adv_numSubFrames (5th int index 4 if exists),
      possible per-subframe periodicities (we pick indices where value seems large like >= 1000).
    Example (shortened):
      API:AdvancedFrameConfig,1,0,0,0,2,255,8000000,0,1,1,8000000,...
    """
    parts = extract_tokens(line, 'API:AdvancedFrameConfig,', 'ar1.AdvancedFrameConfig')
    if not parts:
        m = ADV_FRAME_RE.search(line)
        if not m:
            return None
        parts = [p for p in m.group(1).split(',') if p != '']
    ints = []
    for p in parts:
        try:
            ints.append(int(p))
        except ValueError:
            ints.append(None)
    if len(ints) < 5:
        return None
    adv = {
        "adv_raw": ints,
        "adv_frameHeader": ints[0],
        "adv_numSubFrames": ints[4] if len(ints) > 4 else None
    }
    # Heuristically select periods (values ~millions of us or ticks)
    # If the value is very large (e.g. 2,000,000), it's likely ticks or us.
    # User's case: 2,000,000 ticks = 10ms. 2,000,000 us = 2000ms.
    # We should distinguish them. Usually, if it's > 500,000 and the user says it's 10ms, it's ticks.
    # But wait, 2,000,000 us is also a valid period (2s).
    # However, mmWave Studio often logs periodicity in 5ns ticks in both configs.
    periodicities_raw = [v for v in ints if isinstance(v, int) and v and v >= 1000]
    
    # NEW logic: Convert periodicities. If we suspect they are ticks (e.g. they match FrameConfig ticks),
    # we convert them from 5ns ticks to us. 
    # For now, let's assume they might be ticks and provide both, or prefer ticks if very large.
    # Actually, the user's log shows periodicities_us=[2000000, 2000000]. 
    # If we treat 2000000 as us, it's 2000ms. If we treat it as 5ns ticks, it's 10ms.
    # Since FrameConfig has framePeriodicity_ticks=2000000, it's almost certain AdvancedFrameConfig also uses ticks.
    
    adv["adv_periodicities_ticks"] = periodicities_raw
    periodicities_us = [v * 5.0 / 1000.0 for v in periodicities_raw]
    adv["adv_periodicities_us"] = periodicities_us
    
    if periodicities_us:
        adv["adv_periodicity_1_us"] = periodicities_us[0]
    if len(periodicities_us) >= 2:
        adv["adv_periodicity_2_us"] = periodicities_us[1]
    return adv

def _start_var_to_GHz(start_freq_var_raw: int, band: str) -> float:
    if not isinstance(start_freq_var_raw, int) or start_freq_var_raw == 0:
        return 0.0
    mult = 3.6 if band == '77' else 2.7
    return (start_freq_var_raw / (2**26)) * mult

def find_repeating_period(masks: list) -> int:
    if not masks:
        return 1
    n = len(masks)
    for p in range(1, n + 1):
        match = True
        for i in range(n):
            if masks[i] != masks[i % p]:
                match = False
                break
        if match:
            return p
    return n

def _compute_alt_range_resolutions(params, window_name: str = 'hann'):
    """
    Alternative effective range resolution estimates:
    1. from effective bandwidth (already computed)
    2. from fs/N_eff (sampling rate / effective samples)
    3. window-adjusted resolution
    """
    out = {}
    # 1. from effective bandwidth
    eff_bw_ghz = params.get('effectiveBandwidthGHz')
    if eff_bw_ghz and eff_bw_ghz > 0:
        res_m = 3e8 / (2 * eff_bw_ghz * 1e9)
        out['rangeRes_alt_from_BWeff_m'] = res_m
        out['rangeRes_alt_from_BWeff_cm'] = res_m * 100.0
    # 2. from fs/N_eff (only meaningful for complex data; skip for REAL)
    fs_mhz = params.get('digOutSampleRate_MHz')
    slope_mhz_us = params.get('freqSlopeMHz_us')
    eff_cap_us = params.get('effCaptureWindow_us')
    adc_fmt = str(params.get('dataFmt_adcFmt', '')).upper()
    # Only compute this heuristic for complex formats; for REAL keep it undefined
    if fs_mhz and slope_mhz_us and eff_cap_us and ('COMPLEX' in adc_fmt):
        N_eff = fs_mhz * eff_cap_us
        out['N_effective_used'] = N_eff
        res_m = (3e8 * fs_mhz) / (2 * slope_mhz_us * N_eff) if N_eff > 0 else None
        if res_m:
            out['rangeRes_alt_from_fsN_m'] = res_m
            out['rangeRes_alt_from_fsN_cm'] = res_m * 100.0
    # 3. window-adjusted
    # Window factors for -3dB resolution broadening
    win_factors = {'hann': 1.62, 'hamming': 1.4, 'blackman': 1.97}
    factor = win_factors.get(window_name.lower(), 1.0)
    base_res_cm = out.get('rangeRes_alt_from_BWeff_cm')
    if base_res_cm:
        out['rangeRes_window_factor'] = factor
        out['rangeRes_windowed_cm'] = base_res_cm * factor
    return out


def _merge_recorder_log(params, log_path):
    """
    Fold in the recorder's own log and let it decide the lane count.

    The recorder's lane mode is the width of the container the samples land in, while
    LaneConfig is how many lanes the radar drives. Only the first of those describes the
    file on disk, so it wins: numLanes becomes the recorder's value and the radar's stays
    available as numLanes_radar. When the recorder reserves more than the radar fills,
    the surplus slots may stay empty, and a naive read treats them as samples - which
    halves the range scale and adds a mirror image.
    """
    csv_path = find_raw_log_csv(log_path)
    info = parse_raw_log_csv(csv_path)
    if not info:
        return
    params.update(info)

    dca_lanes = info.get("dca_lane_mode")
    radar_lanes = params.get("numLanes")
    params["numLanes_radar"] = radar_lanes

    if dca_lanes in (1, 2, 4):
        # The demux has to follow the recorder, not the radar. LaneConfig says how many
        # lanes the radar drives; the recorder's lane mode is the width of the container
        # the samples are actually written into, and that is what determines how the
        # int16 stream has to be cut apart. Taking LaneConfig instead decodes the file
        # into the wrong shape and, because the byte count still works out, it does so
        # without raising anything.
        if radar_lanes and int(dca_lanes) != int(radar_lanes):
            params["lane_mode_mismatch"] = True
            print(f"[parse_logfile_full] NOTE: the radar drives {radar_lanes} LVDS "
                  f"lane(s) (LaneConfig) but the recorder ran in {dca_lanes}-lane mode. "
                  f"Decoding with {dca_lanes} lanes, which is the container the samples "
                  f"were written into. Any slots the radar left unfilled are dropped on "
                  f"load when they are systematically zero.")
        params["numLanes"] = int(dca_lanes)
        params["numLanes_source"] = "DCA1000 recorder log"
    elif dca_lanes:
        print(f"[parse_logfile_full] NOTE: the recorder log reports an unusable lane "
              f"mode ({dca_lanes}); keeping {radar_lanes} from the radar configuration.")

    # The capture's wall clock, which is what a reference recording has to be lined up
    # against. The API log only says when the session's first command was issued, and
    # because mmWave Studio appends every capture to the same log that can be many
    # minutes before the data starts - on the vitals set here, between 37 s and 28 min.
    cap_start = info.get("dca_capture_start_timestamp")
    if cap_start:
        log_start = params.get("radar_start_timestamp")
        params["radar_start_timestamp_log"] = log_start
        if log_start is not None:
            drift = cap_start - float(log_start)
            if abs(drift) > 2.0:
                params["capture_start_vs_log_s"] = drift
                print(f"[parse_logfile_full] NOTE: the API log's first timestamp is "
                      f"{drift:+.0f} s from the recorder's capture start. Using the "
                      f"recorder's {info['dca_capture_start']}; syncing a reference "
                      f"recording against the log would have been off by that much.")
        params["radar_start_timestamp"] = cap_start
        params["radar_start_time"] = info["dca_capture_start"]
        params["radar_start_source"] = "DCA1000 recorder log"

    lost = info.get("dca_packets_zero_filled")
    if lost:
        pct = info.get("dca_packet_loss_pct")
        pct_txt = f" ({pct:.1f} % of the stream)" if pct else ""
        print(f"[parse_logfile_full] NOTE: the recorder zero-filled {lost} packets"
              f"{pct_txt} and saw {info.get('dca_out_of_sequence', 0)} out of sequence.")


def _fill_velocity_and_range_limits(p):
    """
    Doppler and range limits, once the frame configuration is known.

    velocityRes_m_s used to be lambda / (2 * rampEndTime), which is neither the
    resolution nor the maximum: it leaves out the number of Doppler bins and uses the
    ramp time instead of the chirp repetition time. Both IWR6843 captures came out as
    the same 62.2 m/s although one has 32 Doppler bins and the other 5, and the AWR1642
    capture read 64.9 m/s where the real figure is 0.074 - a factor of 874.

      Tc    = (idleTime + rampEndTime) * (number of TX in the TDM pattern)
      v_max = lambda / (4 * Tc)
      v_res = lambda / (2 * numLoops * Tc)

    Range limits come from the beat frequency the sampling rate can still carry; real
    sampling only resolves half the spectrum, so its maximum range is half.
    """
    try:
        f_eff = p.get('effectiveStartFreqGHz') or p.get('startFreqGHz')
        idle = p.get('idleTime_us')
        ramp = p.get('rampEndTime_us')
        slope = p.get('freqSlopeMHz_us')
        fs_msps = p.get('digOutSampleRate_MHz')
        if f_eff and idle is not None and ramp is not None:
            lam = C_LIGHT / (float(f_eff) * 1e9)
            if slope and fs_msps and p.get('numAdcSamples'):
                # middle of the ADC window, as capture_params.wavelength_m (roadmap 1.9)
                lam = C_LIGHT / (float(f_eff) * 1e9 + float(slope) * 1e12
                                 * (int(p['numAdcSamples']) - 1) / (2.0 * float(fs_msps) * 1e6))
            p['wavelength_mm'] = lam * 1000.0
            n_tx = len(p.get('tx_order_masks') or []) or 1
            tc = (float(idle) + float(ramp)) * 1e-6 * n_tx
            loops = (p.get('frameConfig') or {}).get('numLoops') or 1
            if tc > 0:
                p['chirpRepetitionTime_us'] = tc * 1e6
                p['maxVelocity_m_s'] = lam / (4.0 * tc)
                p['velocityRes_m_s'] = lam / (2.0 * max(1, int(loops)) * tc)
                p['dopplerBins'] = int(loops)
        if slope and fs_msps:
            r_max = (float(fs_msps) * 1e6) * C_LIGHT / (2.0 * float(slope) * 1e12)
            is_real = 'REAL' in str(p.get('dataFmt_adcFmt', '')).upper()
            p['maxRange_m'] = r_max / 2.0 if is_real else r_max
            p['maxRange_note'] = 'real sampling: half spectrum' if is_real else 'complex sampling'
    except Exception as exc:
        print(f"[parse_logfile_full] Could not derive velocity/range limits: {exc}")


def parse_logfile_full(log_path, print_summary=False):
    params = {}
    chirp_configs = []
    band = None
    ts_re = re.compile(r'^(\d{2}-[A-Za-z]{3}-\d{4} \d{2}:\d{2}:\d{2}):')
    profile_entries = []
    data_fmt_parsed = {}
    adc_out_parsed = {} # NEW: Re-added to store for printing
    chirps_dict = {}
    frame_cfg = None
    frame_cfg_line = None
    adv_frame_cfg = None
    lane_cfg_parsed = {} # NEW: for LaneConfig
    lvds_lane_cfg = {}   # NEW: for LvdsLaneConfig (fallback/confirmation)

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Chip hint: scan the (already loaded) lines instead of opening and reading the
    # whole log a second time. The pattern accepts both spellings that occur in real
    # mmWave Studio logs - select_chip_version and SelectChipVersion - and always
    # keeps the text as logged, so a variant suffix such as "AOP" survives. Losing it
    # here made AOA._get_antenna_geometry() fall back to guessing the antenna layout.
    chip_hint = None
    chip_hint_band = None
    chip_line_pattern = re.compile(r'API:\s*Select_?Chip_?Version\s*,\s*([^,\r\n]+)', re.IGNORECASE)
    for ln in lines:
        m = chip_line_pattern.search(ln)
        if m:
            candidate = m.group(1).strip()
            if not candidate:
                continue
            chip_hint = candidate
            mapped = map_chip_to_band(chip_hint)
            if mapped:
                chip_hint_band = mapped
                break  # first fully resolved hint wins
    if chip_hint and not chip_hint_band:
        print(f"[parse_logfile_full] WARNING: chip '{chip_hint}' not recognised by "
              f"map_chip_to_band(); band will be guessed from the start frequency.")

    # The first timestamp in the API log is when the session's first command was
    # issued, not when the capture began - mmWave Studio keeps appending to one log,
    # so on a log covering several captures this can be tens of minutes early. It is
    # only a fallback; _merge_recorder_log replaces it with the recorder's own capture
    # start when the CSV is there. Kept apart from the selected profile's timestamp.
    log_timestamps = []
    for raw_line in lines:
        match = ts_re.match(raw_line.strip())
        if match:
            try:
                log_timestamps.append(datetime.strptime(match.group(1), "%d-%b-%Y %H:%M:%S"))
            except ValueError:
                pass
    if log_timestamps:
        radar_start_dt = min(log_timestamps)
        params['radar_start_timestamp'] = radar_start_dt.timestamp()
        params['radar_start_time'] = radar_start_dt.strftime("%d-%m-%Y %H:%M:%S")
        params['radar_start_source'] = "API log (first command)"
    params['log_path'] = log_path

    lines = reversed(lines)
    for line in lines:
        line_stripped = line.strip()
        m_ts = ts_re.match(line_stripped)
        current_ts_str = m_ts.group(1) if m_ts else None

        # Fallback if the scan above found no usable chip line: derive only the BAND
        # from the raw line and never overwrite chip_hint with a generic literal.
        # Doing so used to discard variant suffixes (e.g. the "AOP" in IWR6843AOP),
        # which downstream decides the antenna geometry from.
        if ('select_chip_version' in line.lower() or 'selectchipversion' in line.lower()) and not chip_hint_band:
            line_up = line.upper()
            if 'AR1642' in line_up or 'AWR1642' in line_up:
                chip_hint_band = '77'
            elif 'IWR6843' in line_up or 'AWR6843' in line_up:
                chip_hint_band = '60'
        if 'API:ProfileConfig' in line or 'ar1.ProfileConfig' in line:
            tokens = extract_tokens(line, 'API:ProfileConfig,', 'ar1.ProfileConfig')
            if not tokens:
                continue
            raw_int = []
            for t in tokens:
                try:
                    raw_int.append(int(t))
                except ValueError:
                    raw_int.append(None)
            if len(raw_int) < 11:
                continue
            p = {}
            p['profileId'] = raw_int[0]
            p['startFreqConst'] = raw_int[1]
            p['idleTimeConst'] = raw_int[2]
            p['adcStartTimeConst'] = raw_int[3]
            p['rampEndTime'] = raw_int[4]
            p['txOutPowerBackoffCode'] = raw_int[5]
            p['txPhaseShifter'] = raw_int[6]
            p['freqSlopeConst'] = raw_int[7]
            p['txStartTimeConst'] = raw_int[8]
            p['numAdcSamples'] = raw_int[9]
            p['digOutSampleRate'] = raw_int[10]
            p['hpfCornerFreq1'] = raw_int[11] if len(raw_int) > 11 else None
            p['hpfCornerFreq2'] = raw_int[12] if len(raw_int) > 12 else None
            p['rxGain'] = raw_int[13] if len(raw_int) > 13 else None
            p['reserved'] = raw_int[14] if len(raw_int) > 14 else None
            p['profile_tokens'] = raw_int
            freq_60_cand, freq_77_cand = compute_start_freqs(p['startFreqConst'])
            if chip_hint_band:
                band = chip_hint_band
            else:
                if 74 <= freq_77_cand <= 82:
                    band = '77'
                elif 57 <= freq_60_cand <= 64:
                    band = '60'
                else:
                    band = '77'
            p['band'] = band
            p['chip_hint'] = chip_hint
            p['chipHintBand'] = chip_hint_band
            p['startFreqGHz_60_candidate'] = freq_60_cand
            p['startFreqGHz_77_candidate'] = freq_77_cand
            p['startFreqGHz'] = freq_77_cand if band == '77' else freq_60_cand
            p['startFreqGHz_formula'] = f"({p['startFreqConst']} / 2^26) * {'3.6' if band=='77' else '2.7'}"
            p['idleTime_us'] = p['idleTimeConst'] / 100.0 if p['idleTimeConst'] is not None else None
            p['adcStartTime_us'] = p['adcStartTimeConst'] / 100.0 if p['adcStartTimeConst'] is not None else None
            p['txStartTime_us'] = p['txStartTimeConst'] / 100.0 if p['txStartTimeConst'] is not None else None
            p['rampEndTime_us'] = p['rampEndTime'] / 100.0 if p['rampEndTime'] is not None else None
            const_val = p['freqSlopeConst']
            if const_val is not None:
                scale_77 = 0.048279762268
                scale_60 = 0.036209821701
                p['freqSlopeMHz_us_77_candidate'] = const_val * scale_77
                p['freqSlopeMHz_us_60_candidate'] = const_val * scale_60
                if band == '77':
                    p['freqSlopeScale_used'] = scale_77
                    p['freqSlopeMHz_us'] = p['freqSlopeMHz_us_77_candidate']
                else:
                    p['freqSlopeScale_used'] = scale_60
                    p['freqSlopeMHz_us'] = p['freqSlopeMHz_us_60_candidate']
                p['freqSlopeIndex'] = 7
            else:
                p['freqSlopeMHz_us'] = 0.0
                p['freqSlopeScale_used'] = None
                p['freqSlopeIndex'] = None
            if p.get('startFreqGHz') and p.get('freqSlopeMHz_us') and p.get('rampEndTime_us'):
                bw_GHz = p['freqSlopeMHz_us'] * p['rampEndTime_us'] * 1e-3
                p['bandwidthGHz'] = bw_GHz
                p['stopFreqGHz'] = p['startFreqGHz'] + bw_GHz
                p['rangeRes_m'] = (3e8 / (2 * bw_GHz * 1e9)) if bw_GHz > 0 else None
                p['rangeRes_cm'] = (p['rangeRes_m'] * 100.0) if p['rangeRes_m'] else None
            else:
                p['bandwidthGHz'] = None
                p['stopFreqGHz'] = None
                p['rangeRes_m'] = None
                p['rangeRes_cm'] = None
            # Velocity limits are filled in after the frame configuration is known -
            # they depend on the chirp repetition time and the number of Doppler bins,
            # neither of which lives in ProfileConfig.
            p['velocityRes_m_s'] = None
            p['maxVelocity_m_s'] = None
            p['digOutSampleRate_MHz'] = p['digOutSampleRate'] / 1000.0 if p.get('digOutSampleRate') else None
            p['timestamp'] = current_ts_str
            try:
                p['timestamp_obj'] = datetime.strptime(current_ts_str, "%d-%b-%Y %H:%M:%S") if current_ts_str else None
            except Exception:
                p['timestamp_obj'] = None
            # NEW: Effective capture window and bandwidth (profile-level)
            sample_time_us = None
            if p.get('numAdcSamples') and p.get('digOutSampleRate_MHz'):
                if p['digOutSampleRate_MHz'] > 0:
                    sample_time_us = p['numAdcSamples'] / p['digOutSampleRate_MHz']
            ramp_window_us = None
            if p.get('rampEndTime_us') is not None:
                adc_start = p.get('adcStartTime_us') or 0.0
                ramp_window_us = max(0.0, p['rampEndTime_us'] - adc_start)
            eff_capture_us = None
            if ramp_window_us is not None:
                eff_capture_us = min(ramp_window_us, sample_time_us) if sample_time_us else ramp_window_us
            eff_bw_mhz = None
            eff_bw_ghz = None
            eff_range_res_m = None
            eff_range_res_cm = None
            if (p.get('freqSlopeMHz_us') is not None) and (eff_capture_us is not None):
                eff_bw_mhz = p['freqSlopeMHz_us'] * eff_capture_us
                eff_bw_ghz = eff_bw_mhz / 1000.0
                if eff_bw_ghz and eff_bw_ghz > 0:
                    eff_range_res_m = 3e8 / (2 * eff_bw_ghz * 1e9)
                    eff_range_res_cm = eff_range_res_m * 100.0
            p['adcSampleTime_us'] = sample_time_us
            p['rampWindow_us'] = ramp_window_us
            p['effCaptureWindow_us'] = eff_capture_us
            p['effectiveBandwidthMHz'] = eff_bw_mhz
            p['effectiveBandwidthGHz'] = eff_bw_ghz
            p['effectiveRangeRes_m'] = eff_range_res_m
            p['effectiveRangeRes_cm'] = eff_range_res_cm

            # NEW: Effective start frequency at ADC start (profile-level)
            try:
                adc_us = p.get('adcStartTime_us') or 0.0
                slope_mhz_us = p.get('freqSlopeMHz_us')
                if isinstance(slope_mhz_us, (int, float)) and isinstance(adc_us, (int, float)):
                    p['effectiveStartFreqGHz'] = (p.get('startFreqGHz') or 0.0) + (slope_mhz_us * adc_us) / 1000.0
                else:
                    p['effectiveStartFreqGHz'] = None
            except Exception:
                p['effectiveStartFreqGHz'] = None

            profile_entries.append(p)
        elif 'API:DataPathConfig' in line:
            v = re.findall(r'[-+]?\d*\.\d+|\d+', line.split('API:DataPathConfig,',1)[1])
            if len(v) >= 5:
                params['transferFmtPkt0'] = 'ADC_DATA_ONLY' if v[0] == '1' else v[0]
                params['transferFmtPkt1'] = 'Suppress_packet' if v[1] == '1' else v[1]
                params['cqConfig'] = int(v[3]) if len(v) > 3 else None
        elif 'API:DataFmtConfig' in line and not data_fmt_parsed:
            # Parse raw numeric tokens, only take the first one found (last in file)
            try:
                payload = line.split('API:DataFmtConfig,', 1)[1]
            except IndexError:
                payload = ""
            nums = re.findall(r'\d+', payload)
            nums_int = []
            for n in nums:
                try:
                    nums_int.append(int(n))
                except ValueError:
                    pass
            # Use the new decoder
            if len(nums_int) >= 6:
                decoded = _decode_data_fmt(nums_int)
                if decoded:
                    data_fmt_parsed = decoded
                    data_fmt_parsed['raw_values'] = nums_int # Store raw values
                    data_fmt_parsed['timestamp'] = current_ts_str # Store timestamp
        elif 'API:AdcOutConfig' in line and not adc_out_parsed:
            # This is now parsed for logic, with priority. Only take the first one found (last in file).
            try:
                payload = line.split('API:AdcOutConfig,', 1)[1]
                nums = [int(t) for t in payload.strip().split(',') if t != '']
                if len(nums) >= 3:
                    adc_out_parsed['raw_values'] = nums
                    adc_out_parsed['timestamp'] = current_ts_str # Store timestamp
            except (IndexError, ValueError):
                pass
        elif 'API:ChirpConfig' in line or 'ar1.ChirpConfig' in line:
            # NEW: full chirp parsing (with txActive)
            tokens = extract_tokens(line, 'API:ChirpConfig,', 'ar1.ChirpConfig')
            if not tokens:
                continue
            ints = []
            for t in tokens:
                try:
                    ints.append(int(t))
                except ValueError:
                    ints.append(None)
            if len(ints) < 8:
                continue
            chirp = {
                "chirpStartIdx": ints[0],
                "chirpEndIdx": ints[1],
                "profileId": ints[2],
                "startFreqVar": ints[3],
                "freqSlopeVar": ints[4],
                "idleTimeVar": ints[5],
                "adcStartTimeVar": ints[6],
                "txEnable": ints[7],
                "bpfSel": ints[8] if len(ints) > 8 else None
            }
            chirp["txActive"] = decode_tx_enable(chirp["txEnable"])  # NEW
            key = (chirp["chirpStartIdx"], chirp["chirpEndIdx"], chirp["profileId"])
            # Because we read reversed, keep first occurrence (last in original file)
            if key not in chirps_dict:
                chirps_dict[key] = chirp
            # Keep startIdx list for legacy txEnable string
            if ints and ints[0] is not None:
                chirp_configs.append(str(ints[0]))
        elif ('API:FrameConfig' in line or 'ar1.FrameConfig' in line) and frame_cfg is None:
            fc = parse_frame_config_line(line)
            if fc:
                fc['timestamp'] = current_ts_str
                frame_cfg = fc
                frame_cfg_line = line
        elif ('API:AdvancedFrameConfig' in line or 'ar1.AdvancedFrameConfig' in line) and adv_frame_cfg is None:
            afc = parse_advanced_frame_config_line(line)
            if afc:
                afc['timestamp'] = current_ts_str
                adv_frame_cfg = afc
        # NEW: LaneConfig (additional hint for lane count)
        elif 'API:LaneConfig' in line and not lane_cfg_parsed:
            # Parse LaneConfig for LVDS lanes (first number = binary lane mask)
            try:
                payload = line.split('API:LaneConfig,', 1)[1]
                nums = [int(t) for t in payload.strip().split(',') if t != '']
                if len(nums) >= 1:
                    lane_mask = nums[0]
                    lane_cfg_parsed['laneMask'] = lane_mask
                    # Active lanes decoded from bit mask (1-based labels: lane1, lane2, ...)
                    active_idx = [i + 1 for i in range(4) if ((lane_mask >> i) & 1)]
                    lane_cfg_parsed['laneMask_bin'] = bin(lane_mask)
                    lane_cfg_parsed['lanesActive_idx'] = active_idx
                    lane_cfg_parsed['lanesActive'] = ",".join(f"lane{i}" for i in active_idx) if active_idx else "None"
                    # Popcount as lane count
                    pc = bin(lane_mask).count('1')
                    lane_cfg_parsed['numLanes_popcount'] = pc
                    # ACCEPT 1..4 (LaneConfig is a BITMASK)
                    lane_cfg_parsed['numLanes'] = pc if (1 <= pc <= 4) else None
                    lane_cfg_parsed['timestamp'] = current_ts_str
            except (IndexError, ValueError):
                pass
        # NEW: antGeometry
        elif 'API:antGeometry' in line:
            try:
                payload = line.split('API:antGeometry', 1)[1]
                # Format: API:antGeometry0,val0,val1,... or API:antGeometry1,...
                m = re.match(r'(\d),(.+)', payload)
                if m:
                    idx = int(m.group(1))
                    vals = [float(t) for t in m.group(2).strip().split(',') if t != '']
                    params[f'antGeometry{idx}'] = vals
            except Exception:
                pass
        # NEW: compRangeBiasAndRxChanPhase
        elif 'API:compRangeBiasAndRxChanPhase' in line:
            try:
                payload = line.split('API:compRangeBiasAndRxChanPhase,', 1)[1]
                vals = [float(t) for t in payload.strip().split(',') if t != '']
                params['compRangeBiasAndRxChanPhase'] = vals
            except Exception:
                pass
        # NEW: antPhaseRot
        elif 'API:antPhaseRot' in line:
            try:
                payload = line.split('API:antPhaseRot,', 1)[1]
                vals = [float(t) for t in payload.strip().split(',') if t != '']
                params['antPhaseRot'] = vals
            except Exception:
                pass
        # NEW: LvdsLaneConfig (additional hint for lane count)
        elif LVDS_LANE_RE.search(line) and not lvds_lane_cfg:
            try:
                m = LVDS_LANE_RE.search(line)
                parts = [p for p in m.group(1).split(',') if p != '']
                vals = list(map(int, parts))
                lvds_lane_cfg['lvdsLaneConfig_raw'] = vals
                lvds_lane_cfg['timestamp'] = current_ts_str
                # Mapping:
                #   code in {1,2,4} → that many lanes
                #   code == 3 → 4 lanes (common alias)
                def _derive_num_lanes_from_lvds(vs):
                    if not vs or len(vs) < 2:
                        return None
                    code = int(vs[1])
                    if code in (1, 2, 4):
                        return code
                    if code == 3:
                        return 4
                    return None
                lvds_lane_cfg['numLanes_from_lvdsLaneConfig'] = _derive_num_lanes_from_lvds(vals)
            except Exception:
                pass


    # Final selections
    if profile_entries:
        selected = profile_entries[0]
        params.update(selected)
        params['profilesFound'] = len(profile_entries)

    # Combine DataFmtConfig and AdcOutConfig, with AdcOutConfig having priority
    if data_fmt_parsed:
        params.update(data_fmt_parsed)
        params['dataFmtConfig_timestamp'] = data_fmt_parsed.get('timestamp')
    if adc_out_parsed:
        raw = adc_out_parsed.get('raw_values')
        if raw and len(raw) >= 3:
            adc_bits_map = {0: 12, 1: 14, 2: 16}
            params['dataFmt_adcBits'] = adc_bits_map.get(raw[0], 16)
            params['dataFmt_adcBitsEnum'] = raw[0]

            # UPDATED: Use detailed format enum from AdcOutConfig
            adc_fmt_map = {0: "Real", 1: "Complex1X", 2: "Complex2X", 3: "PseudoReal"}
            fmt_enum = raw[1]
            params['dataFmt_adcFmt'] = adc_fmt_map.get(fmt_enum, f"UNK({fmt_enum})")
            params['dataFmt_adcFmtEnum'] = fmt_enum

            # Mapping: 1=interleaved, 0=non-interleaved
            is_interleaved = (raw[2] == 1)
            params['dataFmt_isInterleaved'] = is_interleaved
            params['dataFmt_chInterleave'] = raw[2]

            params['adcOutConfig_raw'] = raw
            params['adcOutConfig_timestamp'] = adc_out_parsed.get('timestamp')

    # NEW: Complex2X – odvoď přímo textový hint "IQIQ"/"QIQI" z iqSwap
    try:
        if str(params.get('dataFmt_adcFmt', '')).upper() == 'COMPLEX2X':
            iq_pref = str(params.get('dataFmt_iqOrder') or '').upper()  # "IQ" / "QI"
            if iq_pref in ('IQ', 'QI'):
                params['dataFmt_iqOrder2x'] = 'IQIQ' if iq_pref == 'IQ' else 'QIQI'
    except Exception:
        pass

    if frame_cfg:
        params['frameConfig'] = frame_cfg
        # NEW: extract periodicity for easier access in other modules
        if 'framePeriodicity_ms' in frame_cfg:
            params['framePeriodicity_ms_selected'] = frame_cfg['framePeriodicity_ms']
    if adv_frame_cfg:
        params['advancedFrameConfig'] = adv_frame_cfg
        # NEW: extract first subframe periodicity as default for Advanced Frame
        if 'adv_periodicity_1_us' in adv_frame_cfg:
            params['framePeriodicity_ms_selected'] = adv_frame_cfg['adv_periodicity_1_us'] / 1000.0

    chirps_list = list(chirps_dict.values())
    chirps_list.sort(key=lambda c: (c["chirpStartIdx"], c["chirpEndIdx"]))
    params['chirps'] = chirps_list

    # The FrameConfig field layout is ambiguous on its own, so resolve it now that the
    # chirps defined by ChirpConfig are known - the frame can only use those.
    if frame_cfg_line is not None:
        chirp_ranges = [(c.get('chirpStartIdx'), c.get('chirpEndIdx')) for c in chirps_list
                        if isinstance(c.get('chirpStartIdx'), int) and isinstance(c.get('chirpEndIdx'), int)]
        resolved = parse_frame_config_line(frame_cfg_line, chirp_ranges=chirp_ranges)
        if resolved:
            resolved['timestamp'] = (frame_cfg or {}).get('timestamp')
            if frame_cfg and (resolved['chirpStartIdx'], resolved['chirpEndIdx']) != \
                    (frame_cfg.get('chirpStartIdx'), frame_cfg.get('chirpEndIdx')):
                print(f"[parse_logfile_full] FrameConfig re-read against the defined chirps: "
                      f"chirps {resolved['chirpStartIdx']}..{resolved['chirpEndIdx']}, "
                      f"{resolved['numLoops']} loops, {resolved['numFrames']} frames.")
            frame_cfg = resolved
            # The re-read layout has to reach params too. It used to stay in this local
            # only: the TX order below followed it while derive_capture_dimensions kept the
            # first guess, which takes layout A whenever numFrames is small. vitals/data_test4
            # (50 frames) came out as 512 frames of chirps 0..50, and a short TDM recording
            # with chirps 0..1 as one chirp per loop - a cube of the wrong shape.
            params['frameConfig'] = frame_cfg
    params['txEnable'] = ' '.join(chirp_configs)

    # NEW: derive TX order (pattern) directly from ChirpConfig ranges (expanded in defined order)
    try:
        tx_order_masks = []
        tx_order_names = []
        positions_by_mask = {}  # mask(int) -> [indices]
        positions_by_name = {}  # "TX0" -> [indices]
        pos = 0

        # Check if we can get active range from frame_cfg or adv_frame_cfg
        start_idx = None
        end_idx = None
        if frame_cfg:
            start_idx = frame_cfg.get('chirpStartIdx')
            end_idx = frame_cfg.get('chirpEndIdx')

        if start_idx is not None and end_idx is not None:
            # We construct tx_order_masks for this range!
            for c_idx in range(start_idx, end_idx + 1):
                # Find matching chirp in chirps_list
                match_chirp = None
                for ch in chirps_list:
                    s = ch.get('chirpStartIdx')
                    e = ch.get('chirpEndIdx')
                    if s is not None and e is not None and s <= c_idx <= e:
                        match_chirp = ch
                        break
                
                # If found, use its txEnable. Otherwise, default mask based on index (e.g. 1 << (c_idx % 3))
                if match_chirp is not None:
                    m = int(match_chirp.get('txEnable') or 0)
                else:
                    # Fallback mask: 1 << (c_idx % 3)
                    m = 1 << (c_idx % 3)
                
                name = decode_tx_enable(m)
                tx_order_masks.append(m)
                tx_order_names.append(name)
                if m != 0:
                    positions_by_mask.setdefault(m, []).append(pos)
                    positions_by_name.setdefault(name, []).append(pos)
                pos += 1
            
            eff_unique = find_repeating_period(tx_order_masks)
            params['unique_chirps_frame'] = eff_unique
            params['uniqueChirps_effective'] = eff_unique
        else:
            for ch in chirps_list:
                s = ch.get('chirpStartIdx')
                e = ch.get('chirpEndIdx')
                m = int(ch.get('txEnable') or 0)
                if isinstance(s, int) and isinstance(e, int) and e >= s:
                    n = e - s + 1
                else:
                    n = 0
                if n > 0:
                    name = decode_tx_enable(m)
                    for _ in range(n):
                        tx_order_masks.append(m)
                        tx_order_names.append(name)
                        if m != 0:
                            positions_by_mask.setdefault(m, []).append(pos)
                            positions_by_name.setdefault(name, []).append(pos)
                        pos += 1
            
            eff_unique = find_repeating_period(tx_order_masks)
            if eff_unique > 0:
                params['uniqueChirps_effective'] = eff_unique
                params['unique_chirps_frame'] = eff_unique

        # Frame vs effective mismatch flag (for GUI diagnostics)
        fc = params.get('frameConfig') or {}
        fc_unique = fc.get('uniqueChirps')
        params['uniqueChirps_log'] = fc_unique
        if isinstance(fc_unique, int) and eff_unique and fc_unique != eff_unique:
            params['uniqueChirps_mismatch'] = True
        else:
            params['uniqueChirps_mismatch'] = False

        if tx_order_masks:
            params['tx_order_masks'] = tx_order_masks
            params['tx_order_names'] = tx_order_names
            active_names = sorted({n for n in tx_order_names if n and n != "None"})
            params['tdm_active_masks'] = active_names
            params['tdm_pattern_len'] = len(tx_order_masks)
            # Also expose TDM positions per TX
            if positions_by_mask:
                params['tdm_positions_by_mask'] = positions_by_mask
            if positions_by_name:
                params['tdm_positions_by_name'] = positions_by_name
    except Exception:
        pass

    _fill_velocity_and_range_limits(params)

    # Add LaneConfig and LvdsLaneConfig info
    if lane_cfg_parsed:
        params.update(lane_cfg_parsed)
        params['laneConfig_timestamp'] = lane_cfg_parsed.get('timestamp')
    if lvds_lane_cfg:
        params.update({
            'lvdsLaneConfig_raw': lvds_lane_cfg.get('lvdsLaneConfig_raw'),
            'lvdsLaneConfig_timestamp': lvds_lane_cfg.get('timestamp'),
            'numLanes_from_lvdsLaneConfig': lvds_lane_cfg.get('numLanes_from_lvdsLaneConfig'),
        })

    # Unify final numLanes selection (prefer LaneConfig popcount 1..4)
    final_num_lanes = None
    try:
        lanes_laneCfg = lane_cfg_parsed.get('numLanes')
    except Exception:
        lanes_laneCfg = None
    lanes_lvdsCfg = lvds_lane_cfg.get('numLanes_from_lvdsLaneConfig') if lvds_lane_cfg else None
    if isinstance(lanes_laneCfg, int) and (1 <= lanes_laneCfg <= 4):
        # An LVDS container is 1, 2 or 4 lanes wide - there is no 3-lane mode. A mask
        # with three bits set is carried in the next width up, the same way
        # LvdsLaneConfig code 3 already means 4. Without this a LaneConfig of 7, 13 or
        # 14 reached load_raw_cube as numLanes=3 and it refused to decode at all.
        final_num_lanes = 4 if lanes_laneCfg == 3 else lanes_laneCfg
        final_src = "LaneConfig" if final_num_lanes == lanes_laneCfg else "LaneConfig (3->4)"
    elif lanes_lvdsCfg in (1, 2, 4):
        final_num_lanes = lanes_lvdsCfg
        final_src = "LvdsLaneConfig"
    else:
        final_src = "unknown"
    params['numLanes'] = final_num_lanes
    params['numLanes_source'] = final_src

    # Needs numLanes, so it runs once the lane configuration is settled.
    _merge_recorder_log(params, log_path)

    # NEW: compute alternative range resolution estimates
    params.update(_compute_alt_range_resolutions(params, window_name='hann'))

    # NEW: Range bin spacing vs. physical effective resolution (+ GUI bin spacing)
    try:
        dR_eff_m = params.get('effectiveRangeRes_m') or params.get('rangeRes_m')
        N = int(params.get('numAdcSamples') or 0)
        if dR_eff_m and N > 0:
            NFFT_pow2 = _next_pow2(N)
            # Physical effective resolution (no FFT padding)
            params['rangeBin_no_pad_m'] = dR_eff_m
            params['rangeBin_no_pad_cm'] = dR_eff_m * 100.0
            # GUI bin spacing with zero-padding: ΔR_bin = ΔR_eff * (N / NFFT)
            params['rangeBin_pow2_m'] = dR_eff_m * (N / float(NFFT_pow2))
            params['rangeBin_pow2_cm'] = params['rangeBin_pow2_m'] * 100.0
            params['rangeBin_NFFT_pow2'] = NFFT_pow2
            # Spacing between FFT bins after zero-padding. This is NOT the range
            # resolution: zero-padding interpolates, it does not resolve. On the
            # AWR1642 capture the bin spacing is 4.7 cm while the radar can only
            # separate two targets 8.0 cm apart, so showing it as "resolution"
            # overstates the radar by 1.7x.
            params['rangeBinSpacing_gui_cm'] = params['rangeBin_pow2_cm']
        else:
            params['rangeBin_no_pad_m'] = None
            params['rangeBin_no_pad_cm'] = None
            params['rangeBin_pow2_m'] = None
            params['rangeBin_pow2_cm'] = None
            params['rangeBin_NFFT_pow2'] = None
            params['rangeBinSpacing_gui_cm'] = None
    except Exception:
        pass

    summary_lines = []
    if print_summary and params:
        # Helpers for safe formatting
        def fmt(val, pattern):
            return pattern.format(val) if isinstance(val, (int, float)) and val is not None else str(val)
        fs60 = params.get('freqSlopeMHz_us_60_candidate')
        fs77 = params.get('freqSlopeMHz_us_77_candidate')
        f60c = params.get('startFreqGHz_60_candidate')
        f77c = params.get('startFreqGHz_77_candidate')
        bw_ghz = params.get('bandwidthGHz')
        bw_mhz = bw_ghz * 1000.0 if isinstance(bw_ghz, (int, float)) else None

        summary_lines.append("Profile (parsed):")
        summary_lines.append(f"  profilesFound: {params.get('profilesFound')} (selected latest at {params.get('timestamp')})")
        summary_lines.append(f"  chipHint: {params.get('chip_hint')}  chipHintBand: {params.get('chipHintBand')}")
        summary_lines.append(f"  band(selected): {params.get('band')}")
        summary_lines.append(f"  startFreqGHz (selected): {fmt(params.get('startFreqGHz'), '{:.5f}')}")
        summary_lines.append(f"  startFreq candidates GHz: 60GHz={fmt(f60c,'{:.5f}')} 77GHz={fmt(f77c,'{:.5f}')}")
        summary_lines.append(f"  freqSlopeConst: {params.get('freqSlopeConst')}")
        summary_lines.append(f"  freqSlope candidates MHz/us: 60GHz={fmt(fs60,'{:.6f}')} 77GHz={fmt(fs77,'{:.6f}')}")
        summary_lines.append(f"  freqSlope selected MHz/us: {fmt(params.get('freqSlopeMHz_us'),'{:.6f}')} (LSB used {fmt(params.get('freqSlopeScale_used'),'{:.12f}')})")
        summary_lines.append(f"  rampEndTime_us: {params.get('rampEndTime_us')}")
        summary_lines.append(f"  idleTime_us: {params.get('idleTime_us')}")
        summary_lines.append(f"  adcStartTime_us: {params.get('adcStartTime_us')}")
        summary_lines.append(f"  txStartTime_us: {params.get('txStartTime_us')}")
        summary_lines.append(f"  numAdcSamples: {params.get('numAdcSamples')}")
        summary_lines.append(f"  digOutSampleRate_MHz: {params.get('digOutSampleRate_MHz')}")
        summary_lines.append(f"  txOutPowerBackoffCode: {params.get('txOutPowerBackoffCode')}")
        summary_lines.append(f"  txPhaseShifter: {params.get('txPhaseShifter')}")
        summary_lines.append(f"  hpfCornerFreq1: {params.get('hpfCornerFreq1')}  hpfCornerFreq2: {params.get('hpfCornerFreq2')}")
        summary_lines.append(f"  rxGain_dB: {params.get('rxGain')}")
        summary_lines.append(f"  Bandwidth: {fmt(bw_ghz,'{:.6f}')} GHz ({fmt(bw_mhz,'{:.1f}')} MHz)")
        summary_lines.append(f"  stopFreqGHz: {fmt(params.get('stopFreqGHz'),'{:.5f}')}")
        summary_lines.append(f"  RangeRes_cm: {params.get('rangeRes_cm')}")
        summary_lines.append(f"  RangeRes_m (ref): {params.get('rangeRes_m')}")

        # NEW: Add R_max calculation to summary
        try:
            fs_hz = (params.get('digOutSampleRate_MHz') or 0) * 1e6
            slope_hz_s = (params.get('freqSlopeMHz_us') or 0) * 1e12
            if fs_hz > 0 and slope_hz_s > 0:
                r_max_half = (3e8 * fs_hz) / (4 * slope_hz_s)
                r_max_full = (3e8 * fs_hz) / (2 * slope_hz_s)  # Full spectrum (like mmWave Studio)
                summary_lines.append(f"  Max Range (Nyquist, half spectrum): {r_max_half:.2f} m")
                summary_lines.append(f"  Max Range (full spectrum, ±): {r_max_full:.2f} m")
        except Exception:
            pass

        # NEW: Add number of range bins to summary
        try:
            nfft = 256 # Default FFT size used in processing
            is_complex = 'COMPLEX' in str(params.get('dataFmt_adcFmt', '')).upper()
            # For complex I/Q we use full spectrum (Nfft bins). For real, rfft gives N/2+1.
            num_bins = nfft if is_complex else (nfft // 2 + 1)
            summary_lines.append(f"  Range Bins (for NFFT={nfft}): {num_bins} ({'full' if is_complex else 'half'} spectrum)")
        except Exception:
            pass

        summary_lines.append(f"  VelocityRes (m/s): {params.get('velocityRes_m_s')}")
        summary_lines.append(f"  reserved: {params.get('reserved')}")

        # NEW: Print Antenna Geometry and Calibration if present
        if 'antGeometry0' in params or 'antGeometry1' in params:
            summary_lines.append("  Antenna Geometry (lambda):")
            if 'antGeometry0' in params:
                summary_lines.append(f"    x: {params['antGeometry0']}")
            if 'antGeometry1' in params:
                summary_lines.append(f"    y: {params['antGeometry1']}")
        
        if 'compRangeBiasAndRxChanPhase' in params:
            summary_lines.append(f"  Calibration (compRangeBiasAndRxChanPhase): {params['compRangeBiasAndRxChanPhase']}")
        
        if 'antPhaseRot' in params:
            summary_lines.append(f"  Calibration (antPhaseRot): {params['antPhaseRot']}")

        # NEW: Detailed DataFmtConfig and AdcOutConfig printing
        summary_lines.append("  Data Format Configs (raw):")
        dfc_raw = params.get('raw_values')
        if dfc_raw:
            ts_dfc = params.get('dataFmtConfig_timestamp') or "N/A"
            summary_lines.append(f"    DataFmtConfig: {dfc_raw} (at {ts_dfc})")
            summary_lines.append(f"      [0] rxMask: {dfc_raw[0]} -> {bin(dfc_raw[0])}")
            summary_lines.append(f"      [1] adcBits enum: {dfc_raw[1]}")
            summary_lines.append(f"      [2] isComplex enum: {dfc_raw[2]} -> {'Complex' if dfc_raw[2]==1 else 'Real'}")
            summary_lines.append(f"      [3] iqSwap enum: {dfc_raw[3]}")
            summary_lines.append(f"      [4] chInterleave (legacy): {dfc_raw[4]}")
            summary_lines.append(f"      [5] interleave flag (0=non-interleaved, 1=interleaved): {dfc_raw[5]} "
                                 f"-> interleaved={params.get('dataFmt_isInterleaved')}")
        else:
            summary_lines.append("    DataFmtConfig: Not found")
        aoc_raw = params.get('adcOutConfig_raw')
        if aoc_raw:
            ts = params.get('adcOutConfig_timestamp') or "N/A"
            # UPDATED: Explain detailed format enum
            adc_fmt_map = {0: "Real", 1: "Complex1X", 2: "Complex2X", 3: "PseudoReal"}
            fmt_str = adc_fmt_map.get(aoc_raw[1], 'Unknown')
            summary_lines.append(f"    AdcOutConfig: {aoc_raw} (at {ts})")
            summary_lines.append(f"      [0] adcBits enum: {aoc_raw[0]}")
            summary_lines.append(f"      [1] format enum: {aoc_raw[1]} -> {fmt_str}")
            summary_lines.append(f"      [2] interleave flag (0=non-interleaved, 1=interleaved): {aoc_raw[2]} "
                                 f"-> interleaved={(aoc_raw[2]==1)}")
        else:
            summary_lines.append("    AdcOutConfig: Not found")

        summary_lines.append("  DataFmt (decoded):")
        source = "AdcOutConfig" if aoc_raw else "DataFmtConfig"
        summary_lines.append(f"    adcFmt: {params.get('dataFmt_adcFmt')} (from {source})")
        summary_lines.append(f"    adcBits: {params.get('dataFmt_adcBits')} (from {source})")
        summary_lines.append(f"    interleaved: {params.get('dataFmt_isInterleaved')} (from {source})")
        summary_lines.append(f"    iqOrder: {params.get('dataFmt_iqOrder')} (from DataFmtConfig)")
        summary_lines.append(f"    rxChanEnMask: {params.get('dataFmt_rxChanEnMask')} (from DataFmtConfig)")

        # FrameConfig
        fc = params.get('frameConfig')
        if fc:
            summary_lines.append("  FrameConfig:")
            summary_lines.append(f"    frameCfgIdx={fc['frameCfgIdx']} chirpStartIdx={fc['chirpStartIdx']} chirpEndIdx={fc['chirpEndIdx']} uniqueChirps={fc['uniqueChirps']}")
            summary_lines.append(f"    numLoops={fc['numLoops']} chirpsPerFrame={fc['chirpsPerFrame']}")
            summary_lines.append(f"    numFrames={fc['numFrames']} framePeriodicity_ticks={fc.get('framePeriodicity_ticks')} -> {fc['framePeriodicity_us']:.3f} us (~{fc['framePeriodicity_ms']:.3f} ms)")
            summary_lines.append(f"    triggerSelect={fc['triggerSelect']} triggerDelay={fc['triggerDelay']}")
        else:
            summary_lines.append("  FrameConfig: (none)")
        # NEW: print derived vs log vs selected
        if params.get('framePeriodicity_ms_selected') is not None:
            summary_lines.append("  Frame periodicity:")
            summary_lines.append(f"    from log: {params.get('framePeriodicity_ms_selected'):.3f} ms")
        # AdvancedFrameConfig
        afc = params.get('advancedFrameConfig')
        if afc:
            summary_lines.append("  AdvancedFrameConfig:")
            summary_lines.append(f"    adv_frameHeader={afc.get('adv_frameHeader')} adv_numSubFrames={afc.get('adv_numSubFrames')}")
            if 'adv_periodicities_us' in afc:
                summary_lines.append(f"    periodicities_us={afc.get('adv_periodicities_us')}")
            summary_lines.append(f"    raw_len={len(afc.get('adv_raw', []))}")
        else:
            summary_lines.append("  AdvancedFrameConfig: (none)")
        # NEW: Effective bandwidth (profile)
        summary_lines.append("  Effective bandwidth (profile):")
        summary_lines.append(f"    adcSampleTime_us={params.get('adcSampleTime_us')}")
        summary_lines.append(f"    rampWindow_us={params.get('rampWindow_us')}")
        summary_lines.append(f"    effCaptureWindow_us={params.get('effCaptureWindow_us')}")
        summary_lines.append(f"    effectiveBW: {params.get('effectiveBandwidthGHz')} GHz ({params.get('effectiveBandwidthMHz')} MHz)")
        summary_lines.append(f"    effectiveRangeRes_cm: {params.get('effectiveRangeRes_cm')} (ref m: {params.get('effectiveRangeRes_m')})")
        summary_lines.append(f"    effectiveStartFreqGHz (@ADC start): {params.get('effectiveStartFreqGHz')}")

        # NEW: Alternative effective range resolution summary
        summary_lines.append("  Effective range resolution (alternatives):")
        summary_lines.append(f"    from BW_eff: {params.get('rangeRes_alt_from_BWeff_cm')} cm (ref m: {params.get('rangeRes_alt_from_BWeff_m')})")
        summary_lines.append(f"    from fs/N_eff: {params.get('rangeRes_alt_from_fsN_cm')} cm (N_eff={params.get('N_effective_used')})")
        if params.get('rangeRes_windowed_cm') is not None:
            summary_lines.append(f"    window-adjusted (~{params.get('rangeRes_window_factor')}×): {params.get('rangeRes_windowed_cm')} cm")

        # NEW: Range resolution vs FFT bin spacing explanation
        summary_lines.append("  Range resolution vs FFT bin spacing:")
        summary_lines.append(f"    physical_eff: {params.get('effectiveRangeRes_cm')} cm (c/(2*B_eff))")
        summary_lines.append(f"    bin_spacing (no zero-pad, NFFT=N={params.get('numAdcSamples')}): {params.get('rangeBin_no_pad_cm')} cm")
        summary_lines.append(f"    bin_spacing (zero-pad to NFFT={params.get('rangeBin_NFFT_pow2')}): {params.get('rangeBin_pow2_cm')} cm")
        # NEW: Explicit GUI number (alias)
        summary_lines.append(f"    GUI bin spacing: {params.get('rangeBinSpacing_gui_cm')} cm "
                             f"(interpolated; the resolution is physical_eff above)")

        # Chirps (extend with effective BW and effective start freq per‑chirp)
        chirps = params.get('chirps', [])
        if chirps:
            summary_lines.append("  Chirps:")
            lsb_77 = 0.048279762268
            lsb_60 = 0.036209821701
            band_sel = params.get('band')
            lsb = lsb_77 if band_sel == '77' else lsb_60
            base_const = params.get('freqSlopeConst') or 0
            ramp_us = params.get('rampEndTime_us') or 0.0
            # FIX: use adcStartTime_us (was adcStartFreqGHz by mistake)
            adc_base_us = params.get('adcStartTime_us') or 0.0
            startFreqGHz_base = params.get('startFreqGHz') or 0.0
            prof_sample_time_us = params.get('adcSampleTime_us')
            for ch in chirps:
                freqSlopeVar = ch.get('freqSlopeVar') or 0
                eff_const = base_const + freqSlopeVar
                eff_slope = eff_const * lsb  # MHz/us
                adcStartVar_us = (ch.get('adcStartTimeVar') or 0) / 100.0
                total_adc_us = adc_base_us + adcStartVar_us
                ramp_window_chirp_us = max(0.0, ramp_us - total_adc_us)
                eff_cap_chirp_us = min(ramp_window_chirp_us, prof_sample_time_us) if prof_sample_time_us else ramp_window_chirp_us
                eff_bw_chirp_mhz = eff_slope * eff_cap_chirp_us
                # NEW: per-chirp effective start frequency (includes startFreqVar)
                startVar_GHz = _start_var_to_GHz(ch.get('startFreqVar') or 0, band_sel or '77')
                eff_start_ghz = startFreqGHz_base + startVar_GHz + (eff_slope * total_adc_us) / 1000.0
                summary_lines.append(
                    f"    chirp {ch['chirpStartIdx']}-{ch['chirpEndIdx']} prof={ch['profileId']} "
                    f"txMask={ch['txEnable']} txActive={ch.get('txActive')} "
                    f"startFreqVar={ch.get('startFreqVar')} freqSlopeVar={freqSlopeVar} "
                    f"idleTimeVar={ch.get('idleTimeVar')} adcStartTimeVar={ch.get('adcStartTimeVar')} "
                    f"bpfSel={ch.get('bpfSel')} effectiveSlope={eff_slope:.4f} MHz/us "
                    f"effWin={eff_cap_chirp_us:.3f} us effBW={eff_bw_chirp_mhz:.1f} MHz "
                    f"effStartGHz={eff_start_ghz:.6f}"
                )
        else:
            summary_lines.append("  Chirps: (none)")

        # NEW: Final data stream order summary
        try:
            is_complex = 'COMPLEX' in str(params.get('dataFmt_adcFmt', '')).upper()
            is_interleaved = bool(params.get('dataFmt_isInterleaved'))
            iq_order = str(params.get('dataFmt_iqOrder') or 'IQ').upper()
            rx_mask = params.get('dataFmt_rxChanEnMask', 0)
            num_rx = sum(((rx_mask >> i) & 1) for i in range(4)) or 4  # fallback to 4

            summary_lines.append("")
            summary_lines.append("Data stream order (fast-time interleave):")
            if is_complex and is_interleaved:
                order = []
                if iq_order == 'QI':
                    for r in range(num_rx):
                        order += [f"RX{r}_Q", f"RX{r}_I"]
                else:
                    for r in range(num_rx):
                        order += [f"RX{r}_I", f"RX{r}_Q"]
                summary_lines.append("  " + ", ".join(order) + ", …")
                summary_lines.append("  Complex combine: " + ("Q + j*I" if iq_order == 'QI' else "I + j*Q"))
            elif not is_complex and is_interleaved:
                order = [f"RX{r}_I" for r in range(num_rx)]
                summary_lines.append("  " + ", ".join(order) + ", …")
                summary_lines.append("  Real combine: I + j*0")
            elif is_complex and not is_interleaved:
                summary_lines.append("  Non-interleaved (per chirp): RX0 all (I,Q) pairs, then RX1, RX2, RX3 …")
                summary_lines.append("  Complex combine per RX: " + ("Q + j*I" if iq_order == 'QI' else "I + j*Q"))
            else:
                summary_lines.append("  Non-interleaved Real (per chirp): RX0 all I, then RX1, RX2, RX3 …")
                summary_lines.append("  Real combine: I + j*0")
        except Exception:
            pass

        # LVDS lane inference summary
        summary_lines.append("LVDS lanes inference:")
        lm = params.get('laneMask')
        if lm is not None:
            pc = bin(int(lm)).count('1')
            sel_txt = (pc if (1 <= pc <= 4) else 'unknown')
            lm_bin = params.get('laneMask_bin')
            active_labels = params.get('lanesActive', 'None')
            summary_lines.append(f"  LaneConfig: laneMask={lm} ({lm_bin}) active={active_labels} "
                                 f"(popcount={pc}) -> {sel_txt} lanes "
                                 f"(timestamp {params.get('laneConfig_timestamp','N/A')})")
        else:
            summary_lines.append("  LaneConfig: not found")
        raw_lvds = params.get('lvdsLaneConfig_raw')
        if raw_lvds:
            summary_lines.append(f"  LvdsLaneConfig: raw={raw_lvds} "
                                 f"-> inferred lanes={params.get('numLanes_from_lvdsLaneConfig')} "
                                 f"(timestamp {params.get('lvdsLaneConfig_timestamp','N/A')})")
        else:
            summary_lines.append("  LvdsLaneConfig: not found")
        dca_lane_mode = params.get('dca_lane_mode')
        if dca_lane_mode:
            summary_lines.append(f"  Recorder log: DCA1000 ran in {dca_lane_mode}-lane mode "
                                 f"(radar drives {params.get('numLanes_radar')})")
        else:
            summary_lines.append("  Recorder log: not found")
        summary_lines.append(f"  numLanes (selected): {params.get('numLanes')}  [source={params.get('numLanes_source')}]")
        summary_lines.append(f"  Detected Lanes: {params.get('numLanes')}")
        summary_lines.append(f"  Lane Pattern: {params.get('lanesActive')}")
        # UPDATED explanation: the recorder owns the container width
        summary_lines.append("  Inference rule: the recorder log's lane mode wins - it is the width of the "
                             "container the samples were written into. Without it, LaneConfig is a bitmask "
                             "(1=lane1, 2=lane2, 3=lane1+2,…) and numLanes = popcount(mask), with 3 lanes "
                             "carried in a 4-lane container; fallback LvdsLaneConfig (1/2/4; 3->4).")

        # NEW: Chirp TX pattern summary
        try:
            if params.get('tx_order_names'):
                summary_lines.append("  Chirp TX Pattern:")
                summary_lines.append(f"    order: {params['tx_order_names']}")
                summary_lines.append(f"    active: {params.get('tdm_active_masks')}")
                summary_lines.append(f"    pattern_len: {params.get('tdm_pattern_len')}")
                # Positions per TX (if any)
                if params.get('tdm_positions_by_name'):
                    summary_lines.append(f"    positions: {params['tdm_positions_by_name']}")
        except Exception:
            pass

        # NEW: Unique chirps mismatch info
        try:
            eff_u = params.get('uniqueChirps_effective')
            log_u = params.get('uniqueChirps_log')
            if eff_u is not None:
                summary_lines.append(f"  Unique Chirps (effective from ChirpConfig): {eff_u}")
            if log_u is not None and eff_u is not None and log_u != eff_u:
                summary_lines.append(f"  WARNING: FrameConfig.uniqueChirps={log_u} ≠ effective={eff_u} (GUI will use effective)")
        except Exception:
            pass

        # ...existing printing...
        for l in summary_lines:
            print(l)
    return params, summary_lines
