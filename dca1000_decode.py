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
    folder, bin_name = os.path.split(bin_path)
    if '_Raw_0.bin' in bin_name:
        base = bin_name.replace('_Raw_0.bin', '')
    else:
        base = os.path.splitext(bin_name)[0]
    log_name = f"{base}_LogFile.txt"
    log_path = os.path.join(folder, log_name)
    return log_path if os.path.isfile(log_path) else None

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

# NEW: Simplified parser based on user rules
def parse_mmwave_log(log_lines: list) -> dict:
    """
    Parses log lines to find the last DataFmtConfig and returns format parameters.
    Rules:
      - adcBits from 2nd param (0=12, 1=14, 2=16)
      - isComplex from 3rd param (1=Complex, 0=Real)
      - iqSwap from 4th param (1=QI, 0=IQ)
      - isInterleaved from 6th param (1=interleaved, 0=non-interleaved)  # UPDATED
      - rxMask from 1st param
    """
    data_fmt_line = None
    for line in reversed(log_lines):
        if 'API:DataFmtConfig' in line:
            data_fmt_line = line
            break
    if not data_fmt_line:
        raise ValueError("DataFmtConfig not found in log.")

    try:
        payload = data_fmt_line.split('API:DataFmtConfig,', 1)[1]
        nums = [int(t) for t in payload.strip().split(',') if t != '']
    except (IndexError, ValueError):
        raise ValueError("Failed to parse DataFmtConfig line.")

    if len(nums) < 6:
        raise ValueError(f"DataFmtConfig has too few parameters: {nums}")

    adc_bits_map = {0: 12, 1: 14, 2: 16}

    return {
        "rxMask": nums[0],
        "adcBits": adc_bits_map.get(nums[1], 16),
        "isComplex": (nums[2] == 1),
        "iqSwap": (nums[3] == 1),
        "isInterleaved": (nums[5] == 1),  # UPDATED mapping
    }


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
    if mask is None:
        return ""
    txs = [f"TX{i}" for i in range(3) if (mask >> i) & 1]
    return ",".join(txs) if txs else "None"

# NEW regex for Frame / AdvancedFrame
FRAME_RE = re.compile(r'API:FrameConfig,([\d,]+)')
ADV_FRAME_RE = re.compile(r'API:AdvancedFrameConfig,([\d,]+)')

def parse_frame_config_line(line: str):
    """
    Expected pattern (8 integers):
      frameCfgIdx, chirpStartIdx, chirpEndIdx, numLoops, framePeriodicity_ticks,
      triggerSelect, numFrames, triggerDelay
    Note: framePeriodicity is in 5ns ticks.
    Example:
      API:FrameConfig,0,1,250,255,8000000,0,300,0,
    """
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
    frameCfgIdx, chirpStartIdx, chirpEndIdx, numLoops, framePeriodicity_ticks, triggerSelect, numFrames, triggerDelay = vals

    # Convert from 5ns ticks to microseconds
    framePeriodicity_us = framePeriodicity_ticks * 5.0 / 1000.0

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
    # Heuristically select periods (values ~millions of us)
    periodicities = [v for v in ints if isinstance(v, int) and v and v >= 100000]
    if periodicities:
        adv["adv_periodicities_us"] = periodicities
    # For convenience the first two
    if len(periodicities) >= 1:
        adv["adv_periodicity_1_us"] = periodicities[0]
    if len(periodicities) >= 2:
        adv["adv_periodicity_2_us"] = periodicities[1]
    return adv

def _start_var_to_GHz(start_freq_var_raw: int, band: str) -> float:
    if not isinstance(start_freq_var_raw, int) or start_freq_var_raw == 0:
        return 0.0
    mult = 3.6 if band == '77' else 2.7
    return (start_freq_var_raw / (2**26)) * mult

def _sum_defined_unique_chirps(chirps_list):
    if not chirps_list:
        return 0
    total = 0
    for ch in chirps_list:
        cs = ch.get('chirpStartIdx')
        ce = ch.get('chirpEndIdx')
        if isinstance(cs, int) and isinstance(ce, int) and ce >= cs:
            total += (ce - cs + 1)
    return total

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


def parse_logfile_full(log_path, print_summary=False):
    # PRE-SCAN (FORWARD) to get chip hint before reverse reading
    chip_hint = None
    chip_hint_band = None
    chip_line_pattern = re.compile(r'API:select_chip_version,([^,]+)', re.IGNORECASE)
    with open(log_path, 'r') as f_scan:
        for ln in f_scan:
            m = chip_line_pattern.search(ln)
            if m:
                chip_hint = m.group(1).strip()
                mapped = map_chip_to_band(chip_hint)
                if mapped:
                    chip_hint_band = mapped
                    break # Found it, no need to scan further
    params = {}
    chirp_configs = []
    band = None
    ts_re = re.compile(r'^(\d{2}-[A-Za-z]{3}-\d{4} \d{2}:\d{2}:\d{2}):')
    profile_entries = []
    data_fmt_parsed = {}
    adc_out_parsed = {} # NEW: Re-added to store for printing
    chirps_dict = {}
    frame_cfg = None
    adv_frame_cfg = None
    lane_cfg_parsed = {} # NEW: for LaneConfig
    lvds_lane_cfg = {}   # NEW: for LvdsLaneConfig (fallback/confirmation)

    with open(log_path, 'r') as f:
        lines = f.readlines()
    lines = reversed(lines)
    for line in lines:
        line_stripped = line.strip()
        m_ts = ts_re.match(line_stripped)
        current_ts_str = m_ts.group(1) if m_ts else None

        # This is now a fallback if pre-scan fails.
        if ('API:select_chip_version' in line or 'API:SelectChipVersion' in line) and not chip_hint_band:
            # Only update if pre-scan failed
            if 'AR1642' in line or 'AWR1642' in line:
                chip_hint = 'AR1642'
                chip_hint_band = '77'
            elif 'IWR6843' in line or 'AWR6843' in line:
                chip_hint = 'IWR6843'
                chip_hint_band = '60'
        if 'API:ProfileConfig' in line:
            try:
                payload = line.split('API:ProfileConfig,', 1)[1]
            except IndexError:
                continue
            tokens = [t for t in payload.strip().split(',') if t != '']
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
            if p.get('startFreqGHz') and p.get('rampEndTime_us'):
                p['velocityRes_m_s'] = 3e8 / (2 * p['startFreqGHz'] * 1e9 * (p['rampEndTime_us'] * 1e-6)) if p['rampEndTime_us'] else None
            else:
                p['velocityRes_m_s'] = None
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
        elif 'API:ChirpConfig' in line:
            # NEW: full chirp parsing (with txActive)
            try:
                payload = line.split('API:ChirpConfig,', 1)[1]
            except IndexError:
                continue
            nums = [tok for tok in payload.strip().split(',') if tok != '']
            ints = []
            for t in nums:
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
        elif 'API:FrameConfig' in line and frame_cfg is None:
            fc = parse_frame_config_line(line)
            if fc:
                fc['timestamp'] = current_ts_str
                frame_cfg = fc
        elif 'API:AdvancedFrameConfig' in line and adv_frame_cfg is None:
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
    if adv_frame_cfg:
        params['advancedFrameConfig'] = adv_frame_cfg

    chirps_list = list(chirps_dict.values())
    chirps_list.sort(key=lambda c: (c["chirpStartIdx"], c["chirpEndIdx"]))
    params['chirps'] = chirps_list
    params['txEnable'] = ' '.join(chirp_configs)

    # NEW: derive TX order (pattern) directly from ChirpConfig ranges (expanded in defined order)
    try:
        tx_order_masks = []
        tx_order_names = []
        positions_by_mask = {}  # mask(int) -> [indices]
        positions_by_name = {}  # "TX0" -> [indices]
        pos = 0
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

        # Effective unique chirps from definitions (not FrameConfig!)
        eff_unique = _sum_defined_unique_chirps(chirps_list)
        if eff_unique > 0:
            params['uniqueChirps_effective'] = eff_unique

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
        final_num_lanes = lanes_laneCfg
        final_src = "LaneConfig"
    elif lanes_lvdsCfg in (1, 2, 4):
        final_num_lanes = lanes_lvdsCfg
        final_src = "LvdsLaneConfig"
    else:
        final_src = "unknown"
    params['numLanes'] = final_num_lanes
    params['numLanes_source'] = final_src

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
            # Convenience alias: what GUI typically shows as “Range Resolution”
            params['rangeResolution_gui_cm'] = params['rangeBin_pow2_cm']
        else:
            params['rangeBin_no_pad_m'] = None
            params['rangeBin_no_pad_cm'] = None
            params['rangeBin_pow2_m'] = None
            params['rangeBin_pow2_cm'] = None
            params['rangeBin_NFFT_pow2'] = None
            params['rangeResolution_gui_cm'] = None
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
        summary_lines.append(f"    GUI Range Resolution: {params.get('rangeResolution_gui_cm')} cm")

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
        summary_lines.append(f"  numLanes (selected): {params.get('numLanes')}  [source={params.get('numLanes_source')}]")
        summary_lines.append(f"  Detected Lanes: {params.get('numLanes')}")
        summary_lines.append(f"  Lane Pattern: {params.get('lanesActive')}")
        # UPDATED explanation: LaneConfig is a bitmask
        summary_lines.append("  Inference rule: LaneConfig is a bitmask (1=lane1, 2=lane2, 3=lane1+2,…). "
                             "numLanes = popcount(mask) in [1..4]; fallback LvdsLaneConfig (1/2/4; 3->4).")

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
