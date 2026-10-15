"""
cascade/decode.py

A cascade take for the rest of DARTscope (Decode, Fastcheck, Analysis): the params a
mmWave Studio log gives (dca1000_decode.parse_logfile_full), built from <name>_cascade.json
and the recorder's <name>_Raw_LogFile.csv, and the radar cube without the LVDS demux.

    params, summary = cascade_params(bin_path)       # None when not a cascade take
    cube = load_cascade_cube(bin_path, params, dims) # (frames, chirps, 8 RX, samples)

Why no demux: the AM2732R passes each chirp's CSI-2 buffers to the DCA1000 as a CBUFF
software session, and the DCA1000's 4-lane raw mode puts the 16-bit words back in the
order they were sent - the file is already [frame][chirp][RX 0..7][sample][I, Q]
(checked against TI's main_read_2chipCascade.m and on the lab's recordings).

The transmitters of the two devices are numbered TX1-TX3 (primary) and TX4-TX6
(secondary): a chirp's mask is primary | secondary << 3.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import datetime
import json
import os
import re
from pathlib import Path

import numpy as np

from .params import FREQ_LSB_HZ, NUM_RX, SLOPE_LSB_MHZ_US, HPF1_KHZ, HPF2_KHZ, CascadeParams

PLATFORM = "AWR2243-2X-CAS-EVM"
CHIP = "AWR2243 cascade"


def sidecar_for(bin_path) -> Path | None:
    m = re.match(r"(.*)_Raw_\d+\.bin$", os.path.basename(str(bin_path)))
    if not m:
        return None
    side = Path(bin_path).with_name(m.group(1) + "_cascade.json")
    return side if side.is_file() else None


def is_cascade_take(bin_path) -> bool:
    return sidecar_for(bin_path) is not None


def chirp_masks(chirps) -> list[int]:
    """One mask per chirp of the loop over the six transmitters (primary TX1-3,
    secondary TX4-6)."""
    return [int(a) | (int(b) << 3) for a, b in chirps]


def cascade_params(bin_path) -> tuple[dict, list[str]] | None:
    """parse_logfile_full's params for a cascade take, or None when it is not one."""
    from dca1000_decode import (_compute_alt_range_resolutions, _fill_velocity_and_range_limits,
                                decode_tx_enable, parse_raw_log_csv)
    side = sidecar_for(bin_path)
    if side is None:
        return None
    meta = json.loads(side.read_text(encoding="utf-8"))
    cp = CascadeParams.from_dict(meta["params"])
    w = cp.words()
    lay = meta["layout"]
    chirps = [tuple(c) for c in lay["tx_chirps"]]
    n_ch, loops = len(chirps), int(lay["num_loops"])
    masks = chirp_masks(chirps)
    names = [decode_tx_enable(m) for m in masks]

    slope = cp.slope_code() * SLOPE_LSB_MHZ_US                       # what the radar ran
    start_ghz = w[2] * FREQ_LSB_HZ / 1e9
    fs_mhz = cp.sample_rate_ksps / 1000.0
    adc_us = cp.num_samples / fs_mhz
    eff_bw_mhz = slope * adc_us
    period_ticks = w[16]
    p: dict = {
        "cascade": True, "cascade_sidecar": str(side), "log_path": str(side),
        "platform": PLATFORM, "chip_hint": CHIP, "band": "77",
        "transferFmtPkt0": "ADC_DATA_ONLY",
        # ProfileConfig as the firmware wrote it (gUserCfg)
        "profileId": 0, "startFreqConst": w[2], "idleTimeConst": w[4], "adcStartTimeConst": w[5],
        "rampEndTime": w[6], "txOutPowerBackoffCode": w[13], "txPhaseShifter": 0,
        "freqSlopeConst": cp.slope_code(), "txStartTimeConst": int(round(cp.tx_start_us * 100)),
        "numAdcSamples": int(cp.num_samples), "digOutSampleRate": int(cp.sample_rate_ksps),
        "hpfCornerFreq1": HPF1_KHZ[int(cp.hpf1_khz)], "hpfCornerFreq2": HPF2_KHZ[int(cp.hpf2_khz)],
        "rxGain": int(cp.rx_gain_db),
        "startFreqGHz": start_ghz, "idleTime_us": cp.idle_us, "adcStartTime_us": cp.adc_start_us,
        "txStartTime_us": cp.tx_start_us, "rampEndTime_us": cp.ramp_end_us,
        "freqSlopeMHz_us": slope, "bandwidthGHz": slope * cp.ramp_end_us / 1000.0,
        "stopFreqGHz": start_ghz + slope * cp.ramp_end_us / 1000.0,
        "digOutSampleRate_MHz": fs_mhz, "adcSampleTime_us": adc_us, "rampWindow_us": cp.ramp_end_us,
        "effCaptureWindow_us": adc_us, "effectiveBandwidthMHz": eff_bw_mhz,
        "effectiveBandwidthGHz": eff_bw_mhz / 1000.0,
        "effectiveRangeRes_m": 3e8 / (2 * eff_bw_mhz * 1e6),
        "effectiveRangeRes_cm": 3e8 / (2 * eff_bw_mhz * 1e6) * 100.0,
        "effectiveStartFreqGHz": start_ghz + slope * cp.adc_start_us / 1000.0,
        "rangeRes_m": 3e8 / (2 * slope * cp.ramp_end_us * 1e6),
        "rangeRes_cm": 3e8 / (2 * slope * cp.ramp_end_us * 1e6) * 100.0,
        "profilesFound": 1,
        # data format: 8 RX (4 per device), complex 16 bit, non-interleaved
        "dataFmt_rxChanEnMask": (1 << NUM_RX) - 1, "dataFmt_rxChannels": " ".join("1" * NUM_RX),
        "dataFmt_adcBits": 16, "dataFmt_adcFmt": "Complex1X", "dataFmt_iqSwapSel": 0,
        "dataFmt_iqOrder": "IQ", "dataFmt_chInterleave": 1, "dataFmt_isInterleaved": False,
        # frame and chirps
        "frameConfig": {"frameCfgIdx": 0, "chirpStartIdx": 0, "chirpEndIdx": n_ch - 1, "numLoops": loops,
                        "framePeriodicity_ticks": period_ticks,
                        "framePeriodicity_us": period_ticks * 5e-3,
                        "framePeriodicity_ms": period_ticks * 5e-6,
                        "numFrames": int(meta["radar"]["num_frames"]), "triggerDelay": 0,
                        "uniqueChirps": n_ch, "chirpsPerFrame": n_ch * loops},
        "framePeriodicity_ms_selected": period_ticks * 5e-6,
        "chirps": [{"chirpStartIdx": i, "chirpEndIdx": i, "profileId": 0, "startFreqVar": 0,
                    "freqSlopeVar": 0, "idleTimeVar": 0, "adcStartTimeVar": 0, "txEnable": m,
                    "bpfSel": 0, "txActive": n} for i, (m, n) in enumerate(zip(masks, names))],
        "unique_chirps_frame": n_ch, "uniqueChirps_effective": n_ch, "uniqueChirps_log": n_ch,
        "uniqueChirps_mismatch": False,
        "tx_order_masks": masks, "tx_order_names": names,
        "tdm_active_masks": sorted({n for n in names if n and n != "None"}),
        "tdm_pattern_len": n_ch,
        # the DCA1000 container: 4 LVDS lanes from the AM2732R
        "numLanes": 4, "numLanes_source": "cascade (AM2732R CBUFF, 4 lanes)", "numLanes_radar": 4,
    }
    by_mask: dict = {}
    by_name: dict = {}
    for i, (m, n) in enumerate(zip(masks, names)):
        by_mask.setdefault(m, []).append(i)
        by_name.setdefault(n, []).append(i)
    p["tdm_positions_by_mask"], p["tdm_positions_by_name"] = by_mask, by_name

    try:
        started = datetime.datetime.strptime(meta["started"], "%Y-%m-%d %H:%M:%S")
        p["timestamp"] = started.strftime("%d-%b-%Y %H:%M:%S")
        p["timestamp_obj"] = started
        p["radar_start_timestamp"] = started.timestamp()
        p["radar_start_time"] = started.strftime("%d-%m-%Y %H:%M:%S")
        p["radar_start_source"] = "cascade sidecar"
    except (KeyError, ValueError):
        pass
    csv = Path(str(side)[: -len("_cascade.json")] + "_Raw_LogFile.csv")
    info = parse_raw_log_csv(str(csv)) if csv.is_file() else None
    if info:
        p.update(info)
        if info.get("dca_lane_mode") in (1, 2, 4):
            p["numLanes"] = int(info["dca_lane_mode"])
            p["numLanes_source"] = "DCA1000 recorder log"
        if info.get("dca_capture_start_timestamp"):
            p["radar_start_timestamp"] = info["dca_capture_start_timestamp"]
            p["radar_start_time"] = info["dca_capture_start"]
            p["radar_start_source"] = "DCA1000 recorder log"
    _fill_velocity_and_range_limits(p)
    p.update(_compute_alt_range_resolutions(p, window_name="hann"))

    rec = meta.get("record") or {}
    summary = [
        f"Cascade take {side.name[: -len('_cascade.json')]} ({PLATFORM})",
        f"  profile: {start_ghz:.3f} GHz, {slope:.3f} MHz/us, {cp.num_samples} samples @ {fs_mhz:g} Msps,"
        f" idle {cp.idle_us:g} us, ramp {cp.ramp_end_us:g} us",
        f"  frame: {n_ch} chirp(s) x {loops} loops every {period_ticks * 5e-6:g} ms,"
        f" {meta['radar']['num_frames']} frames; TX per chirp: {', '.join(names)}",
        f"  8 RX (RX1-4 primary, RX5-8 secondary); range resolution {p['effectiveRangeRes_cm']:.2f} cm,"
        f" max range {p.get('maxRange_m', 0):.2f} m, max velocity {p.get('maxVelocity_m_s', 0):.2f} m/s",
        f"  recorded: {rec.get('frames_in_file', '?')} frames, lost packets {rec.get('zero_filled_packets', '?')}",
    ]
    if meta.get("notes"):
        summary.append(f"  note: {meta['notes']}")
    summary.append(_apply_calibration(p, meta, masks))
    return p, summary


def _apply_calibration(p: dict, meta: dict, masks) -> str:
    """
    The take's calibration into its params - the AoA views and Multi Vitals then use the
    cascade's antenna configuration with it. The take's own copy wins over the board's
    current calibration (a later calibration must not change what an older take shows
    without saying so). Returns the summary line.
    """
    from . import calibration as cc
    cal, where = cc.for_take(meta)
    if cal is None:
        return ("  calibration: none - angles need one (Measure -> Cascade radar -> Calibration); "
                "Multi Vitals uses Range only")
    coeffs = cal.coefficients(masks, cc.centre_frequency_ghz(p))
    if coeffs is None:
        return (f"  calibration: {cal.describe()} - not for this chirp pattern "
                "(a chirp sends on several TX at once)")
    import antenna_config
    p["antenna_config"] = antenna_config.get_config(antenna_config.CASCADE_CONFIG)
    p["channel_calibration"] = [[float(c.real), float(c.imag)] for c in coeffs]
    p["cascade_calibration"] = {"method": cal.method, "created": cal.created, "where": where,
                                "describe": cal.describe()}
    return f"  calibration: {cal.describe()} - {where}"


def load_cascade_cube(bin_path, params: dict, dims: dict) -> np.ndarray:
    """(frames, chirps, RX, samples) complex64 at full scale 1, from every _Raw_<n>.bin
    part; a partial last frame is dropped."""
    from radar_processing import get_all_bin_parts
    n, rx, c = int(dims["num_samples"]), int(dims["num_rx"]), int(dims["chirps_per_frame"])
    parts = [np.fromfile(p, dtype="<i2") for p in get_all_bin_parts(str(bin_path))]
    raw = np.concatenate(parts) if len(parts) > 1 else parts[0]
    per_frame = c * rx * n * 2
    frames = raw.size // per_frame
    if frames == 0:
        raise ValueError(f"{os.path.basename(str(bin_path))} holds no whole frame "
                         f"({raw.size * 2} bytes, a frame is {per_frame * 2})")
    x = raw[: frames * per_frame].astype(np.float32)
    x *= 1.0 / 32767.0
    cube = x.view(np.complex64).reshape(frames, c, rx, n)
    dims["used_interleaved"] = False
    dims["used_iq_order"] = "IQ"
    dims["used_lanes"] = int(params.get("numLanes") or 4)
    dims["num_frames_actual"] = frames
    dims["requested_interleaved"] = False
    params["unique_chirps_frame"] = dims.get("unique_chirps_frame")
    print(f"[load_raw_cube] cascade take: {frames} frames x {c} chirps x {rx} RX x {n} samples "
          f"(no LVDS demux: the AM2732R sends the chirps in order)")
    return cube
