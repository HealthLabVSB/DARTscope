"""
Data integrity and coexistence: bursts of foreign FMCW chirps in the raw ADC data.

    python -m tools.paper_array.interference_check [--days 2026-10-06 2026-10-07] [--only STEM ...]

Detector (per recording):
  * the chirps of a frame are grouped by TX slot (chirp c -> slot c % pattern_len);
  * the median chirp of each (frame, slot) - the element-wise median of I and of Q over
    that slot's chirps, per RX and sample - stands for everything locked to the chirps
    (static scene, slow body motion, TX leakage);
  * residual power of a chirp: mean |x - median|^2 over its samples, per RX;
  * reference: the median residual power over the whole recording, per (slot, RX);
  * a chirp is HIT when its residual exceeds the reference by > 10 dB on >= 3 of the 4
    RX (a foreign chirp crossing the IF band is seen by every receiver at once; a spur
    on one receiver is not counted). A frame is hit when any of its chirps is.
Frames the recorder zero-filled (vitalsigns.detect_damaged_frames) are excluded, and the
recorder's zero-filled packet count (dca_packets_zero_filled) is reported.

Also per recording: the noise floor in dBFS per sample (capture_health.noise_spectrum,
median over 200-frame chunks and RX) and the ADC sampling duty cycle
N_chirps * N_samples / f_s / T_frame; per day the totals.

Collision model (interferer parameters are unknown, so only the formula is evaluated):
see collision_rate() - expected number of foreign-chirp crossings in one ADC window.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

import capture_health
import vitalsigns
from tools.paper_array import common

HIT_DB = 10.0
MIN_RX = 3
CHUNK = 100


def slot_pattern(params) -> int:
    for k in ("tdm_pattern_len", "unique_chirps_frame", "uniqueChirps_effective"):
        v = params.get(k)
        if v:
            return max(1, int(v))
    return 1


def residual_power(cube, pattern: int, chunk: int = CHUNK, mode: str = "mean"):
    """
    (F, C, RX) residual power of every chirp against the median chirp of its frame and
    TX slot.
    mode 'mean'     : mean |residual|^2 over the samples (the rule as specified);
    mode 'specmed'  : median over range bins of |FFT_hann(residual)|^2 / S - a foreign
                      chirp sweeping through the IF band is broadband in range and raises
                      the median; a target that moved within the frame leaves a residual
                      in its few range bins only and does not.
    """
    F, C, RX, S = cube.shape
    out = np.empty((F, C, RX), np.float32)
    w = np.hanning(S).astype(np.float32)
    for f0 in range(0, F, chunk):
        x = cube[f0:f0 + chunk]
        for s in range(pattern):
            xs = x[:, s::pattern]                                     # (f, n, RX, S)
            if xs.shape[1] < 2:
                out[f0:f0 + chunk, s::pattern] = np.nan
                continue
            med = np.median(xs.real, axis=1, keepdims=True) + 1j * np.median(xs.imag, axis=1, keepdims=True)
            d = xs - med
            if mode == "specmed":
                p = np.abs(np.fft.fft(d * w, axis=-1)) ** 2 / np.sum(w * w)
                out[f0:f0 + chunk, s::pattern] = np.median(p, axis=-1)
            else:
                out[f0:f0 + chunk, s::pattern] = np.mean(np.abs(d) ** 2, axis=-1)
    return out


def detect(cube, pattern: int, damaged=None, hit_db=HIT_DB, min_rx=MIN_RX, mode: str = "mean"):
    """
    Returns dict: hit (F, C) bool, frame_hit (F,), ref (pattern, RX) reference power,
    excess_db (F, C, RX). Damaged frames never hit. mode: see residual_power.
    """
    P = residual_power(cube, pattern, mode=mode)
    F, C, RX = P.shape
    ok = np.ones(F, bool) if damaged is None else ~np.asarray(damaged, bool)
    ref = np.empty((pattern, RX))
    for s in range(pattern):
        v = P[ok][:, s::pattern]
        ref[s] = np.nanmedian(v.reshape(-1, RX), axis=0)
    refc = ref[np.arange(C) % pattern]                                # (C, RX)
    ex = 10 * np.log10(P / (refc[None] + 1e-30) + 1e-30)
    need = min(min_rx, RX)
    hit = (np.sum(ex > hit_db, axis=-1) >= need) & ok[:, None]
    return dict(hit=hit, frame_hit=hit.any(1), ref=ref, excess_db=ex)


def noise_floor_dbfs(cube, params, chunk=200):
    stride = int(params.get("unique_chirps_frame") or params.get("uniqueChirps_effective") or 1)
    fl = []
    for f0 in range(0, cube.shape[0], chunk):
        x = cube[f0:f0 + chunk]
        fl.append(np.median([capture_health.noise_spectrum(x[:, :, r, :], stride)["floor_db"]
                             for r in range(x.shape[2])]))
    return float(np.median(fl))


def adc_duty(params) -> dict:
    """ADC sampling and chirp-ramp duty cycles from the log."""
    fc = params.get("frameConfig") or {}
    n_chirps = int(fc.get("chirpsPerFrame") or 0)
    n_s = int(params.get("numAdcSamples") or 0)
    fs = float(params.get("digOutSampleRate_MHz") or (params.get("digOutSampleRate") or 0) / 1e3) * 1e6
    t_frame = float(fc.get("framePeriodicity_ms") or params.get("framePeriodicity_ms_selected") or 0) * 1e-3
    t_adc = n_s / fs if fs else float("nan")
    t_chirp = (float(params.get("idleTime_us") or 0) + float(params.get("rampEndTime_us") or 0)) * 1e-6
    return dict(chirps_per_frame=n_chirps, samples=n_s, fs_msps=fs / 1e6, t_adc_us=t_adc * 1e6,
                t_chirp_us=t_chirp * 1e6, t_frame_ms=t_frame * 1e3,
                adc_duty=n_chirps * t_adc / t_frame if t_frame else float("nan"),
                ramp_duty=n_chirps * t_chirp / t_frame if t_frame else float("nan"),
                slope_mhz_us=float(params.get("freqSlopeMHz_us") or 0))


def collision_rate(t_adc, slope_v, b_if, rate_i, slope_i, overlap=1.0):
    """
    Expected number lambda of foreign chirps whose frequency line crosses the victim's
    within one ADC window, for an asynchronous chirped interferer (S_i != 0):

        lambda = rho_i * kappa * (T_s |S_v - S_i| + B_IF) / |S_i|

    T_s victim ADC window, S_v / S_i victim / interferer slopes (Hz/s, signed: a
    down-chirp is negative), B_IF the victim's IF bandwidth (~f_s for complex
    sampling), rho_i the interferer's mean chirp rate (chirps per frame / frame period),
    kappa the fraction of the victim's sampled sweep that lies inside the interferer's
    sweep span. Derivation: a crossing at victim time t fixes the interferer's start time
    t0 = t - (f_v(t) - f_i0) / S_i, so the set of start times that produce a crossing
    within the window extended by the in-band duration tau = B_IF / |S_v - S_i| has
    measure kappa (T_s + tau) |1 - S_v/S_i|. For parallel chirps (S_i -> S_v) it tends
    to rho_i kappa B_IF / |S_i| (the interferer must start within B_IF/S of the victim).
    P(chirp hit) = 1 - exp(-lambda); P(frame hit) = 1 - exp(-N_chirps lambda).
    """
    ds = abs(slope_v - slope_i)
    return rate_i * overlap * (t_adc * ds + b_if) / abs(slope_i)


def run(caps):
    rows = []
    for cap in caps:
        params, cube = common.load_cube(cap)
        damaged = vitalsigns.detect_damaged_frames(cube)
        pat = slot_pattern(params)
        d = detect(cube, pat, damaged)
        ds = detect(cube, pat, damaged, mode="specmed")
        F, C = d["hit"].shape
        duty = adc_duty(params)
        hpf = d["hit"].sum(1)
        extra = dict(hit_chirps_specmed=int(ds["hit"].sum()), hit_frames_specmed=int(ds["frame_hit"].sum()),
                     hit_chirp_rate_specmed=float(ds["hit"].sum() / (F * C)),
                     hit_frame_rate_specmed=float(ds["frame_hit"].mean()),
                     hit_chirps_per_hit_frame=float(hpf[hpf > 0].mean()) if (hpf > 0).any() else 0.0)
        row = dict(stem=cap.stem, day=cap.start.strftime("%Y-%m-%d"), config=common.config_label(params),
                   frames=F, chirps=F * C, damaged_frames=int(np.sum(damaged)),
                   packets_zero_filled=int(params.get("dca_packets_zero_filled") or 0),
                   bytes_zero_filled=int(params.get("dca_bytes_zero_filled") or 0),
                   packet_loss_pct=float(params.get("dca_packet_loss_pct") or 0),
                   hit_chirps=int(d["hit"].sum()), hit_frames=int(d["frame_hit"].sum()),
                   hit_chirp_rate=float(d["hit"].sum() / (F * C)),
                   hit_frame_rate=float(d["frame_hit"].mean()),
                   max_excess_db=float(np.nanmax(np.sort(d["excess_db"], axis=-1)[..., -MIN_RX])),
                   noise_floor_dbfs=noise_floor_dbfs(cube, params), **extra, **duty)
        rows.append(row)
        print(f"{cap.stem:34s} {row['day']} cfg {row['config']:5s} frames {F:5d} hit chirps "
              f"{row['hit_chirps']:6d} frames {row['hit_frames']:5d} ({row['hit_frame_rate']:.2%}) "
              f"[specmed {row['hit_chirps_specmed']} / {row['hit_frames_specmed']}] "
              f"floor {row['noise_floor_dbfs']:.1f} dBFS zero-filled pkts {row['packets_zero_filled']} "
              f"duty {row['adc_duty']:.2%}", flush=True)
        del cube
    return rows


def per_day(rows):
    out = []
    for day in sorted({r["day"] for r in rows}):
        sel = [r for r in rows if r["day"] == day]
        ch = sum(r["chirps"] for r in sel)
        fr = sum(r["frames"] for r in sel)
        out.append(dict(day=day, recordings=len(sel), frames=fr, chirps=ch,
                        hit_chirps=sum(r["hit_chirps"] for r in sel),
                        hit_frames=sum(r["hit_frames"] for r in sel),
                        hit_chirp_rate=sum(r["hit_chirps"] for r in sel) / ch,
                        hit_frame_rate=sum(r["hit_frames"] for r in sel) / fr,
                        recordings_with_hits=sum(r["hit_chirps"] > 0 for r in sel),
                        noise_floor_dbfs_median=float(np.median([r["noise_floor_dbfs"] for r in sel])),
                        noise_floor_dbfs_min=float(np.min([r["noise_floor_dbfs"] for r in sel])),
                        noise_floor_dbfs_max=float(np.max([r["noise_floor_dbfs"] for r in sel])),
                        packets_zero_filled=sum(r["packets_zero_filled"] for r in sel),
                        damaged_frames=sum(r["damaged_frames"] for r in sel),
                        hit_chirps_specmed=sum(r["hit_chirps_specmed"] for r in sel),
                        hit_frames_specmed=sum(r["hit_frames_specmed"] for r in sel),
                        **{f"noise_floor_dbfs_median[{c}]": float(np.median([r["noise_floor_dbfs"] for r in sel
                                                                             if r["config"] == c]))
                           for c in sorted({r["config"] for r in sel})}))
    return out


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="*", default=["2026-10-06", "2026-10-07"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    caps = [c for d in args.days for c in common.captures(d)]
    if args.only:
        caps = [c for c in caps if c.stem in args.only]
    rows = run(caps)
    os.makedirs(common.OUT_DIR, exist_ok=True)
    write_csv(os.path.join(common.OUT_DIR, f"interference_recordings{args.tag}.csv"), rows)
    days = per_day(rows)
    write_csv(os.path.join(common.OUT_DIR, f"interference_days{args.tag}.csv"), days)
    for d in days:
        print(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
