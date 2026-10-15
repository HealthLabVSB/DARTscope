"""
How the Vital Signs view does against a somnograph (SOMNOtouch) on the recordings that
one of its sessions covers.

    python -m tools.vitals_check <radar_dir> [--reference DIR] [--csv OUT]
                                [--win-br S] [--win-hr S] [--only STEM ...]

For every capture a session of the reference directory covers (default:
somno_vysledky inside the radar directory, as tools.pair_reference finds them) the
Vital Signs chain runs as the view runs it with its defaults (vitalsigns.DEFAULT_CONFIG:
Auto Bin, every receiver, 30 s heart windows), and every window is compared with the
reference over the same span of time:

  heart      the somnograph's Pulse, and the beats of its Pleth waveform (the pulse
             peaks of the PPG). The Pulse read 1.6 bpm above the beats on the six seated
             recordings the tool was written against.
  breathing  these exports hold no belt, so three surrogates of the Pleth are used - its
             baseline (respiratory-induced intensity variation), the pulse amplitude and
             the beat-to-beat rate - and only windows where all three agree within
             2 /min (smart fusion, Karlen et al. 2013). Two are not enough: the baseline
             and the beat rate both carry the ~0.1 Hz Mayer wave of the blood pressure,
             and on one recording they agreed on 6 /min while the chest and the pulse
             amplitude went 17.

The ECG channels of these exports were flat (the electrodes were not attached), hence
the Pleth. Reports per capture and pooled: MAE and bias, the share within 5 bpm (heart)
or 2 /min (breathing), and the coverage - the share of windows whose confidence passed
Min peak/avg.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
from scipy import signal

import capture_params
import vitalsigns
from radar_processing import (compute_range_fft_cube, derive_capture_dimensions,
                              load_raw_cube, range_fft_size)
from tools.captures import find_captures
from tools.pair_reference import DEFAULT_REFERENCE_DIRNAME
from tools.reference import find_sessions, read_values

MARGIN_S = 40.0                  # reference read this far beyond the capture, for filters
RESP_FS = 4.0                    # breathing surrogates resampled to this rate
BR_AGREE_SD = 2.0                # the three surrogates within this of their mean, per minute
HR_WITHIN = 5.0
BR_WITHIN = 2.0


# --- the reference -----------------------------------------------------------------------------

def ppg_beats(v: np.ndarray, fs: float):
    """(peak indices, pulse amplitudes) of a PPG: band-pass 0.5-5 Hz, peaks at least
    0.33 s apart with a prominence of 30 % of the typical pulse, beats whose interval
    strays more than 25 % from the local median dropped."""
    b, a = signal.butter(2, [0.5 / (fs / 2), 5 / (fs / 2)], btype="band")
    y = signal.filtfilt(b, a, np.asarray(v, float))
    prom = 0.3 * (np.percentile(y, 95) - np.percentile(y, 5))
    pk, props = signal.find_peaks(y, distance=max(1, int(0.33 * fs)), prominence=prom)
    amp = props["prominences"]
    if pk.size >= 8:
        rr = np.diff(pk)
        med = signal.medfilt(rr.astype(float), 7)
        keep = np.r_[True, np.abs(rr - med) < 0.25 * med]
        pk, amp = pk[keep], amp[keep]
    return pk, amp


def _band(x, fs, lo=0.08, hi=0.7):
    b, a = signal.butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return signal.filtfilt(b, a, x)


def breathing_surrogates(pleth: np.ndarray, fs: float, t0: float = 0.0):
    """{name: (t, x)} at RESP_FS: the Pleth baseline (riiv), the pulse amplitude (riav)
    and the beat-to-beat rate (rifv), each band-passed to 0.08-0.7 Hz. t0: the time of
    the first Pleth sample."""
    v = np.asarray(pleth, float)
    tp = t0 + np.arange(v.size) / fs
    tg = np.arange(tp[0] + 2.0, tp[-1] - 2.0, 1.0 / RESP_FS)
    out = {}
    if tg.size < 16:
        return out
    b, a = signal.butter(2, 0.7 / (fs / 2))
    out["riiv"] = (tg, _band(np.interp(tg, tp, signal.filtfilt(b, a, v)), RESP_FS))
    pk, amp = ppg_beats(v, fs)
    if pk.size > 10:
        bt = tp[pk]
        out["riav"] = (tg, _band(np.interp(tg, bt, amp), RESP_FS))
        out["rifv"] = (tg, _band(np.interp(tg, bt[1:], 60.0 / np.diff(bt)), RESP_FS))
    return out


def peak_rate(x: np.ndarray, fs: float = RESP_FS, band=(5 / 60, 40 / 60)) -> float:
    """The strongest true peak (higher than both neighbours) in the band, per minute."""
    x = signal.detrend(np.asarray(x, float))
    n = len(x)
    if n < 8:
        return float("nan")
    nfft = max(4096, 1 << (n - 1).bit_length())
    f, p = signal.periodogram(x * np.hanning(n), fs=fs, nfft=nfft)
    k = np.flatnonzero((f >= band[0]) & (f <= band[1]))
    if not k.size or p[k].max() <= 0:
        return float("nan")
    peak = (p[k] >= p[np.maximum(k - 1, 0)]) & (p[k] >= p[np.minimum(k + 1, len(p) - 1)])
    if not peak.any():
        return float("nan")
    j = k[peak][np.argmax(p[k][peak])]
    return float(f[j] * 60.0)


def fused_breathing(rates: dict, max_sd: float = BR_AGREE_SD) -> float:
    """The mean of the three surrogates' rates when all three agree within max_sd, else
    nan (smart fusion)."""
    vals = np.array([rates.get(k, np.nan) for k in ("riiv", "riav", "rifv")], float)
    if np.all(np.isfinite(vals)) and np.std(vals) <= max_sd:
        return float(vals.mean())
    return float("nan")


@dataclass
class Reference:
    """A session's Pleth and Pulse around one capture, times in s from the capture start."""
    session: str
    duration_s: float
    beat_t: np.ndarray
    pulse_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pulse: np.ndarray = field(default_factory=lambda: np.zeros(0))
    resp: dict = field(default_factory=dict)

    def hr_beats(self, t0: float, t1: float) -> float:
        t = self.beat_t[(self.beat_t >= t0) & (self.beat_t <= t1)]
        rr = np.diff(t)
        rr = rr[(rr > 0.33) & (rr < 1.6)]
        return 60.0 / float(rr.mean()) if rr.size >= 2 else float("nan")

    def hr_pulse(self, t0: float, t1: float) -> float:
        m = (self.pulse_t >= t0) & (self.pulse_t <= t1) & (self.pulse > 30) & (self.pulse < 220)
        return float(self.pulse[m].mean()) if m.any() else float("nan")

    def breathing(self, t0: float, t1: float):
        """(fused rate or nan, {surrogate: rate}) over [t0, t1]."""
        rates = {}
        for name, (t, x) in self.resp.items():
            m = (t >= t0) & (t <= t1)
            if m.sum() >= 8 * RESP_FS:
                rates[name] = peak_rate(x[m])
        return fused_breathing(rates), rates


def load_reference(cap, directory: str) -> Reference | None:
    """The session of `directory` that covers the capture, or None."""
    if cap.start is None:
        return None
    sessions = [s for s in find_sessions(directory) if s.start <= cap.start and s.end >= cap.end]
    if not sessions:
        return None
    ses = sessions[0]
    pl = ses.find("Pleth")
    if pl is None:
        return None
    start = cap.start - timedelta(seconds=MARGIN_S)
    span = float(cap.duration_s) + 2 * MARGIN_S
    v = read_values(pl, start, span)
    if v.size < 10 * pl.fs:
        return None
    pk, _ = ppg_beats(v, pl.fs)
    ref = Reference(ses.name, float(cap.duration_s), pk / pl.fs - MARGIN_S)
    ref.resp = breathing_surrogates(v, pl.fs, -MARGIN_S)
    pls = ses.find("Pulse") or ses.find("Puls")
    if pls is not None:
        p = read_values(pls, start, span)
        ref.pulse_t = np.arange(p.size) / pls.fs - MARGIN_S
        ref.pulse = p
    return ref


# --- the radar --------------------------------------------------------------------------------

def vitals_of(cap, config: dict | None = None) -> dict:
    """The Vital Signs view's result for a capture with its default settings."""
    params = dict(cap.params)
    with contextlib.redirect_stdout(io.StringIO()):
        cube = load_raw_cube(cap.bin_paths[0], params, derive_capture_dimensions(dict(params)))
    damaged = vitalsigns.detect_damaged_frames(cube)
    x = cube.mean(axis=1, keepdims=True)       # only the chirp mean enters the chain
    del cube
    n = int(params.get("numAdcSamples") or x.shape[-1])
    with contextlib.redirect_stdout(io.StringIO()):
        rf, meta = compute_range_fft_cube(x, params, return_complex=True,
                                          n_fft_range=max(256, range_fft_size(n, None)))
        fps = capture_params.derive_fps(params)
    r = np.asarray(meta.get("range_m") if meta.get("range_m") is not None else meta["range_axis"])
    cfg = {**vitalsigns.DEFAULT_CONFIG, **(config or {}), "damaged_frames": damaged}
    with contextlib.redirect_stdout(io.StringIO()):
        return vitalsigns.process_vital_signs_logic(rf, params, fps, r, cfg)


# --- the comparison ---------------------------------------------------------------------------

def stats(err, within: float) -> dict:
    e = np.asarray(err, float)
    e = e[np.isfinite(e)]
    if not e.size:
        return {"n": 0}
    return {"n": int(e.size), "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e)),
            "within": float(np.mean(np.abs(e) <= within))}


def compare(res: dict, ref: Reference, win_br: float, win_hr: float) -> dict:
    """Per window: radar against the reference over the same span. Returns the rows and
    the statistics for the heart (Pulse, PPG beats) and the breathing (fused)."""
    th = res["conf_th"]
    rows = []
    for t, v, c in zip(res["hr_times"], res["hr_values"], res["hr_confs"]):
        h = win_hr / 2
        rows.append(("heart", float(t), float(v), float(c), bool(c >= th),
                     ref.hr_pulse(t - h, t + h), ref.hr_beats(t - h, t + h)))
    for t, v, c in zip(res["rr_times"], res["rr_values"], res["rr_confs"]):
        h = win_br / 2
        rows.append(("breathing", float(t), float(v), float(c), bool(c >= th),
                     ref.breathing(t - h, t + h)[0], float("nan")))
    hr = [r for r in rows if r[0] == "heart"]
    br = [r for r in rows if r[0] == "breathing"]

    def err(sel, col, valid_only):
        return [r[2] - r[col] for r in sel if (r[4] or not valid_only)]
    return {
        "rows": rows,
        "hr_pulse": stats(err(hr, 5, False), HR_WITHIN),
        "hr_beats": stats(err(hr, 6, False), HR_WITHIN),
        "hr_coverage": float(np.mean([r[4] for r in hr])) if hr else float("nan"),
        "br": stats(err(br, 5, False), BR_WITHIN),
        "br_valid": stats(err(br, 5, True), BR_WITHIN),
        "br_coverage": float(np.mean([r[4] for r in br])) if br else float("nan"),
        "br_ref_share": float(np.mean([np.isfinite(r[5]) for r in br])) if br else float("nan"),
    }


def _fmt(s: dict, unit: str) -> str:
    if not s.get("n"):
        return "      -        "
    return f"{s['mae']:4.1f} ({s['bias']:+4.1f}) {s['within']:4.0%}"


def pooled(results: dict, key: str, within: float) -> dict:
    errs = []
    for res in results.values():
        col = {"hr_pulse": 5, "hr_beats": 6, "br": 5}[key]
        kind = "breathing" if key == "br" else "heart"
        errs += [r[2] - r[col] for r in res["rows"] if r[0] == kind]
    return stats(errs, within)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("radar_dir")
    ap.add_argument("--reference", help=f"somnograph folder (default: {DEFAULT_REFERENCE_DIRNAME} "
                                        f"in the radar folder)")
    ap.add_argument("--csv", help="every window with the reference over it")
    ap.add_argument("--win-br", type=float, default=vitalsigns.DEFAULT_CONFIG["win_br"])
    ap.add_argument("--win-hr", type=float, default=vitalsigns.DEFAULT_CONFIG["win_hr"])
    ap.add_argument("--only", nargs="*", help="these captures only (stems)")
    args = ap.parse_args(argv)
    ref_dir = args.reference or os.path.join(args.radar_dir, DEFAULT_REFERENCE_DIRNAME)
    config = {"win_br": args.win_br, "win_hr": args.win_hr}
    results = {}
    print(f"{'capture':18s} {'range':>6s} | {'heart vs Pulse':15s} {'vs PPG beats':15s} {'valid':>5s} | "
          f"{'breathing':15s} {'ref':>4s} {'valid':>5s}   (MAE (bias) share within "
          f"{HR_WITHIN:g} bpm / {BR_WITHIN:g} /min)")
    shortest = max(args.win_br, args.win_hr)
    for cap in find_captures(args.radar_dir):
        if args.only and cap.stem not in args.only:
            continue
        ref = load_reference(cap, ref_dir)
        if ref is None:
            continue
        if not cap.duration_s or cap.duration_s < shortest:
            print(f"{cap.stem:18s} skipped: {cap.duration_s or 0:.0f} s is shorter than the "
                  f"{shortest:g} s windows")
            continue
        res = vitals_of(cap, config)
        out = compare(res, ref, args.win_br, args.win_hr)
        results[cap.stem] = out
        print(f"{cap.stem:18s} {res['dist_m']:5.2f}m | {_fmt(out['hr_pulse'], 'bpm'):15s} "
              f"{_fmt(out['hr_beats'], 'bpm'):15s} {out['hr_coverage']:5.0%} | {_fmt(out['br'], '/min'):15s} "
              f"{out['br_ref_share']:4.0%} {out['br_coverage']:5.0%}")
    if not results:
        print(f"no capture in {args.radar_dir} is covered by a session in {ref_dir}")
        return 1
    print(f"{'pooled':18s} {'':6s} | {_fmt(pooled(results, 'hr_pulse', HR_WITHIN), 'bpm'):15s} "
          f"{_fmt(pooled(results, 'hr_beats', HR_WITHIN), 'bpm'):15s} {'':5s} | "
          f"{_fmt(pooled(results, 'br', BR_WITHIN), '/min'):15s}")
    print("breathing 'ref': the share of windows with a breathing reference (the three PPG "
          "surrogates agreeing)")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["capture", "kind", "time_s", "radar", "confidence", "valid",
                        "reference", "reference_ppg_beats"])
            for stem, out in results.items():
                for r in out["rows"]:
                    w.writerow([stem, r[0], f"{r[1]:.1f}", f"{r[2]:.2f}", f"{r[3]:.2f}", int(r[4]),
                                "" if not np.isfinite(r[5]) else f"{r[5]:.2f}",
                                "" if not np.isfinite(r[6]) else f"{r[6]:.2f}"])
        print(f"windows written to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
