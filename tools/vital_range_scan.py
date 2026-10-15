"""
Where in range a person is, judged by breathing rather than by the strongest echo.

Picking the bin with the most reflected power, or the steadiest phase, points at a
wall: a wall is strong and it does not move, and those are exactly what such scores
reward. Breathing is the discriminating signature - millimetres of chest travel at
0.12-0.6 Hz, held for as long as the person is there - so this scans every range bin
and scores how much of its low-frequency energy sits in one breathing peak.

Two guards matter, both learned by watching simpler versions fail:

  - a floor on reflected power. After the static component is removed an empty bin is
    pure noise, and the arctangent of noise is a full-scale phase walk: on one capture
    the largest apparent breathing displacement in the whole scene, 4.8 mm, sat at
    3.84 m where there was nothing at all.
  - agreement across neighbouring bins. A torso spans several bins that share one
    breathing rate; a spurious peak does not. Irregular breathing has no single
    rate - the spectral peak of neighbouring chest bins jumped between 8 and 16 /min -
    so a neighbour also agrees when its breathing waveform follows this bin's.

And two more, from a seated person at 1.1 m who breathed 9 times in a minute, slowly
and unevenly, while the scan named 2.15 m: an uneven chest spreads its breathing over
the band and scores a low concentration, where a still reflector with a faint ripple
scores a high one. So

  - a bin must actually move: less than MIN_BREATH_MM of breathing-band travel is not
    a chest (the bin at 2.15 m moved 0.004 mm, the chest 1.3 mm);
  - the score is weighted by how much the bin's echo moves at breathing rates, its
    complex slow time in the breathing band against the bin that moves most. The chest
    moves most; its reflections further out (2.9 m, 30 dB less) move less.

The breaths are also counted one by one (vitalsigns.count_breaths), the way a person
counts them; for uneven breathing that count is the rate to believe, as the spectral
peak lands on a harmonic or on the longest pause. They are counted in the breathing band
every view shares (vital_bands), filtered as Fastcheck filters it: the scoring band
starts at 7.2 /min, and breaths at 6.6-7.4 /min lost there to their own harmonics
counted 12-13 where the person counted 9 (the shared band: 9 in every chest bin).

The cardiac rate is reported per bin, but only as a readout: chest travel from the
heart is some 20-30 dB below breathing, so it runs out of signal much closer in. A
cardiac figure from a bin whose windowed estimate scatters is not worth trusting, and
the scatter is printed next to it.

Run with:
    python -m tools.vital_range_scan <log_or_bin> [--min-range M] [--max-range M]
                                     [--top N] [--csv OUT]

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
from dataclasses import dataclass, asdict

import numpy as np
from scipy import signal as sps

import vital_bands
import vitalsigns
import capture_params
from ai_vital_analysis import AIVitalAnalysis
from radar_processing import (compute_range_fft_cube, derive_capture_dimensions,
                              load_raw_cube)
from tools.captures import load_capture

RESP_BAND = (0.12, 0.6)
CARD_BAND = (0.7, 2.6)
WIDE_BAND = (0.05, 2.0)
POWER_FLOOR_DB = -20.0          # relative to the strongest bin in the search window
NEIGHBOUR_BINS = 2
RATE_TOLERANCE_HZ = 0.025
WAVEFORM_AGREE = 0.8            # |correlation| of two bins' breathing waveforms
MIN_BREATH_MM = 0.05            # breathing-band travel (95th percentile) below this is no chest


@dataclass
class BinResult:
    range_m: float
    power_db: float
    resp_concentration: float
    resp_per_min: float
    resp_displacement_mm: float
    neighbours_agreeing: int
    score: float
    cardiac_bpm: float
    cardiac_scatter_bpm: float
    motion_db: float = float("nan")         # breathing-band motion of the echo, vs the most moving bin
    breaths: int = 0                         # breaths counted one by one
    counted_per_min: float = float("nan")   # their rate (NaN unless they stand above the noise)
    breath_cv: float = float("nan")         # spread of the intervals between them


def _displacement(an, iq, lam):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        phase = an.extract_phase(an.dc_offset_correction(iq), method="circlefit")
        return an.phase_to_displacement(np.nan_to_num(phase), lam)


def _windowed_rate(sig, fs, band, win_s=20.0, step_s=4.0):
    """Median and scatter of the dominant rate across windows, in cycles per minute."""
    n, k = int(win_s * fs), max(1, int(step_s * fs))
    if len(sig) < n:
        return float("nan"), float("nan")
    rates = []
    for i in range(0, len(sig) - n + 1, k):
        f, p = sps.welch(sig[i:i + n], fs, nperseg=min(n, 4096), nfft=1 << 15)
        m = (f > band[0]) & (f < band[1])
        if m.any():
            rates.append(60.0 * f[m][int(np.argmax(p[m]))])
    if not rates:
        return float("nan"), float("nan")
    return float(np.median(rates)), float(np.std(rates))


def _breathing_to_count(disp, fps, band):
    """disp in band, the slow drift taken out first (as Fastcheck does before counting)."""
    lo, hi = band[0], min(band[1], 0.45 * fps)
    try:
        x = sps.sosfiltfilt(sps.butter(2, 0.6 * lo, btype="highpass", fs=fps, output="sos"), disp)
        return sps.sosfiltfilt(sps.butter(4, [lo, hi], btype="band", fs=fps, output="sos"), x)
    except ValueError:                   # too short to filter
        return None


def _motion(slow, fps):
    """Mean power of each bin's complex echo in the breathing band, static part removed."""
    hi = min(RESP_BAND[1], 0.45 * fps)
    x = slow - slow.mean(axis=0, keepdims=True)
    try:
        sos = sps.butter(2, [RESP_BAND[0], hi], btype="band", fs=fps, output="sos")
        x = sps.sosfiltfilt(sos, x, axis=0)
    except ValueError:                   # too short to filter: all of it is the motion
        pass
    return (np.abs(x) ** 2).mean(axis=(0, 1))


def scan(slow, range_m, fps, wavelength, min_range=0.4, max_range=4.0,
         power_floor_db=POWER_FLOOR_DB) -> list[BinResult]:
    """
    Score every range bin of a slow-time cube shaped [frames, rx, range_bins].

    Returns one entry per bin above the power floor, highest score first.
    """
    an = AIVitalAnalysis()
    b0 = int(np.searchsorted(range_m, min_range))
    b1 = int(np.searchsorted(range_m, max_range))
    if b1 <= b0:
        return []
    amp = np.abs(slow).mean(axis=(0, 1))
    ref = amp[b0:b1].max()
    amp_db = 20.0 * np.log10(amp / (ref + 1e-12) + 1e-12)

    motion = _motion(slow[:, :, b0:b1], fps)
    motion_db = 10.0 * np.log10(motion / (motion.max() + 1e-30) + 1e-30)

    count_band = vital_bands.current().resp_hz
    rows: dict[int, BinResult] = {}
    rate_hz: dict[int, float] = {}
    waves: dict[int, np.ndarray] = {}
    for b in range(b0, b1):
        if amp_db[b] < power_floor_db:
            continue
        rx = int(np.argmax(np.abs(slow[:, :, b]).mean(axis=0)))
        disp = _displacement(an, slow[:, rx, b], wavelength)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            resp = an.safe_bandpass(disp, fps, *RESP_BAND)
            card = an.safe_bandpass(disp, fps, *CARD_BAND)
        nper = min(len(resp), int(60 * fps))
        f, p = sps.welch(resp, fps, nperseg=nper, nfft=1 << 15)
        band = (f > RESP_BAND[0]) & (f < RESP_BAND[1])
        wide = (f > WIDE_BAND[0]) & (f < WIDE_BAND[1])
        if not band.any():
            continue
        conc = float(p[band].max() / (p[wide].sum() + 1e-30))
        peak_hz = float(f[band][int(np.argmax(p[band]))])
        hr, hr_scatter = _windowed_rate(card, fps, CARD_BAND)
        to_count = _breathing_to_count(disp, fps, count_band)
        cnt = (vitalsigns.count_breaths(
                   to_count, fps, count_band,
                   vitalsigns.breathing_noise_rms(disp, fps, count_band, CARD_BAND[1]))
               if to_count is not None else vitalsigns.count_breaths(np.zeros(0), fps, count_band))
        rate_hz[b] = peak_hz
        waves[b] = resp
        rows[b] = BinResult(
            range_m=float(range_m[b]), power_db=float(amp_db[b]),
            resp_concentration=conc, resp_per_min=peak_hz * 60.0,
            resp_displacement_mm=float(np.percentile(np.abs(resp), 95) * 1000.0),
            neighbours_agreeing=0, score=0.0,
            cardiac_bpm=hr, cardiac_scatter_bpm=hr_scatter,
            motion_db=float(motion_db[b - b0]), breaths=cnt["n"],
            counted_per_min=cnt["rate"] if cnt["ok"] else float("nan"), breath_cv=cnt["cv"],
        )

    def agrees(b, k):
        if abs(rate_hz[k] - rate_hz[b]) < RATE_TOLERANCE_HZ:
            return True
        a, c = waves[b], waves[k]
        if a.std() == 0 or c.std() == 0:
            return False
        return abs(float(np.corrcoef(a, c)[0, 1])) >= WAVEFORM_AGREE

    for b, row in rows.items():
        agree = sum(1 for k in range(b - NEIGHBOUR_BINS, b + NEIGHBOUR_BINS + 1)
                    if k != b and k in rate_hz and agrees(b, k))
        score = (row.resp_concentration * (1 + agree) * 10.0 ** (row.motion_db / 20.0)
                 if row.resp_displacement_mm >= MIN_BREATH_MM else 0.0)
        rows[b] = BinResult(**{**asdict(row), "neighbours_agreeing": agree, "score": score})
    return sorted(rows.values(), key=lambda r: r.score, reverse=True)


def scan_capture(log_or_bin: str, min_range=0.4, max_range=4.0, n_fft_range=512):
    """Load a capture, reduce it to slow time and scan it. Returns (rows, info)."""
    log = log_or_bin
    if log.endswith(".bin"):
        stem = os.path.basename(log).split("_Raw_")[0]
        log = os.path.join(os.path.dirname(log), f"{stem}_LogFile.txt")
    cap = load_capture(log)
    if cap is None:
        raise FileNotFoundError(f"No capture beside {log}")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cube = load_raw_cube(cap.bin_paths[0], cap.params,
                             derive_capture_dimensions(cap.params))
        # One sample per frame: averaging the chirps of a frame is 10*log10(N) of SNR
        # and the vital bands are three orders of magnitude below the frame rate.
        reduced = cube.mean(axis=1, keepdims=True)
        del cube
        rfft, meta = compute_range_fft_cube(reduced, cap.params, n_fft_range=n_fft_range,
                                            remove_dc=True, window="hann",
                                            return_complex=True)
    slow = rfft[:, 0]
    range_m = np.asarray(meta["range_m"])
    fps = cap.fps
    lam = capture_params.wavelength_m(cap.params)       # middle of the ADC window (roadmap 1.9)
    rows = scan(slow, range_m, fps, lam, min_range, max_range)
    info = {"stem": cap.stem, "fps": fps, "frames": int(slow.shape[0]),
            "wavelength_mm": lam * 1000.0, "duration_s": slow.shape[0] / fps if fps else 0.0}
    return rows, info


def report(rows: list[BinResult], info: dict, top: int = 10) -> str:
    lines = [f"{info['stem']}: {info['frames']} frames at {info['fps']:.0f} fps "
             f"({info['duration_s']:.0f} s), lambda {info['wavelength_mm']:.2f} mm",
             ""]
    if not rows:
        return "\n".join(lines + ["No bin passed the power floor."])
    lines.append(f"{'range':>7}{'dB':>7}{'move':>7}{'conc':>8}{'peak/min':>9}"
                 f"{'counted':>8}{'/min':>6}{'mm':>7}{'nb':>4}{'score':>8}{'bpm':>8}{'+/-':>7}")
    for r in rows[:top]:
        lines.append(f"{r.range_m:>7.2f}{r.power_db:>7.1f}{r.motion_db:>7.1f}"
                     f"{r.resp_concentration:>8.3f}{r.resp_per_min:>9.1f}"
                     f"{r.breaths:>8d}{r.counted_per_min:>6.1f}{r.resp_displacement_mm:>7.2f}"
                     f"{r.neighbours_agreeing:>4}{r.score:>8.4f}"
                     f"{r.cardiac_bpm:>8.1f}{r.cardiac_scatter_bpm:>7.1f}")
    best = rows[0]
    if best.score <= 0:
        return "\n".join(lines + ["", "No bin moves like a breathing chest "
                                       f"(at least {MIN_BREATH_MM} mm of breathing-band travel)."])
    uneven = np.isfinite(best.breath_cv) and best.breath_cv > 0.25
    if np.isfinite(best.counted_per_min) and best.breaths >= 3:
        rate = (f"{best.counted_per_min:.1f} breaths/min ({best.breaths} counted"
                f"{', uneven' if uneven else ''}; spectral peak {best.resp_per_min:.1f})")
    else:
        rate = f"{best.resp_per_min:.1f} breaths/min (spectral peak)"
    lines += ["", f"Most breathing-like bin: {best.range_m:.2f} m, {rate}, "
                  f"{best.neighbours_agreeing} neighbour(s) agreeing."]
    if best.cardiac_scatter_bpm > 5.0:
        lines.append(f"Its cardiac estimate scatters by {best.cardiac_scatter_bpm:.1f} bpm "
                     f"across windows - report breathing only.")
    if best.neighbours_agreeing == 0:
        lines.append("No neighbouring bin shares the rate, so this is a weak candidate.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("capture", help="the capture's _LogFile.txt or _Raw_0.bin")
    ap.add_argument("--min-range", type=float, default=0.4)
    ap.add_argument("--max-range", type=float, default=4.0)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--csv", help="write every scanned bin to this file")
    args = ap.parse_args(argv)
    try:
        rows, info = scan_capture(args.capture, args.min_range, args.max_range)
    except (FileNotFoundError, KeyError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(report(rows, info, args.top))
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))
        print(f"\nWritten to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
