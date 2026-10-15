"""
Heart rate of a person who may be moving, in a recording (roadmap phase 12), and how it
compares with a reference (12.10).

moving_hr.analyse on a recording: window by window the motion (rest, light, strong),
breathing, step frequency and heart rate, as a report and optionally a CSV. With a
reference it prints, for all windows and for each motion class, the number of windows,
the MAE, the bias with the Bland-Altman limits of agreement, the share within 5 and
10 bpm and the coverage (the share of windows with a valid estimate).

The reference is either
    --reference somno        the Pulse of the somnograph session that covers the
                             recording, found by its wall-clock time in the folder
                             somno_vysledky next to the recording (or --reference-dir);
    --reference FILE.csv     a CSV with a time column (time_s, time, t or timestamp,
                             seconds from the start of the recording unless --offset
                             says otherwise) and a heart-rate column (hr_bpm, hr, bpm,
                             heart_rate or pulse) - an ECG belt, for example.

Run with:
    python -m tools.heart_rate <log_or_bin> [--from S] [--to S] [--method auto]
        [--win 12] [--max-change 3] [--range 0.3-3.0] [--allow-strong]
        [--csv WINDOWS.csv] [--reference somno|FILE.csv] [--reference-dir DIR]
        [--offset S]

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys

import numpy as np

import capture_params
import moving_hr
from radar_processing import derive_capture_dimensions, load_raw_cube
from tools.captures import load_capture

TIME_COLUMNS = ("time_s", "time", "t", "timestamp")
HR_COLUMNS = ("hr_bpm", "hr", "bpm", "heart_rate", "pulse")


def _log_for(path: str) -> str:
    if path.lower().endswith(".bin"):
        stem = os.path.basename(path)
        for tail in ("_Raw_0.bin", "_0.bin", ".bin"):
            if stem.endswith(tail):
                stem = stem[: -len(tail)]
                break
        return os.path.join(os.path.dirname(path), f"{stem}_LogFile.txt")
    return path


def load_csv_reference(path: str):
    """(time_s, bpm) from a CSV with a time and a heart-rate column."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = [c.strip() for c in (reader.fieldnames or [])]
    low = {c.lower(): c for c in fields}
    tcol = next((low[c] for c in TIME_COLUMNS if c in low), None)
    hcol = next((low[c] for c in HR_COLUMNS if c in low), None)
    if tcol is None or hcol is None:
        raise ValueError(f"{path}: needs a time column ({', '.join(TIME_COLUMNS)}) and a "
                         f"heart-rate column ({', '.join(HR_COLUMNS)}); it has {fields}")
    t, v = [], []
    for r in rows:
        try:
            t.append(float(r[tcol]))
            v.append(float(r[hcol]))
        except (TypeError, ValueError):
            continue
    return np.array(t), np.array(v)


def somno_reference(cap, directory: str | None = None):
    """(time_s from the start of the recording, bpm) of the covering somnograph Pulse."""
    from tools.reference import find_sessions, read_values
    directory = directory or os.path.join(os.path.dirname(cap.log_path), "somno_vysledky")
    if cap.start is None:
        raise ValueError("the recording has no start time to pair a reference by")
    sessions = [s for s in find_sessions(directory) if s.start <= cap.start and s.end >= cap.end]
    if not sessions:
        raise ValueError(f"no session in {directory} covers {cap.start} - {cap.end}")
    series = sessions[0].find("Pulse") or sessions[0].find("Puls")
    if series is None:
        raise ValueError(f"session {sessions[0].name} has no Pulse signal")
    v = read_values(series, cap.start, cap.duration_s or None)
    return np.arange(v.size) / series.fs, v, sessions[0].name


def _parse_range(text):
    if not text:
        return moving_hr.R_MIN_M, None
    a, _, b = text.partition("-")
    return float(a), (float(b) if b else None)


def format_stats(stats: dict) -> str:
    lines = [f"{'class':7s} {'windows':>7s} {'MAE':>6s} {'bias':>6s} {'limits of agreement':>21s} "
             f"{'<=5':>5s} {'<=10':>5s} {'coverage':>8s}"]
    for name in ("all", "rest", "light", "strong"):
        s = stats[name]
        if "mae" in s:
            lo, hi = s["loa"]
            lines.append(f"{name:7s} {s['n']:7d} {s['mae']:6.1f} {s['bias']:+6.1f} "
                         f"{lo:+9.1f} to {hi:+6.1f} {s['within5'] * 100:4.0f}% "
                         f"{s['within10'] * 100:4.0f}% {s['coverage'] * 100:7.0f}%")
        else:
            cov = "-" if not np.isfinite(s["coverage"]) else f"{s['coverage'] * 100:.0f}%"
            lines.append(f"{name:7s} {s['n']:7d} {'-':>6s} {'-':>6s} {'-':>21s} {'-':>5s} "
                         f"{'-':>5s} {cov:>8s}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("capture", help="the recording's _LogFile.txt or _Raw_0.bin")
    ap.add_argument("--from", dest="t_from", type=float, help="segment start [s]")
    ap.add_argument("--to", dest="t_to", type=float, help="segment end [s]")
    ap.add_argument("--method", default="auto", choices=["auto", "both", "displacement", "envelope"])
    ap.add_argument("--win", type=float, default=moving_hr.HR_WIN_S, help="window [s]")
    ap.add_argument("--max-change", type=float, default=moving_hr.MAX_CHANGE_BPM_S,
                    help="largest likely change of the heart rate [bpm/s]")
    ap.add_argument("--range", dest="range_m", help="range searched for the trunk, e.g. 0.3-3.0 [m]")
    ap.add_argument("--allow-strong", action="store_true",
                    help="estimate in strong motion too (untested on real recordings)")
    ap.add_argument("--csv", help="write the windows to this file")
    ap.add_argument("--reference", help="'somno' or a CSV with time and heart rate")
    ap.add_argument("--reference-dir", help="somnograph folder (default: somno_vysledky beside the recording)")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="reference time + offset = recording time [s] (CSV reference)")
    args = ap.parse_args(argv)

    try:
        cap = load_capture(_log_for(args.capture))
        if cap is None:
            raise FileNotFoundError(f"No capture beside {args.capture}")
        with contextlib.redirect_stdout(io.StringIO()):
            cube = load_raw_cube(cap.bin_paths[0], cap.params, derive_capture_dimensions(cap.params))
        fps = cap.fps
        f0, f1 = 0, cube.shape[0]
        if args.t_from is not None or args.t_to is not None:
            f0, f1 = capture_params.segment_frames(cube.shape[0], fps, args.t_from, args.t_to)
        r_min, r_max = _parse_range(args.range_m)
        res = moving_hr.analyse(cube[f0:f1], cap.params, fps=fps, method=args.method,
                                win_s=args.win, max_change_bpm_s=args.max_change,
                                r_min=r_min, r_max=r_max, t0_s=f0 / fps,
                                allow_strong=args.allow_strong)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"{cap.stem}: {f1 - f0} frames ({(f1 - f0) / fps:.1f} s from {f0 / fps:.1f} s), "
          f"trunk at {np.median(res.range_m):.2f} m (median)")
    print(moving_hr.describe(res))

    if args.csv:
        arr, header = res.window_table()
        np.savetxt(args.csv, arr, delimiter=",", header=header, comments="", fmt="%.6g")
        print(f"\nWindows written to {args.csv}")

    if args.reference:
        try:
            if args.reference.lower() == "somno":
                t, v, name = somno_reference(cap, args.reference_dir)
                offset = 0.0
                print(f"\nReference: Pulse of '{name}'")
            else:
                t, v = load_csv_reference(args.reference)
                offset = args.offset
                print(f"\nReference: {args.reference}")
        except (OSError, ValueError) as e:
            print(f"reference: {e}", file=sys.stderr)
            return 1
        stats = moving_hr.compare_with_reference(res, t, v, offset_s=offset)
        print("Valid windows against the reference [bpm]:")
        print(format_stats(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
