"""
Swings of legs and arms in a recording and their maximum speed (roadmap phase 10).

The Micro-Doppler tab without the GUI: limb_motion.analyse on a recording, a report of
every swing (when, how long, which way, how fast, from what height), and CSV files of
the swings and of the envelopes. With a motion-capture reference it also compares
every swing's speed with the radial speed of a joint.

The reference is a CSV with a time column (time_s, time, t or timestamp) and the
joint's position, either in columns x, y, z with a `joint` column naming the joint
(long format), or in columns <joint>_x, <joint>_y, <joint>_z (wide format); or the
JSON written by the RemoteCaptury example recorder (root position only). Positions and
--radar-pos must be in the same coordinates; only the radar's position matters, not
which way it faces, because a radial speed is the rate of change of a distance.

Run with:
    python -m tools.limb_speed <log_or_bin> [--from S] [--to S] [--bins 5-100] [--fixed]
        [--width M] [--window-ms 20] [--threshold-db 8] [--min-swing 1.5]
        [--clutter mean] [--tau 5] [--antenna MRC] [--radar-height M]
        [--csv SWINGS.csv] [--envelopes ENV.csv]
        [--reference MOCAP.csv --radar-pos X,Y,Z [--joint NAME] [--offset S]
         [--ref-units mm|m] [--ref-time-units s|ms|us]]

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys

import numpy as np

import capture_params
import clutter
import limb_motion as lm
from radar_processing import derive_capture_dimensions, load_raw_cube
from tools.captures import load_capture

UNITS = {"m": 1.0, "mm": 1e-3, "cm": 1e-2}
TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6}
TIME_COLUMNS = ("time_s", "time", "t", "timestamp")


def _log_for(path: str) -> str:
    if path.lower().endswith(".bin"):
        stem = os.path.basename(path)
        for tail in ("_Raw_0.bin", "_0.bin", ".bin"):
            if stem.endswith(tail):
                stem = stem[: -len(tail)]
                break
        return os.path.join(os.path.dirname(path), f"{stem}_LogFile.txt")
    return path


def _parse_bins(text: str | None):
    if not text:
        return None
    a, _, b = text.partition("-")
    return (int(a), int(b or a))


def load_reference(path: str, joint: str | None = None, pos_scale: float = 1e-3,
                   time_scale: float = 1.0):
    """(time_s, xyz_m) of one joint from a motion-capture export (see the module notes)."""
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        poses = data.get("poses") or {}
        actor = joint if joint in poses else next(iter(poses), None)
        if actor is None:
            raise ValueError(f"{path}: no poses")
        rows = sorted((p for p in poses[actor] if p.get("root_position") is not None),
                      key=lambda p: p["timestamp"])
        t = np.array([p["timestamp"] for p in rows], float) * time_scale
        xyz = np.array([p["root_position"] for p in rows], float) * pos_scale
        return t, xyz
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = [c.strip() for c in (reader.fieldnames or [])]
    low = {c.lower(): c for c in fields}
    tcol = next((low[c] for c in TIME_COLUMNS if c in low), None)
    if tcol is None:
        raise ValueError(f"{path}: no time column ({', '.join(TIME_COLUMNS)})")
    jcol = low.get("joint") or low.get("name")
    if jcol and "x" in low:                                   # long format
        if joint:
            rows = [r for r in rows if r[jcol].strip() == joint]
        if not rows:
            raise ValueError(f"{path}: no rows for joint {joint!r}")
        cols = (low["x"], low["y"], low["z"])
    else:                                                     # wide format
        prefix = joint or ""
        cols = tuple(low.get(f"{prefix}_{a}".lower()) or low.get(f"{prefix}{a}".lower())
                     for a in "xyz")
        if None in cols:
            raise ValueError(f"{path}: no columns {prefix}_x/_y/_z")
    t = np.array([float(r[tcol]) for r in rows]) * time_scale
    xyz = np.array([[float(r[c]) for c in cols] for r in rows]) * pos_scale
    order = np.argsort(t)
    return t[order], xyz[order]


def load(path: str):
    """(capture, raw cube) of a recording, quietly."""
    cap = load_capture(_log_for(path))
    if cap is None:
        raise FileNotFoundError(f"No capture beside {path}")
    with contextlib.redirect_stdout(io.StringIO()):
        cube = load_raw_cube(cap.bin_paths[0], cap.params, derive_capture_dimensions(cap.params))
    return cap, cube


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("capture", help="the recording's _LogFile.txt or _Raw_0.bin")
    ap.add_argument("--from", dest="t_from", type=float, help="segment start [s]")
    ap.add_argument("--to", dest="t_to", type=float, help="segment end [s]")
    ap.add_argument("--bins", help="range bins searched for the person, e.g. 5-100")
    ap.add_argument("--fixed", action="store_true", help="use all of --bins instead of following the person")
    ap.add_argument("--width", type=float, default=lm.DEFAULT_GATE_WIDTH_M, help="automatic gate width [m]")
    ap.add_argument("--window-ms", type=float, default=1e3 * lm.DEFAULT_WINDOW_S)
    ap.add_argument("--threshold-db", type=float, default=lm.DEFAULT_THRESHOLD_DB)
    ap.add_argument("--min-swing", type=float, default=lm.DEFAULT_SWING_M_S, help="[m/s]")
    ap.add_argument("--clutter", default="mean", choices=list(clutter.METHODS))
    ap.add_argument("--tau", type=float, default=5.0, help="EMA time constant [s]")
    ap.add_argument("--rank", type=int, default=clutter.DEFAULT_RANK, help="SVD rank")
    ap.add_argument("--antenna", default="MRC", help="MRC, SUM or RXk")
    ap.add_argument("--radar-height", type=float, help="radar height above the floor [m]")
    ap.add_argument("--csv", help="write the swings to this file")
    ap.add_argument("--envelopes", help="write torso line and envelopes to this file")
    ap.add_argument("--reference", help="motion-capture CSV or RemoteCaptury JSON")
    ap.add_argument("--radar-pos", help="radar position in the reference coordinates, X,Y,Z")
    ap.add_argument("--joint", help="joint (or actor, for JSON) in the reference")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="reference time + offset = radar recording time [s]")
    ap.add_argument("--ref-units", default="mm", choices=list(UNITS))
    ap.add_argument("--ref-time-units", default="s", choices=list(TIME_UNITS))
    args = ap.parse_args(argv)

    try:
        cap, cube = load(args.capture)
        params = dict(cap.params)
        period = capture_params.frame_period_s(params) or 0.0
        params.update(clutter_method=args.clutter, clutter_rank=args.rank,
                      clutter_alpha=clutter.alpha_from_time_constant(args.tau, period)
                      if period else clutter.DEFAULT_ALPHA)
        fps = capture_params.derive_fps(params, cap.stem)
        f0, f1 = 0, cube.shape[0]
        if args.t_from is not None or args.t_to is not None:
            f0, f1 = capture_params.segment_frames(cube.shape[0], fps, args.t_from, args.t_to)
        with contextlib.redirect_stdout(io.StringIO()):
            res = lm.analyse(cube[f0:f1], params, bins=_parse_bins(args.bins),
                             gate="fixed" if args.fixed else "auto", gate_width_m=args.width,
                             window_s=args.window_ms / 1e3, rx_mode=args.antenna,
                             t0_s=f0 / fps, threshold_db=args.threshold_db,
                             swing_m_s=args.min_swing, radar_height_m=args.radar_height)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"{cap.stem}: {f1 - f0} frames ({(f1 - f0) / fps:.1f} s from {f0 / fps:.1f} s)")
    print(lm.describe(res))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(lm.SWING_FIELDS))
            w.writeheader()
            for row in lm.swing_rows(res.swings):
                w.writerow(row)
        print(f"\nSwings written to {args.csv}")
    if args.envelopes:
        arr, header = lm.envelope_table(res.env)
        np.savetxt(args.envelopes, arr, delimiter=",", header=header, comments="")
        print(f"Envelopes written to {args.envelopes}")

    if args.reference:
        if not args.radar_pos:
            print("--reference needs --radar-pos X,Y,Z", file=sys.stderr)
            return 1
        radar = np.array([float(v) for v in args.radar_pos.split(",")]) * UNITS[args.ref_units]
        t, xyz = load_reference(args.reference, args.joint, UNITS[args.ref_units],
                                TIME_UNITS[args.ref_time_units])
        v = lm.radial_velocity_from_positions(t, xyz, radar)
        cmp = lm.compare_with_reference(res.swings, t, v, offset_s=args.offset)
        print(f"\nReference {os.path.basename(args.reference)}"
              + (f" ({args.joint})" if args.joint else "") + f": {cmp['n']} swings matched")
        for sw, ref in cmp["pairs"]:
            print(f"  {sw.t_peak_s:8.2f} s {sw.direction:>8}  radar {sw.v_peak_m_s:+6.2f}  "
                  f"reference {ref:+6.2f}  difference {sw.v_peak_m_s - ref:+5.2f} m/s")
        if cmp["n"]:
            print(f"  bias {cmp['bias_m_s']:+.2f}, MAE {cmp['mae_m_s']:.2f}, RMSE "
                  f"{cmp['rmse_m_s']:.2f} m/s, correlation {cmp['correlation']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
