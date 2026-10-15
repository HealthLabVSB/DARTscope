"""
Antenna calibration from a corner-reflector recording, and a check of it on another.

Estimate (reflector at a known angle, boresight by default):
    python -m tools.calibrate <log_or_bin> --range 1.5 [--az 0 --el 0]
                              [--true-range 1.50] [--empty <log_or_bin>]
                              [--phase-only] [--antenna NAME] --out cal.json

Verify on a second recording with the reflector somewhere else:
    python -m tools.calibrate <log_or_bin> --verify cal.json --range 2.0 --az 25 --el 0

The estimate alone always looks perfect - it is fitted to that recording. Only the
second position says whether it holds. See calibration.py for the procedure.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

import numpy as np

import antenna_config
import calibration
from radar_processing import compute_range_fft_cube, derive_capture_dimensions, load_raw_cube
from tools.captures import load_capture


def _log_for(path: str) -> str:
    if path.endswith(".bin"):
        stem = os.path.basename(path).split("_Raw_")[0]
        return os.path.join(os.path.dirname(path), f"{stem}_LogFile.txt")
    return path


def load_range_cube(path: str, frames: int = 200, antenna: str | None = None,
                    n_fft_range: int = 256):
    """(range cube, params, range axis) of the first `frames` frames of a capture."""
    cap = load_capture(_log_for(path))
    if cap is None:
        raise FileNotFoundError(f"no capture at {path}")
    params = dict(cap.params)
    if antenna:
        cfg = antenna_config.get_config(antenna)
        if cfg is None:
            raise ValueError(f"unknown antenna configuration {antenna!r}; "
                             f"known: {antenna_config.config_names()}")
        params["antenna_config"] = cfg
    with contextlib.redirect_stdout(io.StringIO()):
        cube = load_raw_cube(cap.bin_paths[0], params, derive_capture_dimensions(params))
        cube = cube[:frames] if frames else cube
        rc, meta = compute_range_fft_cube(cube, params, window="hann", pad_pow2=False,
                                          remove_dc=True, n_fft_range=n_fft_range,
                                          return_complex=True)
    return rc, params, np.asarray(meta["range_m"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("capture", help="_LogFile.txt or _Raw_0.bin of the reflector recording")
    ap.add_argument("--range", type=float, required=True, help="approximate reflector range [m]")
    ap.add_argument("--window", type=float, default=0.15, help="search +- this around --range [m]")
    ap.add_argument("--az", type=float, default=0.0, help="reflector azimuth [deg]")
    ap.add_argument("--el", type=float, default=0.0, help="reflector elevation [deg]")
    ap.add_argument("--true-range", type=float, help="measured distance, to report range bias [m]")
    ap.add_argument("--empty", help="the same scene without the reflector, subtracted")
    ap.add_argument("--phase-only", action="store_true", help="do not correct channel gains")
    ap.add_argument("--antenna", help="antenna configuration name (default: from the log)")
    ap.add_argument("--frames", type=int, default=200, help="frames to use (default 200)")
    ap.add_argument("--out", help="write the calibration to this JSON file")
    ap.add_argument("--verify", metavar="CAL_JSON", help="check this calibration instead")
    args = ap.parse_args(argv)

    rc, params, r_axis = load_range_cube(args.capture, args.frames, args.antenna)
    if args.verify:
        cal = calibration.load(args.verify)
        v = calibration.verify(rc, params, cal, args.range, args.az, args.el, args.window,
                               range_axis=r_axis)
        print(f"reflector at {r_axis[v['range_bin']]:.3f} m, known angle az {args.az:+.1f}, "
              f"el {args.el:+.1f} deg (margin {v['reflector_margin_db']:.1f} dB)")
        print(f"  without calibration: az {v['angle_before'][0]:+.2f}, el {v['angle_before'][1]:+.2f}"
              f"  error {v['error_before_deg']:.2f} deg, coherence {v['coherence_before']:.3f}")
        print(f"  with calibration:    az {v['angle_after'][0]:+.2f}, el {v['angle_after'][1]:+.2f}"
              f"  error {v['error_after_deg']:.2f} deg, coherence {v['coherence_after']:.3f}")
        return 0 if v["error_after_deg"] <= v["error_before_deg"] else 2

    background = None
    if args.empty:
        background, _, _ = load_range_cube(args.empty, args.frames, args.antenna)
    cal = calibration.estimate(rc, params, args.range, args.az, args.el, args.window,
                               background=background, true_range_m=args.true_range,
                               range_axis=r_axis, amplitude=not args.phase_only)
    cal.meta["capture"] = os.path.basename(_log_for(args.capture))
    if args.empty:
        cal.meta["empty_scene"] = os.path.basename(_log_for(args.empty))
    print(cal.summary())
    if cal.reflector_margin_db < 20:
        print(f"\nWARNING: the reflector is only {cal.reflector_margin_db:.1f} dB above the "
              f"range cells around it - something else is probably in its cell. Record an "
              f"empty scene (--empty) or move the reflector.", file=sys.stderr)
    if args.out:
        calibration.save(args.out, cal)
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
