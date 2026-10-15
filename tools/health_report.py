"""
Health report for one or more captures: can this recording be trusted?

Checks the logs (recorder log present, start time source, lane width, packet loss,
ADC window, frame rate, antenna configuration), a prefix of the raw file (saturation,
container layout) and - unless --quick - the decoded cube (frame count, dead channels,
DC offset, IQ balance, zero-filled frames). See capture_health for what each means.

Run with:
    python -m tools.health_report <capture_or_dir> [...] [--quick] [--problems] [--json OUT]
    python -m tools.health_report --random N <dir> [--seed S]

Exit status is 2 when any capture has an error, so the tool can gate a batch.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import glob
import io
import json
import os
import random
import sys
from dataclasses import asdict

from capture_health import Finding, assess
from radar_processing import _expected_int16_total, derive_capture_dimensions, load_raw_cube
from tools.captures import load_capture


def _logs_in(path: str) -> list[str]:
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "**", "*_LogFile.txt"), recursive=True)
                      + glob.glob(os.path.join(path, "**", "*_cascade.json"), recursive=True))
    if path.endswith(("_LogFile.txt", "_cascade.json")):
        return [path]
    if path.endswith(".bin"):
        stem = os.path.basename(path).split("_Raw_")[0]
        return [os.path.join(os.path.dirname(path), f"{stem}_LogFile.txt")]
    return []


def report_capture(log_path: str, decode: bool = True):
    """Health report for one capture, or None when it has no raw data beside it."""
    cap = load_capture(log_path)
    if cap is None:
        return None
    cube, ratio = None, None
    if decode:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                dims = derive_capture_dimensions(cap.params)
                expected = _expected_int16_total(dims)
                cube = load_raw_cube(cap.bin_paths[0], cap.params, dims)
            ratio = (cap.bytes_total / 2) / expected if expected else None
        except Exception as e:
            rep = assess(cap.stem, cap.params, cap.bin_paths)
            rep.findings.append(Finding("decode", "error", f"{type(e).__name__}: {e}"))
            return rep
    rep = assess(cap.stem, cap.params, cap.bin_paths, cube=cube, size_ratio=ratio)
    del cube
    gc.collect()
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("paths", nargs="+", help="_LogFile.txt, _Raw_0.bin or a directory")
    ap.add_argument("--quick", action="store_true", help="skip decoding the cube")
    ap.add_argument("--problems", action="store_true", help="show only warnings and errors")
    ap.add_argument("--random", type=int, metavar="N", help="check N captures picked at random")
    ap.add_argument("--seed", type=int, help="seed for --random, printed so a run can be repeated")
    ap.add_argument("--json", help="also write the findings to this file")
    args = ap.parse_args(argv)

    logs = sorted({lg for p in args.paths for lg in _logs_in(p)})
    if args.random:
        seed = args.seed if args.seed is not None else random.randrange(1 << 30)
        print(f"random selection, seed {seed}")
        logs = sorted(random.Random(seed).sample(logs, min(args.random, len(logs))))
    if not logs:
        print("No captures found.", file=sys.stderr)
        return 1

    reports = []
    for lg in logs:
        rep = report_capture(lg, decode=not args.quick)
        if rep is None:
            continue
        reports.append(rep)
        print(rep.format(only_problems=args.problems))
        print()

    counts = {s: sum(r.worst == s for r in reports) for s in ("ok", "info", "warning", "error")}
    print(f"{len(reports)} capture(s): " + ", ".join(f"{v} {k}" for k, v in counts.items() if v))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([{"stem": r.stem, "worst": r.worst,
                        "findings": [asdict(x) for x in r.findings]} for r in reports],
                      f, indent=2, ensure_ascii=False)
        print(f"Written to {args.json}")
    return 2 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
