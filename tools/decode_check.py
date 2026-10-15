"""
Decode a set of captures and check that what came out is self-consistent.

The unit tests prove the decode rules on synthetic streams. This is the other half:
run the real pipeline over a directory of recordings and look at the result, which is
how the lane and REAL container faults were found in the first place. Both looked fine
one file at a time - the byte count worked out, nothing raised - and only showed up as
a disagreement between captures of the same scene.

Per capture it reports what the log resolved to, whether the file size matches the
declared geometry, and whether the cube came out finite and non-empty. Across captures
recorded of one scene it compares the time-averaged range profiles: those should agree
closely whatever the lane configuration or ADC format, so a low correlation is a decode
fault rather than a difference in the room.

Run with:
    python -m tools.decode_check <dir> [--pattern GLOB] [--no-compare] [--json OUT]

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import gc
import io
import json
import sys
from dataclasses import asdict, dataclass, field

import numpy as np

from capture_health import frame_count_finding
from radar_processing import (_expected_int16_total, compute_range_fft_cube,
                              derive_capture_dimensions, load_raw_cube)
from tools.captures import find_captures

PROFILE_BINS = 129
PROFILE_FRAMES = 40


@dataclass
class CaptureCheck:
    stem: str
    status: str
    adc_format: str = ""
    lanes: int | None = None
    lanes_source: str = ""
    lanes_radar: int | None = None
    chirps_per_frame: float = 0.0
    frames_declared: int | None = None
    frames_decoded: int | None = None
    size_ratio: float | None = None
    fps: float | None = None
    recorded_s: float | None = None
    non_finite: int = 0
    all_zero: bool = False
    dead_rx: int = 0
    problems: list[str] = field(default_factory=list)


def _profile(cube, params, n_fft=256, frames=PROFILE_FRAMES):
    """Time-averaged range profile in dB, first `frames` frames, positive half."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mag, _ = compute_range_fft_cube(cube[:frames], params, n_fft_range=n_fft,
                                        remove_dc=True, window="hann")
    prof = np.asarray(mag).mean(axis=tuple(range(mag.ndim - 1)))[:PROFILE_BINS]
    prof = prof / (prof.max() + 1e-30)
    return 20.0 * np.log10(prof + 1e-12)


def check_capture(cap, want_profile=True):
    """Decode one capture and describe what came out."""
    res = CaptureCheck(stem=cap.stem, status="OK")
    p = cap.params
    res.adc_format = str(p.get("dataFmt_adcFmt") or "?")
    res.lanes = p.get("numLanes")
    res.lanes_source = str(p.get("numLanes_source") or "?")
    res.lanes_radar = p.get("numLanes_radar")
    res.frames_declared = (p.get("frameConfig") or {}).get("numFrames")
    res.fps = cap.fps or None
    res.recorded_s = cap.duration_s or None

    profile = None
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            dims = derive_capture_dimensions(p)
            expected = _expected_int16_total(dims)
            cube = load_raw_cube(cap.bin_paths[0], p, dims)
        actual = cap.bytes_total // 2
        res.size_ratio = round(actual / expected, 4) if expected else None
        res.frames_decoded = int(cube.shape[0])
        res.chirps_per_frame = float(cube.shape[1])
        a = np.abs(cube)
        res.non_finite = int((~np.isfinite(a)).sum())
        res.all_zero = bool(not a.any())
        res.dead_rx = int((a.mean(axis=(0, 1, 3)) < 1e-12).sum())
        if want_profile:
            profile = _profile(cube, p)
        del cube, a
        gc.collect()
    except Exception as e:
        res.status = "ERROR"
        res.problems.append(f"{type(e).__name__}: {e}")
        return res, None

    res.problems = problems(res)
    if res.problems:
        res.status = "SUSPECT"
    return res, profile


def problems(res: CaptureCheck) -> list[str]:
    """
    What is wrong with a decoded capture, if anything.

    The frame-count rule, and why it looks at frames rather than file size, lives in
    capture_health.frame_count_finding so the health report and this agree.
    """
    out = []
    if res.non_finite:
        out.append(f"{res.non_finite} non-finite samples")
    if res.all_zero:
        out.append("the whole cube is zero")
    if res.dead_rx:
        out.append(f"{res.dead_rx} RX channel(s) carry nothing")
    frames = frame_count_finding(res.frames_declared, res.frames_decoded, res.size_ratio,
                                 fps=res.fps, recorded_s=res.recorded_s)
    if frames is not None and frames.severity != "ok":
        out.append(frames.message)
    return out


def compare_profiles(profiles: dict[str, np.ndarray]) -> tuple[float, list[tuple]]:
    """Pairwise correlation of range profiles; returns (minimum, worst pairs)."""
    names = sorted(profiles)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = float(np.corrcoef(profiles[a], profiles[b])[0, 1])
            pairs.append((c, a, b))
    if not pairs:
        return float("nan"), []
    pairs.sort()
    return pairs[0][0], pairs[:5]


def report(results: list[CaptureCheck], worst, min_corr, compared: int) -> str:
    lines = [f"{'capture':<22}{'format':<11}{'lanes':>6}  {'source':<24}"
             f"{'chirps':>7}{'frames':>8}{'ratio':>7}  status"]
    for r in results:
        lines.append(f"{r.stem:<22}{r.adc_format:<11}{str(r.lanes):>6}  "
                     f"{r.lanes_source:<24}{r.chirps_per_frame:>7.0f}"
                     f"{str(r.frames_decoded or '-'):>8}"
                     f"{(f'{r.size_ratio:.2f}' if r.size_ratio else '-'):>7}  {r.status}")
        for p in r.problems:
            lines.append(f"{'':22}  - {p}")
    ok = sum(r.status == "OK" for r in results)
    lines += ["", f"{ok} OK, {sum(r.status == 'SUSPECT' for r in results)} suspect, "
                  f"{sum(r.status == 'ERROR' for r in results)} failed, of {len(results)}."]
    if compared >= 2:
        lines.append(f"Range profiles compared across {compared} captures, "
                     f"lowest correlation {min_corr:+.4f}.")
        if min_corr < 0.9:
            lines.append("Below 0.9 the captures disagree about the same room - look at:")
            for c, a, b in worst:
                lines.append(f"   {a} x {b}: {c:+.4f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("directory")
    ap.add_argument("--pattern", default="*", help="only captures whose stem matches")
    ap.add_argument("--no-compare", action="store_true",
                    help="skip the cross-capture range profile comparison")
    ap.add_argument("--json", help="also write the per-capture result to this file")
    args = ap.parse_args(argv)

    captures = [c for c in find_captures(args.directory)
                if fnmatch.fnmatch(c.stem, args.pattern)]
    if not captures:
        print(f"No captures matching {args.pattern!r} in {args.directory}", file=sys.stderr)
        return 1

    results, profiles = [], {}
    for cap in captures:
        res, prof = check_capture(cap, want_profile=not args.no_compare)
        results.append(res)
        if prof is not None:
            profiles[cap.stem] = prof
        print(f"  checked {cap.stem} -> {res.status}", file=sys.stderr)

    min_corr, worst = compare_profiles(profiles) if len(profiles) >= 2 else (float("nan"), [])
    print(report(results, worst, min_corr, len(profiles)))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
        print(f"\nWritten to {args.json}")
    return 0 if all(r.status != "ERROR" for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
