"""
Which reference session covers which radar capture.

Both sides carry an absolute wall clock - the recorder CSV for the radar, the header of
every signal file for the somnograph - so the pairing is a matter of comparing
intervals rather than lining waveforms up by eye. What the tool reports is the offset
into the session at which the capture starts, which is what
AIVitalAnalysis.synchronize_reference_with_radar needs.

Run with:
    python -m tools.pair_reference <radar_dir> [--reference DIR] [--json OUT]

The reference directory defaults to "somno_vysledky" inside the radar directory.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

from tools.captures import Capture, find_captures
from tools.reference import Session, find_sessions

DEFAULT_REFERENCE_DIRNAME = "somno_vysledky"


@dataclass
class Pairing:
    capture: str
    session: str | None
    capture_start: str | None
    duration_s: float
    offset_into_session_s: float | None
    covered_pct: float
    time_source: str
    note: str = ""


def overlap_s(cap: Capture, ses: Session) -> float:
    """Seconds of the capture that the session also covers."""
    if cap.start is None or cap.end is None:
        return 0.0
    return max(0.0, (min(cap.end, ses.end) - max(cap.start, ses.start)).total_seconds())


def pair(captures: list[Capture], sessions: list[Session]) -> list[Pairing]:
    """Best-covering session for each capture, or none when nothing overlaps."""
    out = []
    for cap in captures:
        if cap.start is None:
            out.append(Pairing(cap.stem, None, None, cap.duration_s, None, 0.0,
                               cap.time_source, "capture has no wall clock"))
            continue
        best, best_ov = None, 0.0
        for ses in sessions:
            ov = overlap_s(cap, ses)
            if ov > best_ov:
                best, best_ov = ses, ov
        dur = cap.duration_s or 0.0
        pct = 100.0 * best_ov / dur if dur > 0 else 0.0
        out.append(Pairing(
            capture=cap.stem,
            session=best.name if best else None,
            capture_start=cap.start.strftime("%Y-%m-%d %H:%M:%S"),
            duration_s=dur,
            offset_into_session_s=((cap.start - best.start).total_seconds() if best else None),
            covered_pct=round(pct, 1),
            time_source=cap.time_source,
            note="" if best else "no reference session covers this capture",
        ))
    return out


def report(pairings: list[Pairing], sessions: list[Session]) -> str:
    lines = ["Reference sessions", "=" * 78]
    for s in sessions:
        warn = "" if s.starts_agree() else "   !! signals disagree on the start time"
        lines.append(f"{s.name}")
        lines.append(f"   {s.start:%Y-%m-%d %H:%M:%S} -> {s.end:%H:%M:%S}  "
                     f"({(s.end - s.start).total_seconds() / 60:.1f} min, "
                     f"{len(s.physiological)} signals){warn}")
    if not sessions:
        lines.append("   none found")

    lines += ["", "Captures", "=" * 78,
              f"{'capture':<22}{'start':<21}{'s':>6}  {'session':<34}{'offset':>9}"]
    for p in pairings:
        ses = p.session or "-- none --"
        off = f"{p.offset_into_session_s:+.1f} s" if p.offset_into_session_s is not None else "-"
        lines.append(f"{p.capture:<22}{p.capture_start or '-':<21}{p.duration_s:>6.0f}  "
                     f"{ses:<34}{off:>9}")
    covered = [p for p in pairings if p.session and p.covered_pct >= 99.9]
    lines += ["", f"{len(covered)} of {len(pairings)} captures are fully covered by a reference."]
    weak = [p for p in pairings if p.session and p.covered_pct < 99.9]
    if weak:
        lines.append("Partially covered: " +
                     ", ".join(f"{p.capture} ({p.covered_pct:.0f} %)" for p in weak))
    odd = {p.time_source for p in pairings if p.session}
    if odd - {"DCA1000 recorder log"}:
        lines.append("NOTE: some captures date themselves from the API log, which is "
                     "when the session's first command was issued, not the capture.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("radar_dir", help="directory holding the captures")
    ap.add_argument("--reference", help=f"reference root (default: <radar_dir>/{DEFAULT_REFERENCE_DIRNAME})")
    ap.add_argument("--json", help="also write the pairing to this file")
    args = ap.parse_args(argv)

    ref_dir = args.reference or os.path.join(args.radar_dir, DEFAULT_REFERENCE_DIRNAME)
    captures = find_captures(args.radar_dir)
    if not captures:
        print(f"No captures in {args.radar_dir}", file=sys.stderr)
        return 1
    sessions = find_sessions(ref_dir)
    pairings = pair(captures, sessions)
    print(report(pairings, sessions))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in pairings], f, indent=2, ensure_ascii=False)
        print(f"\nWritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
