"""
Finding radar captures on disk and reading what they say about themselves.

A capture is an mmWave Studio API log, the recorder's own CSV and one or more raw
binaries sharing a stem:

    test_D_LogFile.txt
    test_D_Raw_LogFile.csv
    test_D_Raw_0.bin

The wall clock comes from the recorder CSV, not from the API log - mmWave Studio
appends every capture to the same log, so its first timestamp is when the session's
first command was issued, which on the captures this was written against ran from 37 s
to 28 min before the data actually starts. parse_logfile_full already prefers the
recorder's value and records which one it used in radar_start_source.

An AWR2243 cascade take has <stem>_cascade.json in place of the API log
(cascade/decode.py gives the same params); its log_path is that file.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import contextlib
import glob
import io
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

from capture_params import derive_fps
from dca1000_decode import find_raw_log_csv, parse_logfile_full
from radar_processing import get_all_bin_parts


@dataclass
class Capture:
    """One recording and the parameters parsed from its log."""
    stem: str
    log_path: str
    bin_paths: tuple[str, ...]
    params: dict

    @property
    def csv_path(self) -> str | None:
        if self.log_path.endswith("_cascade.json"):
            csv = self.log_path[: -len("_cascade.json")] + "_Raw_LogFile.csv"
            return csv if os.path.isfile(csv) else None
        return find_raw_log_csv(self.log_path)

    @property
    def bytes_total(self) -> int:
        return sum(os.path.getsize(p) for p in self.bin_paths)

    @property
    def start(self) -> datetime | None:
        ts = self.params.get("radar_start_timestamp")
        return datetime.fromtimestamp(float(ts)) if ts else None

    @property
    def duration_s(self) -> float:
        return float(self.params.get("dca_duration_s") or 0.0)

    @property
    def end(self) -> datetime | None:
        s = self.start
        return s + timedelta(seconds=self.duration_s) if s else None

    @property
    def fps(self) -> float:
        # The same rule the GUI uses, including its fallbacks, so a tool and a tab
        # cannot disagree about the frame rate of one recording.
        return derive_fps(self.params, self.stem)

    @property
    def time_source(self) -> str:
        return str(self.params.get("radar_start_source") or "unknown")


def first_bin_for(log_path: str) -> str | None:
    """
    The first raw binary belonging to a log, whichever way it was named.

    mmWave Studio writes <stem>_Raw_0.bin; the DCA1000 command line writes <stem>.bin
    (adc_data.bin on the AWR2544, data_test.bin in chuze_dca_noha). find_log_file in
    the decoder accepts all of these, so the tools have to as well - otherwise they
    skip those recordings without saying so.
    """
    folder = os.path.dirname(log_path)
    stem = os.path.basename(log_path)[: -len("_LogFile.txt")]
    for name in (f"{stem}_Raw_0.bin", f"{stem}_0.bin", f"{stem}.bin"):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return None


def load_capture(log_path: str, quiet: bool = True) -> Capture | None:
    """Parse one capture's log. Returns None when it has no raw data beside it. A
    cascade take is found by its <stem>_cascade.json - given directly, or in place of a
    <stem>_LogFile.txt that does not exist (what the tools derive from a _Raw_0.bin)."""
    if log_path.endswith("_cascade.json"):
        return load_cascade_capture(log_path, quiet=quiet)
    if log_path.endswith("_LogFile.txt") and not os.path.isfile(log_path):
        side = log_path[: -len("_LogFile.txt")] + "_cascade.json"
        if os.path.isfile(side):
            return load_cascade_capture(side, quiet=quiet)
    stem = os.path.basename(log_path)[: -len("_LogFile.txt")]
    first_bin = first_bin_for(log_path)
    if first_bin is None:
        return None
    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
    with ctx:
        params, _ = parse_logfile_full(log_path)
    return Capture(stem=stem, log_path=log_path,
                   bin_paths=tuple(get_all_bin_parts(first_bin)), params=params)


def load_cascade_capture(sidecar: str, quiet: bool = True) -> Capture | None:
    """An AWR2243 cascade take, from its _cascade.json (None without raw data)."""
    from cascade.decode import cascade_params
    stem = os.path.basename(sidecar)[: -len("_cascade.json")]
    first_bin = os.path.join(os.path.dirname(sidecar), f"{stem}_Raw_0.bin")
    if not os.path.isfile(first_bin):
        return None
    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
    with ctx:
        params, _ = cascade_params(first_bin)
    return Capture(stem=stem, log_path=sidecar,
                   bin_paths=tuple(get_all_bin_parts(first_bin)), params=params)


def find_captures(directory: str, quiet: bool = True) -> list[Capture]:
    """Every capture in a directory (cascade takes too), oldest first; undated ones last,
    by name."""
    out = []
    for log in sorted(glob.glob(os.path.join(directory, "*_LogFile.txt"))):
        cap = load_capture(log, quiet=quiet)
        if cap is not None:
            out.append(cap)
    for side in sorted(glob.glob(os.path.join(directory, "*_cascade.json"))):
        cap = load_cascade_capture(side, quiet=quiet)
        if cap is not None:
            out.append(cap)
    return sorted(out, key=lambda c: (c.start is None, c.start or datetime.min, c.stem))


def chirps_per_frame(cap: Capture) -> float:
    """
    How many chirps a frame actually holds, counted from the file rather than the log.

    The byte count cannot disagree with itself, and across the captures this was
    written against the same profile was recorded with 5, 10, 50 and 200 chirps.
    """
    p = cap.params
    frames = (p.get("frameConfig") or {}).get("numFrames")
    samples = p.get("numAdcSamples")
    rx_mask = p.get("dataFmt_rxChanEnMask")
    if not (frames and samples and rx_mask):
        return 0.0
    n_rx = bin(int(rx_mask)).count("1")
    is_complex = "COMPLEX" in str(p.get("dataFmt_adcFmt", "")).upper()
    per_sample = 2 if is_complex else 1
    int16_total = cap.bytes_total / 2
    return int16_total / per_sample / (n_rx * int(samples)) / int(frames)
