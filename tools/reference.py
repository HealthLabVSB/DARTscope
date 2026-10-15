"""
Reading a reference recording (somnograph export) at the level of whole sessions.

A somnograph export is one directory per session, one text file per signal, each with
a header that carries the absolute wall clock:

    Signal Type: Pleth_Type
    Start Time: 10.07.2026 14:31:41
    Sample Rate: 128
    Length: 80896

    Data:
    10.07.2026 14:31:41,000; 0

AIVitalAnalysis.parse_pleth_txt_file already reads one such file into a signal. This
module deliberately stops at the header for the discovery pass: lining a session up
against a capture needs its start, rate and length, and a session holds several files
of hundreds of thousands of samples each.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

_HEADER_BYTES = 512
_START_RE = re.compile(r"Start Time:\s*(.+)")
_RATE_RE = re.compile(r"Sample Rate:\s*([\d.]+)")
_LENGTH_RE = re.compile(r"Length:\s*(\d+)")
_TYPE_RE = re.compile(r"Signal Type:\s*(.+)")

# Battery state is logged like a signal but starts before the recording proper, so it
# would widen every session by a few seconds for no reason.
NON_PHYSIOLOGICAL = {"accu"}


@dataclass(frozen=True)
class Series:
    """One signal file: what it is, when it starts and how long it runs."""
    name: str
    path: str
    signal_type: str
    start: datetime
    fs: float
    samples: int

    @property
    def duration_s(self) -> float:
        return self.samples / self.fs if self.fs > 0 else 0.0

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.duration_s)


@dataclass(frozen=True)
class Session:
    """A directory of signals recorded together."""
    name: str
    path: str
    series: tuple[Series, ...]

    @property
    def physiological(self) -> tuple[Series, ...]:
        return tuple(s for s in self.series if s.name not in NON_PHYSIOLOGICAL)

    @property
    def start(self) -> datetime:
        return min(s.start for s in self.physiological or self.series)

    @property
    def end(self) -> datetime:
        return min(s.end for s in self.physiological or self.series)

    def starts_agree(self) -> bool:
        """
        Whether every physiological signal shares one start time.

        When they do, a capture is lined up with one offset for the whole session
        instead of one per signal.
        """
        starts = {s.start for s in self.physiological}
        return len(starts) <= 1

    def find(self, prefix: str) -> Series | None:
        """First signal whose file name starts with `prefix`, case-insensitively."""
        low = prefix.lower()
        for s in self.series:
            if s.name.lower().startswith(low):
                return s
        return None


def read_series(path: str) -> Series | None:
    """One signal's header, or None when the file carries no usable one."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(_HEADER_BYTES)
    except OSError:
        return None
    m_start, m_rate, m_len = _START_RE.search(head), _RATE_RE.search(head), _LENGTH_RE.search(head)
    if not (m_start and m_rate and m_len):
        return None
    try:
        start = datetime.strptime(m_start.group(1).strip(), "%d.%m.%Y %H:%M:%S")
        fs = float(m_rate.group(1))
        samples = int(m_len.group(1))
    except ValueError:
        return None
    if fs <= 0:
        return None
    m_type = _TYPE_RE.search(head)
    base = os.path.basename(path)
    return Series(
        name=base.split(" - ")[0].strip(),
        path=path,
        signal_type=(m_type.group(1).strip() if m_type else "?"),
        start=start, fs=fs, samples=samples,
    )


def read_session(directory: str) -> Session | None:
    """Every signal in one session directory, or None when it holds none."""
    if not os.path.isdir(directory):
        return None
    series = []
    for entry in sorted(os.listdir(directory)):
        if not entry.lower().endswith(".txt"):
            continue
        s = read_series(os.path.join(directory, entry))
        if s is not None:
            series.append(s)
    if not series:
        return None
    return Session(name=os.path.basename(directory.rstrip(os.sep)),
                   path=directory, series=tuple(series))


def find_sessions(root: str) -> list[Session]:
    """Every session directory under `root`, oldest first."""
    if not os.path.isdir(root):
        return []
    found = []
    for entry in sorted(os.listdir(root)):
        s = read_session(os.path.join(root, entry))
        if s is not None:
            found.append(s)
    return sorted(found, key=lambda s: s.start)


def read_values(series: Series, start: datetime | None = None,
                duration_s: float | None = None):
    """
    The samples of one signal, optionally only the window a capture covers.

    Rows are "<timestamp>; <value>"; the timestamps are regular, so the window is cut
    by index off the header's start rather than by parsing every row.
    """
    import numpy as np

    values = []
    with open(series.path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("Data:"):
                break
        for line in f:
            parts = line.rsplit(";", 1)
            if len(parts) == 2:
                try:
                    values.append(float(parts[1]))
                except ValueError:
                    pass
    out = np.asarray(values, dtype=float)
    if start is None:
        return out
    i0 = int(round((start - series.start).total_seconds() * series.fs))
    i0 = max(0, i0)
    i1 = len(out) if duration_s is None else i0 + int(round(duration_s * series.fs))
    return out[i0:min(i1, len(out))]
