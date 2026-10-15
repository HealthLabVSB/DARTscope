"""
acquisition/recorder.py

DCA1000 data datagrams -> <prefix>_Raw_<n>.bin and <prefix>_Raw_LogFile.csv, byte for
byte as TI's DCA1000EVM_CLI_Record writes them (inline processing, recorddatarecv.cpp),
which is what mmWave Studio runs underneath and what every recording in the data folders
came from.

Per datagram (10-byte header: sequence number, bytes the FPGA sent before it):

    in order     the payload is appended
    a gap        as many zero bytes as the byte counter says went missing, then the payload
                 (not packets x 1456: the counter is exact even for a short last packet)
    late         written back over the zeros it left, when it lands inside a zero-filled
                 stretch; counted like TI does (received + 1, zero-filled - 1)

The file never contains headers, never reorders bytes (reorderEnable = 0 - with it the
loader decodes garbage, see docs/ACQUISITION.md) and is split into parts of at most
max_file_bytes, cut after a whole number of 1456-byte packets: 1 GiB parts come out at
1 073 741 760 bytes, as in testovani/vitals.

The CSV repeats TI's text exactly: CRLF line ends, one ",offset,bytes," line per
zero-filled stretch, the summary with MSVC's ctime ("Fri Dec 05 14:00:33 2025"). The
parser (dca1000_decode.parse_raw_log_csv) takes the lane mode, the packet counts and the
capture start time from it.

Two deliberate differences from TI, both for damaged streams only:
  - a late packet can be written back anywhere in the recording, not only inside the
    75 MB the CLI keeps in memory;
  - a datagram that repeats data already written is counted as a duplicate and left out,
    where TI would write it again and decrement the zero-filled count below the truth.
A third, for a stream whose first packets never arrived: when the first packet received
is not packet 1 but its byte counter shows exactly the packets before it were lost, the
file starts with that many zero bytes. The loader assumes the file starts on a frame
boundary; TI would start mid-frame and shift every frame of the recording.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

from .config import PAYLOAD_BYTES_PER_PACKET

MAX_LEADING_ZERO_FILL = 16 * 1024 * 1024
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def msvc_ctime(t: float) -> str:
    """C's ctime() as the MSVC runtime prints it: zero-padded day, trailing newline."""
    d = datetime.fromtimestamp(int(t))
    return (f"{_WEEKDAYS[d.weekday()]} {_MONTHS[d.month - 1]} {d.day:02d} "
            f"{d.hour:02d}:{d.minute:02d}:{d.second:02d} {d.year}\n")


def part_path(folder: str, prefix: str, index: int) -> str:
    return os.path.join(folder, f"{prefix}_Raw_{index}.bin")


def csv_path(folder: str, prefix: str) -> str:
    return os.path.join(folder, f"{prefix}_Raw_LogFile.csv")


@dataclass
class RecorderStats:
    first_seq: int | None = None
    last_seq: int | None = None
    received: int = 0                     # TI's "Number of received packets"
    out_of_sequence: int = 0              # TI's count: packets not following the previous one
    oos_from: int = 0
    oos_to: int = 0
    zero_filled_packets: int = 0
    zero_filled_bytes: int = 0
    late_recovered: int = 0
    late_dropped: int = 0
    duplicates: int = 0
    leading_zero_bytes: int = 0
    bytes_written: int = 0                # stream length on disk, zeros included
    start_time: float | None = None       # wall clock of the first packet, s
    end_time: float | None = None
    first_packet_ns: int | None = None    # time.time_ns() of the first packet
    last_packet_ns: int | None = None
    files: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class RawRecorder:
    """
    Feed it datagrams with feed(); close() finishes the files and the CSV.

    Not thread-safe: one receiver thread feeds it (receiver.UdpReceiver).
    """

    def __init__(self, folder: str, prefix: str, *, lvds_lanes: int = 2,
                 max_file_mb: int = 1024, write_buffer: int = 16 * 1024 * 1024,
                 clock_ns=time.time_ns):
        if lvds_lanes not in (2, 4):
            raise ValueError("the DCA1000 records 2 or 4 lanes")
        if not 1 <= int(max_file_mb) <= 1024:
            raise ValueError("maxRecFileSize must be 1..1024 MB")
        self.folder = folder
        self.prefix = prefix
        self.lvds_lanes = lvds_lanes
        self.max_file_mb = int(max_file_mb)
        self.max_file_bytes = int(max_file_mb) * 1024 * 1024
        self.write_buffer = write_buffer
        self.clock_ns = clock_ns
        self.stats = RecorderStats()
        self._csv = None
        self._fh = None
        self._count = 0                   # bytes in the current part
        self._parts: list[list] = []      # [stream start, path, size]
        self._pos = 0                     # stream offset of the next byte
        self._base = 0                    # FPGA byte counter at stream offset 0
        self._next = None                 # sequence number expected next
        self._prev = None                 # sequence number of the previous packet
        self._holes: list[list[int]] = []  # zero-filled [start, end) stream offsets
        self._closed = False
        self._open_time = time.time()

    # --- files -----------------------------------------------------------------------------
    def open(self) -> "RawRecorder":
        """Create the CSV with the record settings; refuse to overwrite anything."""
        os.makedirs(self.folder, exist_ok=True)
        for p in (csv_path(self.folder, self.prefix), part_path(self.folder, self.prefix, 0)):
            if os.path.exists(p):
                raise FileExistsError(f"{p} exists; a recording is never overwritten")
        # newline="\r\n": the CLI writes the CSV in text mode on Windows
        self._csv = open(csv_path(self.folder, self.prefix), "x", encoding="ascii",
                         newline="\r\n")
        self._open_time = time.time()
        self._csv.write("Start record configuration : \n,"
                        "\nLog mode : Raw"
                        f"\nLVDS lane mode : {self.lvds_lanes} lane"
                        "\nRecord stop mode : Infinite"
                        f"\nMax file size (MB) : {self.max_file_mb},"
                        "\n,*DT 1,")
        self._csv.flush()
        return self

    def _open_part(self):
        idx = len(self._parts)
        path = part_path(self.folder, self.prefix, idx)
        self._fh = open(path, "xb", buffering=self.write_buffer)
        self._parts.append([self._pos, path, 0])
        self._count = 0

    def _append(self, data) -> None:
        """Append to the stream, starting a new part the way the CLI does."""
        n = len(data)
        if self._count + n <= self.max_file_bytes:
            self._fh.write(data)
            self._count += n
            self._pos += n
            self._parts[-1][2] = self._count
            return
        mv = memoryview(data)
        while len(mv):
            room = self.max_file_bytes - self._count
            if len(mv) <= room:
                self._fh.write(mv)
                self._count += len(mv)
                self._pos += len(mv)
                self._parts[-1][2] = self._count
                return
            k = (room // PAYLOAD_BYTES_PER_PACKET) * PAYLOAD_BYTES_PER_PACKET
            if k:
                self._fh.write(mv[:k])
                self._count += k
                self._pos += k
                self._parts[-1][2] = self._count
                mv = mv[k:]
            self._fh.close()
            self._open_part()

    def _zeros(self, n: int) -> None:
        start = self._pos
        chunk = bytes(min(n, 1 << 20))
        left = n
        while left:
            k = min(left, len(chunk))
            self._append(chunk[:k] if k < len(chunk) else chunk)
            left -= k
        if self._holes and self._holes[-1][1] == start:
            self._holes[-1][1] = self._pos
        else:
            self._holes.append([start, self._pos])

    def _write_at(self, offset: int, data) -> None:
        """Write over bytes already in the stream (a late packet)."""
        mv = memoryview(data)
        while len(mv):
            for i, (start, path, size) in enumerate(self._parts):
                if start <= offset < start + size:
                    break
            else:
                raise ValueError("offset outside the recording")
            k = min(len(mv), start + size - offset)
            if i == len(self._parts) - 1:
                self._fh.seek(offset - start)
                self._fh.write(mv[:k])
                self._fh.seek(0, os.SEEK_END)
            else:
                with open(path, "r+b") as f:
                    f.seek(offset - start)
                    f.write(mv[:k])
            offset += k
            mv = mv[k:]

    # --- packets -----------------------------------------------------------------------------
    def feed(self, pkt, n: int | None = None, t_ns: int | None = None) -> None:
        """One data datagram (header included); n = its length when pkt is a larger buffer."""
        mv = memoryview(pkt)
        if n is None:
            n = len(mv)
        if n <= 10:
            return
        t_ns = self.clock_ns() if t_ns is None else t_ns
        seq = int.from_bytes(mv[0:4], "little")
        bc = int.from_bytes(mv[4:10], "little")
        st = self.stats
        if self._next is None:
            self._first_packet(seq, bc, t_ns)
        st.last_seq = seq
        st.last_packet_ns = t_ns
        if seq != self._prev + 1:
            st.out_of_sequence += 1
        self._prev = seq
        off = bc - self._base
        if seq == self._next:
            st.received += 1
            if off != self._pos:
                self._resync(off)
            self._append(mv[10:n])
            self._next = seq + 1
        elif seq > self._next:
            st.received += 1
            st.zero_filled_packets += seq - self._next
            st.oos_from, st.oos_to = self._next - 1, seq
            gap = off - self._pos
            if gap > 0:
                st.zero_filled_bytes += gap
                self._zeros(gap)
            elif gap < 0:
                self._resync(off)
            self._append(mv[10:n])
            self._next = seq + 1
        else:
            self._late(seq, off, mv, n)

    def _first_packet(self, seq: int, bc: int, t_ns: int):
        st = self.stats
        st.first_seq = seq
        st.first_packet_ns = t_ns
        st.start_time = t_ns / 1e9
        self._next = seq
        self._prev = seq - 1
        self._open_part()
        # packets lost before the first one that arrived: keep the file frame-aligned
        lost = seq - 1
        if lost > 0 and bc == lost * PAYLOAD_BYTES_PER_PACKET and bc <= MAX_LEADING_ZERO_FILL:
            self._base = 0
            st.zero_filled_packets += lost
            st.zero_filled_bytes += bc
            st.leading_zero_bytes = bc
            self._zeros(bc)
        else:
            self._base = bc

    def _resync(self, off: int):
        """
        The byte counter disagrees with what was written although no packet is missing.
        Nothing in TI's recorder covers this; follow the data rather than the counter.
        """
        self._base += off - self._pos

    def _late(self, seq: int, off: int, pkt, n: int):
        st = self.stats
        size = n - 10
        end = off + size
        for i, (h0, h1) in enumerate(self._holes):
            if h0 <= off and end <= h1:
                self._write_at(off, memoryview(pkt)[10:n])
                pieces = [p for p in ([h0, off], [end, h1]) if p[1] > p[0]]
                self._holes[i:i + 1] = pieces
                st.received += 1
                st.late_recovered += 1
                st.zero_filled_packets -= 1
                st.zero_filled_bytes -= size
                st.oos_from, st.oos_to = self._next - 1, seq
                return
        if end <= self._pos:
            st.duplicates += 1
        else:
            st.late_dropped += 1

    # --- the end -----------------------------------------------------------------------------
    @property
    def stream_bytes(self) -> int:
        return self._pos

    @property
    def files(self) -> list[str]:
        return [p for _, p, _ in self._parts]

    def close(self) -> RecorderStats:
        if self._closed:
            return self.stats
        self._closed = True
        st = self.stats
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        st.bytes_written = self._pos
        st.files = [{"path": p, "bytes": s, "stream_offset": o} for o, p, s in self._parts]
        if self._csv is not None:
            for h0, h1 in self._holes:
                self._csv.write(f"\n,{h0},{h1 - h0},")
            start = st.start_time if st.start_time is not None else self._open_time
            end = (st.last_packet_ns / 1e9) if st.last_packet_ns else start
            st.end_time = end
            self._csv.write(
                "\n\nRaw Data :"
                f"\nOut of sequence count - {st.out_of_sequence}"
                f"\nOut of sequence seen from {st.oos_from} to {st.oos_to}"
                f"\nFirst Packet ID - {st.first_seq or 0}"
                f"\nLast Packet ID - {st.last_seq or 0}"
                f"\nNumber of received packets - {st.received}"
                f"\nNumber of zero filled packets - {st.zero_filled_packets}"
                f"\nNumber of zero filled bytes - {st.zero_filled_bytes}"
                f"\nCapture start time - {msvc_ctime(start)}"
                f"Capture end time - {msvc_ctime(end)}"
                f"Duration(sec) - {int(end) - int(start)}")
            self._csv.close()
            self._csv = None
        return st
