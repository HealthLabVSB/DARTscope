"""
acquisition/simulator.py

A DCA1000 on localhost: it answers the configuration port like the FPGA does and streams
a byte source as data datagrams, so the whole acquisition path - controller, receiver,
recorder, log, verification - runs without hardware.

What it reproduces, from TI's CLI source and the Wireshark capture of the lab's board:
  - responses A5 5A | cmd | status | AA EE to CONFIG_FPGA, CONFIG_PACKET_DATA,
    READ_FPGA_VERSION (status 0x0482 = FPGA 2.9), SYSTEM_ALIVENESS, RECORD_START/STOP;
  - datagrams of up to 1456 payload bytes, sequence numbers from 1, the 48-bit count of
    bytes sent before each packet;
  - the "record completed" status (bit 8) some time after the stream ends;
  - optionally, the DCA1000 keeping the last bytes back (the lab's board withheld 128);
  - faults: packets dropped, swapped, duplicated.

The byte source is whatever a recording should contain: layout.encode_cube() of known
samples, an existing mmWave Studio recording, or the payloads of a pcap.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass, field

from .dca1000 import (ASYNC_RECORD_COMPLETED, FOOTER, HEADER, PAYLOAD_BYTES, Cmd,
                      decode_command, encode_data_packet)

FPGA_VERSION_STATUS = 0x0482


@dataclass
class Faults:
    drop: set = field(default_factory=set)          # sequence numbers never sent
    swap: set = field(default_factory=set)          # seq s is sent after s + 1
    duplicate: set = field(default_factory=set)     # seq s is sent twice
    withhold_tail: int = 0                          # bytes at the end never sent


def packetize(data, faults: Faults | None = None, payload: int = PAYLOAD_BYTES):
    """(seq, byte_count, payload) in send order, faults applied."""
    f = faults or Faults()
    mv = memoryview(data)
    total = len(mv) - int(f.withhold_tail)
    pkts = []
    seq, off = 1, 0
    while off < total:
        n = min(payload, total - off)
        pkts.append((seq, off, mv[off:off + n]))
        seq += 1
        off += n
    out = []
    i = 0
    while i < len(pkts):
        s = pkts[i][0]
        if s in f.swap and i + 1 < len(pkts):
            order = [pkts[i + 1], pkts[i]]
            i += 2
        else:
            order = [pkts[i]]
            i += 1
        for p in order:
            if p[0] in f.drop:
                continue
            out.append(p)
            if p[0] in f.duplicate:
                out.append(p)
    return out


def _response(cmd: int, status: int) -> bytes:
    return struct.pack("<HHHH", HEADER, cmd, status, FOOTER)


class Dca1000Simulator:
    """
    Binds a config port on 127.0.0.1 (pass it to Dca1000Controller as dca_addr) and sends
    data to host_data_addr. The stream starts with start_stream() - the radar's job on the
    real board - and only while a record is active, as on the FPGA.
    """

    def __init__(self, source=b"", host_data_addr=("127.0.0.1", 4098), config_port: int = 0,
                 faults: Faults | None = None, rate_mb_s: float | None = None,
                 completed_after_s: float = 0.3, ip: str = "127.0.0.1"):
        self.source = source
        self.host_data_addr = host_data_addr
        self.faults = faults or Faults()
        self.rate_mb_s = rate_mb_s
        self.completed_after_s = completed_after_s
        self.cfg_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cfg_sock.bind((ip, config_port))
        self.cfg_sock.settimeout(0.1)
        self.addr = self.cfg_sock.getsockname()
        self.data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)
        self.received: list[tuple[int, bytes]] = []
        self.fpga_mode: bytes | None = None
        self.packet_config: bytes | None = None
        self.recording = False
        self.host_cfg_addr = None
        self.sent_packets = 0
        self._stop = threading.Event()
        self._streamer: threading.Thread | None = None
        self._thread = threading.Thread(target=self._serve, name="dca1000-sim", daemon=True)
        self._thread.start()

    # config port
    def _serve(self):
        while not self._stop.is_set():
            try:
                buf, addr = self.cfg_sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                return
            cmd = decode_command(buf)
            if cmd is None:
                continue
            code, data = cmd
            self.received.append((code, data))
            self.host_cfg_addr = addr
            status = 0
            if code == Cmd.CONFIG_FPGA:
                ok = len(data) == 6 and data[0] in (1, 2) and data[1] in (1, 2) and data[4] in (1, 2, 3)
                status = 0 if ok else 1
                if ok:
                    self.fpga_mode = data
            elif code == Cmd.CONFIG_PACKET_DATA:
                self.packet_config = data
            elif code == Cmd.READ_FPGA_VERSION:
                status = FPGA_VERSION_STATUS
            elif code == Cmd.RECORD_START:
                self.recording = True
            elif code == Cmd.RECORD_STOP:
                self.recording = False
            self.cfg_sock.sendto(_response(code, status), addr)

    # data
    def start_stream(self):
        """What the radar's StartFrame does to the DCA1000: LVDS data begins to flow."""
        self._streamer = threading.Thread(target=self._stream, name="dca1000-sim-data", daemon=True)
        self._streamer.start()

    def _stream(self):
        pkts = packetize(self.source, self.faults)
        t0 = time.monotonic()
        sent_bytes = 0
        for seq, bc, payload in pkts:
            if self._stop.is_set() or not self.recording:
                break
            self.data_sock.sendto(encode_data_packet(seq, bc, bytes(payload)), self.host_data_addr)
            self.sent_packets += 1
            sent_bytes += len(payload)
            if self.rate_mb_s:
                ahead = sent_bytes / (self.rate_mb_s * 1e6) - (time.monotonic() - t0)
                if ahead > 0.002:
                    time.sleep(ahead)
            elif self.sent_packets % 64 == 0:
                time.sleep(0)                   # let the receiver run on a busy machine
        time.sleep(self.completed_after_s)
        if self.recording and self.host_cfg_addr is not None and not self._stop.is_set():
            self.cfg_sock.sendto(_response(Cmd.SYSTEM_ASYNC_STATUS, ASYNC_RECORD_COMPLETED),
                                 self.host_cfg_addr)

    def wait_stream(self, timeout: float = 30.0):
        if self._streamer is not None:
            self._streamer.join(timeout)

    def close(self):
        self._stop.set()
        self.wait_stream(5)
        self._thread.join(timeout=2)
        self.cfg_sock.close()
        self.data_sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def free_udp_port(ip: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind((ip, 0))
        return s.getsockname()[1]
