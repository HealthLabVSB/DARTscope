"""
acquisition/receiver.py

The thread that takes the DCA1000's data datagrams off UDP port 4098 and hands them to
the recorder.

A Gigabit DCA1000 can send ~85 000 datagrams a second; the lab's heaviest configuration
sends ~21 000 (30.6 MB/s). What keeps Python from losing them:
  - the socket's receive buffer is made as large as the OS allows (TI asks for 2 GB), so a
    pause - a disk flush, the GIL held elsewhere - is absorbed by the kernel, not dropped;
  - one reused buffer and recv_into, no allocation per datagram;
  - a blocking socket: a timeout would make CPython call select() before every receive.
    To stop, a one-byte datagram is sent to the socket itself; it wakes the thread, and
    what is still queued is drained with a short timeout.

Losses that still happen are not hidden: the recorder zero-fills them from the DCA1000's
byte counter exactly as TI's recorder does, and counts them.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import socket
import threading
import time

WAKE = b"\x00"
RECV_BUFFER_REQUEST = 0x7FFFFFFF        # what TI's CLI requests; the OS grants what it can


class UdpReceiver:
    def __init__(self, recorder, port: int = 4098, bind_ip: str = "",
                 rcvbuf: int = RECV_BUFFER_REQUEST):
        self.recorder = recorder
        self.port = int(port)
        self.bind_ip = bind_ip
        self.rcvbuf_requested = rcvbuf
        self.rcvbuf_effective: int | None = None
        self.sock: socket.socket | None = None
        self.thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self.datagrams = 0
        self.last_rx_monotonic: float | None = None
        self.first_rx_monotonic: float | None = None
        self._stopping = threading.Event()
        self._drain_s = 0.5

    def start(self) -> "UdpReceiver":
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.bind((self.bind_ip, self.port))
        except OSError as e:
            s.close()
            raise OSError(f"cannot bind UDP port {self.port} ({e}). Is mmWave Studio or "
                          f"DCA1000EVM_CLI_Record still holding it?") from e
        for size in (self.rcvbuf_requested, 1 << 30, 512 << 20, 256 << 20, 64 << 20):
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, size)
                break
            except OSError:
                continue
        self.rcvbuf_effective = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        self.sock = s
        self.thread = threading.Thread(target=self._run, name="dca1000-data", daemon=True)
        self.thread.start()
        return self

    def _run(self):
        buf = bytearray(65536)
        mv = memoryview(buf)
        s = self.sock
        rec = self.recorder
        mono = time.monotonic
        clock = time.time_ns
        try:
            while True:
                try:
                    n = s.recv_into(buf)
                except socket.timeout:
                    return                                   # drained
                except OSError:
                    if self._stopping.is_set():
                        return
                    raise
                if n <= 1:
                    if self._stopping.is_set():
                        s.settimeout(self._drain_s)          # drain what is queued, then end
                    continue
                t = mono()
                if self.first_rx_monotonic is None:
                    self.first_rx_monotonic = t
                self.last_rx_monotonic = t
                self.datagrams += 1
                rec.feed(mv, n, clock())
        except BaseException as e:                           # disk full, bad data - reported
            self.error = e

    def stop(self, drain_s: float = 0.5):
        """Stop after the datagrams already queued have been recorded."""
        if self.sock is None:
            return
        self._drain_s = drain_s
        self._stopping.set()
        try:
            addr = self.sock.getsockname()
            host = addr[0] if addr[0] not in ("0.0.0.0", "") else "127.0.0.1"
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as w:
                w.sendto(WAKE, (host, addr[1]))
        except OSError:
            pass
        if self.thread is not None:
            self.thread.join(timeout=drain_s + 5.0)
            if self.thread.is_alive():                      # wake-up lost: force it
                self.sock.close()
                self.thread.join(timeout=2.0)
        try:
            self.sock.close()
        except OSError:
            pass
        self.sock = None

    def idle_s(self) -> float | None:
        """Seconds since the last datagram, None before the first."""
        return None if self.last_rx_monotonic is None else time.monotonic() - self.last_rx_monotonic
