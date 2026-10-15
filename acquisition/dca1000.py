"""
acquisition/dca1000.py

The DCA1000EVM's UDP protocol and a controller for its configuration port.

Everything here follows TI's DCA1000 CLI source that ships with mmWave Studio
(ReferenceCode/DCA1000/SourceCode: rf_api.h, defines.h, commandsprotocol.cpp,
configdatarecv.cpp) and was checked against a Wireshark capture of mmWave Studio talking
to the lab's DCA1000 (C:/Data/Radar/Wireshark/dca1000.pcapng):

    command   A5 5A | cmd u16 | size u16 | data | AA EE          host:4096 -> DCA:4096
    response  A5 5A | cmd u16 | status u16 | AA EE               DCA:1024  -> host:4096
    data      seq u32 | bytes-sent-before u48 | <= 1456 bytes    DCA:1024  -> host:4098

all little-endian. What mmWave Studio sends for ar1.CaptureCardConfig_Mode(1, 2, 1, 2, 3,
30) and _PacketDelay(25), in this order:

    5a a5 03 00 06 00 01 02 01 02 03 1e aa ee    CONFIG_FPGA  raw, 2 lanes, LVDS capture,
                                                 Ethernet, 16 bit, timer 30
    5a a5 0e 00 00 00 aa ee                      READ_FPGA_VERSION -> status 0x0482 = 2.9
    5a a5 0b 00 06 00 be 05 35 0c 00 00 aa ee    CONFIG_PACKET_DATA 1470 bytes, 25 us
                                                 (3125 FPGA clocks of 8 ns), 0
    5a a5 05 00 00 00 aa ee                      RECORD_START
    ... data ..., then from the DCA1000: 5a a5 0a 00 00 01 aa ee  async status, bit 8 =
    record completed
    5a a5 06 00 00 00 aa ee                      RECORD_STOP

The asynchronous status is a bit mask (configdatarecv.cpp tests each bit), not a number.

Not implemented on purpose: CONFIG_EEPROM. It rewrites the DCA1000's stored IP and MAC
addresses; a mistake there makes the board unreachable until it is reset by its DIP
switch. Use TI's tool for that.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass
from enum import IntEnum

from .config import Dca1000Config

HEADER = 0xA55A
FOOTER = 0xEEAA
DATA_HEADER_BYTES = 10              # seq u32 + byte count u48
MAX_PACKET_BYTES = 1470             # what the CLI configures and the recorder reads
PAYLOAD_BYTES = 1456
FPGA_CLOCK_NS = 8
MIN_PACKET_DELAY_US, MAX_PACKET_DELAY_US = 5, 500


class Cmd(IntEnum):
    RESET_FPGA = 0x01
    RESET_AR_DEV = 0x02
    CONFIG_FPGA = 0x03
    CONFIG_EEPROM = 0x04
    RECORD_START = 0x05
    RECORD_STOP = 0x06
    PLAYBACK_START = 0x07
    PLAYBACK_STOP = 0x08
    SYSTEM_ALIVENESS = 0x09
    SYSTEM_ASYNC_STATUS = 0x0A
    CONFIG_PACKET_DATA = 0x0B
    CONFIG_AR_DEV = 0x0C
    INIT_FPGA_PLAYBACK = 0x0D
    READ_FPGA_VERSION = 0x0E


# SYS_ASYNC_STATUS in rf_api.h: bit n of the status word
ASYNC_STATUS = {
    0: "no LVDS data",
    1: "no header",
    2: "EEPROM failure",
    3: "SD card detected",
    4: "SD card removed",
    5: "SD card full",
    6: "mode config failure",
    7: "DDR full",
    8: "record completed",
    9: "LVDS buffer full",
    10: "playback completed",
    11: "playback out of sequence",
}
ASYNC_RECORD_COMPLETED = 1 << 8
ASYNC_NO_LVDS_DATA = 1 << 0


class Dca1000Error(RuntimeError):
    pass


class Dca1000Timeout(Dca1000Error):
    pass


# --- encoding ------------------------------------------------------------------------------

def encode_command(cmd: int, data: bytes = b"") -> bytes:
    return struct.pack("<HHH", HEADER, int(cmd), len(data)) + bytes(data) + struct.pack("<H", FOOTER)


@dataclass
class Response:
    cmd: int
    status: int

    @property
    def is_async(self) -> bool:
        return self.cmd == Cmd.SYSTEM_ASYNC_STATUS


def decode_response(buf: bytes) -> Response | None:
    """A config-port datagram from the DCA1000, or None when it is not one."""
    if len(buf) < 8:
        return None
    header, cmd, status, footer = struct.unpack("<HHHH", buf[:8])
    if header != HEADER or footer != FOOTER:
        return None
    return Response(cmd, status)


def decode_command(buf: bytes) -> tuple[int, bytes] | None:
    """(cmd, data) of a command datagram - what the simulator receives."""
    if len(buf) < 8:
        return None
    header, cmd, size = struct.unpack("<HHH", buf[:6])
    if header != HEADER or len(buf) < 8 + size:
        return None
    if struct.unpack("<H", buf[6 + size:8 + size])[0] != FOOTER:
        return None
    return cmd, bytes(buf[6:6 + size])


def fpga_config_data(dca: Dca1000Config) -> bytes:
    """CONFIG_FPGA payload: six bytes in strFpgaConfigMode order (rf_api.cpp)."""
    vals = (dca.log_mode, dca.lvds_mode, dca.transfer_mode, dca.capture_mode, dca.data_format,
            dca.timer_s)
    for v in vals:
        if not 0 <= int(v) <= 255:
            raise ValueError(f"DCA1000 mode value {v} does not fit a byte")
    return bytes(int(v) for v in vals)


def packet_config_data(delay_us: int, packet_bytes: int = MAX_PACKET_BYTES) -> bytes:
    """CONFIG_PACKET_DATA payload: packet size, delay in 8 ns FPGA clocks, reserved."""
    if not MIN_PACKET_DELAY_US <= int(delay_us) <= MAX_PACKET_DELAY_US:
        raise ValueError(f"packet delay {delay_us} us outside {MIN_PACKET_DELAY_US}.."
                         f"{MAX_PACKET_DELAY_US} us")
    clocks = int(delay_us) * 1000 // FPGA_CLOCK_NS
    return struct.pack("<HHH", int(packet_bytes), clocks, 0)


def decode_fpga_version(status: int) -> tuple[int, int, bool]:
    """(major, minor, playback bitfile) from READ_FPGA_VERSION's status (rf_api.cpp)."""
    return status & 0x7F, (status >> 7) & 0x7F, bool(status & 0x4000)


def async_status_names(status: int) -> list[str]:
    return [name for bit, name in ASYNC_STATUS.items() if status & (1 << bit)] or \
           [f"unknown status 0x{status:04x}"]


def parse_data_header(buf) -> tuple[int, int]:
    """(sequence number, bytes the DCA1000 sent before this packet) of a data datagram."""
    return int.from_bytes(buf[0:4], "little"), int.from_bytes(buf[4:10], "little")


def encode_data_packet(seq: int, byte_count: int, payload: bytes) -> bytes:
    return (int(seq).to_bytes(4, "little") + int(byte_count).to_bytes(6, "little") + payload)


# --- controller ----------------------------------------------------------------------------

@dataclass
class AsyncEvent:
    time: float
    status: int
    names: list


class Dca1000Controller:
    """
    The DCA1000's configuration port: commands, their responses and the asynchronous
    status reports the FPGA sends on its own (record completed, no LVDS data, ...).

    One reader thread owns the socket's receive side, as in TI's CLI, so an async
    report arriving while a command waits is neither lost nor taken for the response.
    """

    def __init__(self, cfg: Dca1000Config, bind_ip: str = "", timeout_s: float = 3.0,
                 on_async=None, dca_addr: tuple[str, int] | None = None):
        self.cfg = cfg
        self.bind_ip = bind_ip
        self.timeout_s = timeout_s
        self.on_async = on_async
        self.dca_addr = dca_addr or (cfg.dca_ip, cfg.config_port)
        self.sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._responses: dict[int, list[Response]] = {}
        self.async_events: list[AsyncEvent] = []
        self.fpga_version: str | None = None
        self.sent: list[bytes] = []                    # every command sent, for the sidecar

    # socket
    def open(self) -> "Dca1000Controller":
        if self.sock is not None:
            return self
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.bind((self.bind_ip, self.cfg.config_port))
        except OSError as e:
            s.close()
            raise Dca1000Error(
                f"cannot bind UDP port {self.cfg.config_port} ({e}). Is mmWave Studio or "
                f"DCA1000EVM_CLI_Record still running and holding it?") from e
        s.settimeout(0.2)
        self.sock = s
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, name="dca1000-config", daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def _reader(self):
        while not self._stop.is_set():
            try:
                buf, _ = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            r = decode_response(buf)
            if r is None:
                continue
            if r.is_async:
                ev = AsyncEvent(time.time(), r.status, async_status_names(r.status))
                self.async_events.append(ev)
                if self.on_async is not None:
                    try:
                        self.on_async(ev)
                    except Exception:
                        pass
                continue
            with self._cond:
                self._responses.setdefault(r.cmd, []).append(r)
                self._cond.notify_all()

    def request(self, cmd: int, data: bytes = b"", timeout_s: float | None = None) -> int:
        """Send a command, wait for its response; raise when it times out or fails."""
        if self.sock is None:
            self.open()
        pkt = encode_command(cmd, data)
        with self._cond:
            self._responses.pop(int(cmd), None)
        self.sock.sendto(pkt, self.dca_addr)
        self.sent.append(pkt)
        deadline = time.monotonic() + (timeout_s or self.timeout_s)
        with self._cond:
            while True:
                got = self._responses.get(int(cmd))
                if got:
                    r = got.pop(0)
                    break
                left = deadline - time.monotonic()
                if left <= 0:
                    raise Dca1000Timeout(
                        f"no answer from the DCA1000 at {self.dca_addr[0]}:{self.dca_addr[1]} to "
                        f"{Cmd(cmd).name} within {timeout_s or self.timeout_s:g} s - is it powered, "
                        f"cabled, and is this PC's adapter set to {self.cfg.host_ip}?")
                self._cond.wait(left)
        if cmd != Cmd.READ_FPGA_VERSION and r.status != 0:
            raise Dca1000Error(f"{Cmd(cmd).name} failed with status {r.status}")
        return r.status

    # commands
    def system_aliveness(self) -> bool:
        try:
            self.request(Cmd.SYSTEM_ALIVENESS)
            return True
        except Dca1000Timeout:
            return False

    def read_fpga_version(self) -> str:
        status = self.request(Cmd.READ_FPGA_VERSION)
        major, minor, playback = decode_fpga_version(status)
        self.fpga_version = f"{major}.{minor}" + (" [playback]" if playback else "")
        return self.fpga_version

    def configure_fpga(self):
        self.request(Cmd.CONFIG_FPGA, fpga_config_data(self.cfg))

    def configure_packets(self):
        self.request(Cmd.CONFIG_PACKET_DATA, packet_config_data(self.cfg.packet_delay_us))

    def configure(self) -> str:
        """What mmWave Studio does before a recording: mode, version, packet delay."""
        self.configure_fpga()
        version = self.read_fpga_version()
        if version.endswith("[playback]"):
            raise Dca1000Error("the DCA1000 runs the playback FPGA image; it cannot record")
        self.configure_packets()
        return version

    def start_record(self):
        self.request(Cmd.RECORD_START)

    def stop_record(self):
        self.request(Cmd.RECORD_STOP)

    def reset_fpga(self):
        self.request(Cmd.RESET_FPGA)

    def reset_radar(self):
        """Pulses the radar's reset line through the DCA1000 (RESET_AR_DEV)."""
        self.request(Cmd.RESET_AR_DEV)

    def completed_since(self, t0: float) -> AsyncEvent | None:
        """The first 'record completed' or 'no LVDS data' report after t0, if any."""
        for ev in list(self.async_events):
            if ev.time >= t0 and ev.status & (ASYNC_RECORD_COMPLETED | ASYNC_NO_LVDS_DATA):
                return ev
        return None


def local_ip_towards(ip: str, port: int = 4096) -> str | None:
    """The address this PC would send from to reach ip - no packet is sent."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((ip, port))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return None
