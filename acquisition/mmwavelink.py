"""
acquisition/mmwavelink.py

TI's RadarLinkDLL.dll in Python, for computers that cannot load it (macOS, Linux): the
radar's SOP lines, reset and SPI through the DCA1000's FTDI chip, and the mmWaveLink
protocol (RHCP) over that SPI. MmwLink answers the same RadarLinkImpl_* calls with the
same arguments as radarlink.RadarLinkClient, so native.NativeRadar drives the radar
the same way on every platform.

Sources, both shipped with mmWave Studio 2.1.1 or TI's radar toolbox:
  - the FTDI port layer RadarLinkDLL is built from (mmWaveStudio/ReferenceCode/FTDILib/
    SourceCode/mmwl_port_ftdi.c, "mmWaveLink Studio 1.1.5.4"), checked against the DLL's
    machine code (disassembled: the same MPSSE bytes, pins and port roles);
  - mmWaveLink's driver (ti/control/mmwavelink/src/rl_driver.c): sync patterns, header,
    checksum, CRC, sub-blocks, ACK and async events;
  - the DLL's RadarLinkImpl_* wrappers (disassembled): which rl* API each one calls and
    how it fills the struct - see docs/NATIVE_RADAR_NOTES.md.

The DCA1000's FT4232H (USB 0451:FD03, "AR-DevPack-EVM-012"):

    A  SPI        MPSSE, 10 MHz, mode 0: SCK ADBUS0, MOSI 1, MISO 2, CS 3 (one CS frame
                  per 16-bit word, MSB first); bit 6 PMIC reset kept high
    B  HOST_IRQ   MPSSE; the radar's HOST_INTR is ADBUS5, read with 0x81
    C  board      async bit-bang, outputs 0xF9; NRESET of the radar is bit 6
    D  GPIO       async bit-bang, outputs 0xFF; SOP lines are bits 2, 3, 4

USB access is through libusb (pyftdi + pyusb). On macOS the system's own FTDI serial
driver does not get in the way: an interface is claimed only while a port is open.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import os
import struct
import sys
import time

from .radarlink import RadarLinkError

# ---- FTDI -------------------------------------------------------------------------------------

DCA_VID, DCA_PID = 0x0451, 0xFD03
PORT_SPI, PORT_IRQ, PORT_BOARD, PORT_GPIO = 1, 2, 3, 4          # pyftdi interfaces A..D

SPI_DIR, SPI_IDLE, SPI_CS_LOW, SPI_READ_LOW = 0x4B, 0xC8, 0xC0, 0xC2
IRQ_DIR, IRQ_IDLE, IRQ_BIT = 0x5B, 0x13, 1 << 5
BOARD_DIR, NRESET_BIT = 0xF9, 6
GPIO_DIR = 0xFF
SOP_PINS = (2, 3, 4)
SOP_LEVELS = {1: (0, 1, 0), 2: (0, 1, 1), 3: (0, 0, 0), 4: (0, 0, 1),     # rlsSOPControl
              5: (1, 0, 1), 6: (1, 1, 0), 7: (1, 1, 1)}

# MPSSE commands (FTDI AN_108)
_SET_LOW, _SEND_NOW = 0x80, 0x87
_OUT_NEG_MSB, _IN_POS_MSB = 0x11, 0x20
_GET_LOW = 0x81
SPI_INIT = bytes([0x8A, 0x97, 0x8D, 0x86, 0x02, 0x00, _SET_LOW, SPI_IDLE, SPI_DIR])  # 60/((1+2)*2) = 10 MHz
IRQ_INIT = bytes([0x8A, 0x97, 0x8C, _SET_LOW, IRQ_IDLE, IRQ_DIR, 0x86, 0x4A, 0x00, 0x85])

WORDS_PER_USB_WRITE = 64            # SPI words batched into one USB transfer


class LinkError(RadarLinkError):
    pass


_LIBUSB_CANDIDATES = (
    "/opt/homebrew/lib/libusb-1.0.dylib",            # Homebrew, Apple silicon
    "/usr/local/lib/libusb-1.0.dylib",               # Homebrew, Intel
    "/opt/local/lib/libusb-1.0.dylib",               # MacPorts
)


def _find_libusb(_name=None):
    env = os.environ.get("DARTSCOPE_LIBUSB")
    if env and os.path.isfile(env):
        return env
    try:
        import libusb_package                         # pip install libusb-package: a bundled copy
        p = libusb_package.get_library_path()
        if p and os.path.isfile(str(p)):
            return str(p)
    except Exception:
        pass
    for p in _LIBUSB_CANDIDATES:
        if os.path.isfile(p):
            return p
    import ctypes.util
    return ctypes.util.find_library("usb-1.0")


def usb_backend():
    """pyusb's libusb backend, found also where ctypes does not look (Homebrew on macOS).
    pyusb keeps the library it loaded, so pyftdi's own lookups find it afterwards."""
    try:
        import usb.backend.libusb1 as libusb1
    except ImportError as e:
        raise LinkError("pyusb/pyftdi are not installed (pip install pyftdi)") from e
    be = libusb1.get_backend(find_library=_find_libusb)
    if be is None:
        raise LinkError("libusb is not installed: on macOS 'brew install libusb' "
                        "(or pip install libusb-package)")
    return be


def find_dca1000():
    """The DCA1000's FTDI as a pyusb device, or None."""
    import usb.core
    return usb.core.find(idVendor=DCA_VID, idProduct=DCA_PID, backend=usb_backend())


def link_status() -> list[str]:
    """What the radar path without the DLL lacks on this computer; empty when it has all."""
    out = []
    try:
        import pyftdi  # noqa: F401
        import serial  # noqa: F401
    except ImportError:
        out.append("pyftdi and pyserial are needed to drive the radar here: "
                   "pip install pyftdi pyserial")
        return out
    try:
        if find_dca1000() is None:
            out.append("the DCA1000's USB (FTDI 'AR-DevPack-EVM-012', 0451:FD03) is not plugged in")
    except LinkError as e:
        out.append(str(e))
    except Exception as e:
        out.append(f"USB access failed: {e}")
    return out


class FtdiPort:
    """One channel of the FT4232H through pyftdi, used the way mmwl_port_ftdi.c uses D2XX."""

    def __init__(self, usb_dev, interface: int):
        from pyftdi.ftdi import Ftdi
        self.Ftdi = Ftdi
        self.f = Ftdi()
        try:
            self.f.open_from_device(usb_dev, interface=interface)   # purge, reset, bit mode reset
        except Exception as e:
            raise LinkError(f"DCA1000 FTDI port {'ABCD'[interface - 1]} cannot be opened: {e}") from e
        self.interface = interface

    def bitbang(self, direction: int):
        self.f.set_bitmode(direction, self.Ftdi.BitMode.BITBANG)

    def mpsse(self, direction: int):
        self.f.set_bitmode(direction, self.Ftdi.BitMode.MPSSE)

    def write(self, data: bytes):
        n = self.f.write_data(bytes(data))
        if n != len(data):
            raise LinkError(f"FTDI port {'ABCD'[self.interface - 1]}: {n} of {len(data)} bytes written")

    def read(self, n: int, timeout: float = 1.0) -> bytes:
        out = bytearray()
        deadline = time.monotonic() + timeout
        while len(out) < n:
            out += self.f.read_data_bytes(n - len(out), attempt=1)
            if len(out) < n and time.monotonic() > deadline:
                raise LinkError(f"FTDI port {'ABCD'[self.interface - 1]}: {len(out)} of {n} bytes read")
        return bytes(out)

    def pins(self) -> int:
        return int(self.f.read_pins())

    def purge(self):
        self.f.purge_buffers()

    def release(self):
        """rlsClose*If: bit mode reset (the pins float back to the board's pull-ups), port reset."""
        try:
            self.f.set_bitmode(0, self.Ftdi.BitMode.RESET)
            self.f.reset()
        finally:
            self.close()

    def close(self):
        """FT_Close: release the interface, leave the outputs as they are. Not pyftdi's
        close(): it hands the interface back to the kernel driver, which on macOS
        re-enumerates the whole FT4232H and cuts the other three ports off."""
        dev = getattr(self.f, "_usb_dev", None)
        if dev is None:
            return
        try:
            import usb.util
            usb.util.release_interface(dev, self.interface - 1)
        except Exception:
            pass
        self.f._usb_dev = None


class Dca1000Ftdi:
    """The four FTDI channels: SOP, reset, SPI and HOST_IRQ (rlsOpen*If, rlsComm*, rlsSpi*)."""

    def __init__(self, usb_dev=None):
        self.usb_dev = usb_dev
        self.gpio: FtdiPort | None = None
        self.board: FtdiPort | None = None
        self.spi: FtdiPort | None = None
        self.irq: FtdiPort | None = None

    def _port(self, interface: int) -> FtdiPort:
        """Open a channel. An FT4232H that was left without the board's power can stop
        answering control requests (seen 2026-10-06 after the DCA1000 was switched off and
        on): then, if no other channel is open, the USB device is reset once and tried again."""
        try:
            return FtdiPort(self._dev(), interface)
        except LinkError:
            if any(getattr(self, n) is not None for n in ("gpio", "board", "spi", "irq")):
                raise
            try:
                self.usb_dev.reset()
            except Exception:
                pass
            self.usb_dev = None
            time.sleep(1.0)
            return FtdiPort(self._dev(), interface)

    def _dev(self):
        if self.usb_dev is None:
            self.usb_dev = find_dca1000()
            if self.usb_dev is None:
                raise LinkError("DCA1000 not found on USB (0451:FD03) - is its USB cable in and "
                                "the board powered?")
        return self.usb_dev

    # rlsOpenGenericGpioIf / rlsSOPControl / rlsCloseGenericGpioIf
    def open_gpio(self):
        if self.gpio is None:
            self.gpio = self._port(PORT_GPIO)
            self.gpio.bitbang(GPIO_DIR)

    def sop(self, mode: int):
        if mode not in SOP_LEVELS:
            raise LinkError(f"no SOP mode {mode}")
        v = self.gpio.pins()
        for pin, level in zip(SOP_PINS, SOP_LEVELS[mode]):
            v = (v | (1 << pin)) if level else (v & ~(1 << pin))
        self.gpio.write(bytes([v & 0xFF]))

    def close_gpio(self):
        if self.gpio is not None:
            self.gpio.release()
            self.gpio = None

    # rlsOpenBoardControlIf / rlsFullReset / rlsCloseBoardControlIf
    def open_board(self):
        if self.board is None:
            self.board = self._port(PORT_BOARD)
            self.board.bitbang(BOARD_DIR)

    def nreset(self, state: int):
        v = self.board.pins()
        v = (v | (1 << NRESET_BIT)) if state else (v & ~(1 << NRESET_BIT))
        self.board.write(bytes([v & 0xFF]))
        time.sleep(0.002)

    def close_board(self):
        if self.board is not None:
            self.board.release()
            self.board = None

    # rlsCommOpen: SPI, HOST_IRQ, and the board and GPIO ports back in bit-bang mode, which
    # drives their last written levels again (rlsInitBoardControlPort, rlsInitGenericGPIOPort)
    def open_comm(self, board_ports: bool = True):
        if board_ports:
            self.open_board()
            self.open_gpio()
        if self.spi is None:
            self.spi = self._port(PORT_SPI)
            self.spi.mpsse(SPI_DIR)
            self.spi.write(SPI_INIT)
            self.spi.purge()
        if self.irq is None:
            self.irq = self._port(PORT_IRQ)
            self.irq.mpsse(IRQ_DIR)
            self.irq.write(IRQ_INIT)
            self.irq.purge()

    def close_comm(self):
        for name in ("spi", "irq"):
            port = getattr(self, name)
            if port is not None:
                port.close()
                setattr(self, name, None)

    def close(self):
        self.close_comm()
        self.close_board()
        self.close_gpio()
        if self.usb_dev is not None:
            try:
                import usb.util
                usb.util.dispose_resources(self.usb_dev)
            except Exception:
                pass

    def host_irq(self) -> int:
        self.irq.write(bytes([_GET_LOW, _SEND_NOW]))
        return 1 if self.irq.read(1, 0.5)[0] & IRQ_BIT else 0

    # rlsSpiWrite / rlsSpiRead: 16-bit words, little-endian in memory, each its own CS frame
    def spi_write(self, data: bytes):
        if len(data) % 2:
            raise LinkError("SPI transfers are whole 16-bit words")
        cmds = bytearray()
        words = 0
        for i in range(0, len(data), 2):
            cmds += bytes([_SET_LOW, SPI_CS_LOW, SPI_DIR, _OUT_NEG_MSB, 0x01, 0x00,
                           data[i + 1], data[i], _SET_LOW, SPI_IDLE, SPI_DIR])
            words += 1
            if words == WORDS_PER_USB_WRITE:
                self.spi.write(bytes(cmds) + bytes([_SEND_NOW]))
                cmds.clear()
                words = 0
        if cmds:
            self.spi.write(bytes(cmds) + bytes([_SEND_NOW]))

    def spi_read(self, n: int) -> bytes:
        if n % 2:
            raise LinkError("SPI transfers are whole 16-bit words")
        out = bytearray()
        left = n // 2
        while left:
            k = min(left, WORDS_PER_USB_WRITE)
            cmd = bytes([_SET_LOW, SPI_READ_LOW, SPI_DIR, _IN_POS_MSB, 0x01, 0x00,
                         _SET_LOW, SPI_IDLE, SPI_DIR]) * k + bytes([_SEND_NOW])
            self.spi.write(cmd)
            raw = self.spi.read(2 * k, 1.0)
            for i in range(0, len(raw), 2):
                out += bytes([raw[i + 1], raw[i]])
            left -= k
        return bytes(out)


# ---- mmWaveLink protocol (RHCP) ---------------------------------------------------------------

H2D_SYNC = struct.pack("<HH", 0x1234, 0x4321)
D2H_SYNC = struct.pack("<HH", 0xDCBA, 0xABCD)
H2D_CNYS = struct.pack("<8H", 0x5678, 0x8765, *([0xFFFF] * 6))
HEADER_LEN = 12
SYNC_SCAN_LIMIT = 252
MAX_PAYLOAD = 232                     # RL_CMD_PL_LEN_MAX for 256-byte messages

CLASS_CMD, CLASS_RSP, CLASS_NACK, CLASS_ASYNC = 0, 1, 2, 3
DIR_HOST_TO_BSS, DIR_HOST_TO_DSS, DIR_HOST_TO_MSS = 1, 3, 5

RL_OK = 0
RL_PROTOCOL_ERROR, RL_INVALID_INPUT, RL_RADAR_IF_ERROR = -1, -2, -4
RL_CRC_FAILED, RL_CHKSUM_FAILED, RL_RESP_TIMEOUT = -6, -7, -8
RL_INVALID_STATE, RL_NACK_ERROR, RL_HOSTIRQ_TIMEOUT, RL_SEQ_MISMATCH = -11, -16, -17, -18

# message ids (rl_messages.h) and the sub-blocks used here
MSG_RF_STATIC_SET, MSG_RF_INIT, MSG_RF_DYNAMIC_SET, MSG_RF_FRAME_TRIG = 0x04, 0x06, 0x08, 0x0A
MSG_RF_MISC_SET = 0x16
MSG_DEV_POWERUP, MSG_DEV_CONFIG_SET, MSG_DEV_CONFIG_APPLY = 0x200, 0x202, 0x206


def sbid(msg_id: int, sb: int) -> int:
    """RL_GET_UNIQUE_SBID: the id a sub-block carries on the wire."""
    return msg_id * 32 + sb


def checksum(header10: bytes) -> int:
    """rlDriverCalChkSum: one's complement of the folded sum of the header's half-words."""
    total = sum(struct.unpack(f"<{len(header10) // 2}H", header10))
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def _crc16_table():
    t = []
    for i in range(256):
        c = i << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x1021) if c & 0x8000 else (c << 1)
        t.append(c & 0xFFFF)
    return t


_CRC_TABLE = _crc16_table()


def crc16(data: bytes) -> int:
    """RadarLinkDLL's rlComputeCRC: CRC-16/CCITT (0x1021, start 0xFFFF), bytes in order."""
    c = 0xFFFF
    for b in data:
        c = ((c << 8) & 0xFFFF) ^ _CRC_TABLE[((c >> 8) ^ b) & 0xFF]
    return c


def cmd_direction(msg_id: int) -> int:
    """rlDeviceIdentifyCmdDir for a host."""
    if 0x00 < msg_id < 0x80:
        return DIR_HOST_TO_BSS
    if 0x200 <= msg_id < 0x280:
        return DIR_HOST_TO_MSS
    if 0x100 <= msg_id < 0x180:
        return DIR_HOST_TO_DSS
    raise LinkError(f"message id 0x{msg_id:x} has no direction")


def build_command(msg_id: int, blocks: list[tuple[int, bytes]], seq: int, retry: bool = False
                  ) -> bytes:
    """
    A command as rlDriverCmdInvoke + rlDriverMsgWrite put it on the wire: sync, header
    (opcode, length, flags, remaining chunks, sub-block count, checksum), sub-blocks,
    0xFF padding to 4 bytes, 16-bit CRC of header and payload.
    """
    payload = b"".join(struct.pack("<HH", sb, len(d) + 4) + bytes(d) for sb, d in blocks)
    if len(payload) > MAX_PAYLOAD:
        raise LinkError(f"message 0x{msg_id:x}: {len(payload)} bytes of payload, at most {MAX_PAYLOAD}")
    length = HEADER_LEN + len(payload)
    pad = (-length) % 4
    payload += b"\xff" * pad
    length += pad + 2
    opcode = cmd_direction(msg_id) | (CLASS_CMD << 4) | ((msg_id & 0x3FF) << 6)
    # flags: retry 0/3 (bits 0-1), ACK requested 0 (2-3), version 0 (4-7), CRC present 0
    # (8-9), 16-bit CRC 0 (10-11), sequence number (12-15)
    flags = (3 if retry else 0) | ((seq & 0xF) << 12)
    head = struct.pack("<5H", opcode, length, flags, 0, len(blocks))
    head += struct.pack("<H", checksum(head))
    body = head + payload
    return H2D_SYNC + body + struct.pack("<H", crc16(body))


class Message:
    def __init__(self, header: bytes, payload: bytes, crc: bytes = b""):
        (self.opcode, self.length, self.flags, self.rem_chunks, self.nsbc,
         self.chksum) = struct.unpack("<6H", header)
        self.header, self.payload, self.crc = header, payload, crc

    @property
    def msg_class(self) -> int:
        return (self.opcode >> 4) & 0x3

    @property
    def msg_id(self) -> int:
        return (self.opcode >> 6) & 0x3FF

    @property
    def seq(self) -> int:
        return (self.flags >> 12) & 0xF

    def blocks(self) -> list[tuple[int, bytes]]:
        out, pos = [], 0
        for _ in range(self.nsbc):
            if pos + 4 > len(self.payload):
                break
            sb, ln = struct.unpack_from("<HH", self.payload, pos)
            out.append((sb, self.payload[pos + 4:pos + max(4, ln)]))
            pos += max(4, ln)
        return out


def parse_header(raw: bytes) -> tuple[Message | None, int]:
    """A received header (12 bytes after the sync) -> (Message without payload, error)."""
    m = Message(raw, b"")
    if checksum(raw[:10]) != m.chksum:
        return m, RL_CHKSUM_FAILED
    return m, RL_OK


class MmwLink:
    """
    The RadarLinkImpl_* functions NativeRadar calls, over the DCA1000's FTDI. Calls and
    events have the shapes of radarlink.RadarLinkClient: call(fn, *(type, value)) -> int,
    wait_event(sbids, timeout) -> {"dev", "msg", "sbid", "payload", "time"} or None.
    """

    def __init__(self, ftdi: Dca1000Ftdi | None = None, log=None, ack_timeout: float = 1.0):
        self.ftdi = ftdi or Dca1000Ftdi()
        self.trace = bool(os.environ.get("DARTSCOPE_MMWL_TRACE"))
        if log is None and self.trace:
            log = lambda s: print(f"mmwl {time.monotonic():.3f} {s}", file=sys.stderr, flush=True)
        self.log = log or (lambda s: None)
        self.ack_timeout = ack_timeout
        self.seq = 0
        self.powered = False
        self.pending: list[dict] = []          # async events not yet asked for
        self.event_log: list[dict] = []
        self.info = {"ready": True, "impl": "python mmWaveLink over the DCA1000 FTDI",
                     "platform": sys.platform}

    # ---- the RadarLinkClient interface ----
    def call(self, fn: str, *args, timeout: float = 60.0) -> int:
        name = fn.replace("RadarLinkImpl_", "")
        impl = getattr(self, "_fn_" + name, None)
        if impl is None:
            raise LinkError(f"{fn} is not implemented without RadarLinkDLL")
        values = [_arg(a) for a in args]
        ret = int(impl(*values))
        self.log(f"{fn}{tuple(v for v in values if not isinstance(v, bytes))} -> {ret}")
        return ret

    def wait_event(self, sbids, timeout: float) -> dict | None:
        """The next event with one of these sub-block ids (earlier other events are dropped)."""
        want = {sbids} if isinstance(sbids, int) else set(sbids)
        deadline = time.monotonic() + timeout
        while True:
            while self.pending:
                ev = self.pending.pop(0)
                if ev["sbid"] in want:
                    return ev
            if time.monotonic() > deadline:
                return None
            self._poll(min(0.05, max(0.0, deadline - time.monotonic())))

    def close(self):
        self.ftdi.close()
        self.powered = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- board control (ar1.SOPControl, ar1.FullReset) ----
    def _fn_setSystemType(self, kind):
        return RL_OK

    def _fn_OpenGenericGpioIf(self, dev):
        self.ftdi.open_gpio()
        return RL_OK

    def _fn_SOPControl(self, dev, mode):
        self.ftdi.sop(int(mode))
        return RL_OK

    def _fn_OpenBoardControlIf(self, dev):
        self.ftdi.open_board()
        return RL_OK

    def _fn_FullReset(self, dev, state):
        if self.powered:                       # a reset ends the session with the old firmware
            self.ftdi.close_comm()
            self.powered = False
            self.pending.clear()
        self.ftdi.nreset(int(state))
        return RL_OK

    def _fn_CloseBoardControlIf(self, dev):
        self.ftdi.close_board()
        return RL_OK

    def _fn_CloseGenericGpioIf(self, dev):
        self.ftdi.close_gpio()
        return RL_OK

    # ---- power ----
    def _fn_PowerOn(self, dev, crc_type=0, ack_timeout=1000, intf=0, port=0, callback=None):
        """rlDevicePowerOn: open SPI and HOST_IRQ; the MSS reports its power-up (0x5000)."""
        if int(crc_type) != 0:
            return RL_INVALID_INPUT            # only the 16-bit CRC mmWave Studio powers on with
        self.ack_timeout = max(0.1, int(ack_timeout) / 1000.0)
        self.ftdi.open_comm()
        self.seq = 0
        self.powered = True
        return RL_OK if self._await(sbid(0x280, 0), 2.0) else RL_RESP_TIMEOUT

    def _fn_RfEnable(self, dev):
        """rlDeviceRfStart; RadarLinkDLL waits up to 1 s for the BSS power-up (0x5001)."""
        ret = self._command(MSG_DEV_POWERUP, [(sbid(MSG_DEV_POWERUP, 0), b"")])
        if ret != RL_OK:
            return ret
        return RL_OK if self._await(sbid(0x280, 1), 1.0) else RL_RESP_TIMEOUT

    # ---- configuration ----
    def _fn_ChannelConfig(self, dev, buf):
        return self._set(MSG_RF_STATIC_SET, 0, buf)

    def _fn_AdcOutConfig(self, dev, buf):
        return self._set(MSG_RF_STATIC_SET, 2, buf)

    def _fn_LowPowerConfig(self, dev, lp_adc_mode):
        return self._set(MSG_RF_STATIC_SET, 3, struct.pack("<HH", 0, int(lp_adc_mode)))

    def _fn_SetHsiClock(self, dev, clk):
        return self._set(MSG_RF_STATIC_SET, 5, struct.pack("<HH", int(clk), 0))

    def _fn_RfInit(self, dev):
        """rlRfInit; RadarLinkDLL waits up to 1 s for the calibration report (0x1004)."""
        ret = self._set(MSG_RF_INIT, 0, b"")
        if ret != RL_OK:
            return ret
        return RL_OK if self._await(sbid(0x80, 4), 1.0) else RL_RESP_TIMEOUT

    def _fn_DataFmtConfig(self, dev, rx, bits, fmt, iq_swap, interleave):
        return self._set(MSG_DEV_CONFIG_SET, 1, struct.pack("<HHHBBI", int(rx), int(bits), int(fmt),
                                                            int(iq_swap), int(interleave), 0))

    def _fn_DataPathConfig(self, dev, *b):
        return self._set(MSG_DEV_CONFIG_SET, 2, bytes(int(x) & 0xFF for x in b[:8]).ljust(8, b"\0"))

    def _fn_LaneConfig(self, dev, lanes):
        return self._set(MSG_DEV_CONFIG_SET, 3, struct.pack("<HH", int(lanes), 0))

    def _fn_LvdsClkConfig(self, dev, lane_clk, rate):
        return self._set(MSG_DEV_CONFIG_SET, 4, struct.pack("<BBH", int(lane_clk), int(rate), 0))

    def _fn_LvdsLaneConfig(self, dev, fmt_map, param):
        return self._set(MSG_DEV_CONFIG_SET, 5, struct.pack("<HH", int(fmt_map), int(param)))

    def _fn_ProfileConfig(self, dev, buf):
        return self._set(MSG_RF_DYNAMIC_SET, 0, buf)

    def _fn_ChirpConfig(self, dev, start, end, profile, f_var, slope_var, idle_var, adc_var, tx):
        return self._set(MSG_RF_DYNAMIC_SET, 1, struct.pack(
            "<HHHHIhHHH", int(start), int(end), int(profile), 0, int(f_var) & 0xFFFFFFFF,
            int(slope_var), int(idle_var), int(adc_var), int(tx)))

    def _fn_EnableTestSource(self, dev, enable, mode=1):
        return self._set(MSG_RF_MISC_SET, 3, struct.pack("<HH", int(enable), 0))

    def _fn_FrameConfig(self, dev, start, end, frames, loops, period, delay, samples, dummy,
                        trigger, reserved=0):
        """rlSetFrameConfig and, on a host, rlDeviceFrameConfigApply with the chirp count."""
        cfg = struct.pack("<HHHHHHIHBBI", 0, int(start), int(end), int(loops), int(frames),
                          int(samples), int(period), int(trigger), 0, int(dummy), int(delay))
        ret = self._set(MSG_RF_DYNAMIC_SET, 2, cfg)
        if ret != RL_OK:
            return ret
        chirps = ((int(end) - int(start) + 1) * int(loops) - int(dummy)) & 0xFFFF
        return self._set(MSG_DEV_CONFIG_APPLY, 0, struct.pack("<IHH", chirps, int(samples), 0))

    def _fn_SensorStart(self, dev):
        return self._set(MSG_RF_FRAME_TRIG, 0, struct.pack("<HH", 1, 0))

    def _fn_SensorStop(self, dev):
        return self._set(MSG_RF_FRAME_TRIG, 0, struct.pack("<HH", 0, 0))

    # ---- RHCP ----
    def _set(self, msg_id: int, sb: int, data: bytes) -> int:
        """rlDriverExecuteSetApi: one sub-block, wait for the response."""
        return self._command(msg_id, [(sbid(msg_id, sb), bytes(data))])

    def _command(self, msg_id: int, blocks, retries: int = 3) -> int:
        if not self.powered:
            return RL_INVALID_STATE
        self._drain()                               # events the radar raised meanwhile
        retry = False
        ret = RL_RESP_TIMEOUT
        for _ in range(retries + 1):
            seq = self.seq % 16
            raw = build_command(msg_id, blocks, seq, retry=retry)
            if self.trace:
                self.log(f"  SPI <- {raw.hex()}")
            self.ftdi.spi_write(raw)
            ret = self._response(msg_id, seq)
            if ret in (RL_CRC_FAILED, RL_RESP_TIMEOUT):
                retry = True                         # the same sequence number, retry flag set
                continue
            self.seq += 1
            if ret == RL_NACK_ERROR:
                retry = False                        # a NACK: next sequence number, no retry flag
                continue
            return ret
        return ret

    def _response(self, msg_id: int, seq: int) -> int:
        deadline = time.monotonic() + self.ack_timeout
        while True:
            if not self._wait_irq(deadline):
                return RL_RESP_TIMEOUT
            msg, err = self._read_message()
            if msg is None or err not in (RL_OK,):
                if msg is not None and msg.msg_class == CLASS_ASYNC:
                    continue                         # a damaged event; keep waiting
                return err if err else RL_PROTOCOL_ERROR
            if msg.msg_class == CLASS_ASYNC:
                self._events(msg)
                continue
            if msg.msg_class == CLASS_NACK:
                return RL_NACK_ERROR
            if msg.msg_class != CLASS_RSP:
                return RL_INVALID_STATE
            if msg.msg_id != msg_id:
                # an error response: the first sub-block holds the error type (rlErrorResp_t)
                blocks = msg.blocks()
                if blocks and len(blocks[0][1]) >= 2:
                    return struct.unpack_from("<H", blocks[0][1], 0)[0]
                return RL_PROTOCOL_ERROR
            if msg.seq != seq:
                return RL_SEQ_MISMATCH
            return RL_OK

    def _wait_irq(self, deadline: float) -> bool:
        while True:
            if self.ftdi.host_irq():
                return True
            if time.monotonic() > deadline:
                return False
            time.sleep(0.001)

    def _poll(self, timeout: float):
        """Read whatever the radar has raised HOST_IRQ for within the timeout."""
        if not self.powered:
            time.sleep(timeout)
            return
        if self._wait_irq(time.monotonic() + timeout):
            msg, err = self._read_message()
            if msg is not None and err == RL_OK and msg.msg_class == CLASS_ASYNC:
                self._events(msg)

    def _drain(self):
        for _ in range(16):
            if not self.ftdi.host_irq():
                return
            msg, err = self._read_message()
            if msg is not None and err == RL_OK and msg.msg_class == CLASS_ASYNC:
                self._events(msg)

    def _await(self, sb: int, timeout: float) -> bool:
        """Wait until an event with this sub-block id has arrived; leave it for wait_event."""
        deadline = time.monotonic() + timeout
        while True:
            if any(ev["sbid"] == sb for ev in self.pending):
                return True
            if time.monotonic() > deadline:
                return False
            self._poll(min(0.05, max(0.0, deadline - time.monotonic())))

    def _events(self, msg: Message):
        for sb, data in msg.blocks():
            ev = {"dev": 1, "msg": sb >> 5, "sbid": sb, "payload": data.hex(), "time": time.time()}
            self.pending.append(ev)
            self.event_log.append(ev)
            self.log(f"event msg=0x{sb >> 5:x} sbid=0x{sb:04x}")

    def _read_message(self) -> tuple[Message | None, int]:
        """rlDriverRxHdrRead + rlDriverRdVerifyMsg for a host: CNYS, wait for HOST_IRQ to
        drop, find the sync, read header, payload and CRC."""
        f = self.ftdi
        f.spi_write(H2D_CNYS)
        if not self._wait_irq_low(0.5):
            f.spi_write(H2D_CNYS)
            if not self._wait_irq_low(0.5):
                return None, RL_HOSTIRQ_TIMEOUT
        buf = bytearray(f.spi_read(4))
        shift = 0
        while bytes(buf[:4]) not in (D2H_SYNC, H2D_SYNC):
            if shift >= SYNC_SCAN_LIMIT:
                f.spi_read(2)                                   # keep the reads 4n + CRC long
                return None, RL_PROTOCOL_ERROR
            if shift % 4 == 0:
                buf += f.spi_read(4)
            del buf[0]
            shift += 1
        rest = bytes(buf[4:])                                   # header bytes read with the sync
        need = HEADER_LEN - len(rest)
        header = rest + (f.spi_read(need + (need % 2)) if need > 0 else b"")
        extra = header[HEADER_LEN:]
        header = header[:HEADER_LEN]
        msg, err = parse_header(header)
        if self.trace:
            self.log(f"  SPI -> {(bytes(buf[:4]) + header).hex()}")
        if err != RL_OK:
            # rlDriverMsgRead flushes the device's SPI buffer after a bad header
            for _ in range(15):
                f.spi_read(16)
            f.spi_read(2)
            return msg, err
        crc_present = ((msg.flags >> 8) & 0x3) == 0
        crc_len = (2 << ((msg.flags >> 10) & 0x3)) if crc_present else 0
        pay_len = msg.length - HEADER_LEN - crc_len
        if pay_len < 0 or msg.length > 2048:
            return msg, RL_PROTOCOL_ERROR
        body = extra + f.spi_read(max(0, pay_len + crc_len - len(extra) + ((pay_len + crc_len - len(extra)) % 2)))
        msg.payload, msg.crc = body[:pay_len], body[pay_len:pay_len + crc_len]
        if self.trace:
            self.log(f"  SPI -> {body.hex()}")
        if crc_present and crc_len == 2:
            if struct.pack("<H", crc16(header + msg.payload)) != msg.crc:
                return msg, RL_CRC_FAILED
        return msg, RL_OK

    def _wait_irq_low(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while self.ftdi.host_irq():
            if time.monotonic() > deadline:
                return False
            time.sleep(0.002)
        return True


def _arg(a):
    """A RadarLinkClient argument (type, value) -> the value the Python side takes."""
    if not isinstance(a, (tuple, list)) or len(a) != 2:
        return a
    kind, value = a
    if kind == "buf":
        return bytes.fromhex(value) if isinstance(value, str) else bytes(value)
    if kind == "callback":
        return None
    return int(value)
