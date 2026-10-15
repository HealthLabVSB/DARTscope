"""
acquisition/bootloader.py

The radar's UART monitor as mmWave Studio uses it before the firmware runs:
ar1.Connect (connect, switch the device from 115200 to 921600 baud) and
ar1.DownloadBSSFw / DownloadMSSFw (RPRC images written to the device's RAM).

A line-for-line transcription of mmWave Studio's SerialPortWrapper
(Clients/AR1xController/AR1xController.dll), including its waits:
    wake      "x0 \\r\\n"
    read      "rd %08x\\r"          -> the reply is the value in hex
    write     "wr %08x %08x\\r"
    block     "wr %08x w1 w2 ... \\r"  (little-endian 32-bit words, 4096 bytes a line)
    baud      reg 0xFFFFE144 |= 0x7800, then 0xFFFFE264 <- 227552299 (921600) or
              1820417371 (115200), then reconnect at the new rate

Around the RPRC writes the device's memories are prepared with the register sequence of
AR1xxxWrapper.Calling_ATE_DownloadFW (the xWR16xx/18xx/68xx branch - mmWave Studio's
"AR16xx" device flag, which IWR6843 selects); without it the images do not start and
RadarLinkImpl_PowerOn never returns.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import struct
import time

from .serialport import IS_WINDOWS, open_serial, port_label

WAKE = "x0 \r\n"
ALIVE_REG = 0xFFFFE2FC
UART_CTRL_REG = 0xFFFFE144
UART_BAUD_REG = 0xFFFFE264
BAUD_MAGIC = {921600: 227552299, 115200: 1820417371}   # what to write to go to that rate
CHUNK = 4096
RPRC_MAGIC = 0x43525052          # "RPRC"


class BootloaderError(RuntimeError):
    pass


class RadarUart:
    """
    paced: after every write wait until the monitor answers a read, so a command is never
    sent while the previous one is still being executed. The monitor has no flow control;
    on macOS (pyserial, Apple's CP210x driver) the lines arrive back to back and it drops
    characters at their boundaries - the MSS reset vector and the last word of a 4 KB block
    came out wrong and the firmware never started. Default: on everywhere but Windows,
    where mmWave Studio's unpaced timing is the verified one.
    """

    def __init__(self, port, log=None, paced: bool | None = None):
        self.port = port                 # COM number on Windows, the device path elsewhere
        self.log = log or (lambda s: None)
        self.paced = (not IS_WINDOWS) if paced is None else bool(paced)
        self.sp = None
        self.reconnected = False

    # SerialPortWrapper.SendAndRead: flush, write, wait the timeout, collect what came
    def send_and_read(self, command: str, timeout_ms: int = 100) -> str:
        if self.sp is None or not self.sp.is_open:
            raise BootloaderError("port not open")
        self.sp.read_existing()
        self.sp.write(command)
        time.sleep(timeout_ms / 1000.0)
        out = b""
        while self.sp.in_waiting():
            out += self.sp.read_existing()
            if self.sp.in_waiting():
                time.sleep(timeout_ms / 1000.0)
        return out.decode("ascii", "replace")

    def read(self, addr: int) -> int | None:
        resp = self.send_and_read(f"rd {addr:08x}\r", 100)
        try:
            return int(resp.strip(), 16)
        except ValueError:
            return None

    def write(self, addr: int, value: int):
        self.sp.write(f"wr {addr:08x} {value & 0xFFFFFFFF:08x}\r")
        self._settle()

    def write_block(self, addr: int, data: bytes):
        words = "".join(f"{v:08x} " for (v,) in struct.iter_unpack("<I", data[:len(data) // 4 * 4]))
        self.sp.write(f"wr {addr:08x} {words}\r")
        self._settle()

    def _settle(self, timeout: float = 2.0, attempts: int = 3):
        """paced mode: a read of the alive register; its answer means the monitor is idle.
        An answer that does not come is asked for again (the read itself can be lost)."""
        if not self.paced:
            return
        for _ in range(attempts):
            self.sp.write(f"rd {ALIVE_REG:08x}\r")
            buf = b""
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                buf += self.sp.read_existing()
                if buf.endswith(b"\r"):
                    return
                time.sleep(0.0005)
        raise BootloaderError(f"{port_label(self.port)}: the radar's bootloader stopped answering "
                              f"during the download")

    def is_connected(self) -> bool:
        if self.sp is None or not self.sp.is_open:
            self.close()
            return False
        self.sp.write(WAKE)
        time.sleep(0.1)
        return self.read(ALIVE_REG) is not None

    def _open(self, baud: int):
        self.close()
        time.sleep(0.4)
        try:
            self.sp = open_serial(self.port, baud)
        except OSError as e:
            raise BootloaderError(f"{port_label(self.port)}: {e}") from e
        for attempt in range(4):
            try:
                self.sp.open()
                return
            except OSError as e:
                if attempt == 3:
                    raise BootloaderError(f"{port_label(self.port)}: {e}") from e
                time.sleep(0.1)

    def close(self):
        if self.sp is not None:
            self.sp.close()

    # SerialPortWrapper.Init: 0 = connected
    def connect(self, baud: int = 921600) -> int:
        self._open(baud)
        self.sp.write(WAKE)
        time.sleep(0.4)
        if self.is_connected():
            self.sp.write(WAKE)
            self.log(f"connected at {baud} baud")
            return 0
        if self.reconnected:
            self.close()
            self.reconnected = False
            return 2
        return self._reconnect_change_baud(baud)

    # SerialPortWrapper.iReconnectChangeBaudRate
    def _reconnect_change_baud(self, baud: int) -> int:
        wait = 0.3
        self.reconnected = True
        other = 115200 if baud == 921600 else 921600
        magic = BAUD_MAGIC[baud]
        time.sleep(0.5)
        self._open(other)
        time.sleep(0.5)
        if not self.is_connected():
            self.reconnected = False
            self.close()
            return 3
        time.sleep(wait)
        self.log(f"connected with baudrate {other}")
        v = self.read(UART_CTRL_REG) or 0
        self.write(UART_CTRL_REG, v | 0x7800)
        time.sleep(wait)
        # no settle: the device answers at the new rate from here on
        self.sp.write(f"wr {UART_BAUD_REG:08x} {magic:08x}\r")
        time.sleep(wait)
        self.close()
        time.sleep(wait)
        self.log(f"disconnected; trying {baud} baud")
        time.sleep(wait * 2)
        self.reconnected = False
        return self.connect(baud)

    def _poll(self, addr: int, done, tries: int = 10) -> int:
        """Read addr until done(value), at most `tries` more times 100 ms apart."""
        v = self.read(addr) or 0
        while not done(v) and tries > 0:
            time.sleep(0.1)
            v = self.read(addr) or 0
            tries -= 1
        return v

    # AR1xxxWrapper.Calling_ATE_DownloadFW, BSS part, xWR16xx/18xx/68xx branch
    def download_bss(self, path: str, progress=None) -> int:
        blob = open(path, "rb").read(28)
        magic, kind = struct.unpack_from("<I", blob, 0)[0], struct.unpack_from("<I", blob, 24)[0]
        if magic != RPRC_MAGIC:
            raise BootloaderError(f"{path}: not an RPRC image (magic 0x{magic:08x})")
        self.write(0xFFFFE108, 0xADAD00AD)
        self.write(0xFFFFE108, 0xAD0000AD)
        self.write(0xFFFFE108, 0xAD000000)
        self.write(0xFFFFE1D8, 0xAD000100)
        v = self._poll(0xFFFFE1DC, lambda v: v & 0xAD010100)
        self.write(0xFFFFE1D8, 0)
        self.log(f"BSS memory init 0x{v:08x}")
        if kind == 0:                                   # "BSS ROM RPRC" (a patch skips this)
            self.write(0xFFFFE10C, 0)
        self.write(0xFFFFE3B8, 0x67)
        v = self.read(0xFFFFE3CC) or 0
        self.write(0xFFFFE3CC, (v & 0xFFFF) | 0x10100000)
        self.write(0xFFFFE140, 0xAD if kind == 0 else 0xAD00)
        self.write(0xFFFFE1D8, 0xAD000000)
        self.write(0xFFFFE3A8, 0xC0)
        v = self._poll(0xFFFFE3AC, lambda v: (v & 0xC0) == 0xC0)
        self.write(0xFFFFE1D8, 0)
        self.log(f"BSS RAM init 0x{v:08x}")
        return self.download_rprc(path, bss=True, progress=progress)

    # AR1xxxWrapper.Calling_ATE_DownloadFW, MSS part
    def download_mss(self, path: str, progress=None) -> int:
        magic = struct.unpack_from("<I", open(path, "rb").read(4), 0)[0]
        if magic != RPRC_MAGIC:
            raise BootloaderError(f"{path}: not an RPRC image (magic 0x{magic:08x})")
        self.write(0xFFFFE26C, 1)
        self.write(0xFFFFFF5C, 0xAD000003)
        v = self._poll(0xFFFFFF6C, lambda v: (v & 3) == 3)
        self.write(0xFFFFFF5C, 0)
        self.write(0xFFFFFF20, 0xAD00)
        self.log(f"MSS RAM init 0x{v:08x}")
        rc = self.download_rprc(path, bss=False, progress=progress)
        self.write(0xFFFFE26C, 0)
        return rc

    # SerialPortWrapper.DownloadToDevice (RPRC)
    def download_rprc(self, path: str, bss: bool, progress=None) -> int:
        blob = open(path, "rb").read()
        total = len(blob)
        nsec = struct.unpack_from("<I", blob, 12)[0]
        data = blob[24:]
        denom = max(1, total - 24 - nsec * 24)
        done = 0
        while nsec > 0:
            addr = struct.unpack_from("<I", data, 0)[0]
            if bss:
                addr = (addr + 0x39000000) if addr >= 0x08000000 else (addr + 0x40000000)
            addr &= 0xFFFFFFFF
            size = struct.unpack_from("<I", data, 8)[0]
            size += size % 8
            sec = data[24:24 + size]
            used = 0
            for i in range(size // CHUNK + 1):
                chunk = sec[CHUNK * i:CHUNK * i + CHUNK]
                self.write_block(addr, chunk)
                addr += len(chunk)
                step = len(chunk) * 2 if size < 8 else len(chunk)
                used += step
                done += step
                if progress:
                    progress(min(100.0, 100.0 * done / denom))
            data = data[24 + used:]
            nsec -= 1
        return 0
