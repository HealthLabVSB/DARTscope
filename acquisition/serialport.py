"""
acquisition/serialport.py

The radar's UART on any platform, with the interface bootloader.RadarUart uses (the one
of winserial.WinSerial): open(), close(), is_open, set_baud(), write(), in_waiting(),
read_existing().

    Windows        WinSerial (Win32 API through ctypes, no pyserial) - unchanged
    macOS, Linux   PosixSerial: pyserial on the device node (/dev/cu.usbserial-...,
                   /dev/ttyUSB0), the same settings as .NET's SerialPort defaults that
                   mmWave Studio opens the port with: 8N1, no handshake, DTR and RTS off

A port is named the way the platform names it: a number (15) or "COM15" on Windows, the
device path elsewhere. port_label() gives the text for messages.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import sys

IS_WINDOWS = sys.platform.startswith("win")


class SerialError(OSError):
    pass


def com_number(port) -> int | None:
    """15 for 15, "15" or "COM15"; None for a device path."""
    if isinstance(port, int):
        return port
    s = str(port).strip()
    if s.upper().startswith("COM"):
        s = s[3:]
    return int(s) if s.isdigit() else None


def port_label(port) -> str:
    """COM15 for a Windows port number, the path itself for a device node."""
    n = com_number(port)
    return f"COM{n}" if n is not None else str(port)


class PosixSerial:
    """pyserial behind WinSerial's interface; reads never block."""

    def __init__(self, port: str, baud: int):
        self.port, self.baud = str(port), int(baud)
        self.sp = None

    @property
    def is_open(self) -> bool:
        return self.sp is not None and self.sp.is_open

    def open(self):
        try:
            import serial
        except ImportError as e:
            raise SerialError("pyserial is not installed (pip install pyserial)") from e
        sp = serial.Serial()
        sp.port, sp.baudrate = self.port, self.baud
        sp.bytesize, sp.parity, sp.stopbits = serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE
        sp.timeout, sp.write_timeout = 0, 5.0
        sp.xonxoff = sp.rtscts = sp.dsrdtr = False
        sp.dtr = sp.rts = False                 # set before open: no pulse on the lines
        try:
            sp.open()
        except (OSError, serial.SerialException) as e:
            raise SerialError(f"{self.port} cannot be opened ({e}) - is another program using it?") from e
        sp.reset_input_buffer()
        sp.reset_output_buffer()
        self.sp = sp

    def set_baud(self, baud: int):
        self.baud = int(baud)
        if self.sp is not None:
            self.sp.baudrate = self.baud

    def close(self):
        if self.sp is not None:
            try:
                self.sp.close()
            finally:
                self.sp = None

    def write(self, data: bytes | str):
        if isinstance(data, str):
            data = data.encode("ascii")
        try:
            n = self.sp.write(data)
            self.sp.flush()
        except Exception as e:
            raise SerialError(f"write to {self.port} failed ({e})") from e
        if n != len(data):
            raise SerialError(f"write to {self.port} failed ({n} of {len(data)} bytes)")

    def in_waiting(self) -> int:
        return int(self.sp.in_waiting)

    def read_existing(self) -> bytes:
        n = self.in_waiting()
        return self.sp.read(n) if n > 0 else b""


def open_serial(port, baud: int):
    """An unopened serial port object for this platform (call .open())."""
    if IS_WINDOWS:
        from .winserial import WinSerial
        n = com_number(port)
        if n is None:
            raise SerialError(f"{port!r} is not a COM port")
        return WinSerial(n, baud)
    if com_number(port) is not None:
        raise SerialError(f"{port_label(port)} is a Windows port name; on this computer the radar's "
                          f"UART is a device such as /dev/cu.usbserial-...0 (Detect finds it)")
    return PosixSerial(port, baud)
