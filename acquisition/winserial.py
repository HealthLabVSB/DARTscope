"""
acquisition/winserial.py

A Windows serial port through the Win32 API (ctypes) - enough for the radar's UART
bootloader, without pyserial. Settings as .NET's SerialPort defaults, which is what
mmWave Studio opens the port with: 8 data bits, no parity, one stop bit, no handshake,
DTR and RTS off.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes

GENERIC_READ, GENERIC_WRITE, OPEN_EXISTING = 0x80000000, 0x40000000, 3
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
PURGE_ALL = 0x0001 | 0x0002 | 0x0004 | 0x0008
MAXDWORD = 0xFFFFFFFF

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)


class DCB(ctypes.Structure):
    _fields_ = [("DCBlength", wintypes.DWORD), ("BaudRate", wintypes.DWORD),
                ("flags", wintypes.DWORD), ("wReserved", wintypes.WORD),
                ("XonLim", wintypes.WORD), ("XoffLim", wintypes.WORD),
                ("ByteSize", wintypes.BYTE), ("Parity", wintypes.BYTE),
                ("StopBits", wintypes.BYTE), ("XonChar", ctypes.c_char),
                ("XoffChar", ctypes.c_char), ("ErrorChar", ctypes.c_char),
                ("EofChar", ctypes.c_char), ("EvtChar", ctypes.c_char),
                ("wReserved1", wintypes.WORD)]


class COMMTIMEOUTS(ctypes.Structure):
    _fields_ = [("ReadIntervalTimeout", wintypes.DWORD),
                ("ReadTotalTimeoutMultiplier", wintypes.DWORD),
                ("ReadTotalTimeoutConstant", wintypes.DWORD),
                ("WriteTotalTimeoutMultiplier", wintypes.DWORD),
                ("WriteTotalTimeoutConstant", wintypes.DWORD)]


class COMSTAT(ctypes.Structure):
    _fields_ = [("flags", wintypes.DWORD), ("cbInQue", wintypes.DWORD),
                ("cbOutQue", wintypes.DWORD)]


_k32.CreateFileW.restype = wintypes.HANDLE
_k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                             wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
for _name, _args in (("GetCommState", [wintypes.HANDLE, ctypes.POINTER(DCB)]),
                     ("SetCommState", [wintypes.HANDLE, ctypes.POINTER(DCB)]),
                     ("SetCommTimeouts", [wintypes.HANDLE, ctypes.POINTER(COMMTIMEOUTS)]),
                     ("SetupComm", [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD]),
                     ("PurgeComm", [wintypes.HANDLE, wintypes.DWORD]),
                     ("ClearCommError", [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD),
                                         ctypes.POINTER(COMSTAT)]),
                     ("WriteFile", [wintypes.HANDLE, wintypes.LPCVOID, wintypes.DWORD,
                                    ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]),
                     ("ReadFile", [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                   ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]),
                     ("CloseHandle", [wintypes.HANDLE])):
    getattr(_k32, _name).argtypes = _args
    getattr(_k32, _name).restype = wintypes.BOOL


class SerialError(OSError):
    pass


class WinSerial:
    def __init__(self, port: int, baud: int, read_buffer: int = 131072):
        self.port, self.baud, self.read_buffer = int(port), int(baud), read_buffer
        self.h = None

    @property
    def is_open(self) -> bool:
        return self.h is not None

    def open(self):
        h = _k32.CreateFileW(rf"\\.\COM{self.port}", GENERIC_READ | GENERIC_WRITE, 0, None,
                             OPEN_EXISTING, 0, None)
        if h in (None, INVALID_HANDLE_VALUE):
            raise SerialError(ctypes.get_last_error(),
                              f"COM{self.port} cannot be opened - is another program using it?")
        self.h = h
        _k32.SetupComm(h, self.read_buffer, 8192)
        self._apply_baud()
        t = COMMTIMEOUTS(MAXDWORD, 0, 0, 0, 5000)          # reads return at once
        _k32.SetCommTimeouts(h, ctypes.byref(t))
        _k32.PurgeComm(h, PURGE_ALL)

    def _apply_baud(self):
        dcb = DCB()
        dcb.DCBlength = ctypes.sizeof(DCB)
        if not _k32.GetCommState(self.h, ctypes.byref(dcb)):
            raise SerialError(ctypes.get_last_error(), "GetCommState failed")
        dcb.BaudRate = self.baud
        dcb.flags = 0x0001                                  # fBinary; no flow control, DTR/RTS off
        dcb.ByteSize, dcb.Parity, dcb.StopBits = 8, 0, 0
        if not _k32.SetCommState(self.h, ctypes.byref(dcb)):
            raise SerialError(ctypes.get_last_error(), f"SetCommState {self.baud} failed")

    def set_baud(self, baud: int):
        self.baud = int(baud)
        if self.h is not None:
            self._apply_baud()

    def close(self):
        if self.h is not None:
            _k32.CloseHandle(self.h)
            self.h = None

    def write(self, data: bytes | str):
        if isinstance(data, str):
            data = data.encode("ascii")
        n = wintypes.DWORD(0)
        if not _k32.WriteFile(self.h, data, len(data), ctypes.byref(n), None) or n.value != len(data):
            raise SerialError(ctypes.get_last_error(), f"write to COM{self.port} failed")

    def in_waiting(self) -> int:
        err, st = wintypes.DWORD(0), COMSTAT()
        _k32.ClearCommError(self.h, ctypes.byref(err), ctypes.byref(st))
        return int(st.cbInQue)

    def read_existing(self) -> bytes:
        n = self.in_waiting()
        if n <= 0:
            return b""
        buf = ctypes.create_string_buffer(n)
        got = wintypes.DWORD(0)
        _k32.ReadFile(self.h, buf, n, ctypes.byref(got), None)
        return buf.raw[:got.value]
