"""
acquisition/ports.py

Which COM port is the radar's? Found from the USB identity of the device behind each
port, not from its number - Windows numbers ports per USB socket, so COM15 today can be
COM6 after the cable moves.

    IWR6843AOPEVM   Silicon Labs CP2105 (USB 10C4:EA70). Interface 0, "Enhanced COM
                    Port", is the radar's UART - the bootloader mmWave Studio's
                    ar1.Connect talks to and native.py loads the firmware through.
                    Interface 1, "Standard COM Port", is the demo data UART (unused here).
                    Both stay present in the 60-pin (DCA1000) switch position.
    xWR6843 ISK/ODS TI XDS110 (0451:BEF3): interface 0 "Application/User UART" is the
                    radar's UART, interface 3 the data port. (Not on the lab's PC; from
                    TI's documentation.)
    DCA1000EVM      FTDI FT4232H with TI's IDs (0451:FD03), "AR-DevPack-EVM-012" A-D.
                    The radar's SPI, SOP and reset lines go through it; RadarLinkDLL
                    opens it by itself, so its ports only tell that it is plugged in.

Windows: read from the registry with the standard library (no pyserial, nothing to
install): HKLM\\HARDWARE\\DEVICEMAP\\SERIALCOMM lists the ports that exist now; the Enum
tree gives each port's device (hardware key, friendly name, container of the functions of
one USB device).
macOS and Linux: pyserial's list of serial devices, which carries the USB identity; a port
is named by its device node (/dev/cu.usbserial-01081FDF0, /dev/ttyUSB0), the interface
comes from the digit macOS appends to the serial number or from Linux's USB location.
The readers are injectable, so the tests run on recorded contents.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import re
import subprocess
import sys

IS_WINDOWS = sys.platform.startswith("win")
from dataclasses import dataclass, field

ENUM_BUSES = ("USB", "FTDIBUS", "BTHENUM")


@dataclass(frozen=True)
class Known:
    vid: int
    pid: int
    interface: int | None        # None: every interface of the device
    role: str                    # "radar", "radar-data", "dca1000"
    board: str
    what: str


KNOWN = (
    Known(0x10C4, 0xEA70, 0, "radar", "IWR6843AOPEVM",
          "the radar's UART (CP2105 Enhanced COM Port) - firmware download"),
    Known(0x10C4, 0xEA70, 1, "radar-data", "IWR6843AOPEVM",
          "the radar's data UART (CP2105 Standard COM Port) - not used for recording"),
    # the AWR2243 cascade's AM273x has the same XDS110: its ports look the same
    Known(0x0451, 0xBEF3, 0, "radar", "xWR6843 ISK/ODS or AWR2243 cascade (XDS110)",
          "the radar's UART (XDS110 Application/User UART)"),
    Known(0x0451, 0xBEF3, 3, "radar-data", "xWR6843 ISK/ODS or AWR2243 cascade (XDS110)",
          "the radar's data port (XDS110 Auxiliary Data Port)"),
    Known(0x0451, 0xFD03, None, "dca1000", "DCA1000EVM",
          "DCA1000 FTDI port (the radar's SPI, SOP and reset go through this device)"),
)


@dataclass
class PortInfo:
    name: str                    # "COM15"
    description: str = ""        # the driver's friendly name, without "(COMn)"
    vid: int | None = None
    pid: int | None = None
    interface: int | None = None
    serial: str = ""
    container: str = ""          # ContainerID: the same for all functions of one device
    role: str = ""
    board: str = ""
    what: str = ""

    @property
    def number(self) -> int:
        """15 for COM15; 0 for a device path (/dev/cu.usbserial-...)."""
        m = re.fullmatch(r"COM(\d+)", self.name, re.I)
        return int(m.group(1)) if m else 0

    @property
    def usb_id(self) -> str:
        if self.vid is None:
            return ""
        s = f"{self.vid:04X}:{self.pid:04X}"
        return s + (f" if{self.interface}" if self.interface is not None else "")

    def label(self) -> str:
        """For a drop-down list: 'COM15 - IWR6843AOPEVM: the radar's UART (...)'."""
        if self.role:
            return f"{self.name} - {self.board}: {self.what}"
        return f"{self.name} - {self.description or 'unknown device'}"


@dataclass
class Detection:
    ports: list = field(default_factory=list)          # every present port, radar first
    radar: PortInfo | None = None
    radar_candidates: list = field(default_factory=list)
    dca1000_ports: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def dca1000_present(self) -> bool:
        return bool(self.dca1000_ports)

    def summary(self) -> str:
        r = f"radar UART {self.radar.name} ({self.radar.board})" if self.radar else "radar UART not found"
        d = (f"DCA1000 USB present ({', '.join(p.name for p in self.dca1000_ports)})"
             if self.dca1000_ports else "DCA1000 USB not found")
        return f"{r}; {d}"


# --- the registry ------------------------------------------------------------------------------

class WinRegistry:
    """What detect() needs from the registry (the real one)."""

    def serialcomm(self) -> list[str]:
        import winreg
        out = []
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\SERIALCOMM")
        except OSError:
            return out
        with key:
            i = 0
            while True:
                try:
                    _, value, _ = winreg.EnumValue(key, i)
                except OSError:
                    break
                out.append(str(value))
                i += 1
        return out

    def instances(self) -> list[dict]:
        """Every device instance with a PortName: bus, hardware key, instance, values."""
        import winreg
        out = []
        base = r"SYSTEM\CurrentControlSet\Enum"
        for bus in ENUM_BUSES:
            try:
                bus_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, f"{base}\\{bus}")
            except OSError:
                continue
            with bus_key:
                for hw in _subkeys(bus_key):
                    try:
                        hw_key = winreg.OpenKey(bus_key, hw)
                    except OSError:
                        continue
                    with hw_key:
                        for inst in _subkeys(hw_key):
                            try:
                                inst_key = winreg.OpenKey(hw_key, inst)
                            except OSError:
                                continue
                            with inst_key:
                                port = _value(inst_key, "PortName", sub="Device Parameters")
                                if not port:
                                    continue
                                out.append({"bus": bus, "hardware": hw, "instance": inst,
                                            "port": str(port),
                                            "friendly": str(_value(inst_key, "FriendlyName") or ""),
                                            "desc": str(_value(inst_key, "DeviceDesc") or ""),
                                            "container": str(_value(inst_key, "ContainerID") or "")})
        return out


def _subkeys(key) -> list[str]:
    import winreg
    out, i = [], 0
    while True:
        try:
            out.append(winreg.EnumKey(key, i))
        except OSError:
            return out
        i += 1


def _value(key, name, sub=None):
    import winreg
    try:
        if sub:
            with winreg.OpenKey(key, sub) as k:
                return winreg.QueryValueEx(k, name)[0]
        return winreg.QueryValueEx(key, name)[0]
    except OSError:
        return None


# --- detection -------------------------------------------------------------------------------

_USB_RE = re.compile(r"VID_([0-9A-F]{4})&PID_([0-9A-F]{4})(?:&MI_([0-9A-F]{2}))?", re.I)
_FTDI_RE = re.compile(r"VID_([0-9A-F]{4})\+PID_([0-9A-F]{4})\+(\w+)", re.I)


def _classify(p: PortInfo) -> PortInfo:
    for k in KNOWN:
        if p.vid == k.vid and p.pid == k.pid and (k.interface is None or p.interface == k.interface):
            p.role, p.board, p.what = k.role, k.board, k.what
            break
    return p


def _identify(inst: dict) -> PortInfo:
    p = PortInfo(name=inst["port"].upper(), container=inst.get("container", ""))
    friendly = inst.get("friendly") or inst.get("desc") or ""
    if ";" in friendly and friendly.startswith("@"):         # "@oem12.inf,%x%;Name"
        friendly = friendly.split(";")[-1]
    p.description = re.sub(r"\s*\(COM\d+\)\s*$", "", friendly).strip()
    hw = inst.get("hardware", "")
    if inst.get("bus") == "FTDIBUS":
        m = _FTDI_RE.search(hw)
        if m:
            p.vid, p.pid, p.serial = int(m.group(1), 16), int(m.group(2), 16), m.group(3)
            letter = p.serial[-1:].upper()
            p.interface = ord(letter) - ord("A") if "A" <= letter <= "H" else None
    else:
        m = _USB_RE.search(hw)
        if m and inst.get("bus") == "USB":
            p.vid, p.pid = int(m.group(1), 16), int(m.group(2), 16)
            p.interface = int(m.group(3), 16) if m.group(3) else None
    return _classify(p)


class PosixPorts:
    """What detect() needs on macOS and Linux: pyserial's list of serial devices."""

    def comports(self) -> list:
        from serial.tools import list_ports
        return list(list_ports.comports())


def _posix_interface(device: str, serial: str, location: str) -> int | None:
    """
    The USB interface behind a device node. macOS names the node after the serial number
    and the interface (/dev/cu.usbserial-01081FDF0 = CP2105 01081FDF, interface 0; FTDI
    FT80MJN4 + 0..3 = A..D); Linux gives the location "1-1.2:1.0" (configuration.interface).
    """
    if serial:
        m = re.search(re.escape(serial) + r"(\d)$", device)
        if m:
            return int(m.group(1))
    m = re.search(r":\d+\.(\d+)$", location or "")
    return int(m.group(1)) if m else None


def _identify_posix(c) -> PortInfo:
    serial = getattr(c, "serial_number", None) or ""
    p = PortInfo(name=c.device, description=getattr(c, "description", "") or "",
                 vid=getattr(c, "vid", None), pid=getattr(c, "pid", None), serial=serial,
                 container=f"{serial}@{getattr(c, 'location', '') or ''}".split(":")[0])
    if p.vid is not None:
        p.interface = _posix_interface(c.device, serial, getattr(c, "location", "") or "")
    return _classify(p)


def _windows_ports(reader) -> list:
    present = {str(v).upper() for v in reader.serialcomm()}
    seen: dict[str, PortInfo] = {}
    for inst in reader.instances():
        info = _identify(inst)
        if info.name not in present:
            continue                   # an unplugged device's old entry
        old = seen.get(info.name)
        if old is None or (info.role and not old.role):
            seen[info.name] = info
    for name in present:
        seen.setdefault(name, PortInfo(name=name))
    return list(seen.values())


def _posix_ports(reader) -> list:
    out = []
    for c in reader.comports():
        dev = str(getattr(c, "device", ""))
        # macOS lists every port twice (/dev/tty.* and /dev/cu.*); cu is the one to open.
        # Ports without a USB identity (Bluetooth, debug console) are left out.
        if not dev or dev.startswith("/dev/tty.") or getattr(c, "vid", None) is None:
            continue
        out.append(_identify_posix(c))
    return out


def detect(reader=None) -> Detection:
    """The present serial ports, which of them is the radar's UART, and whether the DCA1000 is in."""
    reader = reader or (WinRegistry() if IS_WINDOWS else PosixPorts())
    found = _posix_ports(reader) if hasattr(reader, "comports") else _windows_ports(reader)
    order = {"radar": 0, "radar-data": 1, "dca1000": 2, "": 3}
    # with several boards, the one listed first in KNOWN (the IWR6843AOPEVM this tab
    # records with) before an XDS110 that may be the cascade's - not the lower COM number
    board_rank = {k.board: i for i, k in enumerate(KNOWN)}
    ports = sorted(found, key=lambda p: (order.get(p.role, 3), board_rank.get(p.board, len(KNOWN)),
                                         p.number, p.name))
    det = Detection(ports=ports)
    det.radar_candidates = [p for p in ports if p.role == "radar"]
    det.dca1000_ports = [p for p in ports if p.role == "dca1000"]
    if len(det.radar_candidates) == 1:
        det.radar = det.radar_candidates[0]
    elif len(det.radar_candidates) > 1:
        det.radar = det.radar_candidates[0]
        det.notes.append("more than one radar board is plugged in (" + ", ".join(
            f"{p.name} {p.board}" for p in det.radar_candidates) + f"); {det.radar.name} is taken - "
            "choose another one if needed")
    else:
        det.notes.append("no radar UART found: plug in the radar board's USB cable (on the "
                         "IWR6843AOPEVM its own micro-USB; the board shows up as the Silicon Labs "
                         "CP2105 dual UART with an Enhanced and a Standard port)")
    if not det.dca1000_ports:
        det.notes.append("the DCA1000's USB (FTDI 'AR-DevPack-EVM-012') is not plugged in or has "
                         "no driver: the radar's reset, SOP and SPI go through it")
    return det


def process_running(image: str = "mmWaveStudio.exe") -> bool:
    """Whether a program runs (mmWave Studio holds the FTDI device and the radar's COM port).
    Elsewhere than on Windows it can only run under Wine or CrossOver."""
    if not IS_WINDOWS:
        try:
            out = subprocess.run(["pgrep", "-if", image], capture_output=True, text=True,
                                 timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            return False
        return bool(out.strip())
    try:
        out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {image}", "/NH", "/FO", "CSV"],
                             capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return image.lower() in out.lower()
