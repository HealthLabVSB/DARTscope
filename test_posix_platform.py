"""
Tests for the macOS/Linux side of the acquisition: port detection from pyserial's list,
port names, the paced UART bootloader, firmware found in Wine bottles, and NativeRadar
picking the Python link. They run on every platform (the platform-specific parts are
injected or monkeypatched).

Run with:  python -m pytest test_posix_platform.py -q
"""
import os
from types import SimpleNamespace

import pytest

from acquisition import config as config_mod
from acquisition import serialport
from acquisition.bootloader import RadarUart
from acquisition.ports import detect


def _c(device, vid=None, pid=None, serial=None, location=None, description=""):
    return SimpleNamespace(device=device, vid=vid, pid=pid, serial_number=serial,
                           location=location, description=description)


class FakeMacPorts:
    """This Mac on 2026-10-05: IWR6843AOPEVM (CP2105) + DCA1000 (FT4232H), Bluetooth."""

    def comports(self):
        out = [_c("/dev/cu.debug-console"), _c("/dev/cu.Bluetooth-Incoming-Port")]
        out += [_c(f"/dev/cu.usbserial-FT80MJN4{i}", 0x451, 0xFD03, "FT80MJN4", "0-1.2",
                   "AR-DevPack-EVM-012") for i in (2, 3, 0, 1)]
        out += [_c(f"/dev/cu.usbserial-01081FDF{i}", 0x10C4, 0xEA70, "01081FDF", "0-1.3",
                   "CP2105 Dual USB to UART Bridge Controller") for i in (0, 1)]
        out += [_c("/dev/tty.usbserial-01081FDF0", 0x10C4, 0xEA70, "01081FDF", "0-1.3")]
        return out


class FakeLinuxPorts:
    def comports(self):
        return [_c("/dev/ttyUSB0", 0x10C4, 0xEA70, "01081FDF", "1-1.2:1.0"),
                _c("/dev/ttyUSB1", 0x10C4, 0xEA70, "01081FDF", "1-1.2:1.1")]


def test_mac_detection_finds_the_radar_uart_and_the_dca1000():
    det = detect(FakeMacPorts())
    assert det.radar.name == "/dev/cu.usbserial-01081FDF0"
    assert det.radar.board == "IWR6843AOPEVM"
    assert sorted(p.interface for p in det.dca1000_ports) == [0, 1, 2, 3]
    assert [p.name for p in det.ports][:2] == ["/dev/cu.usbserial-01081FDF0",
                                                "/dev/cu.usbserial-01081FDF1"]
    assert not any(p.name.startswith("/dev/tty.") for p in det.ports)   # cu only
    assert det.notes == []


def test_linux_detection_takes_the_interface_from_the_location():
    det = detect(FakeLinuxPorts())
    assert det.radar.name == "/dev/ttyUSB0"
    assert any("DCA1000" in n for n in det.notes)            # not plugged in here


def test_port_names():
    assert serialport.com_number(15) == 15
    assert serialport.com_number("COM15") == 15
    assert serialport.com_number("/dev/cu.usbserial-01081FDF0") is None
    assert serialport.port_label(15) == "COM15"
    assert serialport.port_label("/dev/ttyUSB0") == "/dev/ttyUSB0"


@pytest.mark.skipif(serialport.IS_WINDOWS, reason="a COM number is a valid port on Windows")
def test_a_windows_com_number_is_refused_elsewhere():
    with pytest.raises(serialport.SerialError):
        serialport.open_serial(12, 921600)


class EchoMonitor:
    """The radar's UART monitor: 'rd' answers 'ffffffff\\r', everything is logged."""

    def __init__(self):
        self.sent = []
        self.pending = b""
        self.is_open = True

    def write(self, data):
        data = data.encode() if isinstance(data, str) else data
        self.sent.append(data.decode())
        if data.startswith(b"rd "):
            self.pending += b"ffffffff\r"

    def read_existing(self):
        out, self.pending = self.pending, b""
        return out

    def in_waiting(self):
        return len(self.pending)

    def close(self):
        self.is_open = False


def test_paced_writes_wait_for_the_monitor():
    u = RadarUart("/dev/cu.x", paced=True)
    u.sp = EchoMonitor()
    u.write(0xFFFFE26C, 1)
    u.write_block(0x0, bytes(range(16)))
    assert u.sp.sent == ["wr ffffe26c 00000001\r", "rd ffffe2fc\r",
                         "wr 00000000 03020100 07060504 0b0a0908 0f0e0d0c \r", "rd ffffe2fc\r"]


def test_unpaced_writes_are_mmwave_studios():
    u = RadarUart(15, paced=False)
    u.sp = EchoMonitor()
    u.write(0xFFFFE26C, 0)
    assert u.sp.sent == ["wr ffffe26c 00000000\r"]


@pytest.mark.skipif(serialport.IS_WINDOWS, reason="no Wine bottles on Windows")
def test_firmware_is_found_in_a_wine_bottle(tmp_path, monkeypatch):
    bottle = tmp_path / "Library" / "Containers" / "com.isaacmarovitz.Whisky" / "Bottles" / "B1"
    fw = bottle / "drive_c" / "ti" / "mmwave_studio_02_01_01_00" / "rf_eval_firmware" / "radarss"
    fw.mkdir(parents=True)
    (fw / "xwr68xx_radarss.bin").write_bytes(b"RPRC")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DARTSCOPE_STUDIO_DIR", raising=False)
    monkeypatch.setattr(config_mod.sys, "platform", "darwin")
    bss, mss = config_mod.DeviceSetup().local_firmware_paths()
    assert bss == str(fw / "xwr68xx_radarss.bin")
    assert mss.startswith("C:\\")                        # not there: left as it was
    other = tmp_path / "studio"
    (other / "rf_eval_firmware" / "masterss").mkdir(parents=True)
    (other / "rf_eval_firmware" / "masterss" / "xwr68xx_masterss.bin").write_bytes(b"RPRC")
    monkeypatch.setenv("DARTSCOPE_STUDIO_DIR", str(other))
    assert config_mod.DeviceSetup().local_firmware_paths()[1] == \
        os.path.join(str(other), "rf_eval_firmware/masterss/xwr68xx_masterss.bin")


def test_paths_stay_as_they_are_on_windows(monkeypatch):
    monkeypatch.setattr(config_mod.sys, "platform", "win32")
    assert config_mod.local_path(r"C:\ti\x.bin") == r"C:\ti\x.bin"


def test_native_radar_uses_the_python_link_off_windows(monkeypatch):
    from acquisition import native
    monkeypatch.setattr(native, "IS_WINDOWS", False)
    made = []

    class FakeLink:
        def __init__(self, *a, **k):
            made.append(self)
    import acquisition.mmwavelink as ml
    monkeypatch.setattr(ml, "MmwLink", FakeLink)
    r = native.NativeRadar()
    r.connect()
    assert made and r.link is made[0]


def test_native_radar_detects_the_uart_off_windows(monkeypatch):
    from acquisition import native, ports
    from acquisition.config import RadarConfig
    monkeypatch.setattr(native, "IS_WINDOWS", False)
    monkeypatch.setattr(ports, "detect", lambda reader=None: detect(FakeMacPorts()))
    r = native.NativeRadar()
    assert r._uart_port(RadarConfig()) == "/dev/cu.usbserial-01081FDF0"     # COM12 of a script ignored
    r2 = native.NativeRadar(com_port="/dev/ttyUSB3")
    assert r2._uart_port(RadarConfig()) == "/dev/ttyUSB3"


def test_open_folder_uses_the_file_manager_of_the_platform(monkeypatch):
    """The Measure tab's Open folder: os.startfile exists only on Windows - Finder (open)
    on macOS, xdg-open on Linux."""
    import measure_gui
    calls = []
    monkeypatch.setattr(measure_gui.subprocess, "Popen", lambda args, **k: calls.append(list(args)))
    monkeypatch.setattr(measure_gui.os, "startfile", lambda p: calls.append(["startfile", p]), raising=False)
    for platform, folder in (("darwin", "/Users/dan/rec"), ("linux", "/home/dan/rec"), ("win32", "C:/rec")):
        monkeypatch.setattr(measure_gui.sys, "platform", platform)
        assert measure_gui.open_in_file_manager(folder)
    assert calls == [["open", "/Users/dan/rec"], ["xdg-open", "/home/dan/rec"], ["startfile", "C:/rec"]]

