"""
Tests for acquisition.native: the RadarLinkDLL calls NativeRadar makes must carry the
values mmWave Studio logged for the same configuration (its API: lines are the arguments
of the same DLL functions), and the structs must have mmWaveLink's layout.

No hardware: the helper process and the UART are replaced by fakes.

Run with:  python -m pytest test_acquisition_native.py -q
"""
import struct

import pytest

from acquisition import studio_log
from acquisition.native import (SB_BSS_POWERUP, SB_MSS_POWERUP, SB_RF_INIT_CALIB, NativeRadar,
                                adc_out_struct, channel_struct, config_calls, profile_struct)
from acquisition.radar import RadarError

# the configuration block of chuze_dca_noha/data_test_LogFile.txt (mmWave Studio 2.1)
STUDIO_LOG = """\
02-Apr-2026 16:23:40: API:select_capture_device,DCA1000,0,
02-Apr-2026 16:23:41: API:select_chip_version,IWR6843,0,
02-Apr-2026 16:23:50: API:ChannelConfig,7,15,0,
02-Apr-2026 16:23:50: API:AdcOutConfig,2,1,0,
02-Apr-2026 16:23:50: API:DataFmtConfig,15,2,1,0,1,0,
02-Apr-2026 16:23:50: API:LowPowerConfig,0,0,0,
02-Apr-2026 16:23:51: API:DataPathConfig,1,1,0,2,0,
02-Apr-2026 16:23:51: API:LvdsClkConfig,1,1,0,
02-Apr-2026 16:23:51: API:SetHsiClock,9,0,
02-Apr-2026 16:23:51: API:LaneConfig,3,0,
02-Apr-2026 16:23:51: API:LvdsLaneConfig,0,1,0,
02-Apr-2026 16:23:52: API:ProfileConfig,0,1509949440,6000,100,1500,0,0,5523,100,150,10785,1,0,36,0,
02-Apr-2026 16:23:52: API:ChirpConfig,0,0,0,0,0,0,0,2,0,
02-Apr-2026 16:23:52: API:ChirpConfig,1,1,0,0,0,0,0,5,0,
02-Apr-2026 16:23:53: API:EnableTestSource,0,1,0,
02-Apr-2026 16:23:53: API:FrameConfig,0,1,250,255,8000000,0,300,0,
"""


def _api(verb):
    return [list(map(int, f[:-1])) for v, f, _ in studio_log.api_lines(STUDIO_LOG) if v == verb]


@pytest.fixture
def cfg():
    return studio_log.read_config(STUDIO_LOG)


def _by_name(calls):
    out = {}
    for fn, args, _ in calls:
        out.setdefault(fn.replace("RadarLinkImpl_", ""), []).append([v for _, v in args])
    return out


def test_struct_sizes(cfg):
    assert len(channel_struct(cfg)) == 8            # rlChanCfg_t
    assert len(adc_out_struct(cfg)) == 8            # rlAdcOutCfg_t
    assert len(profile_struct(cfg)) == 44           # rlProfileCfg_t


def test_channel_and_adc_structs_match_the_log(cfg):
    tx, rx = _api("ChannelConfig")[0][:2]
    assert struct.unpack("<4H", channel_struct(cfg)) == (rx, tx, 0, 0)
    bits, fmt = _api("AdcOutConfig")[0][:2]
    assert struct.unpack("<BBHHH", adc_out_struct(cfg)) == (bits, 0, fmt, 0, 0)


def test_profile_struct_matches_the_log(cfg):
    logged = _api("ProfileConfig")[0]
    f = struct.unpack("<HBBIIIIIIHHHHBBHHH", profile_struct(cfg))
    (pid, vco, lut, start, idle, adc_start, ramp_end, backoff, phase, slope, tx_start,
     samples, rate, hpf1, hpf2, tx_cal, rx_gain, reserved) = f
    assert [pid, start, idle, adc_start, ramp_end, backoff, phase, slope, tx_start, samples,
            rate, hpf1, hpf2, rx_gain] == logged
    assert (vco, lut, tx_cal, reserved) == (2, 0, 0, 0)     # the lab's hpf2 flags 0x20000


def test_scalar_calls_match_the_log(cfg):
    calls = _by_name(config_calls(cfg))
    assert calls["DataFmtConfig"][0][1:] == _api("DataFmtConfig")[0]
    assert calls["LowPowerConfig"][0][1:] == _api("LowPowerConfig")[0][1:]
    assert calls["DataPathConfig"][0][1:4] == _api("DataPathConfig")[0][:3]
    assert calls["DataPathConfig"][0][4:] == [0, 0, 0, 0, 0]
    assert calls["LvdsClkConfig"][0][1:] == _api("LvdsClkConfig")[0]
    assert calls["SetHsiClock"][0][1:] == _api("SetHsiClock")[0]
    assert calls["LaneConfig"][0][1:] == _api("LaneConfig")[0]
    assert calls["LvdsLaneConfig"][0][1:] == _api("LvdsLaneConfig")[0]
    assert sorted(c[1:] for c in calls["ChirpConfig"]) == sorted(_api("ChirpConfig"))
    assert calls["EnableTestSource"][0][1:] == _api("EnableTestSource")[0]
    frame = calls["FrameConfig"][0][1:]
    assert frame[:7] == _api("FrameConfig")[0]              # 300 = 150 complex samples x 2
    assert frame[7:] == [0, 1, 0]                           # no dummy chirps, SW trigger


def test_order_is_the_lab_scripts(cfg):
    names = [fn.replace("RadarLinkImpl_", "") for fn, _, _ in config_calls(cfg)]
    assert names == ["ChannelConfig", "AdcOutConfig", "DataFmtConfig", "LowPowerConfig", "RfInit",
                     "DataPathConfig", "LvdsClkConfig", "SetHsiClock", "LaneConfig",
                     "LvdsLaneConfig", "ProfileConfig", "ChirpConfig", "ChirpConfig",
                     "EnableTestSource", "FrameConfig"]
    assert [e for fn, _, e in config_calls(cfg) if e] == [SB_RF_INIT_CALIB]


def test_unknown_hsi_clock_is_refused_before_the_board_is_touched(cfg):
    cfg.lvds.data_rate = 3
    radar = NativeRadar(link=FakeLink(), uart_factory=lambda port: FakeUart())
    with pytest.raises(RadarError, match="HSI clock"):
        radar.configure(cfg)
    assert radar.link.called == []


class FakeLink:
    EVENTS = {"RadarLinkImpl_PowerOn": SB_MSS_POWERUP, "RadarLinkImpl_RfEnable": SB_BSS_POWERUP,
              "RadarLinkImpl_RfInit": SB_RF_INIT_CALIB}

    def __init__(self, fail=None, silent=None):
        self.called, self.fail, self.silent = [], fail or {}, silent or set()
        self.pending = []

    def call(self, fn, *args, timeout=60):
        self.called.append((fn, [v for _, v in args]))
        if fn in self.EVENTS and fn not in self.silent:
            self.pending.append({"sbid": self.EVENTS[fn], "payload": "00"})
        return self.fail.get(fn, 0)

    def wait_event(self, sbids, timeout):
        want = {sbids} if isinstance(sbids, int) else set(sbids)
        for ev in self.pending:
            if ev["sbid"] in want:
                self.pending.remove(ev)
                return ev
        return None

    def close(self):
        pass


class FakeUart:
    def __init__(self, rc=0):
        self.rc, self.done = rc, []

    def connect(self, baud):
        return self.rc

    def download_bss(self, path):
        self.done.append(("bss", path))

    def download_mss(self, path):
        self.done.append(("mss", path))

    def close(self):
        self.done.append("close")


def test_boot_configure_start_stop(cfg):
    link, uart = FakeLink(), FakeUart()
    ports = []
    radar = NativeRadar(com_port=15, link=link,
                        uart_factory=lambda port: ports.append(port) or uart)
    radar.configure(cfg)
    radar.start()
    radar.stop()
    names = [fn.replace("RadarLinkImpl_", "") for fn, _ in link.called]
    assert names[:8] == ["setSystemType", "OpenGenericGpioIf", "SOPControl", "OpenBoardControlIf",
                         "FullReset", "FullReset", "CloseBoardControlIf", "CloseGenericGpioIf"]
    assert names[8:10] == ["PowerOn", "RfEnable"]
    assert names[-2:] == ["SensorStart", "SensorStop"]
    assert ports == [15]
    assert [d[0] for d in uart.done[:2]] == ["bss", "mss"] and uart.done[-1] == "close"
    power_on = link.called[8][1]
    assert power_on == [1, 0, 1000, 0, 0, 0]                 # as mmWave Studio: CRC type 0
    assert radar.booted


def test_failed_call_names_the_function(cfg):
    radar = NativeRadar(link=FakeLink(fail={"RadarLinkImpl_ProfileConfig": -3}),
                        uart_factory=lambda port: FakeUart())
    with pytest.raises(RadarError, match="ProfileConfig returned -3"):
        radar.configure(cfg)


def test_missing_power_up_event_is_an_error(cfg):
    radar = NativeRadar(link=FakeLink(silent={"RadarLinkImpl_PowerOn"}),
                        uart_factory=lambda port: FakeUart())
    with pytest.raises(RadarError, match="0x5000"):
        radar.configure(cfg)


def test_bootloader_silence_is_an_error(cfg):
    uart = FakeUart(rc=3)
    radar = NativeRadar(com_port=14, link=FakeLink(), uart_factory=lambda port: uart)
    with pytest.raises(RadarError, match="COM14"):
        radar.configure(cfg)
    assert uart.done == ["close"]
