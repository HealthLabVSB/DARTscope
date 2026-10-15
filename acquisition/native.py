"""
acquisition/native.py

NativeRadar: configures and starts the IWR6843 without mmWave Studio. It makes the same
calls into TI's RadarLinkDLL.dll that mmWave Studio's controller (AR1xController.dll)
makes for the lab's ar1.* commands, with the same arguments - read from its IL, see
docs/NATIVE_RADAR_NOTES.md - so the radar runs the same RF-eval firmware with the same
configuration and the LVDS data are those of an mmWave Studio recording.

    boot       SOP 2 + reset (GPIO through the DCA1000's FTDI), UART bootloader on the
               radar's own COM port: BSS and MSS firmware, then PowerOn over SPI and
               RfEnable, each confirmed by the device's power-up event
    configure  ChanNAdcConfig, LPModConfig, RfInit (+ calibration event), DataPathConfig,
               LvdsClkConfig, LVDSLaneConfig, ProfileConfig, ChirpConfig, FrameConfig
    start/stop ar1.StartFrame / ar1.StopFrame = SensorStart / SensorStop

The DLL is 32-bit, so on Windows it lives in a helper process (radarlink_host.py under
Radar/.py32). On macOS and Linux, where it cannot be loaded, mmwavelink.MmwLink does the
DLL's work in Python over the DCA1000's FTDI (the same calls, the same bytes on the SPI),
and the radar's UART is found by its USB identity (ports.detect). mmWave Studio must be
closed - it holds the same FTDI device and COM port.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import struct
import time

from .bootloader import RadarUart
from .config import RadarConfig
from .radar import RadarController, RadarError
from .radarlink import DLL_DIR, PY32, RadarLinkClient, RadarLinkError, reset_into_sop
from .serialport import IS_WINDOWS, com_number, port_label

DEVICE = 1                      # device map of the single radar (g_RadarDeviceId)

# mmWaveLink async-event sub-block ids (msgId 0x80 RF, 0x280 device; x 32 + sub-block)
SB_MSS_POWERUP = 0x5000
SB_BSS_POWERUP = 0x5001
SB_RF_INIT_CALIB = 0x1004

CRC_TYPE = 0                    # RL_CRC_TYPE_16BIT_CCITT, as mmWave Studio powers on
SENSOR_ALREADY_STOPPED = 21     # SensorStop's answer once a finite recording's frames ended

# ar1.LvdsClkConfig(laneClk, dataRate) -> RadarLinkImpl_SetHsiClock: only the pair the
# lab uses has been read off mmWave Studio's logs (API:SetHsiClock,9)
HSI_CLOCK = {(1, 1): 0x09}


def _u32(v):
    return ("u32", int(v) & 0xFFFFFFFF)


def _u16(v):
    return ("u16", int(v) & 0xFFFF)


def _u8(v):
    return ("u8", int(v) & 0xFF)


def _char(v):
    return ("char", int(v) & 0xFF)


def _buf(b: bytes):
    return ("buf", b.hex())


# ---- the structs mmWave Studio passes by pointer (AR1xController's own definitions,
# field for field the mmWaveLink ones: rlChanCfg_t, rlAdcOutCfg_t, rlProfileCfg_t) ----

def channel_struct(cfg: RadarConfig) -> bytes:
    """TxRxChannelConfig: rx enable, tx enable, cascading 0, cascading pin-out 0."""
    ch = cfg.channel
    return struct.pack("<4H", ch.rx_mask, ch.tx_mask, 0, 0)


def adc_out_struct(cfg: RadarConfig) -> bytes:
    """ADCOutConfig: bits, full-scale reduction 0, format, two reserved half-words."""
    ch = cfg.channel
    return struct.pack("<BBHHH", ch.adc_bits, 0, ch.adc_fmt, 0, 0)


def profile_struct(cfg: RadarConfig) -> bytes:
    """
    ProfileConfiguration (44 bytes). The VCO-select and calibration-LUT bytes are the
    bits the lab's Lua scripts pass above the HPF2 code (xwr68: 0x20000 -> VCO byte 2),
    as mmWave Studio's own log of the command packs them back.
    """
    p = cfg.profile
    vco = (p.hpf2_lua_flags >> 16) & 0xFF
    lut = (p.hpf2_lua_flags >> 24) & 0xFF
    return struct.pack(
        "<HBBIIIIIIHHHHBBHHH",
        p.profile_id, vco, lut,
        p.start_freq_const, p.idle_time, p.adc_start_time, p.ramp_end_time,
        p.tx_out_power_backoff, p.tx_phase_shifter,
        p.freq_slope_const & 0xFFFF, p.tx_start_time & 0xFFFF,
        p.num_adc_samples, p.dig_out_sample_rate,
        p.hpf_corner_freq1, p.hpf_corner_freq2,
        0,                                     # tx calibration enables (GUI default off)
        p.rx_gain, 0)


def complex_adc(cfg: RadarConfig) -> bool:
    """mmWave Studio's rule: formats 1 and 2 are complex (0 real, 3 pseudo-real)."""
    return cfg.channel.adc_fmt in (1, 2)


def config_calls(cfg: RadarConfig) -> list[tuple[str, tuple, int | None]]:
    """
    (function, args, event to wait for) for the configuration after boot, in the order
    of the lab's scripts - what mmWave Studio calls for ChanNAdcConfig ... FrameConfig.
    """
    ch, lv, fr = cfg.channel, cfg.lvds, cfg.frame
    hsi = HSI_CLOCK.get((lv.lane_clk, lv.data_rate))
    if hsi is None:
        raise RadarError(f"LvdsClkConfig({lv.lane_clk}, {lv.data_rate}): the HSI clock for this "
                         f"pair is not known yet (only {sorted(HSI_CLOCK)})")
    calls = [
        ("RadarLinkImpl_ChannelConfig", (_u8(DEVICE), _buf(channel_struct(cfg))), None),
        ("RadarLinkImpl_AdcOutConfig", (_u8(DEVICE), _buf(adc_out_struct(cfg))), None),
        ("RadarLinkImpl_DataFmtConfig", (_u32(DEVICE), _u16(ch.rx_mask), _u16(ch.adc_bits),
                                         _u16(1 if complex_adc(cfg) else 0), _char(ch.iq_swap),
                                         _char(1)), None),
        ("RadarLinkImpl_LowPowerConfig", (_u32(DEVICE), _u16(ch.lp_adc_mode)), None),
        ("RadarLinkImpl_RfInit", (_u32(DEVICE),), SB_RF_INIT_CALIB),
        ("RadarLinkImpl_DataPathConfig", (_u32(DEVICE), _u8(lv.intf_sel), _u8(lv.transfer_fmt_pkt0),
                                          _u8(lv.transfer_fmt_pkt1), _u8(0), _u8(0), _u8(0),
                                          _u8(0), _u8(0)), None),
        ("RadarLinkImpl_LvdsClkConfig", (_u32(DEVICE), _char(lv.lane_clk), _char(lv.data_rate)), None),
        ("RadarLinkImpl_SetHsiClock", (_u32(DEVICE), _u16(hsi)), None),
        ("RadarLinkImpl_LaneConfig", (_u32(DEVICE), _u16(lv.lane_mask)), None),
        ("RadarLinkImpl_LvdsLaneConfig", (_u32(DEVICE), _u16(lv.lane_fmt_map), _u16(lv.lane_param)),
         None),
        ("RadarLinkImpl_ProfileConfig", (_u8(DEVICE), _buf(profile_struct(cfg))), None),
    ]
    for c in cfg.chirps:
        calls.append(("RadarLinkImpl_ChirpConfig",
                      (_u32(DEVICE), _u16(c.start_idx), _u16(c.end_idx), _u16(c.profile_id),
                       _u32(c.start_freq_var), ("i16", int(c.freq_slope_var)),
                       _u16(c.idle_time_var), _u16(c.adc_start_time_var), _u16(c.tx_mask)), None))
    samples = cfg.profile.num_adc_samples * (2 if complex_adc(cfg) else 1)
    calls += [
        # FrameConfig first switches the test source off (enable 0, mode 1)
        ("RadarLinkImpl_EnableTestSource", (_u32(DEVICE), _u16(0), _u16(1)), None),
        ("RadarLinkImpl_FrameConfig", (_u32(DEVICE), _u16(fr.chirp_start), _u16(fr.chirp_end),
                                       _u16(fr.num_frames), _u16(fr.num_loops),
                                       _u32(fr.periodicity), _u32(fr.trigger_delay),
                                       _u32(samples), _u8(0), _u16(fr.trigger_select), _u16(0)),
         None),
    ]
    return calls


class NativeRadar(RadarController):
    """The radar driven through RadarLinkDLL from a 32-bit helper - no mmWave Studio."""
    name = "native"

    def __init__(self, boot: bool = True, com_port: int | None = None, dll_dir: str = DLL_DIR,
                 python32: str = PY32, on_command=None, link=None, uart_factory=None):
        self.boot = boot
        self.com_port = com_port
        self.dll_dir, self.python32 = dll_dir, python32
        self.on_command = on_command or (lambda s: None)
        self.link = link
        self.real_uart = uart_factory is None
        self.uart_factory = uart_factory or (lambda port: RadarUart(port, log=self.on_command))
        self.calls: list[tuple[str, int]] = []
        self.events: list[dict] = []
        self.booted = False
        self.frames = 0                     # frame count of the configuration (0 = endless)
        self.configured = 0
        self.reconfigure_failures: list[str] = []

    def connect(self) -> None:
        if self.link is None:
            try:
                if IS_WINDOWS:
                    self.link = RadarLinkClient(self.dll_dir, self.python32)
                else:
                    from .mmwavelink import MmwLink
                    self.link = MmwLink()
            except RadarLinkError as e:
                raise RadarError(str(e)) from e

    def _uart_port(self, cfg: RadarConfig):
        """The radar's UART: the one given, or on macOS/Linux (where a COM number from a
        Lua script means nothing) the one found by its USB identity."""
        port = self.com_port or cfg.device.com_port
        if IS_WINDOWS or not self.real_uart or (port and com_number(port) is None):
            return port
        from .ports import detect
        det = detect()
        if det.radar is None:
            raise RadarError("the radar's UART was not found - " + "; ".join(det.notes))
        return det.radar.name

    def _call(self, fn: str, *args, event: int | None = None, event_timeout: float = 3.0,
              timeout: float = 30.0) -> dict | None:
        self.on_command(f"{fn}({', '.join(str(v) for _, v in args)})")
        try:
            ret = self.link.call(fn, *args, timeout=timeout)
        except RadarLinkError as e:
            raise RadarError(f"{fn}: {e} - mmWave Studio closed? radar powered?") from e
        self.calls.append((fn, ret))
        if ret != 0:
            raise RadarError(f"{fn} returned {ret}")
        if event is None:
            return None
        ev = self.link.wait_event(event, event_timeout)
        if ev is None:
            raise RadarError(f"{fn}: the radar did not report event 0x{event:04x} within "
                             f"{event_timeout:g} s")
        self.events.append(ev)
        self.on_command(f"  event 0x{event:04x}: {ev.get('payload', '')[:48]}")
        return ev

    def _boot(self, cfg: RadarConfig) -> None:
        port = self._uart_port(cfg)
        self.on_command("SOPControl(2) + reset")
        r = reset_into_sop(self.link, sop=2, device=DEVICE)
        bad = {k: v for k, v in r.items() if v != 0}
        if bad:
            raise RadarError(f"SOP 2 / reset through the DCA1000 failed: {bad}")
        uart = self.uart_factory(port)
        try:
            rc = uart.connect(cfg.device.baud_rate)
            if rc != 0:
                raise RadarError(f"no answer from the radar's bootloader on {port_label(port)} (code "
                                 f"{rc}) - is it the radar's own port and the board in SOP 2?")
            bss, mss = cfg.device.local_firmware_paths()
            self.on_command(f"BSS firmware {bss}")
            uart.download_bss(bss)
            self.on_command(f"MSS firmware {mss}")
            uart.download_mss(mss)
        finally:
            uart.close()
        # ar1.PowerOn(x, 1000, 0, 0): mmWave Studio passes the CRC type of its GUI (16-bit,
        # 0) whatever the script's first argument - its log reads "ar1.PowerOn(0, 1000, 0, 0)"
        # for the lab's PowerOn(1, ...) (RSTD Log0082); with 1 the device stops answering
        # after RfEnable (RL_RET_CODE_RESP_TIMEOUT). ACK timeout 1000, SPI, port 0.
        self._call("RadarLinkImpl_PowerOn", _u32(DEVICE), _char(CRC_TYPE),
                   _u32(1000), _char(0), _u32(0), ("callback", 0), event=SB_MSS_POWERUP,
                   event_timeout=4.0)
        self._call("RadarLinkImpl_RfEnable", _u32(DEVICE), event=SB_BSS_POWERUP, event_timeout=4.0)
        self.booted = True

    def configure(self, cfg: RadarConfig) -> None:
        """
        boot=True: reset, firmware and power-up every time (a fresh radar, ~20 s).
        boot=False: only the first time; later the configuration calls alone are sent to
        the running radar - a series of recordings. Should the running radar refuse them,
        it is booted once more and configured from scratch.
        """
        calls = config_calls(cfg)              # refuses what is not supported before touching the board
        self.connect()
        fresh = self.boot or not self.booted
        if fresh:
            self._boot(cfg)
        try:
            for fn, args, event in calls:
                self._call(fn, *args, event=event)
        except RadarError as e:
            if fresh:
                raise
            self.on_command(f"the running radar refused the new configuration ({e}); booting it again")
            self.reconfigure_failures.append(str(e))
            self._boot(cfg)
            for fn, args, event in calls:
                self._call(fn, *args, event=event)
        self.frames = int(cfg.frame.num_frames)
        self.configured += 1

    def start(self) -> None:
        self._call("RadarLinkImpl_SensorStart", _u32(DEVICE))

    def stop(self) -> None:
        """SensorStop. After the last frame of a finite recording the radar has stopped by
        itself and answers 21; mmWave Studio does not even send it then ("Frame Ended")."""
        try:
            self._call("RadarLinkImpl_SensorStop", _u32(DEVICE))
        except RadarError as e:
            if self.frames and str(e).endswith(f"returned {SENSOR_ALREADY_STOPPED}"):
                self.on_command("  (the frames had ended already)")
                return
            raise

    def close(self) -> None:
        if self.link is not None:
            self.link.close()
            self.link = None
        self.booted = False

    def info(self) -> dict:
        return {"controller": self.name, "boot": self.boot, "com_port": self.com_port,
                "calls": len(self.calls), "events": [e.get("sbid") for e in self.events],
                "configurations": self.configured,
                "reconfigure_failures": list(self.reconfigure_failures)}
