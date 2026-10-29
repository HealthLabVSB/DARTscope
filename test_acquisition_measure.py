"""
Tests for recording from DARTscope: finding the radar's COM port, the settings form,
series planning, the quick report on a take, a whole series against the DCA1000
simulator, the start gate, reconfiguring the running radar, and the Measure tab itself.

No hardware: the registry, the radar and the DCA1000 are replaced by recordings of them,
fakes and the simulator; the scenes are synthetic with a known answer.

Run with:  python -m pytest test_acquisition_measure.py -q
"""
import json
import os
import threading
import time

import numpy as np
import pytest

from acquisition import read_script_file
from acquisition.backend import SimulatorBackend
from acquisition.capture import CaptureRequest
from acquisition.config import ConfigError
from acquisition.params import MeasureParams, checks, derived, format_tx, parse_rx, parse_tx
from acquisition.ports import detect
from acquisition.quality import analyze_cube, folded, score
from acquisition.scene import Scene, Target, scene_cube
from acquisition.series import Series, parse_values, plan

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..", "config", "Dataset_motion.lua")

# the kleky_dataset1 configuration (3-TX TDM, 64 loops, 256 samples, 10 fps), Complex1X
KLEKY = """
ar1.FullReset()
ar1.SOPControl(2)
ar1.Connect(10, 921600, 1000)
ar1.frequencyBandSelection("60G")
ar1.SelectChipVersion("IWR6843")
ar1.deviceVariantSelection("IWR6843")
ar1.DownloadBSSFw("C:\\\\ti\\\\mmwave_studio_02_01_01_00\\\\rf_eval_firmware\\\\radarss\\\\xwr68xx_radarss.bin")
ar1.DownloadMSSFw("C:\\\\ti\\\\mmwave_studio_02_01_01_00\\\\rf_eval_firmware\\\\masterss\\\\xwr68xx_masterss.bin")
ar1.PowerOn(1, 1000, 0, 0)
ar1.RfEnable()
ar1.ChanNAdcConfig(1, 1, 1, 1, 1, 1, 1, 2, 1, 0)
ar1.LPModConfig(0, 0)
ar1.RfInit()
ar1.DataPathConfig(1, 1, 0)
ar1.LvdsClkConfig(1, 1)
ar1.LVDSLaneConfig(0, 1, 1, 0, 0, 1, 0, 0)
ar1.ProfileConfig(0, 60, 143, 7, 57.14, 0, 0, 0, 0, 0, 0, 69.994, 0, 256, 10000, 0, 131072, 30)
ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 1, 0, 0)
ar1.ChirpConfig(1, 1, 0, 0, 0, 0, 0, 0, 1, 0)
ar1.ChirpConfig(2, 2, 0, 0, 0, 0, 0, 0, 0, 1)
ar1.FrameConfig(0, 2, 100, 64, 100, 0, 0, 1)
"""


@pytest.fixture
def kleky():
    from acquisition import read_script
    return read_script(KLEKY, source="kleky").config


# --- the radar's COM port --------------------------------------------------------------------

class FakeRegistry:
    """The lab PC's registry on 2026-09-30 (IWR6843AOPEVM + DCA1000 + Bluetooth ports)."""

    def __init__(self, active=None, extra=()):
        self.active = active or ["COM4", "COM9", "COM11", "COM12", "COM13", "COM14", "COM15"]
        self.extra = list(extra)

    def serialcomm(self):
        return list(self.active)

    def instances(self):
        cp = "{efe44f12-0000}"
        ft = "{5f2f6806-0000}"
        rows = [
            {"bus": "USB", "hardware": "VID_10C4&PID_EA70&MI_00", "instance": "8&f8f822b&3&0000",
             "port": "COM15", "friendly": "Silicon Labs Dual CP210x USB to UART Bridge: Enhanced COM Port (COM15) ",
             "container": cp},
            {"bus": "USB", "hardware": "VID_10C4&PID_EA70&MI_01", "instance": "8&f8f822b&3&0001",
             "port": "COM9", "friendly": "Silicon Labs Dual CP210x USB to UART Bridge: Standard COM Port (COM9) ",
             "container": cp},
            {"bus": "BTHENUM", "hardware": "{00001101-0000-1000-8000-00805F9B34FB}_LOCALMFG&0000",
             "instance": "x", "port": "COM4", "friendly": "Standard Serial over Bluetooth link (COM4)"},
            # an unplugged board seen earlier on another socket: not in SERIALCOMM
            {"bus": "USB", "hardware": "VID_10C4&PID_EA70&MI_00", "instance": "old", "port": "COM6",
             "friendly": "Silicon Labs Dual CP210x USB to UART Bridge: Enhanced COM Port (COM6)"},
        ]
        rows += [{"bus": "FTDIBUS", "hardware": f"VID_0451+PID_FD03+FT80MJN4{c}", "instance": "0000",
                  "port": f"COM{11 + i}", "friendly": f"AR-DevPack-EVM-012 (COM{11 + i})", "container": ft}
                 for i, c in enumerate("ABCD")]
        return rows + self.extra


def test_the_radar_uart_is_found_by_usb_identity():
    det = detect(FakeRegistry())
    assert det.radar.name == "COM15" and det.radar.board == "IWR6843AOPEVM"
    assert det.radar.interface == 0 and det.radar.usb_id == "10C4:EA70 if0"
    assert [p.name for p in det.dca1000_ports] == ["COM11", "COM12", "COM13", "COM14"]
    assert det.ports[0].name == "COM15"                         # the radar first in the list
    assert "COM6" not in [p.name for p in det.ports]            # unplugged
    assert det.notes == []
    assert "Enhanced COM Port" in det.radar.description


def test_a_moved_cable_changes_nothing():
    reg = FakeRegistry(active=["COM6", "COM11", "COM12", "COM13", "COM14"])
    assert detect(reg).radar.name == "COM6"


def test_missing_devices_are_explained():
    det = detect(FakeRegistry(active=["COM4"]))
    assert det.radar is None and not det.dca1000_present
    assert any("no radar UART" in n for n in det.notes)
    assert any("DCA1000" in n for n in det.notes)


def test_two_radars_are_named():
    extra = [{"bus": "USB", "hardware": "VID_0451&PID_BEF3&MI_00", "instance": "i", "port": "COM20",
              "friendly": "XDS110 Class Application/User UART (COM20)"}]
    det = detect(FakeRegistry(active=["COM15", "COM20", "COM11"], extra=extra))
    assert [p.name for p in det.radar_candidates] == ["COM15", "COM20"]
    assert any("more than one radar" in n for n in det.notes)


# --- the settings form ---------------------------------------------------------------------------

def test_tx_and_rx_notation():
    assert parse_tx("1, 2, 3") == [1, 2, 4]
    assert parse_tx("2, 1+3") == [2, 5]
    assert parse_tx("TX1 TX3") == [1, 4]
    assert format_tx([2, 5]) == "2, 1+3"
    assert parse_rx("1234") == [1, 2, 3, 4] and parse_rx("RX1, RX3") == [1, 3]
    for bad in ("4", "", "0"):
        with pytest.raises(ConfigError):
            parse_tx(bad)
    with pytest.raises(ConfigError):
        parse_rx("5")


@pytest.mark.skipif(not os.path.isfile(LAB), reason="the lab's scripts are not next to the repository")
def test_a_lab_script_survives_the_form():
    cfg = read_script_file(LAB).config
    back = MeasureParams.from_config(cfg).to_config(cfg)
    assert (back.profile, back.chirps, back.frame, back.channel) == (
        cfg.profile, cfg.chirps, cfg.frame, cfg.channel)


def test_the_form_sets_what_it_shows(kleky):
    p = MeasureParams.from_config(kleky)
    assert (p.tx, p.rx, p.loops, p.samples) == ([1, 2, 4], [1, 2, 3, 4], 64, 256)
    p.tx, p.rx, p.loops, p.duration_s, p.period_ms = [1], [1, 2], 128, 5.0, 50.0
    cfg = p.to_config(kleky)
    assert cfg.tx_pattern == [1] and cfg.channel.rx_mask == 0b0011 and cfg.frame.num_loops == 128
    assert cfg.frame.num_frames == 100 and abs(cfg.frame.period_ms - 50.0) < 1e-6
    assert cfg.profile.hpf2_lua_flags == kleky.profile.hpf2_lua_flags        # kept from the script
    d = derived(cfg)
    assert abs(d["max_vel_ms"] - 6.10) < 0.02 and d["virtual_antennas"] == 2
    assert not [c for c in checks(cfg) if c[0] == "error"]


def test_auto_ramp_follows_the_sampling(kleky):
    p = MeasureParams.from_config(kleky)
    p.ramp_end_us, p.samples, p.fs_ksps, p.adc_start_us = 0.0, 512, 12500, 5.0
    assert abs(p.ramp_end_effective - (5.0 + 512 / 12.5 + 1.0)) < 0.011


def test_checks_name_what_cannot_work(kleky):
    p = MeasureParams.from_config(kleky)
    p.loops = 300
    errs = [c for c in checks(p.to_config(kleky)) if c[0] == "error"]
    assert any("loops" in c[1].lower() for c in errs)
    p.loops, p.ramp_end_us, p.samples = 64, 20.0, 256                       # ADC ends after the ramp
    assert any(c[0] == "error" and "ADC" in c[1] for c in checks(p.to_config(kleky)))


# --- series planning -------------------------------------------------------------------------------

def test_values_notation():
    assert parse_values("loops", "32, 64 128") == [32, 64, 128]
    assert parse_values("slope_mhz_us", "50..200/50") == [50.0, 100.0, 150.0, 200.0]
    assert parse_values("tx", "1 | 1, 2, 3 | 2, 1+3") == [[1], [1, 2, 4], [2, 5]]
    assert parse_values("rx", "1234 | 12") == [[1, 2, 3, 4], [1, 2]]
    with pytest.raises(ConfigError):
        parse_values("nonsense", "1")
    with pytest.raises(ConfigError):
        parse_values("loops", "abc")


def test_plan_combines_checks_and_orders(kleky):
    base = MeasureParams.from_config(kleky)
    variables = [("tx", parse_values("tx", "1 | 1,2,3")), ("loops", parse_values("loops", "64, 300"))]
    steps = plan(base, kleky, variables)
    assert [s.key for s in steps] == ["tx=1 loops=64", "tx=1 loops=300", "tx=1, 2, 3 loops=64",
                                      "tx=1, 2, 3 loops=300"]
    assert [s.ok for s in steps] == [True, False, True, False]
    rep = plan(base, kleky, variables[:1], repeats=2)
    assert [(s.key, s.take) for s in rep] == [("tx=1", 1), ("tx=1", 2), ("tx=1, 2, 3", 1), ("tx=1, 2, 3", 2)]
    shuf = plan(base, kleky, variables[:1], repeats=3, shuffle=True, seed=1)
    assert sorted((s.key, s.take) for s in shuf) == sorted(
        (k, t) for k in ("tx=1", "tx=1, 2, 3") for t in (1, 2, 3))
    assert [s.index for s in shuf] == list(range(1, 7))


def test_plan_lets_the_ramp_follow_the_samples(kleky):
    base = MeasureParams.from_config(kleky)                  # ramp end 57.14 us
    steps = plan(base, kleky, [("samples", [128, 384, 512])])
    assert [s.ok for s in steps] == [True, True, False]
    assert steps[0].params.ramp_end_effective == pytest.approx(7.0 + 12.8 + 1.0, abs=0.011)
    assert steps[1].params.ramp_end_effective == pytest.approx(7.0 + 38.4 + 1.0, abs=0.011)
    # 512 samples at 70 MHz/us sweep 4.14 GHz: beyond 60-64 GHz, refused with the reason
    assert any(p[0] == "error" and p[1] == "band" for p in steps[2].problems)


# --- the quick report ---------------------------------------------------------------------------------

def _cube(cfg, frames, scene):
    return (scene_cube(cfg, frames, scene) / 32767.0).astype(np.complex64)


def test_report_finds_a_slow_movement(kleky):
    kleky.frame.num_frames = 40
    q = analyze_cube(_cube(kleky, 40, Scene(targets=[Target(1.5, 0.2, 0.5)])), kleky)
    m = q.metrics
    assert m["motion_detected"] and 1.25 <= m["range_p05_m"] and m["range_p95_m"] <= 1.8
    assert m["motion_snr_db"] > 30
    assert 0.5 < m["v95_ms"] < 0.9                            # peak speed 0.63 m/s
    assert not folded(m)
    assert [r for r, _ in m["static_reflectors"][:2]] == pytest.approx([3.3, 0.8], abs=0.1)


def test_report_sees_velocity_folding(kleky):
    kleky.frame.num_frames = 40
    q = analyze_cube(_cube(kleky, 40, Scene(targets=[Target(1.5, 0.5, 1.0)])), kleky)   # 3.1 m/s
    assert folded(q.metrics)
    assert any(f[1] == "velocity folding" for f in q.findings)
    assert q.metrics["v_range_p80_ms"] > q.metrics["max_vel_ms"]        # the range track sees it


def test_a_few_loops_give_no_velocity_figures(kleky):
    """The vital-signs preset: 4 chirps per frame - a velocity bin is ~10 m/s wide."""
    p = MeasureParams.from_config(kleky)
    p.tx, p.loops, p.period_ms, p.duration_s = [1], 4, 10.0, 0.4
    cfg = p.to_config(kleky)
    q = analyze_cube(_cube(cfg, 40, Scene(targets=[Target(1.5, 0.01, 0.3)])), cfg)
    assert q.metrics["motion_detected"]
    assert q.metrics["v95_ms"] is None and "|v|" not in q.headline()
    assert any(c[0] == "info" and c[1] == "Doppler" for c in checks(cfg))


def test_presets_keep_their_length_unless_it_overfills_a_file():
    """
    Vital signs keep their 2 minutes - breathing is analysed in 30 s windows, and the 30 s
    every preset used to be cut to gave a single estimate. Only a preset that would pass
    1 GiB is shortened, to whole seconds.
    """
    import chirp_design as cd
    from acquisition.config import from_design
    from acquisition.params import ONE_FILE_BYTES, fit_duration

    def preset(purpose):
        return from_design(cd.design(cd.DEVICES["IWR6843AOP"], cd.goals_for(purpose)))

    assert fit_duration(120.0, preset("Vital signs (seated person)")) == (120.0, False)
    hr = preset("Heart rate of a moving person")
    s, cut = fit_duration(120.0, hr)
    rate = hr.bytes_per_frame / (hr.frame.period_ms * 1e-3)
    assert cut and s == int(s) and s * rate <= ONE_FILE_BYTES < (s + 1) * rate
    assert fit_duration(0.0, hr) == (0.0, False)                # 0: until stopped


def test_report_on_an_empty_scene(kleky):
    kleky.frame.num_frames = 20
    q = analyze_cube(_cube(kleky, 20, Scene(targets=[])), kleky)
    assert not q.metrics["motion_detected"] and q.metrics["target_range_m"] is None
    assert q.verdict == "info" and "no clear motion" in q.headline()


def test_score_terms():
    base = {"motion_snr_db": 30.0, "rate_mb_s": 10.0, "virtual_antennas": 12, "fps": 10.0}
    s_gen, terms = score(base, "general")
    assert s_gen == pytest.approx(29.5) and ("motion SNR", 30.0) in terms
    lost, _ = score(dict(base, zero_filled_packets=3), "general")
    assert lost < s_gen - 90
    fold, _ = score(dict(base, edge_fraction=0.2), "limbs")
    assert fold < s_gen - 10
    assert score(base, "angles")[0] > s_gen                 # 12 virtual antennas
    assert score(base, "vitals")[0] < s_gen                 # 10 fps is too few


# --- the start gate ---------------------------------------------------------------------------------

def test_cancel_at_the_gate_writes_nothing(tmp_path, kleky):
    kleky.frame.num_frames = 5
    gate, stop = threading.Event(), threading.Event()
    events = []

    def on_event(phase, info):
        events.append(phase)
        if phase == "ready":
            stop.set()                                     # the operator cancels
    res = SimulatorBackend().run(CaptureRequest(config=kleky, folder=str(tmp_path), name="t",
                                                start_gate=gate), on_event=on_event, stop_event=stop)
    assert res.status == "cancelled" and "ready" in events
    assert os.listdir(tmp_path) == []


def test_the_gate_holds_the_recording(tmp_path, kleky):
    kleky.frame.num_frames = 5
    gate = threading.Event()
    t_ready = {}

    def on_event(phase, info):
        if phase == "ready":
            t_ready["t"] = time.monotonic()
            threading.Timer(0.5, gate.set).start()
    res = SimulatorBackend().run(CaptureRequest(config=kleky, folder=str(tmp_path), name="t",
                                                start_gate=gate), on_event=on_event)
    assert res.status == "complete"
    started = res.sidecar["timing"]["record_start_utc"]
    assert started is not None and time.monotonic() - t_ready["t"] >= 0.5



def test_the_name_is_the_one_at_start(tmp_path, kleky):
    # prepared as "first"; the operator renames it and adds a subject while the radar waits
    kleky.frame.num_frames = 5
    gate = threading.Event()

    def at_start(req):
        req.name = "second"
        req.notes = dict(req.notes, subject="P2")

    def on_event(phase, info):
        if phase == "ready":
            threading.Timer(0.2, gate.set).start()
    res = SimulatorBackend().run(CaptureRequest(config=kleky, folder=str(tmp_path), name="first",
                                                notes={"movement": "sitting"}, start_gate=gate,
                                                at_start=at_start), on_event=on_event)
    assert res.status == "complete", res.error
    assert res.measurement.stem == "second"
    assert res.sidecar["stem"] == "second"
    assert res.sidecar["notes"]["subject"] == "P2" and res.sidecar["notes"]["movement"] == "sitting"
    assert os.path.isfile(tmp_path / "second_Raw_0.bin")
    assert not [f for f in os.listdir(tmp_path) if f.startswith("first")]

# --- a whole series against the simulator -----------------------------------------------------------------

def test_a_series_ranks_and_recommends(tmp_path, kleky):
    base = MeasureParams.from_config(kleky)
    base.duration_s = 2.0
    variables = [("tx", parse_values("tx", "1 | 1,2,3"))]
    steps = plan(base, kleky, variables)
    series = Series(str(tmp_path), "sweep", steps, objective="limbs", notes={"movement": "sway"},
                    base_params=base, variables=variables)
    backend = SimulatorBackend(Scene(targets=[Target(1.5, 0.5, 1.0)]))      # faster than 2 m/s
    from acquisition.quality import analyze_recording
    for s in steps:
        gate = threading.Event()
        gate.set()
        res = backend.run(series.request(s, start_gate=gate))
        assert res.status == "complete", res.error
        stem = os.path.join(res.measurement.folder, res.measurement.stem)
        series.add(s, res, analyze_recording(stem, stats=res.stats, verification=res.verification))
    paths = series.write()
    rank = series.ranking()
    assert rank[0]["key"] == "tx=1"                           # v_max 6.1 m/s: no folding
    rec = " ".join(series.recommendations())
    assert "velocity folding" in rec and "TX pattern: 1 best" in rec
    for p in paths.values():
        assert os.path.isfile(p)
    html = open(paths["html"], encoding="utf-8").read()
    assert "Ranking of the configurations" in html and "tx=1, 2, 3" in html
    csv_text = open(paths["csv"], encoding="utf-8").read().splitlines()
    assert csv_text[0].startswith("step,take,stem,status,score,tx") and len(csv_text) == 3
    assert os.path.isdir(os.path.join(str(tmp_path), "sweep"))
    series.rescore("general")
    assert series.objective == "general" and all(r.score is not None for r in series.results)


# --- the running radar takes a new configuration ---------------------------------------------------------

class _Link:
    def __init__(self, fail_once=None):
        self.called, self.fail_once = [], set(fail_once or ())
        self.pending = []

    def call(self, fn, *args, timeout=60):
        self.called.append(fn)
        ev = {"RadarLinkImpl_PowerOn": 0x5000, "RadarLinkImpl_RfEnable": 0x5001,
              "RadarLinkImpl_RfInit": 0x1004}.get(fn)
        if ev:
            self.pending.append({"sbid": ev})
        if fn in self.fail_once:
            self.fail_once.discard(fn)
            return -8
        return 21 if fn == "RadarLinkImpl_SensorStop" else 0

    def wait_event(self, sbids, timeout):
        want = {sbids} if isinstance(sbids, int) else set(sbids)
        for e in self.pending:
            if e["sbid"] in want:
                self.pending.remove(e)
                return e
        return None

    def close(self):
        pass


class _Uart:
    def connect(self, baud):
        return 0

    def download_bss(self, p):
        pass

    def download_mss(self, p):
        pass

    def close(self):
        pass


def test_a_series_boots_once_and_reconfigures(kleky):
    from acquisition.native import NativeRadar
    link = _Link()
    radar = NativeRadar(boot=False, link=link, uart_factory=lambda port: _Uart())
    radar.configure(kleky)
    radar.configure(kleky)
    assert link.called.count("RadarLinkImpl_PowerOn") == 1
    assert link.called.count("RadarLinkImpl_ProfileConfig") == 2
    radar.stop()                                             # 21 after the last frame: fine
    assert radar.info()["configurations"] == 2


def test_a_refused_reconfiguration_boots_again(kleky):
    from acquisition.native import NativeRadar
    link = _Link()
    radar = NativeRadar(boot=False, link=link, uart_factory=lambda port: _Uart())
    radar.configure(kleky)
    link.fail_once = {"RadarLinkImpl_ChannelConfig"}
    radar.configure(kleky)
    assert link.called.count("RadarLinkImpl_PowerOn") == 2
    assert radar.reconfigure_failures and "ChannelConfig" in radar.reconfigure_failures[0]


def test_sensor_stop_still_fails_for_an_endless_recording(kleky):
    from acquisition.native import NativeRadar
    from acquisition.radar import RadarError
    kleky.frame.num_frames = 0
    radar = NativeRadar(boot=False, link=_Link(), uart_factory=lambda port: _Uart())
    radar.configure(kleky)
    with pytest.raises(RadarError, match="SensorStop returned 21"):
        radar.stop()


# --- inside DARTscope.exe ---------------------------------------------------------------------------------------

def test_the_helper_is_found_inside_the_exe(tmp_path, monkeypatch):
    import sys
    from acquisition import radarlink
    bundle = tmp_path / "_MEI1234"
    (bundle / "py32").mkdir(parents=True)
    (bundle / "py32" / "python.exe").write_bytes(b"")
    (bundle / "acquisition").mkdir()
    (bundle / "acquisition" / "radarlink_host.py").write_text("# host")
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.delenv("DARTSCOPE_PY32", raising=False)
    assert radarlink.find_python32() == str(bundle / "py32" / "python.exe")
    assert radarlink.find_host() == str(bundle / "acquisition" / "radarlink_host.py")
    other = tmp_path / "py32b"
    other.mkdir()
    (other / "python.exe").write_bytes(b"")
    monkeypatch.setenv("DARTSCOPE_PY32", str(other))                # the variable wins
    assert radarlink.find_python32() == str(other / "python.exe")


def test_a_pc_without_mmwave_studio_is_told_why(tmp_path):
    from acquisition.radarlink import native_status
    probs = native_status(dll_dir=str(tmp_path / "nothing"), firmware=[str(tmp_path / "bss.bin")],
                          windows=True)
    assert any("mmWave Studio 2.1.1 is not installed" in p for p in probs)
    assert any("firmware missing" in p for p in probs)


# --- the Measure tab ------------------------------------------------------------------------------------------

def test_the_measure_tab_records_a_take(tmp_path, monkeypatch, tk_root, tk_pump):
    # In a window of the session's one Tk (conftest.py): a second Tk root in the same
    # process hangs on macOS.
    # The take is renamed while the radar waits ("prepare the next take automatically"
    # prepares before the operator types the next name): the fields at Start name it.
    import tkinter as tk
    root = tk.Toplevel(tk_root)
    root.withdraw()
    import measure_gui
    from pathlib import Path
    monkeypatch.setattr(measure_gui, "_settings_path", lambda: Path(tmp_path) / "settings.json")
    opened = []
    try:
        tab = measure_gui.MeasureTab(root, open_recording=opened.append)
        tab.mode_var.set("simulator")
        tab._on_mode()
        tab.folder_var.set(str(tmp_path))
        tab.name_var.set("first")
        tab.vars["duration_s"].set("1")
        tab.countdown_var.set(0)
        tab.beep_var.set(False)
        tab.auto_next_var.set(False)
        tk_pump(root)
        assert tab._evaluate() is not None
        tab._prepare_single()
        end = time.time() + 60
        while tab.state != measure_gui.READY and time.time() < end:
            tk_pump(root)
            time.sleep(0.02)
        assert tab.state == measure_gui.READY
        tab.name_var.set("second")
        tab.subject_var.set("P2")
        tab._start()
        while (tab.state != measure_gui.IDLE or tab.last_stem is None) and time.time() < end:
            tk_pump(root)
            time.sleep(0.02)
        assert tab.last_stem is not None
        assert os.path.basename(tab.last_stem) == "second"
        assert not [f for f in os.listdir(tmp_path) if f.startswith("first")]
        with open(tab.last_stem + "_acquisition.json", encoding="utf-8") as f:
            assert json.load(f)["notes"]["subject"] == "P2"
        assert "complete" in tab.result_text.get("1.0", "end")
        tab._open_last()
        assert opened and opened[0].endswith("_Raw_0.bin") and os.path.isfile(opened[0])
        assert os.path.isfile(tab.last_stem + "_quality.json")
        tab.close()
    finally:
        root.destroy()

