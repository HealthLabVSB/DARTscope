"""
Tests for the cascade radar (cascade/, cascade_gui.py) that need no hardware: the settings
and their checks, the gUserCfg words, the firmware's symbol, the error decoding, reading a
take split into parts, and the tab.

Run with:  python -m pytest test_cascade.py -q
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade import hardware
from cascade.data import CascadeTake
from cascade.params import (CFG_WORDS, MAX_RATE, PRESETS, CascadeParams, packet_delay_us,
                            parse_tx_chirps)


# --- settings ----------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_can_be_recorded(name):
    rep = CascadeParams.preset(name).check()
    assert rep["errors"] == []
    assert len(CascadeParams.preset(name).words()) == CFG_WORDS


def test_the_sdk_profile_gives_the_sdk_words():
    """The preset reproduces the profile the SDK test hardcodes (common.c, frame mode)."""
    w = CascadeParams.preset("SDK test profile (what the repo ran)").words()
    assert w[:18] == [0xCA5CAD01, 2, 1435384036, 229, 700, 700, 2083, 0, 128, 20000, 42, 0, 0, 0,
                      1, 128, 6_000_000, 1000]
    assert w[18] == 7 and w[30] == 7 and w[19:30] == [0] * 11


def test_derived_values():
    rep = CascadeParams().check()
    assert rep["range_res_m"] == pytest.approx(0.0586, abs=2e-4)     # 100 MHz/us x 25.6 us
    assert rep["bytes_per_frame"] == 32 * 8 * 128 * 4
    assert rep["chirps_per_frame"] == 32
    assert CascadeParams() == CascadeParams.preset("Vital signs - 1 person, 100 fps")
    sdk = CascadeParams.preset("SDK test profile (what the repo ran)").check()
    assert sdk["range_res_m"] == pytest.approx(2.118, abs=0.01)          # what the old data have


def test_tdm_words_and_virtual_channels():
    p = CascadeParams.preset("Vital signs - multi-person (TDM 48 ch)")
    w = p.words()
    assert w[14] == 6
    assert w[18:24] == [1, 2, 4, 0, 0, 0] and w[30:36] == [0, 0, 0, 1, 2, 4]
    assert p.check()["derived"]["Virtual channels"].startswith("48")


@pytest.mark.parametrize("change, fragment", [
    (dict(ramp_end_us=20.0), "Sampling ends"),
    (dict(slope_mhz_us=200.0), "outside 76-81 GHz"),
    (dict(num_samples=1000), "Samples per chirp"),
    (dict(num_loops=255, frame_period_ms=5.0), "longer than the frame period"),
    (dict(num_loops=200, frame_period_ms=12.0), "more than the DCA1000"),
    (dict(tx_chirps="8/0"), "TX mask"),
    (dict(rx_gain_db=31), "RX gain"),
    (dict(duration_s=100000), "Frames"),
])
def test_what_cannot_be_recorded_is_refused(change, fragment):
    p = CascadeParams(**{**CascadeParams().to_dict(), **change})
    errors = p.check()["errors"]
    assert any(fragment in e for e in errors), errors


def test_packet_gap_follows_the_data_rate():
    assert packet_delay_us(0) == 50
    assert packet_delay_us(13e6) == 74
    assert packet_delay_us(MAX_RATE) == 36
    assert packet_delay_us(1e6) == 100


def test_tx_chirps_text():
    assert parse_tx_chirps("7/7") == [(7, 7)]
    assert parse_tx_chirps("1/0; 2/0, 0/4") == [(1, 0), (2, 0), (0, 4)]


# --- hardware helpers ----------------------------------------------------------------------

def test_the_firmware_has_gusercfg():
    addr = hardware.gusercfg_address()
    assert 0x10200000 <= addr < 0x10300000              # MSS L2


def test_mmwave_error_codes_are_decoded():
    # what the radar answered to a TX start time of 300 us (2026-10-09)
    txt = hardware.describe_status({"state": 4, "errStage": 6, "errCode": -203685698, "framesDone": 0})
    assert "MMWave_config" in txt and "EPROFILECFG" in txt and "front-end error 47" in txt
    assert hardware.describe_status({"state": 3, "errStage": 0, "errCode": 0, "framesDone": 500}) == \
        "done, 500 frames"


# --- reading a take ------------------------------------------------------------------------

def _write_take(folder, prefix, params, frames, part_bytes):
    rep = params.check()
    n = params.num_samples
    rng = np.random.default_rng(1)
    iq = rng.integers(-2000, 2000, size=(frames, rep["chirps_per_frame"], 8, n, 2), dtype=np.int16)
    stream = iq.astype("<i2").tobytes()
    for i in range(0, len(stream), part_bytes):
        (folder / f"{prefix}_Raw_{i // part_bytes}.bin").write_bytes(stream[i:i + part_bytes])
    meta = {"layout": {"num_samples": n, "num_rx": 8, "chirps_per_frame": rep["chirps_per_frame"],
                       "chirps_in_loop": len(params.chirps), "num_loops": params.num_loops},
            "radar": {"range_res_m": rep["range_res_m"], "frame_rate_hz": rep["frame_rate_hz"],
                      "wavelength_m": rep["wavelength_m"], "chirp_time_us": rep["chirp_time_us"]}}
    (folder / f"{prefix}_cascade.json").write_text(json.dumps(meta))
    return iq[..., 0] + 1j * iq[..., 1].astype(np.float32)


def test_a_take_split_into_parts_reads_as_one_stream(tmp_path):
    p = CascadeParams(num_samples=64, num_loops=4, tx_chirps="1/0; 2/0")
    ref = _write_take(tmp_path, "t", p, frames=7, part_bytes=10_000)   # parts cut mid-chirp
    take = CascadeTake.open(tmp_path / "t_Raw_0.bin")
    assert take.num_frames == 7 and len(take.parts) > 3
    assert np.array_equal(take.frames(2, 3), ref[2:5])
    assert np.array_equal(take.first_chirps(), ref[:, 0])
    assert np.array_equal(take.first_chirps(1), ref[:, 1])
    assert CascadeTake.is_cascade(tmp_path / "t_Raw_1.bin")


def test_a_single_chip_recording_is_not_a_cascade_take(tmp_path):
    (tmp_path / "x_Raw_0.bin").write_bytes(b"\0" * 64)
    assert not CascadeTake.is_cascade(tmp_path / "x_Raw_0.bin")
    with pytest.raises(FileNotFoundError):
        CascadeTake.open(tmp_path / "x_Raw_0.bin")


# --- the tab -------------------------------------------------------------------------------

def test_the_tab_without_hardware(tk_root, tk_pump, tmp_path, monkeypatch):
    import tkinter as tk
    import cascade_gui
    monkeypatch.setattr(cascade_gui, "_settings_path", lambda: tmp_path / "cascade.json")
    monkeypatch.setattr(cascade_gui, "check_dca1000", lambda: None)
    monkeypatch.setattr(cascade_gui.hardware, "xds110_present", lambda: False)
    top = tk.Toplevel(tk_root)
    try:
        tab = cascade_gui.CascadeTab(top)
        end = time.time() + 3
        while time.time() < end and tab.dca_lbl.cget("text") == "DCA1000":
            tk_pump(top)
            time.sleep(0.02)
        assert "no answer" in tab.dca_lbl.cget("text")
        rep = tab.update_checks()
        assert rep["errors"] == [] and str(tab.start_btn.cget("state")) == "normal"
        assert tab.params() == CascadeParams()
        tab.preset_var.set("Vital signs - multi-person (TDM 48 ch)")
        tab._set_params(CascadeParams.preset("Vital signs - multi-person (TDM 48 ch)"))
        assert tab.txmode_var.get().startswith("TDM-MIMO 6 TX")
        tab.vars["ramp_end_us"].set("10")
        rep = tab.update_checks()
        assert rep["errors"] and str(tab.start_btn.cget("state")) == "disabled"
        # the preview opens a take
        _write_take(tmp_path, "v", CascadeParams(num_samples=64, num_loops=8, duration_s=1), 40, 1 << 20)
        tab.viewer.open(tmp_path / "v_cascade.json")
        assert tab.viewer.take.num_frames == 40
    finally:
        top.destroy()


def test_the_single_chip_tab_releases_the_dca1000_only_when_idle():
    import measure_gui

    class Fake:
        state = measure_gui.IDLE
        closed = False

        def _close_backend(self):
            self.closed = True

        def _update_radar_state(self):
            pass
    f = Fake()
    assert measure_gui.MeasureTab.release_hardware(f) is None and f.closed
    f = Fake()
    f.state = measure_gui.RECORDING
    assert "Stop / cancel" in measure_gui.MeasureTab.release_hardware(f) and not f.closed


# --- a cascade take in the rest of DARTscope (cascade/decode.py) ---------------------------

def _write_full_take(folder, prefix, params, frames, part_bytes):
    """A take as CascadeRecording leaves it: parts, the sidecar (no recorder CSV)."""
    from cascade.recording import CascadeRecording
    ref = _write_take(folder, prefix, params, frames, part_bytes)
    rep = params.check()
    meta = json.loads((folder / f"{prefix}_cascade.json").read_text())
    meta.update({"params": params.to_dict(), "started": "2026-10-09 21:44:24", "notes": "",
                 "record": {"frames_in_file": frames, "zero_filled_packets": 0}})
    meta["radar"]["num_frames"] = params.num_frames
    meta["layout"]["tx_chirps"] = params.chirps
    (folder / f"{prefix}_cascade.json").write_text(json.dumps(meta))
    assert CascadeRecording  # the sidecar's keys are the ones CascadeRecording writes
    return ref


def test_a_cascade_take_decodes_like_a_log(tmp_path):
    from cascade.decode import cascade_params
    from radar_processing import derive_capture_dimensions, load_raw_cube
    p = CascadeParams.preset("Vital signs - multi-person (TDM 48 ch)")
    p.num_samples, p.num_loops, p.duration_s = 64, 4, 0.2
    ref = _write_full_take(tmp_path, "c", p, frames=6, part_bytes=50_000)
    params, summary = cascade_params(tmp_path / "c_Raw_0.bin")
    assert params["cascade"] and params["numLanes"] == 4 and params["numAdcSamples"] == 64
    assert params["tx_order_masks"] == [1, 2, 4, 8, 16, 32]
    assert params["tx_order_names"] == ["TX1", "TX2", "TX3", "TX4", "TX5", "TX6"]
    assert params["freqSlopeMHz_us"] == pytest.approx(p.check()["slope_mhz_us"])
    assert params["maxVelocity_m_s"] == pytest.approx(
        params["wavelength_mm"] / 1000 / (4 * 6 * 40e-6), rel=1e-6)
    dims = derive_capture_dimensions(params)
    assert (dims["num_rx"], dims["chirps_per_frame"], dims["num_samples"]) == (8, 24, 64)
    cube = load_raw_cube(str(tmp_path / "c_Raw_0.bin"), params, dims)
    assert cube.shape == (6, 24, 8, 64)
    assert np.allclose(cube, ref / 32767.0)
    # the TX selector keeps one slot of every loop
    sel = load_raw_cube(str(tmp_path / "c_Raw_0.bin"), dict(params), dict(dims),
                        chirp_selection={"positions": [3], "pattern_len": 6})
    assert np.allclose(sel, ref[:, 3::6] / 32767.0)


def test_four_rx_masks_still_count_four():
    from radar_processing import derive_capture_dimensions
    assert derive_capture_dimensions({"dataFmt_rxChanEnMask": 15, "numAdcSamples": 128})["num_rx"] == 4
    assert derive_capture_dimensions({"dataFmt_rxChanEnMask": 255, "numAdcSamples": 128})["num_rx"] == 8


def test_the_angle_views_refuse_the_cascade_without_its_layout():
    import AOA
    with pytest.raises(ValueError, match="needs its antenna configuration selected"):
        AOA._get_antenna_geometry({"cascade": True, "chip_hint": "AWR2243 cascade"}, 8, 8)


def test_the_cascade_tx_run_the_other_way_within_each_device():
    # measured with a corner reflector on 8 floor marks (2026-10-11): masks 4, 2, 1 at
    # 0, 1 (raised), 2 wavelengths and 32, 16, 8 at 3.5, 5.5, 7.5 - the user guide's
    # positions, its labels in the other order
    import antenna_config as ac
    cfg = ac.get_config(ac.CASCADE_CONFIG)
    at = {m: tuple(o) for m, o in zip(cfg["tx_masks"], cfg["tx_offsets"])}
    assert at == {4: (0.0, 0.0), 2: (1.0, 0.8), 1: (2.0, 0.0), 32: (3.5, 0.0), 16: (5.5, 0.0), 8: (7.5, 0.0)}
    x = sorted({o[0] + r[0] for o in cfg["tx_offsets"] if o[1] == 0 for r in cfg["rx_positions"]})
    assert x == list(np.arange(0, 18.51, 0.5))         # the 18.5-wavelength row has no gap


def test_the_command_line_tools_find_cascade_takes(tmp_path):
    from tools.captures import find_captures, load_capture
    p = CascadeParams(num_samples=64, num_loops=4, duration_s=0.2)
    _write_full_take(tmp_path, "k", p, frames=5, part_bytes=1 << 20)
    caps = find_captures(str(tmp_path))
    assert [c.stem for c in caps] == ["k"] and caps[0].params["cascade"]
    # what the tools derive from a _Raw_0.bin: <stem>_LogFile.txt, which does not exist
    cap = load_capture(str(tmp_path / "k_LogFile.txt"))
    assert cap is not None and cap.log_path.endswith("k_cascade.json") and len(cap.bin_paths) == 1


def test_the_hr_tab_offers_every_rx():
    import hr_gui
    assert hr_gui.HRDetectionTab._rx_choices(8)[:8] == [f"RX{i}" for i in range(1, 9)]
    assert hr_gui.HRDetectionTab._rx_choices(4) == ["RX1", "RX2", "RX3", "RX4", "Fused", "PCA", "ICA"]


def test_few_doppler_bins_are_drawn_bin_by_bin():
    from plot_interactivity import few_bins_interpolation
    assert few_bins_interpolation(32) == "nearest"          # the cascade's vital-signs preset
    assert few_bins_interpolation(255) is None               # the lab's single-chip captures


def test_a_chirp_must_outlast_its_lvds_transfer_with_margin():
    """512 samples at 20 Msps stalled after one chirp with a 63 us chirp and ran with 72 us
    (2026-10-09); 448 at 20 Msps with 55 us ran."""
    base = dict(start_freq_ghz=76.0, slope_mhz_us=150.0, num_samples=512, sample_rate_ksps=20000,
                ramp_end_us=33.0, num_loops=8, frame_period_ms=20.0, tx_chirps="1/0")
    stalled = CascadeParams(**{**CascadeParams().to_dict(), **base, "idle_us": 30.0}).check()["errors"]
    assert any("LVDS transfer" in e for e in stalled)
    ran = CascadeParams(**{**CascadeParams().to_dict(), **base, "idle_us": 60.0}).check()["errors"]
    assert not any("LVDS transfer" in e for e in ran)


def test_the_cascade_antenna_configuration_reproduces_ti_figure_2_4():
    import antenna_config as a
    cfg = a.get_config(a.CASCADE_CONFIG)
    assert a.validate(cfg)[0]
    x, y = a.build_geometry(cfg, masks=[1, 2, 4, 8, 16, 32], num_virt_rx=48, num_phys_rx=8)
    low = np.isclose(y, y.min())
    assert len(x) == 48 and low.sum() == 40
    assert len(set(np.round(x[low], 3))) == 38                 # 38 positions, two doubled
    assert x.max() - x.min() == pytest.approx(18.5)              # the 18.5 lambda aperture
    assert sorted(set(np.round(y - y.min(), 3))) == [0.0, 0.8]   # TX3 raised 0.8 lambda


# --- TI's debugger, the XDS110 and a take, on Windows and macOS ---------------------------

def _fake_ti(top, uniflash=(), ccs=()):
    """UniFlash and CCS installations as their scripting debuggers are laid out."""
    for name in uniflash:
        for osd in ("osx", "win", "linux"):
            (top / name / "deskdb/content/TICloudAgent" / osd / "scripting").mkdir(parents=True)
    for name, scripting in ccs:
        root = top / name / "ccs"
        (root / "ccs_base/cloudagent/src").mkdir(parents=True)
        for node in ("node", "node.exe"):
            (root / "ccs_base/cloudagent" / node).write_bytes(b"")
        if scripting:                          # CCS 20+ (Theia); older (Eclipse) ones have none
            (root / "scripting").mkdir()
            (root / "scripting/launcher.mjs").write_text("")


def test_the_scripting_debugger_comes_from_uniflash_or_ccs(tmp_path, monkeypatch):
    monkeypatch.delenv(hardware.TI_ROOT_ENV, raising=False)
    monkeypatch.setattr(hardware, "UNIFLASH_GLOBS", (str(tmp_path / "uniflash_*"),))
    monkeypatch.setattr(hardware, "CCS_GLOBS", (str(tmp_path / "ccs*" / "ccs"),))
    assert hardware.scripting("darwin") is None and hardware.scripting("win32") is None
    _fake_ti(tmp_path, ccs=[("ccs1281", False), ("ccs2040", True)])
    for plat, node in (("darwin", "node"), ("win32", "node.exe")):
        s = hardware.scripting(plat)
        assert (s.kind, s.name, s.node.name) == ("CCS", "ccs2040", node)
        assert s.ccs_root() == tmp_path / "ccs2040" / "ccs"      # complete: nothing to link
        assert s.launcher == tmp_path / "ccs2040/ccs/scripting/launcher.mjs"
    _fake_ti(tmp_path, uniflash=["uniflash_8.9.1", "uniflash_8.10.0"])
    s = hardware.scripting("darwin")                             # the Mac keeps UniFlash first
    assert (s.kind, s.name) == ("UniFlash", "uniflash_8.10.0")   # 8.10 is newer than 8.9
    assert s.launcher == tmp_path / "uniflash_8.10.0/deskdb/content/TICloudAgent/osx/scripting/launcher.mjs"
    assert hardware.scripting("win32").kind == "CCS"             # Windows: CCS, no links needed
    monkeypatch.setenv(hardware.TI_ROOT_ENV, str(tmp_path / "uniflash_8.9.1"))
    assert hardware.scripting("win32").name == "uniflash_8.9.1"
    monkeypatch.setenv(hardware.TI_ROOT_ENV, str(tmp_path / "ccs2040"))   # the ccs inside it
    assert hardware.scripting("darwin").kind == "CCS"


def _fake_uniflash_with_files(top):
    root = top / "uniflash_8.7.0"
    base = root / "deskdb/content/TICloudAgent" / hardware._os_dir() / "ccs_base"
    (base / "DebugServer").mkdir(parents=True)
    (base / "DebugServer" / "x.txt").write_text("debug server")
    (base / "version.txt").write_text("8.7.0")
    host = root / "TICloudAgentHostApp"
    (host / "src").mkdir(parents=True)
    (host / "src" / "main.js").write_text("// agent")
    hardware._node(root).write_bytes(b"node")
    return root


def _check_ccs_root(base, root):
    assert (base / "ccs_base/DebugServer/x.txt").read_text() == "debug server"
    assert (base / "ccs_base/version.txt").read_text() == "8.7.0"
    assert (base / "ccs_base/cloudagent/src/main.js").read_text() == "// agent"
    assert (base / "ccs_base/cloudagent" / hardware._node(root).name).read_bytes() == b"node"


def test_uniflash_gets_its_cloudagent_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("DARTSCOPE_CACHE", str(tmp_path / "cache"))
    root = _fake_uniflash_with_files(tmp_path)
    base = hardware.ccs_root(root)
    _check_ccs_root(base, root)
    assert hardware.ccs_root(root) == base                       # made once


@pytest.mark.skipif(not hardware._win(), reason="junctions are Windows'")
def test_uniflash_on_windows_without_developer_mode(tmp_path, monkeypatch):
    """No right to make symbolic links: junctions for the folders, a hard link for node.exe."""
    def no_symlinks(*a, **k):
        raise OSError(1314, "A required privilege is not held by the client")
    monkeypatch.setattr(hardware.os, "symlink", no_symlinks)
    monkeypatch.setenv("DARTSCOPE_CACHE", str(tmp_path / "cache"))
    root = _fake_uniflash_with_files(tmp_path)
    base = hardware.ccs_root(root)
    _check_ccs_root(base, root)
    assert hardware.ccs_root(root) == base


def _fake_node(tmp_path, body):
    """A 'node' that is this Python running a launcher script: jtag()'s process plumbing."""
    launcher = tmp_path / "launcher.py"
    launcher.write_text("import json, os, sys\n" + body, encoding="utf-8")
    return hardware.Scripting("CCS", tmp_path, Path(sys.executable), launcher)


def test_jtag_runs_the_debugger_and_reads_its_result(tmp_path, monkeypatch):
    monkeypatch.setattr(hardware, "scripting", lambda: None)
    assert hardware.jtag("probe") == {"ok": False, "error": hardware.NOT_INSTALLED}
    # argv: launcher, jtag_cfg.js, mode, args; the answer in UTF-8 whatever the locale
    ti = _fake_node(tmp_path, (
        "res = {'ok': True, 'argv': sys.argv[1:], 'root': os.environ['CASCADE_CCS_ROOT'], 'log': 'µs ≤ Ř'}\n"
        "sys.stdout.buffer.write(b'info: AM273x\\n')\n"
        "sys.stdout.buffer.write(('RESULT ' + json.dumps(res, ensure_ascii=False) + '\\n').encode('utf-8'))\n"))
    monkeypatch.setattr(hardware, "scripting", lambda: ti)
    res = hardware.jtag("stop", "0x10200000")
    assert res == {"ok": True, "argv": [str(hardware.JTAG_SCRIPT), "stop", "0x10200000"],
                   "root": str(tmp_path), "log": "µs ≤ Ř"}
    # no RESULT line: the last line it said
    ti = _fake_node(tmp_path, "print('Error: cannot connect to the XDS110', file=sys.stderr); sys.exit(1)\n")
    monkeypatch.setattr(hardware, "scripting", lambda: ti)
    assert hardware.jtag("probe") == {"ok": False, "error": "Error: cannot connect to the XDS110"}
    ti = _fake_node(tmp_path, "import time; time.sleep(30)\n")
    monkeypatch.setattr(hardware, "scripting", lambda: ti)
    t0 = time.time()
    res = hardware.jtag("probe", timeout=1)
    assert not res["ok"] and "did not answer" in res["error"] and time.time() - t0 < 10


def test_the_xds110_is_found_on_windows_and_macos():
    from test_acquisition_measure import FakeRegistry
    xds = [{"bus": "USB", "hardware": f"VID_0451&PID_BEF3&MI_0{i}", "instance": f"7&2c3&0&000{i}",
            "port": port, "friendly": f"XDS110 Class {what} ({port})"}
           for i, port, what in ((0, "COM16", "Application/User UART"), (3, "COM18", "Auxiliary Data Port"))]
    assert not hardware.xds110_present(FakeRegistry())
    assert hardware.xds110_present(FakeRegistry(active=["COM4", "COM16", "COM18"], extra=xds))
    assert not hardware.xds110_present(FakeRegistry(extra=xds))          # unplugged: not in SERIALCOMM

    class Mac:
        def __init__(self, xds):
            self.xds = xds

        def comports(self):
            c = [SimpleNamespace(device="/dev/cu.Bluetooth-Incoming-Port", vid=None, pid=None,
                                 serial_number=None, location=None, description="")]
            if self.xds:
                c += [SimpleNamespace(device=f"/dev/cu.usbmodemR00910021{i}", vid=0x0451, pid=0xBEF3,
                                      serial_number="R0091002", location="0-1.1", description="XDS110")
                      for i in (1, 4)]
            return c
    assert hardware.xds110_present(Mac(True)) and not hardware.xds110_present(Mac(False))


def test_a_take_through_the_simulated_dca1000(tmp_path, monkeypatch):
    """The whole take - DCA1000, firmware start, record, firmware stop, _cascade.json - with
    the DCA1000 simulated on localhost and the JTAG answers faked. The sidecar is UTF-8 on
    every system: its derived values say '≤' and a note may be Czech."""
    from acquisition.config import Dca1000Config
    from acquisition.dca1000 import Dca1000Controller
    from acquisition.receiver import UdpReceiver
    from acquisition.simulator import Dca1000Simulator, free_udp_port
    from cascade import recording

    p = CascadeParams(num_samples=64, num_loops=4, tx_chirps="1/0", duration_s=0.5)
    rep = p.check()
    rng = np.random.default_rng(2)
    source = rng.integers(-2000, 2000, rep["bytes_total"] // 2, dtype=np.int16).astype("<i2").tobytes()
    host_cfg, host_data = free_udp_port(), free_udp_port()
    sim = Dca1000Simulator(source, host_data_addr=("127.0.0.1", host_data))
    monkeypatch.setattr(recording, "dca1000_config", lambda delay: Dca1000Config(
        host_ip="127.0.0.1", config_port=host_cfg, data_port=host_data, lvds_mode=1, packet_delay_us=delay))
    monkeypatch.setattr(recording, "Dca1000Controller", lambda cfg, timeout_s: Dca1000Controller(
        cfg, bind_ip="127.0.0.1", dca_addr=sim.addr, timeout_s=timeout_s))
    monkeypatch.setattr(recording, "UdpReceiver", lambda rec: UdpReceiver(rec, port=host_data, bind_ip="127.0.0.1"))
    calls = []

    def jtag(mode, *args, **kw):
        calls.append((mode, args))
        if mode == "start":
            assert json.loads(Path(args[1]).read_text())["words"] == p.words()
            sim.start_stream()                 # what the firmware's frames do
            return {"ok": True}
        return {"ok": True, "status": {"state": 3, "errStage": 0, "errCode": 0,
                                       "framesDone": p.num_frames, "heartbeat": 7}}
    monkeypatch.setattr(hardware, "jtag", jtag)
    try:
        meta = recording.CascadeRecording(p, tmp_path, "take", notes="Řízené dýchání, ň").run()
    finally:
        sim.close()
    assert [c[0] for c in calls] == ["start", "stop"]
    assert calls[1][1] == (hex(hardware.gusercfg_address()),)
    assert meta["record"]["complete"] and meta["record"]["frames_in_file"] == p.num_frames
    side = json.loads((tmp_path / "take_cascade.json").read_bytes().decode("utf-8"))
    assert side["notes"] == "Řízené dýchání, ň" and "≤" in side["derived"]["DCA1000 packet gap"]
    take = CascadeTake.open(tmp_path / "take_Raw_0.bin")
    assert take.num_frames == p.num_frames
    ref = np.frombuffer(source, "<i2").reshape(p.num_frames, rep["chirps_per_frame"], 8, 64, 2)
    assert np.array_equal(take.frames(0, p.num_frames), ref[..., 0] + 1j * ref[..., 1].astype(np.float32))


def test_the_single_chip_tab_takes_its_own_board_before_the_cascades_xds110():
    """Both boards plugged in: the IWR6843AOPEVM's CP2105 is the radar UART, whatever the
    COM numbers - the cascade's XDS110 got COM16 here, a CP2105 can get COM20."""
    from acquisition.ports import detect
    from test_acquisition_measure import FakeRegistry
    xds = [{"bus": "USB", "hardware": "VID_0451&PID_BEF3&MI_00", "instance": "7&2c3&0&0000",
            "port": "COM3", "friendly": "XDS110 Class Application/User UART (COM3)"}]
    det = detect(FakeRegistry(active=["COM3", "COM9", "COM11", "COM12", "COM13", "COM14", "COM15"], extra=xds))
    assert [p.name for p in det.radar_candidates] == ["COM15", "COM3"]
    assert det.radar.name == "COM15" and "cascade" in det.radar_candidates[1].board
