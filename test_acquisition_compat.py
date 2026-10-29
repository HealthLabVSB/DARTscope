"""
The compatibility test for recording without mmWave Studio.

A recording made by the acquisition layer is correct only when the existing loader -
dca1000_decode.parse_logfile_full, radar_processing.derive_capture_dimensions and
load_raw_cube, unchanged - reads it as the configuration says. Every test here runs the
real capture engine (DCA1000 controller, UDP receiver, recorder, log writer, verification)
against a DCA1000 simulator on localhost and then opens the result with that loader:

  - for the lab's configurations (1, 2 and 3 TX in TDM, Complex1X and Complex2X), known
    samples come back exactly: frames, chirps, RX, samples, I/Q, TX sequence, bytes;
  - lost, late and duplicated packets are handled the way TI's recorder handles them, and
    the lost ones show up where the capture-health code looks for them;
  - with the lab's data next to the repository: a real mmWave Studio recording replayed
    through the new path comes out byte-identical and decodes identically, and the real
    DCA1000 stream from the Wireshark capture decodes like an mmWave Studio file.

Run with:  python -m pytest test_acquisition_compat.py -q
"""
import contextlib
import copy
import hashlib
import io
import os
import struct

import numpy as np
import pytest

import vitalsigns as vs
from acquisition import lua, studio_log
from acquisition.capture import CaptureEngine, CaptureRequest
from acquisition.dca1000 import Dca1000Controller
from acquisition.layout import encode_for, random_cube
from acquisition.measurement import Measurement, NamingError, choose_stem, sanitize
from acquisition.radar import NullRadar, RadarError, StudioRemoteRadar
from acquisition.simulator import Dca1000Simulator, Faults, free_udp_port
from dca1000_decode import parse_logfile_full
from radar_processing import derive_capture_dimensions, load_raw_cube
from test_acquisition_config import DATASET_MOTION, KLEKY
from tools.captures import chirps_per_frame, load_capture

DATA = os.environ.get("DARTSCOPE_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

SINGLE_TX_VITALS = """
ar1.frequencyBandSelection("60G")
ar1.SelectChipVersion("IWR6843")
ar1.ChanNAdcConfig(1, 1, 1, 1, 1, 1, 1, 2, 1, 0)
ar1.LVDSLaneConfig(0, 1, 1, 0, 0, 1, 0, 0)
ar1.ProfileConfig(0, 60, 100, 1, 60, 0, 0, 0, 0, 0, 0, 66.01, 1, 64, 5000, 1, 131072, 36)
ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 0, 1, 0)
ar1.ChirpConfig(1, 1, 0, 0, 0, 0, 0, 1, 0, 1)
ar1.FrameConfig(0, 0, 1200, 5, 10, 0, 0, 1)
"""
# config_AWR6843.lua: three chirps defined, the frame uses the first two
FRAME_SUBSET = KLEKY.replace("FrameConfig(0, 2, 100, 64, 100", "FrameConfig(0, 1, 100, 32, 100") \
                    .replace("1, 2, 2, 0)", "1, 2, 1, 0)")

CONFIGS = {"dataset_motion": DATASET_MOTION, "kleky_3tx_complex2x": KLEKY,
           "vitals_1tx": SINGLE_TX_VITALS, "frame_subset": FRAME_SUBSET}


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def capture(tmp_path, cfg, data: bytes, faults=None, name="rec", rate=None, radar=None,
            duration_s=None):
    cfg = copy.deepcopy(cfg)
    cfg.dca.host_ip = "127.0.0.1"
    cfg.dca.config_port, cfg.dca.data_port = free_udp_port(), free_udp_port()
    sim = Dca1000Simulator(data, host_data_addr=("127.0.0.1", cfg.dca.data_port), faults=faults,
                           rate_mb_s=rate)
    dca = Dca1000Controller(cfg.dca, bind_ip="127.0.0.1", dca_addr=sim.addr)
    radar = radar(sim) if radar else NullRadar(on_start=sim.start_stream)
    try:
        res = CaptureEngine(radar, dca=dca).run(CaptureRequest(
            config=cfg, folder=str(tmp_path), name=name, bind_ip="127.0.0.1",
            duration_s=duration_s, idle_timeout_s=1.0))
    finally:
        dca.close()
        sim.close()
    return res


def cfg_for(key: str, frames: int):
    c = lua.read_script(CONFIGS[key]).config
    c.frame.num_frames = frames
    return c


@pytest.mark.parametrize("key", sorted(CONFIGS))
def test_known_samples_come_back_exactly(tmp_path, key):
    cfg = cfg_for(key, 4 if key != "vitals_1tx" else 40)
    cube = random_cube(cfg, cfg.frame.num_frames, seed=7)
    res = capture(tmp_path, cfg, encode_for(cfg, cube))
    assert res.ok, res.verification
    params, dims, got = res.measurement.load_cube()
    e = cfg.expectation()
    assert got.shape == cube.shape == (e["frames"], e["chirps_per_frame"], e["rx"], e["samples"])
    assert got.dtype == np.complex64
    assert np.array_equal(np.rint(got.real * 32767), cube.real)
    assert np.array_equal(np.rint(got.imag * 32767), cube.imag)
    assert params["tx_order_masks"] == e["tx_pattern"]
    assert params["dataFmt_adcFmt"] == e["adc_format"]
    assert params["numLanes"] == 2 and params["numLanes_source"] == "DCA1000 recorder log"
    assert params["dca_packets_received"] == e["packets"]
    assert sum(os.path.getsize(p) for p in res.measurement.bin_paths) == e["total_bytes"]
    # the tools see it like any other recording
    cap = load_capture(res.measurement.log_path)
    assert cap.bin_paths == tuple(res.measurement.bin_paths)
    assert chirps_per_frame(cap) == e["chirps_per_frame"]
    side = res.measurement.sidecar()
    assert side["status"] == "complete" and side["expected"]["total_bytes"] == e["total_bytes"]
    assert side["timing"]["first_packet_ns"] <= side["timing"]["last_packet_ns"]


def test_the_frame_count_follows_the_file_like_before(tmp_path):
    """The DCA1000 kept 128 bytes back on the lab's board: the loader drops the last frame."""
    cfg = cfg_for("dataset_motion", 4)
    cube = random_cube(cfg, 4, seed=1)
    res = capture(tmp_path, cfg, encode_for(cfg, cube), Faults(withhold_tail=128))
    assert res.ok and res.verification["info"]["bytes_missing_at_end"] == 128
    _, _, got = res.measurement.load_cube()
    assert got.shape[0] == 3
    assert np.array_equal(np.rint(got.real * 32767), cube[:3].real)


def test_lost_packets_are_zeros_where_health_checks_look(tmp_path):
    cfg = cfg_for("dataset_motion", 4)
    cube = random_cube(cfg, 4, seed=2)
    # packets 2101..2130 hold bytes 3 057 600 .. 3 101 280: inside frame 2 (2 448 000 .. 3 672 000)
    res = capture(tmp_path, cfg, encode_for(cfg, cube), Faults(drop=set(range(2101, 2131))))
    assert res.ok and res.stats["zero_filled_packets"] == 30
    params, _, got = res.measurement.load_cube()
    assert params["dca_packets_zero_filled"] == 30
    dmg = vs.detect_damaged_frames(got)
    assert list(np.flatnonzero(dmg)) == [2]
    ok = np.ones(4, bool)
    ok[2] = False
    assert np.array_equal(np.rint(got[ok].real * 32767), cube[ok].real)


def test_late_and_duplicate_packets_change_nothing(tmp_path):
    cfg = cfg_for("kleky_3tx_complex2x", 3)
    cube = random_cube(cfg, 3, seed=3)
    res = capture(tmp_path, cfg, encode_for(cfg, cube),
                  Faults(swap={5, 500, 1000}, duplicate={7, 800}))
    assert res.ok and res.stats["late_recovered"] == 3 and res.stats["duplicates"] == 2
    _, _, got = res.measurement.load_cube()
    assert np.array_equal(np.rint(got.imag * 32767), cube.imag)


def test_an_open_ended_recording_stops_on_time(tmp_path):
    cfg = cfg_for("vitals_1tx", 0)                      # numFrames 0: until stopped
    src = random_cube(cfg, 400, seed=4)                 # 400 frames at 2 MB/s ~ 2.6 s
    res = capture(tmp_path, cfg, encode_for(cfg, src), rate=2.0, duration_s=0.6)
    assert res.stop_reason.startswith("duration")
    assert res.ok, res.verification
    params, dims, got = res.measurement.load_cube()
    assert 0 < got.shape[0] < 400
    assert np.array_equal(np.rint(got.real * 32767), src[:got.shape[0]].real)


def test_mmwave_studio_remote_radar_sends_the_radar_commands(tmp_path):
    class FakeRstd:
        def __init__(self, sim):
            self.sim, self.sent = sim, []

        def connect(self, host, port):
            pass

        def send(self, cmd):
            self.sent.append(cmd)
            if cmd == "return " + lua.START_COMMAND:
                self.sim.start_stream()
            return 0, [0.0] if cmd.startswith("return ar1.") else [], ""

        def close(self):
            pass

    cfg = cfg_for("dataset_motion", 2)
    holder = {}

    def radar(sim):
        holder["t"] = FakeRstd(sim)
        return StudioRemoteRadar(transport=holder["t"])

    res = capture(tmp_path, cfg, encode_for(cfg, random_cube(cfg, 2)), radar=radar)
    assert res.ok
    sent = holder["t"].sent
    unwrap = [c[len("return "):] if c.startswith("return ar1.") else c for c in sent]
    assert unwrap[:-2] == lua.radar_commands(cfg)
    assert unwrap[-2:] == [lua.START_COMMAND, lua.STOP_COMMAND]

    class LuaError(FakeRstd):          # what mmWave Studio answers when the Lua raises
        def send(self, cmd):
            return (20000, [], "bad argument") if "ProfileConfig" in cmd else (0, [0.0], "")
    with pytest.raises(RadarError, match="ProfileConfig.*bad argument"):
        StudioRemoteRadar(transport=LuaError(None)).configure(cfg)

    class Ar1Failure(FakeRstd):        # the Lua ran, the ar1 function reported a failure
        def send(self, cmd):
            return 0, [-1.0] if "RfInit" in cmd else [0.0], ""
    with pytest.raises(RadarError, match="RfInit.*returned -1"):
        StudioRemoteRadar(transport=Ar1Failure(None)).configure(cfg)


def test_names_and_no_overwrite(tmp_path):
    assert sanitize("Měření sedící 1") == "Mereni_sedici_1"
    for bad in ("x_Raw_1", "a_LogFile", "", " "):
        with pytest.raises(NamingError):
            choose_stem(str(tmp_path), bad)
    (tmp_path / "take_Raw_0.bin").write_bytes(b"")
    with pytest.raises(NamingError):
        choose_stem(str(tmp_path), "take")
    assert choose_stem(str(tmp_path), "take", auto_increment=True) == "take_2"
    assert Measurement.from_path(str(tmp_path / "take_Raw_0.bin")).stem == "take"
    assert Measurement.from_path(str(tmp_path / "take_acquisition.json")).stem == "take"


def test_receiver_keeps_up_with_a_fast_stream(tmp_path):
    """~34 MB as fast as Python can send it on localhost - more than the lab's 30.6 MB/s."""
    cfg = cfg_for("dataset_motion", 28)
    data = encode_for(cfg, random_cube(cfg, 28, seed=5))
    res = capture(tmp_path, cfg, data)
    assert res.ok and res.stats["zero_filled_packets"] == 0
    got = b"".join(open(p, "rb").read() for p in res.measurement.bin_paths)
    assert hashlib.sha256(got).digest() == hashlib.sha256(data).digest()


# --- the lab's real data --------------------------------------------------------------------

CHUZE = os.path.join(DATA, "chuze_dca_noha")


@pytest.mark.skipif(not os.path.isfile(os.path.join(CHUZE, "data_test.bin")),
                    reason="the lab's recordings are not next to the repository")
def test_a_real_mmwave_studio_recording_replayed_comes_out_identical(tmp_path):
    """
    chuze_dca_noha/data_test.bin (walking, 255 frames, 65 MB) as mmWave Studio + DCA1000
    recorded it, sent again through the new path with the configuration read from its
    log: the new file is byte for byte the old one, and the loader decodes both alike.
    """
    old_bin = os.path.join(CHUZE, "data_test.bin")
    old_log = os.path.join(CHUZE, "data_test_LogFile.txt")
    cfg = studio_log.read_config(old_log)
    data = open(old_bin, "rb").read()
    assert cfg.total_bytes == len(data) == 65_280_000
    res = capture(tmp_path, cfg, data, name="replay")
    assert res.ok, res.verification
    new = b"".join(open(p, "rb").read() for p in res.measurement.bin_paths)
    assert hashlib.sha256(new).digest() == hashlib.sha256(data).digest()
    with _quiet():
        p_old, _ = parse_logfile_full(old_log)
        d_old = derive_capture_dimensions(p_old)
        p_new = res.measurement.load_params()
        d_new = derive_capture_dimensions(p_new)
    assert d_old == d_new
    for k in ("tx_order_masks", "numLanes", "dataFmt_adcFmt", "dataFmt_iqOrder",
              "framePeriodicity_ms_selected", "freqSlopeMHz_us", "startFreqGHz"):
        assert p_old[k] == p_new[k], k
    with _quiet():
        c_old = load_raw_cube(old_bin, p_old, dict(d_old))
        c_new = load_raw_cube(res.measurement.first_bin, p_new, dict(d_new))
    assert np.array_equal(c_old, c_new)


def _pcap_payloads(path: str, max_bytes: int) -> bytes:
    """DCA1000 data payloads from the Wireshark capture, split where the NIC coalesced them."""
    out = bytearray()
    with open(path, "rb") as f:
        while len(out) < max_bytes:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            btype, blen = struct.unpack("<II", hdr)
            body = f.read(blen - 8)
            if btype != 6:
                continue
            frame = body[20:20 + struct.unpack("<I", body[12:16])[0]]
            if len(frame) < 42 or frame[23] != 17:
                continue
            u = 14 + (frame[14] & 0x0F) * 4
            if struct.unpack(">H", frame[u + 2:u + 4])[0] != 4098:
                continue
            # data frames are never padded; the NIC coalesced up to ten datagrams into one
            payload, pos = frame[u + 8:], 0
            while pos < len(payload):
                n = min(1456, len(payload) - pos - 10)
                out += payload[pos + 10:pos + 10 + n]
                pos += 10 + n
    return bytes(out[:max_bytes])


@pytest.mark.skipif(not os.path.isfile(os.path.join(DATA, "Wireshark", "dca1000.pcapng")),
                    reason="the Wireshark capture is not next to the repository")
def test_the_real_dca1000_stream_decodes_like_an_mmwave_studio_file(tmp_path):
    """
    20 frames of the real DCA1000 stream (the Dataset_motion configuration) through the new
    path. Real ADC samples - not ones this code generated - must come out as complex
    baseband: I and Q uncorrelated, consecutive I samples strongly correlated, much more
    energy at positive than at negative beat frequencies, as in M1_01_Raw_0.bin.
    """
    cfg = cfg_for("dataset_motion", 20)
    data = _pcap_payloads(os.path.join(DATA, "Wireshark", "dca1000.pcapng"), cfg.total_bytes)
    assert len(data) == cfg.total_bytes
    res = capture(tmp_path, cfg, data, name="pcap")
    assert res.ok, res.verification
    new = b"".join(open(p, "rb").read() for p in res.measurement.bin_paths)
    assert new == data
    signs = res.verification["info"]["iq_signs"]
    assert signs["corr_I_I"] > 0.8 and abs(signs["corr_I_Q"]) < 0.1
    assert signs["positive_vs_negative_beat_dB"] > 10
