"""
Tests for the acquisition configuration: the lab's Lua scripts read into RadarConfig,
written back, and turned into the mmWave Studio log that dca1000_decode reads.

The expected numbers are the lab's own: the codes mmWave Studio logged for the same
scripts (dataset_square/M1_01_LogFile.txt, kleky_dataset1) and the sizes of the files it
recorded with them.

Run with:  python -m pytest test_acquisition_config.py -q
"""
import contextlib
import glob
import io
import os

import pytest

import chirp_design as cd
from acquisition import lua, studio_log
from acquisition.config import (Chirp, ConfigError, Frame, Profile, RadarConfig,
                                from_design)
from dca1000_decode import parse_logfile_full
from radar_processing import derive_capture_dimensions

DATA = os.environ.get("DARTSCOPE_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# config/Dataset_motion.lua as the lab ran it (dataset_square, kleky_dataset_published, rect)
DATASET_MOTION = r'''
ar1.FullReset()
ar1.SOPControl(2)
ar1.Connect(12, 921600, 1000)
ar1.Calling_IsConnected()
ar1.frequencyBandSelection("60G")
ar1.SelectChipVersion("IWR6843")
ar1.deviceVariantSelection("IWR6843")
BSS_FW = "C:\\ti\\mmwave_studio_02_01_01_00\\rf_eval_firmware\\radarss\\xwr68xx_radarss.bin"
MSS_FW = "C:\\ti\\mmwave_studio_02_01_01_00\\rf_eval_firmware\\masterss\\xwr68xx_masterss.bin"
ar1.DownloadBSSFw(BSS_FW)
ar1.DownloadMSSFw(MSS_FW)
ar1.PowerOn(1, 1000, 0, 0)
ar1.RfEnable()
ar1.ChanNAdcConfig(1, 1, 1, 1, 1, 1, 1, 2, 1, 0)
ar1.LPModConfig(0, 0)
ar1.RfInit()
RSTD.Sleep(1000)
ar1.DataPathConfig(1, 1, 0)
ar1.LvdsClkConfig(1, 1)
ar1.LVDSLaneConfig(0, 1, 1, 0, 0, 1, 0, 0)
-- ar1.ProfileConfig(0, 60, 100, 6, 60, 0, 0, 0, 0, 0, 0, 29.982, 0, 256, 10000, 0, 131072, 30)
ar1.ProfileConfig(0, 60.75, 60, 1, 15, 0, 0, 0, 0, 0, 0, 199.987, 1, 150, 10785, 1, 131072, 36)
ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 0, 1, 0)
ar1.ChirpConfig(1, 1, 0, 0, 0, 0, 0, 1, 0, 1)
ar1.SelectCaptureDevice("DCA1000")
ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098)
ar1.CaptureCardConfig_Mode(1, 2, 1, 2, 3, 30)
ar1.CaptureCardConfig_PacketDelay(25)
ar1.DisableTestSource(0)
ar1.FrameConfig(0, 1, 250, 255, 40, 0, 0, 1)
ar1.CaptureCardConfig_StartRecord("C:\\Data\\Radar\\testovani\\feb26\\\data_test.bin",1)
ar1.StartFrame()
'''

# the configuration lines mmWave Studio logged for that script (dataset_square/M1_01_LogFile.txt)
M1_LOG_LINES = [
    "API:select_chip_version,IWR6843,0,",
    "API:ChannelConfig,7,15,0,",
    "API:AdcOutConfig,2,1,0,",
    "API:DataFmtConfig,15,2,1,0,1,0,",
    "API:LowPowerConfig,0,0,0,",
    "API:DataPathConfig,1,1,0,2,0,",
    "API:LvdsClkConfig,1,1,0,",
    "API:SetHsiClock,9,0,",
    "API:LaneConfig,3,0,",
    "API:LvdsLaneConfig,0,1,0,",
    "API:ProfileConfig,0,1509949440,6000,100,1500,0,0,5523,100,150,10785,1,0,36,0,",
    "API:ChirpConfig,0,0,0,0,0,0,0,2,0,",
    "API:ChirpConfig,1,1,0,0,0,0,0,5,0,",
    "API:EnableTestSource,0,1,0,",
    "API:FrameConfig,0,1,250,255,8000000,0,300,0,",
    "API:AdvancedFrameConfig,1,0,0,0,2,255,8000000,0,1,1,8000000," + "0," * 27 + "250,"
    + "0," * 16,
]

# kleky_dataset1/luaconfig.lua: three TX in TDM, Complex2X
KLEKY = r'''
ar1.frequencyBandSelection("60G")
ar1.SelectChipVersion("IWR6843")
ar1.ChanNAdcConfig(1, 1, 1, 1, 1, 1, 1, 2, 2, 0)
ar1.LVDSLaneConfig(0, 1, 1, 0, 0, 1, 0, 0)
ar1.ProfileConfig(0, 60, 143, 7, 57.14, 0, 0, 0, 0, 0, 0, 69.994, 0, 256, 10000, 0, 131072, 30)
ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 1, 0, 0)
ar1.ChirpConfig(1, 1, 0, 0, 0, 0, 0, 0, 1, 0)
ar1.ChirpConfig(2, 2, 0, 0, 0, 0, 0, 0, 0, 1)
ar1.CaptureCardConfig_Mode(1, 2, 1, 2, 3, 30)
ar1.FrameConfig(0, 2, 100, 64, 100, 0, 0, 1)
'''


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


# --- Lua -> configuration ------------------------------------------------------------------

def test_dataset_motion_reads_to_the_codes_mmwave_studio_logged():
    s = lua.read_script(DATASET_MOTION)
    c = s.config
    p = c.profile
    assert (p.start_freq_const, p.idle_time, p.adc_start_time, p.ramp_end_time) == \
           (1509949440, 6000, 100, 1500)
    assert (p.freq_slope_const, p.tx_start_time, p.num_adc_samples, p.dig_out_sample_rate) == \
           (5523, 100, 150, 10785)
    assert (p.hpf_corner_freq1, p.hpf_corner_freq2, p.rx_gain) == (1, 0, 36)
    assert [(ch.start_idx, ch.end_idx, ch.tx_mask) for ch in c.chirps] == [(0, 0, 2), (1, 1, 5)]
    assert (c.frame.chirp_start, c.frame.chirp_end, c.frame.num_frames, c.frame.num_loops,
            c.frame.periodicity) == (0, 1, 250, 255, 8_000_000)
    assert (c.channel.tx_mask, c.channel.rx_mask, c.channel.adc_fmt) == (7, 15, 1)
    assert c.lvds.lane_mask == 3 and c.dca.lvds_mode == 2 and c.dca.packet_delay_us == 25
    assert s.record_path.endswith("data_test.bin") and s.uses_capture_card
    assert not c.errors()


def test_expected_size_is_the_size_of_the_lab_recordings():
    """306 000 000 bytes: dataset_square/*_Raw_0.bin hold 305 999 872 (the DCA1000 kept 128 back)."""
    e = lua.read_script(DATASET_MOTION).config.expectation()
    assert e["chirps_per_frame"] == 510 and e["bytes_per_frame"] == 1_224_000
    assert e["total_bytes"] == 306_000_000
    assert e["tx_pattern"] == [2, 5]
    assert e["packets"] == 210_165                   # the recorder logs 210165 packets
    k = lua.read_script(KLEKY).config.expectation()
    assert k["total_bytes"] == 78_643_200            # size of every kleky_dataset1 recording
    assert k["tx_pattern"] == [1, 2, 4] and k["adc_format"] == "Complex2X"


def test_rounding_is_mmwave_studios():
    k = lua.read_script(KLEKY).config.profile
    assert k.start_freq_const == 1491308089           # 60 GHz
    assert k.ramp_end_time == 5714                    # 57.14 us, not 5713
    assert k.freq_slope_const == 1933                 # 69.994 MHz/us
    assert k.idle_time == 14300 and k.adc_start_time == 700


def test_a_decimal_comma_is_reported_not_guessed():
    bad = DATASET_MOTION.replace("60.75, 60, 1, 15", "60,75, 60, 1, 15")
    with pytest.raises(lua.LuaConfigError, match="decimal comma"):
        lua.read_script(bad)


def test_only_the_branch_that_runs_counts():
    """config_my.lua chooses LVDSLaneConfig in an if/elseif on partId."""
    text = DATASET_MOTION.replace("ar1.LVDSLaneConfig(0, 1, 1, 0, 0, 1, 0, 0)", r'''
partId = 6843
if (partId ~= 6843) then
    WriteToLog("This script is designed for IWR6843 only!\n", "red")
    return
end
if((partId == 1642) or (partId == 1843) or (partId == 6843)) then
    if (ar1.LVDSLaneConfig(0, 1, 1, 0, 0, 1, 0, 0) == 0) then
        WriteToLog("LVDSLaneConfig Success\n", "green")
    else
        WriteToLog("LVDSLaneConfig failure\n", "red")
    end
elseif ((partId == 1243) or (partId == 1443)) then
    ar1.LVDSLaneConfig(0, 1, 1, 1, 1, 1, 0, 0)
end''')
    c = lua.read_script(text).config
    assert c.lvds.lane_mask == 0b0011
    assert c.frame.num_frames == 250                   # the script ran on after the ifs


def test_a_return_in_a_taken_branch_ends_the_script():
    text = 'x = 1\nif (x == 1) then return end\n' + DATASET_MOTION
    with pytest.raises(lua.LuaConfigError, match="no ChirpConfig"):
        lua.read_script(text)


def test_variables_and_concatenation_in_firmware_paths():
    text = DATASET_MOTION.replace(
        'BSS_FW = "C:\\\\ti\\\\mmwave_studio_02_01_01_00\\\\rf_eval_firmware\\\\radarss\\\\xwr68xx_radarss.bin"',
        'fw_path = "D:\\\\studio\\\\rf_eval_firmware"\nBSS_FW = fw_path.."\\\\radarss\\\\xwr68xx_radarss.bin"')
    c = lua.read_script(text).config
    assert c.device.bss_firmware == r"D:\studio\rf_eval_firmware\radarss\xwr68xx_radarss.bin"


def test_commands_read_back_to_the_same_configuration():
    for text in (DATASET_MOTION, KLEKY):
        c = lua.read_script(text).config
        c2 = lua.read_script("\n".join(lua.radar_commands(c))).config
        assert (c2.profile, c2.chirps, c2.frame, c2.channel, c2.lvds) == \
               (c.profile, c.chirps, c.frame, c.channel, c.lvds)


def test_radar_commands_leave_the_dca1000_alone():
    cmds = lua.radar_commands(lua.read_script(DATASET_MOTION).config)
    assert not any("CaptureCard" in c or "SelectCaptureDevice" in c for c in cmds)
    assert cmds[-1].startswith("ar1.FrameConfig(0, 1, 250, 255, 40,")


@pytest.mark.skipif(not glob.glob(os.path.join(DATA, "config", "*.lua")),
                    reason="the lab's Lua scripts are not next to the repository")
def test_every_lab_script_reads_or_is_rejected_with_a_reason():
    scripts = [p for p in glob.glob(os.path.join(DATA, "**", "*.lua"), recursive=True)
               if "MatlabExamples" not in p]
    ok = 0
    for path in scripts:
        try:
            s = lua.read_script_file(path)
        except lua.LuaConfigError as e:
            assert "config_my.lua" in path and "decimal comma" in str(e)
            continue
        assert not s.config.errors(), (path, s.config.errors())
        ok += 1
    assert ok >= 6


# --- configuration checks ------------------------------------------------------------------

def test_problems_name_what_would_break():
    c = lua.read_script(DATASET_MOTION).config
    c.frame.chirp_end = 2                              # chirp 2 is not defined
    c.profile.ramp_end_time = 1000                     # 150 samples at 10.785 Msps need 13.9 us
    msgs = " ".join(str(p) for p in c.errors())
    assert "chirps [2]" in msgs and "ADC window" in msgs
    with pytest.raises(ConfigError):
        c.check()


def test_json_round_trip():
    c = lua.read_script(KLEKY).config
    assert RadarConfig.from_dict(c.to_dict()) == c


def test_the_helpers_design_and_its_lua_agree():
    """chirp_design.lua_script and from_design describe the same radar."""
    dev = cd.DEVICES["IWR6843AOP"]
    for purpose in cd.PURPOSES:
        d = cd.design(dev, cd.goals_for(purpose))
        via_lua = lua.read_script(cd.lua_script(d)).config
        direct = from_design(d)
        assert (via_lua.profile, via_lua.chirps, via_lua.frame) == \
               (direct.profile, direct.chirps, direct.frame), purpose
        assert direct.bytes_per_chirp == d.perf["chirp_bytes"]
        assert direct.chirps_per_frame == d.perf["chirps_per_frame"]


# --- configuration -> mmWave Studio log -------------------------------------------------------

def test_log_lines_are_mmwave_studios():
    ours = studio_log.config_lines(lua.read_script(DATASET_MOTION).config)
    for line in M1_LOG_LINES:
        assert line in ours, line


def test_the_parser_reads_the_written_log_as_configured(tmp_path):
    c = lua.read_script(DATASET_MOTION).config
    log = tmp_path / "x_LogFile.txt"
    studio_log.write_log(str(log), c)
    with _quiet():
        p, _ = parse_logfile_full(str(log))
        d = derive_capture_dimensions(p)
    assert p["numAdcSamples"] == 150 and p["dataFmt_adcFmt"] == "Complex1X"
    assert abs(p["startFreqGHz"] - 60.75) < 1e-9
    assert abs(p["freqSlopeMHz_us"] - 199.987) < 0.01
    assert p["tx_order_masks"] == [2, 5] and p["numLanes"] == 2
    assert p["framePeriodicity_ms_selected"] == 40.0 and p["chip_hint"] == "IWR6843"
    assert (d["num_rx"], d["num_samples"], d["chirps_per_frame"], d["num_frames"]) == (4, 150, 510, 250)


@pytest.mark.parametrize("frames", [2, 5, 50, 64, 65, 250])
def test_short_tdm_recordings_keep_their_chirps(tmp_path, frames):
    """
    FrameConfig 0,1,5,255,... also reads as 'chirps 1..5' (a leading index field). The
    parser re-reads it against the defined chirps; params['frameConfig'] has to follow,
    or a 5-frame TDM recording decodes with 255 chirps per frame instead of 510.
    """
    c = lua.read_script(DATASET_MOTION).config
    c.frame.num_frames = frames
    log = tmp_path / f"f{frames}_LogFile.txt"
    studio_log.write_log(str(log), c)
    with _quiet():
        p, _ = parse_logfile_full(str(log))
        d = derive_capture_dimensions(p)
    assert d["chirps_per_frame"] == 510 and d["num_frames"] == frames
    assert p["frameConfig"]["chirpStartIdx"] == 0 and p["frameConfig"]["chirpEndIdx"] == 1


def test_a_log_the_loader_would_misread_is_refused_before_recording():
    """
    One frame of chirps 0..1 logs FrameConfig,0,1,1,255,...: read with a leading index
    field that is 'chirps 1..1', which the defined chirps fit just as well, and the parser
    prefers it. The capture refuses the configuration instead of recording it.
    """
    from acquisition.verify import loader_mismatches
    c = lua.read_script(DATASET_MOTION).config
    assert loader_mismatches(c) == []
    c.frame.num_frames = 1
    assert any("chirps per frame" in m for m in loader_mismatches(c))


def test_log_is_never_overwritten(tmp_path):
    c = lua.read_script(DATASET_MOTION).config
    log = tmp_path / "x_LogFile.txt"
    studio_log.write_log(str(log), c)
    with pytest.raises(FileExistsError):
        studio_log.write_log(str(log), c)


def test_reading_a_session_log_takes_the_last_configuration():
    text = "\n".join(f"05-Dec-2025 12:19:5{k % 10}: {line}" for k, line in enumerate(
        ["API:ProfileConfig,0,1491308089,10000,600,6000,0,0,828,0,256,10000,0,0,30,0,",
         "API:FrameConfig,0,0,100,128,8000000,0,512,0,"] + M1_LOG_LINES))
    c = studio_log.read_config(text)
    assert studio_log.config_lines(c)[1:] == M1_LOG_LINES


def test_a_dca1000_cli_log_is_not_taken_for_a_configuration():
    with pytest.raises(ConfigError):
        studio_log.read_config("Mon Mar 04 15:16:48 2024\nFPGA Configuration command : Success\n")


@pytest.mark.skipif(not glob.glob(os.path.join(DATA, "dataset_square", "*_LogFile.txt")),
                    reason="the lab's recordings are not next to the repository")
def test_every_real_log_is_rewritten_line_for_line(tmp_path):
    """
    Every mmWave Studio log in the data folders, read into a RadarConfig and written back:
    each line matches the last line of its kind in the original, and parse_logfile_full
    derives the same parameters from both. The one allowed difference is FrameConfig's
    sample field in logs where the ADC format was switched after FrameConfig (the lane
    sweep): mmWave Studio logged the value from before the switch.
    """
    keys = ["numAdcSamples", "startFreqGHz", "freqSlopeMHz_us", "idleTime_us", "rampEndTime_us",
            "digOutSampleRate_MHz", "dataFmt_adcFmt", "dataFmt_iqOrder", "dataFmt_rxChanEnMask",
            "numLanes", "tx_order_masks", "framePeriodicity_ms_selected", "chip_hint"]
    logs = [p for p in glob.glob(os.path.join(DATA, "**", "*_LogFile.txt"), recursive=True)
            if "Docs" not in p and "RSTD" not in p and not os.path.basename(p).startswith("CLI_")]
    assert len(logs) > 100
    for k, path in enumerate(logs):
        text = open(path, encoding="utf-8", errors="replace").read()
        c = studio_log.read_config(text)
        last = {}
        chirps = {}
        order = {}
        for i, (verb, flds, s) in enumerate(studio_log.api_lines(text)):
            if flds and flds[-1] == "0":
                last[verb], order[verb] = s, i
                if verb == "ChirpConfig":
                    chirps[tuple(flds[:3])] = s
        format_after_frame = order.get("AdcOutConfig", -1) > order.get("FrameConfig", -1)
        for s in studio_log.config_lines(c):
            verb = s[4:s.index(",")]
            ref = chirps.get(tuple(s.split(",")[1:4])) if verb == "ChirpConfig" else last.get(verb)
            if ref is None or (verb == "FrameConfig" and format_after_frame):
                continue
            assert ref == s, (path, ref, s)
        a, b = tmp_path / f"a{k}_LogFile.txt", tmp_path / f"b{k}_LogFile.txt"
        a.write_text(text, encoding="utf-8")
        studio_log.write_log(str(b), c)
        with _quiet():
            pa, _ = parse_logfile_full(str(a))
            pb, _ = parse_logfile_full(str(b))
        for key in keys:
            assert pa.get(key) == pb.get(key), (path, key, pa.get(key), pb.get(key))
        with _quiet():
            assert derive_capture_dimensions(pa) == derive_capture_dimensions(pb), path


def test_lua_math_functions_are_evaluated():
    """testovani/lua/IWR6843AOP_DCA1000_CAPON_TDM_MIMO.lua computes its frame count:
    NUM_FRAMES = math.floor(MEASUREMENT_DURATION_S * 1000 / FRAME_PERIODICITY_MS)."""
    calls, _ = lua.read_calls(
        "D = 60\nP = 50\nN = math.floor(\n    D * 1000 / P\n)\n"
        "ar1.FrameConfig(0, 2, N, math.max(4, 32), math.ceil(P * 1e6 / 5 - 0.5), 0, 1)\n"
        "X = math.huge\n")
    frame = [c for c in calls if c.name == "FrameConfig"][0]
    assert frame.args[2] == 1200 and frame.args[3] == 32 and frame.args[4] == 10000000


def test_the_radar_needs_300_us_between_frames():
    # 2026-10-06, IWR6843AOPEVM: FrameConfig answered 80 (frame ON time > periodicity) with
    # 250 us between frames and accepted 300 us - TI's InterFrameBlankTime
    from acquisition.config import FRAME_BLANK_MS, RadarConfig, Frame
    cfg = RadarConfig()
    active = cfg.active_ms
    cfg.frame.periodicity = Frame.ticks(active + 0.25)
    assert any(p.what == "frame" and p.level == "error" for p in cfg.problems())
    cfg.frame.periodicity = Frame.ticks(active + FRAME_BLANK_MS + 0.01)
    assert not any(p.what == "frame" for p in cfg.problems())


def test_every_purpose_leaves_the_blank_between_frames():
    import chirp_design as cd
    for dev in cd.DEVICES.values():
        for purpose, goals in cd.PURPOSES.items():
            d = cd.design(dev, cd.Goals(**goals))
            assert d.perf["active_ms"] + cd.FRAME_BLANK_MS <= d.period_ms + 1e-9, (dev, purpose)
