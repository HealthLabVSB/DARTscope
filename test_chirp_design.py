"""
Tests for the chirp helper (chirp_design.py): the designs meet their goals within the
device limits, impossible goals say what to change, the numbers follow the FMCW
relations, and the Lua script has the argument lists of the lab's working scripts.

Run with:  python -m pytest test_chirp_design.py -q
"""
import re

import pytest

import chirp_design as cd

C0 = 299_792_458.0


@pytest.mark.parametrize("device", list(cd.DEVICES))
@pytest.mark.parametrize("purpose", list(cd.PURPOSES))
def test_every_purpose_is_designed_within_the_limits(device, purpose):
    dev = cd.DEVICES[device]
    g = cd.goals_for(purpose)
    d = cd.design(dev, g)
    errors = [c for c in d.checks if c.level == "error"]
    assert not errors, errors
    p = d.perf
    assert p["range_res_cm"] <= g.range_res_cm * 1.02
    assert p["max_range_m"] >= g.max_range_m * 0.98
    assert p["max_vel_ms"] >= g.max_vel_ms * 0.98
    if not g.back_to_back:
        assert p["fps"] == pytest.approx(g.fps, rel=0.02)
    assert p["active_ms"] <= d.period_ms + 1e-9
    assert d.slope_mhz_us <= dev.max_slope and d.fs_ksps <= dev.max_fs_ksps
    assert p["stop_ghz"] <= dev.f_hi and p["sweep_ghz"] <= dev.max_bw_ghz
    assert 1 <= d.loops <= cd.MAX_LOOPS
    assert p["rate_mb_s"] <= cd.PROVEN_RATE_MB_S           # the presets stay in what worked


def test_the_slope_is_on_the_chips_grid():
    d = cd.design(cd.DEVICES["IWR6843AOP"], cd.goals_for("People and angles (AoA, tracking)"))
    lsb = cd.SLOPE_LSB["60G"]
    assert abs(d.slope_mhz_us / lsb - round(d.slope_mhz_us / lsb)) < 0.02
    assert cd.quantise_slope(29.982, "60G") == pytest.approx(29.982, abs=1e-3)   # TI's example


def test_impossible_goals_say_what_to_change():
    g = cd.Goals(max_range_m=10.0, range_res_cm=1.0, max_vel_ms=6.0, vel_res_ms=0.1, fps=20,
                 tdm_all_tx=True)
    d = cd.design(cd.DEVICES["IWR6843AOP"], g)
    errors = [c for c in d.checks if c.level == "error"]
    assert errors
    text = " ".join(c.message for c in errors)
    assert "coarser" in text or "fewer TX" in text or "lower" in text


def test_the_numbers_follow_the_fmcw_relations():
    """The lab's Dataset_motion.lua: 199.987 MHz/us, 150 samples at 10785 ksps, idle 60,
    ramp end 15, TX2 and TX1 in turn, 255 loops, 40 ms."""
    dev = cd.DEVICES["IWR6843AOP"]
    p = cd.performance(dev, 60.75, 60.0, 1.0, 15.0, 199.987, 150, 10785, 2, 255, 40.0)
    t_adc = 150 / 10.785e6
    assert p["range_res_cm"] == pytest.approx(C0 / (2 * 199.987e12 * t_adc) * 100, rel=1e-6)
    assert p["range_res_cm"] == pytest.approx(5.39, abs=0.01)
    lam = C0 / (60.75e9 + 199.987e12 * (1e-6 + t_adc / 2))
    assert p["max_vel_ms"] == pytest.approx(lam / (4 * 2 * 75e-6), rel=1e-6)
    assert p["vel_res_ms"] == pytest.approx(lam / (2 * 255 * 2 * 75e-6), rel=1e-6)
    assert p["rate_mb_s"] == pytest.approx(150 * 4 * 4 * 510 * 25 / 1e6, rel=1e-6)   # 30.6
    assert p["max_range_m"] == pytest.approx(min(0.9 * 10.785e6, 10e6) * C0 / (2 * 199.987e12))


def test_a_configuration_as_entered_is_checked():
    dev = cd.DEVICES["IWR6843AOP"]
    ok = cd.evaluate(dev, None, start_ghz=60.25, idle_us=100, adc_start_us=6, ramp_end_us=60,
                     slope_mhz_us=29.982, tx_start_us=0, n_samples=256, fs_ksps=10000,
                     rx_gain_db=30, tx_order=[1, 2, 3], loops=64, frames=100, period_ms=40)
    assert not [c for c in ok.checks if c.level == "error"]
    bad = cd.evaluate(dev, None, start_ghz=60.25, idle_us=10, adc_start_us=6, ramp_end_us=30,
                      slope_mhz_us=29.982, tx_start_us=0, n_samples=1100, fs_ksps=10000,
                      rx_gain_db=30, tx_order=[1, 5], loops=300, frames=10, period_ms=5)
    what = {c.what for c in bad.checks if c.level == "error"}
    assert {"ADC window", "chirps", "ADC buffer", "loops", "frame"} <= what


def test_the_configuration_of_a_recording():
    params = {"chip_hint": "IWR6843", "startFreqGHz": 60.75000000001, "idleTime_us": 60.0,
              "adcStartTime_us": 1.0, "rampEndTime_us": 15.0, "freqSlopeMHz_us": 199.98712,
              "txStartTime_us": 1.0, "numAdcSamples": 150, "digOutSampleRate_MHz": 10.785,
              "rxGain": 30, "tx_order_masks": [2, 1],
              "frameConfig": {"numLoops": 255, "numFrames": 250}, "framePeriodicity_ms_selected": 40.0}
    vals = cd.from_params(params)
    assert vals["tx_order"] == [2, 1] and vals["fs_ksps"] == 10785 and vals["start_ghz"] == 60.75
    assert cd.device_for(params).name == "IWR6843AOP"
    assert cd.device_for({"chip_hint": "AWR2544"}).name == "AWR2544"
    d = cd.evaluate(cd.device_for(params), None, **vals)
    assert d.perf["rate_mb_s"] == pytest.approx(30.6, abs=0.1)


def _args(script, call):
    return [re.search(rf"ar1\.{call}\((.*)\)", line).group(1).split(", ")
            for line in script.splitlines() if line.startswith(f"ar1.{call}(")]


def test_the_lua_script_of_the_iwr6843():
    d = cd.design(cd.DEVICES["IWR6843AOP"], cd.goals_for("People and angles (AoA, tracking)"))
    lua = cd.lua_script(d, com_port=7, studio_dir=r"C:\ti\studio", out_bin=r"D:\rec\run1.bin")
    prof = _args(lua, "ProfileConfig")
    assert len(prof) == 1 and len(prof[0]) == 18
    assert [len(a) for a in _args(lua, "ChirpConfig")] == [10, 10, 10]
    assert [a[7:] for a in _args(lua, "ChirpConfig")] == [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]]
    frame = _args(lua, "FrameConfig")[0]
    assert len(frame) == 8 and frame[:4] == ["0", "2", str(d.frames), str(d.loops)]
    assert len(_args(lua, "ChanNAdcConfig")[0]) == 10
    assert "ar1.Connect(7, 921600, 1000)" in lua
    assert 'StartRecord("D:\\\\rec\\\\run1.bin", 1)' in lua
    assert "C:\\\\ti\\\\studio\\\\rf_eval_firmware" in lua
    # Lua takes a decimal point only: a decimal comma splits an argument in two
    assert prof[0][1] == "60.25" and "." in prof[0][11]
    assert not re.search(r"\d,\d", lua)


def test_the_lua_script_of_the_awr2544():
    d = cd.design(cd.DEVICES["AWR2544"], cd.goals_for("People and angles (AoA, tracking)"))
    lua = cd.lua_script(d)
    assert len(_args(lua, "ProfileConfig")[0]) == 23
    assert [len(a) for a in _args(lua, "ChirpConfig")] == [11] * 4
    assert len(_args(lua, "FrameConfig")[0]) == 7
    assert len(_args(lua, "ChanNAdcConfig")[0]) == 11
    assert 'frequencyBandSelection("77G")' in lua and "ar1.DisableTestSource(0)" in lua
