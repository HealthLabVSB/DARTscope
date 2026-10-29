"""
Tests for capture_health - the one report on whether a capture can be trusted.

Every scene here is synthetic, with the fault planted, so each check is shown to fire
on the fault and to stay quiet on a clean capture.

Run with:  python -m pytest test_capture_health.py -q
"""
import numpy as np
import pytest

import capture_health as ch


def clean_params(**over):
    p = {
        "dca_lane_mode": 2, "dca_packets_received": 1000, "dca_packets_zero_filled": 0,
        "dca_out_of_sequence": 0,
        "radar_start_source": "DCA1000 recorder log",
        "radar_start_time": "10-07-2026 14:33:42",
        "numLanes": 2, "numLanes_source": "DCA1000 recorder log", "numLanes_radar": 2,
        "rampEndTime_us": 40.0, "adcStartTime_us": 1.0, "numAdcSamples": 384,
        "digOutSampleRate_MHz": 10.0, "freqSlopeMHz_us": 90.0,
        "frameConfig": {"framePeriodicity_ms": 10.0, "numFrames": 50},
        "dataFmt_rxChanEnMask": 15, "tx_order_masks": [2], "chip_hint": "IWR6843",
        "dataFmt_adcBits": 16, "dataFmt_adcFmt": "Complex1X",
    }
    p.update(over)
    return p


def by_check(findings):
    return {f.check: f for f in findings}


def synthetic_cube(frames=50, chirps=4, rx=4, ns=256, beats=(20, 45, 70), seed=0,
                   imbalance=None, swap=False):
    """Complex ADC samples: a few targets at positive beat bins plus noise."""
    rng = np.random.default_rng(seed)
    n = np.arange(ns)
    sig = sum(np.exp(2j * np.pi * k * n / ns) * (1.0 / (i + 1)) for i, k in enumerate(beats))
    cube = np.broadcast_to(sig, (frames, chirps, rx, ns)).astype(np.complex64).copy()
    cube += 0.01 * (rng.standard_normal(cube.shape) + 1j * rng.standard_normal(cube.shape))
    if imbalance is not None:
        eps, phi = imbalance
        i, q = cube.real, cube.imag
        cube = (i + 1j * (1 + eps) * (q * np.cos(phi) - i * np.sin(phi))).astype(np.complex64)
    if swap:
        cube = np.conj(cube)
    return cube


# --- the report itself -------------------------------------------------------------

def test_severities_are_validated():
    with pytest.raises(ValueError):
        ch.Finding("x", "catastrophic", "nope")


def test_the_worst_finding_sets_the_verdict():
    rep = ch.HealthReport("c", [ch.Finding("a", "ok", ""), ch.Finding("b", "warning", "w"),
                                ch.Finding("c", "info", "")])
    assert rep.worst == "warning"
    assert "1 warning(s)" in rep.summary()
    assert ch.HealthReport("empty").worst == "ok"


def test_only_problems_hides_the_rest():
    rep = ch.HealthReport("c", [ch.Finding("fine", "ok", "all good"),
                                ch.Finding("bad", "error", "broken")])
    text = rep.format(only_problems=True)
    assert "broken" in text and "all good" not in text


def test_a_failing_check_is_reported_not_raised(monkeypatch):
    def boom(_params):
        raise RuntimeError("parser exploded")
    monkeypatch.setattr(ch, "assess_params", boom)
    rep = ch.assess("c", {})
    assert rep.worst == "warning"
    assert "parser exploded" in rep.findings[0].message


# --- what the logs say -------------------------------------------------------------

def test_a_clean_log_raises_nothing():
    f = ch.assess_params(clean_params())
    assert all(x.severity in ("ok", "info") for x in f), [x for x in f if x.severity != "ok"]


def test_a_missing_recorder_log_is_a_warning():
    p = clean_params()
    for k in ("dca_lane_mode", "dca_packets_received"):
        p.pop(k)
    assert by_check(ch.assess_params(p))["recorder log"].severity == "warning"


def test_a_start_time_from_the_api_log_is_a_warning():
    p = clean_params(radar_start_source="API log (first command)")
    f = by_check(ch.assess_params(p))["capture start"]
    assert f.severity == "warning" and "minutes off" in f.message


def test_the_drift_from_the_api_log_is_shown():
    p = clean_params(capture_start_vs_log_s=37.0)
    assert "+37 s" in by_check(ch.assess_params(p))["capture start"].message


def test_a_lane_mismatch_is_explained_not_alarmed():
    p = clean_params(lane_mode_mismatch=True, numLanes_radar=1)
    f = by_check(ch.assess_params(p))["lanes"]
    assert f.severity == "info" and "radar drives 1" in f.message


def test_an_unusable_lane_count_is_an_error():
    assert by_check(ch.assess_params(clean_params(numLanes=3)))["lanes"].severity == "error"


def test_packet_loss_is_a_warning():
    """The figures are test_D's own recorder log."""
    p = clean_params(dca_packets_zero_filled=4525, dca_out_of_sequence=74,
                     dca_packet_loss_pct=5.36)
    f = by_check(ch.assess_params(p))["packet loss"]
    assert f.severity == "warning" and "4525" in f.message and "5.36" in f.message


def test_sampling_past_the_ramp_is_an_error():
    p = clean_params(numAdcSamples=512)          # 51.2 us of sampling on a 40 us ramp
    f = by_check(ch.assess_params(p))["ADC window"]
    assert f.severity == "error" and "after the ramp" in f.message


def test_an_undersampled_ramp_is_pointed_out():
    """test_D: 256 samples at 10 MSps on a 40 us ramp leaves a third unsampled."""
    f = by_check(ch.assess_params(clean_params(numAdcSamples=256)))["ADC window"]
    assert f.severity == "info" and "6.5 cm" in f.message


def test_a_frame_rate_fallback_is_a_warning():
    p = clean_params(frameConfig={"numFrames": 50})
    assert by_check(ch.assess_params(p))["frame rate"].severity == "warning"


def test_a_known_board_resolves_to_its_preset():
    f = by_check(ch.assess_params(clean_params()))["antenna"]
    assert f.severity == "ok" and "AOP" in f.message


def test_an_unknown_board_is_flagged_before_it_gives_wrong_angles():
    f = by_check(ch.assess_params(clean_params(chip_hint="XWR9999")))["antenna"]
    assert f.severity == "warning" and "linear array" in f.message


# --- a prefix of the raw file ------------------------------------------------------

def write_raw(tmp_path, data):
    p = tmp_path / "cap_Raw_0.bin"
    np.asarray(data, dtype=np.int16).tofile(p)
    return str(p)


def test_clean_raw_data_is_ok(tmp_path):
    rng = np.random.default_rng(0)
    path = write_raw(tmp_path, rng.integers(-8000, 8000, 1 << 16))
    f = by_check(ch.assess_raw(clean_params(), path))["ADC saturation"]
    assert f.severity == "ok" and "dBFS" in f.message


def test_clipping_is_caught(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.integers(-8000, 8000, 1 << 16)
    data[::20] = 32767                           # 5 % pinned at full scale
    f = by_check(ch.assess_raw(clean_params(), write_raw(tmp_path, data)))["ADC saturation"]
    assert f.severity == "error"


def test_zero_padded_real_slots_are_explained(tmp_path):
    rng = np.random.default_rng(2)
    data = np.zeros(1 << 16, dtype=np.int16)
    data[0::2] = rng.integers(-500, 500, 1 << 15)
    data[data == 0] = 1
    data[1::2] = 0
    f = by_check(ch.assess_raw(clean_params(dataFmt_adcFmt="Real"), write_raw(tmp_path, data)))
    assert f["REAL container"].severity == "info" and "empty" in f["REAL container"].message


def test_an_empty_binary_is_an_error(tmp_path):
    path = write_raw(tmp_path, [])
    assert ch.assess_raw(clean_params(), path)[0].severity == "error"


# --- the decoded cube --------------------------------------------------------------

def test_a_clean_cube_is_ok():
    f = by_check(ch.assess_cube(clean_params(), synthetic_cube()))
    for name in ("frame count", "RX channels", "DC offset", "IQ balance"):
        assert f[name].severity == "ok", f[name]


@pytest.mark.parametrize("declared, decoded, fps, recorded_s, expected", [
    (250, 249, 25, 12, "ok"),        # recorder stopped mid-frame
    (250, 250, 25, 12, "ok"),
    # REAL samples read out of their container straight through: 20 s of frames from a
    # 12 s recording cannot be right
    (250, 499, 25, 12, "error"),
    # feb26/data_test2: 10 371 frames at 20 fps is 519 s, the recorder ran 61 s
    (1200, 10371, 20, 61, "error"),
    # kleky_dataset1 before the chirp-count fix: 5 s of frames from a 12 s recording
    (100, 50, 10, 12, "warning"),
    # a 4 s capture whose recorder started a couple of seconds before the radar
    (100, 20, 10, 4, "warning"),
    # a recorder left on "Infinite" collects more than numFrames, all of it genuine
    (100, 300, 10, 31, "warning"),
])
def test_the_frame_count_is_judged_against_the_recording(declared, decoded, fps,
                                                         recorded_s, expected):
    f = ch.frame_count_finding(declared, decoded, 2.0, fps=fps, recorded_s=recorded_s)
    assert f.severity == expected, f.message


def test_too_little_data_for_the_recording_does_not_blame_one_cause():
    """Half the recording's length is what frames decoded twice too large look like."""
    f = ch.frame_count_finding(100, 50, 0.5, fps=10, recorded_s=12)
    assert "stopped early" in f.message and "larger than what the radar sent" in f.message


def test_a_short_capture_with_recorder_overhead_is_not_suspected():
    """vitals/data_test4: 2 s of frames in a 4 s recording, all of it the recorder's lead-in."""
    f = ch.frame_count_finding(20, 20, 1.0, fps=10, recorded_s=4)
    assert f.severity == "ok"


def test_without_a_recording_length_a_mismatch_is_only_reported():
    """No recorder log means no way to tell a decode fault from a long recording."""
    f = ch.frame_count_finding(250, 499, 2.0)
    assert f.severity == "warning" and "499 frames" in f.message and "2.00x" in f.message


def test_a_dead_channel_is_an_error():
    cube = synthetic_cube()
    cube[:, :, 2, :] = 0
    f = by_check(ch.assess_cube(clean_params(), cube))["RX channels"]
    assert f.severity == "error" and "RX3" in f.message


def test_a_large_dc_offset_is_a_warning():
    cube = synthetic_cube()
    cube[:, :, 1, :] += 3.0
    f = by_check(ch.assess_cube(clean_params(), cube))["DC offset"]
    assert f.severity == "warning" and "RX2" in f.message


def test_iq_imbalance_shows_as_weak_image_rejection():
    cube = synthetic_cube(imbalance=(0.3, 0.3))
    f = by_check(ch.assess_cube(clean_params(), cube))["IQ balance"]
    assert f.severity == "warning"


def test_swapped_i_and_q_are_an_error():
    f = by_check(ch.assess_cube(clean_params(), synthetic_cube(swap=True)))["IQ balance"]
    assert f.severity == "error" and "swapped" in f.message


def test_image_rejection_matches_the_textbook_figure():
    """For small errors IRR ~ 4 / (eps^2 + phi^2)."""
    eps, phi = 0.1, 0.1
    irr = ch.image_rejection_db(synthetic_cube(imbalance=(eps, phi))[:, :, 0, :])
    assert irr == pytest.approx(10 * np.log10(4 / (eps ** 2 + phi ** 2)), abs=2.0)


def test_a_scene_without_strong_targets_cannot_judge_iq():
    """
    The rect and kleky recordings: nothing stood more than 16 dB above the floor, and
    comparing noise with noise read as swapped I and Q. It must say "not measurable".
    """
    cube = synthetic_cube(beats=())                      # noise only
    f = by_check(ch.assess_cube(clean_params(), cube))["IQ balance"]
    assert f.severity == "info" and "not measurable" in f.message
    assert np.isnan(ch.image_rejection_db(cube[:, :, 0, :]))


def test_a_weak_target_is_not_used_for_iq():
    rng = np.random.default_rng(5)
    n = np.arange(256)
    # Hann-windowed, 256 points: a 0.003 tone clears the median of 0.01 complex noise
    # by ~10 dB - visible, but below the 20 dB a target needs to count
    x = (0.003 * np.exp(2j * np.pi * 40 * n / 256)
         + 0.01 * (rng.standard_normal((200, 256)) + 1j * rng.standard_normal((200, 256))))
    assert np.isnan(ch.image_rejection_db(x))


def test_a_real_cube_skips_the_iq_check():
    cube = synthetic_cube().real.astype(np.float32)
    assert "IQ balance" not in by_check(ch.assess_cube(clean_params(dataFmt_adcFmt="Real"), cube))


def test_the_slope_of_the_dc_spike_is_not_a_target():
    """
    Near-field coupling sits tens of dB above the floor at DC and the window spreads it
    to both sides alike. On five recordings the only bin clearing 20 dB was that slope,
    just past the excluded DC bins, and it read as 5-8 dB of image rejection.
    """
    rng = np.random.default_rng(7)
    n = np.arange(150)
    coupling = 30.0 * np.exp(2j * np.pi * 0.4 * n / 150)        # strong, right at DC
    x = coupling + 0.01 * (rng.standard_normal((200, 150)) + 1j * rng.standard_normal((200, 150)))
    assert np.isnan(ch.image_rejection_db(x))                   # nothing else to judge by


def test_a_real_target_beside_the_dc_slope_still_counts():
    rng = np.random.default_rng(8)
    n = np.arange(150)
    x = (30.0 * np.exp(2j * np.pi * 0.4 * n / 150) + 0.5 * np.exp(2j * np.pi * 40 * n / 150)
         + 0.01 * (rng.standard_normal((200, 150)) + 1j * rng.standard_normal((200, 150))))
    assert ch.image_rejection_db(x) > 20


# --- whether the files belong together ---------------------------------------------

def _touch(path, when):
    import os
    path.write_bytes(b"\x00\x00")
    os.utime(path, (when, when))
    return str(path)


def test_files_written_together_are_ok(tmp_path):
    t = 1_780_000_000
    raw = _touch(tmp_path / "c_Raw_0.bin", t)
    csv = _touch(tmp_path / "c_Raw_LogFile.csv", t + 5)
    f = by_check(ch.assess_files({"dca_csv_path": csv}, [raw]))
    assert f["raw files"].severity == "ok"


def test_raw_data_from_another_day_is_caught(tmp_path):
    """feb26/data_test: log and CSV from 19 June, the .bin from 17 April."""
    t = 1_780_000_000
    raw = _touch(tmp_path / "c.bin", t - 63 * 86400)
    csv = _touch(tmp_path / "c_Raw_LogFile.csv", t)
    f = by_check(ch.assess_files({"dca_csv_path": csv}, [raw]))["raw files"]
    assert f.severity == "error" and "63 day(s) older" in f.message


def test_a_later_part_older_than_the_first_is_caught(tmp_path):
    """feb26/data_test2: part 1 was a day older than part 0."""
    t = 1_780_000_000
    p0 = _touch(tmp_path / "c_Raw_0.bin", t)
    p1 = _touch(tmp_path / "c_1.bin", t - 86400)
    f = ch.assess_files({}, [p0, p1])
    assert f and f[0].severity == "error" and "cannot be the next part" in f[0].message


def test_a_long_capture_split_into_parts_is_fine(tmp_path):
    t = 1_780_000_000
    parts = [_touch(tmp_path / f"c_Raw_{i}.bin", t + 120 * i) for i in range(3)]
    csv = _touch(tmp_path / "c_Raw_LogFile.csv", t + 245)
    assert all(x.severity == "ok" for x in ch.assess_files({"dca_csv_path": csv}, parts))


def test_a_lone_peak_on_the_wrong_side_is_not_a_verdict():
    """
    kleky rlk90s2_4: one peak at bin -34, 20 dB up, on three channels. A spur or a
    reflector folded in from beyond the range looks like that; a swap moves the scene.
    """
    rng = np.random.default_rng(9)
    n = np.arange(150)
    lone = 0.6 * np.exp(-2j * np.pi * 34 * n / 150)             # negative frequency
    x = lone + 0.01 * (rng.standard_normal((40, 4, 4, 150)) + 1j * rng.standard_normal((40, 4, 4, 150)))
    f = by_check(ch.assess_cube(clean_params(frameConfig={"framePeriodicity_ms": 10.0,
                                                          "numFrames": 40}), x))["IQ balance"]
    assert f.severity == "info" and "single peak" in f.message
    # complex 1x spans 0..fs: the far half is range too, not "beyond the range"
    assert "complex 1x" in f.message and "beyond the range" not in f.message
    f2 = by_check(ch.assess_cube(clean_params(frameConfig={"framePeriodicity_ms": 10.0,
                                                           "numFrames": 40},
                                              dataFmt_adcFmt="Complex2X"), x))["IQ balance"]
    assert "beyond the range" in f2.message


def test_one_clean_target_is_enough_to_clear_a_channel():
    rng = np.random.default_rng(10)
    n = np.arange(150)
    x = 0.6 * np.exp(2j * np.pi * 34 * n / 150) + \
        0.01 * (rng.standard_normal((40, 4, 4, 150)) + 1j * rng.standard_normal((40, 4, 4, 150)))
    f = by_check(ch.assess_cube(clean_params(frameConfig={"framePeriodicity_ms": 10.0,
                                                          "numFrames": 40}), x))["IQ balance"]
    assert f.severity == "ok"


# --- noise spectrum ----------------------------------------------------------------

def test_the_noise_floor_is_the_noise_and_the_static_scene_is_not_in_it():
    rng = np.random.default_rng(5)
    sigma = 1e-3
    x = synthetic_cube(frames=20, chirps=8, rx=1, beats=(20, 45))[:, :, 0] * 0     # no noise
    n = np.arange(256)
    x = x + 0.3 * np.exp(2j * np.pi * 30 * n / 256)                                # a wall
    x = x + sigma / np.sqrt(2) * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    s = ch.noise_spectrum(x)
    assert s["floor_db"] == pytest.approx(20 * np.log10(sigma), abs=0.5)
    assert not s["lines"]                                   # the wall cancelled out


def test_chirps_of_the_same_tx_are_what_gets_subtracted():
    """TDM: consecutive chirps come from different TX, so the scene differs between them."""
    rng = np.random.default_rng(6)
    n = np.arange(256)
    x = np.zeros((20, 8, 256), complex)
    x[:, 0::2] += 0.3 * np.exp(2j * np.pi * 30 * n / 256)                   # TX1 sees it,
    x[:, 1::2] += 0.3 * np.exp(2j * np.pi * 30 * n / 256 + 2.0j)            # TX2 at another phase
    x += 1e-3 / np.sqrt(2) * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    assert ch.noise_spectrum(x, stride=2)["floor_db"] == pytest.approx(-60, abs=0.5)
    assert ch.noise_spectrum(x, stride=1)["lines"]            # wrong stride: the wall leaks


def test_interference_not_locked_to_the_chirps_shows_as_a_line():
    rng = np.random.default_rng(7)
    n = np.arange(256)
    x = 1e-3 / np.sqrt(2) * (rng.standard_normal((20, 8, 256)) + 1j * rng.standard_normal((20, 8, 256)))
    x = x + 0.02 * np.exp(2j * np.pi * (77 * n / 256 + rng.random((20, 8, 1))))
    s = ch.noise_spectrum(x)
    assert s["lines"] and s["lines"][0][0] == 77


def test_a_noisy_receiver_is_a_warning():
    cube = synthetic_cube(frames=20, chirps=4, rx=4)
    rng = np.random.default_rng(8)
    cube[:, :, 2] += 0.1 * (rng.standard_normal(cube[:, :, 2].shape)
                            + 1j * rng.standard_normal(cube[:, :, 2].shape))
    f = by_check(ch.noise_findings(clean_params(), cube))["noise floor"]
    assert f.severity == "warning" and "RX3" in f.message
    ok = by_check(ch.noise_findings(clean_params(), synthetic_cube(frames=20, chirps=4)))
    assert ok["noise floor"].severity == "ok"
