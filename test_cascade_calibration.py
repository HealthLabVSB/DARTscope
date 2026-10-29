"""
Tests for the cascade calibration (cascade/calibration.py): synthetic cascade takes of
point reflectors seen through channels with known errors - a random phase on every TX
and RX, the second chip turned by 120 deg, paths up to 8 mm longer - have to give the
errors back (reflector and autofocus), the reflector somewhere else has to come out at
its angle, and decoding a take has to apply the calibration by itself.

Run with:  python -m pytest test_cascade_calibration.py -q
"""
import json

import numpy as np
import pytest

from cascade import calibration as cc
from cascade.decode import cascade_params
from cascade.params import TDM6, CascadeParams

FRAMES = 24


def write_take(tmp_path, name, tx_chirps=TDM6, start_freq_ghz=77.0, calibration=None):
    """A sidecar like the recorder writes (no data file needed for the params)."""
    cp = CascadeParams.preset("Vital signs - multi-person (TDM 48 ch)")
    cp.start_freq_ghz = start_freq_ghz
    cp.tx_chirps = tx_chirps
    cp.num_loops = 2
    meta = {"params": cp.to_dict(), "started": "2026-10-10 12:00:00",
            "layout": {"tx_chirps": cp.chirps, "num_loops": 2}, "radar": {"num_frames": FRAMES}}
    if calibration is not None:
        meta["calibration"] = calibration
    (tmp_path / f"{name}_cascade.json").write_text(json.dumps(meta))
    params, summary = cascade_params(tmp_path / f"{name}_Raw_0.bin")
    return params, summary


def errors(seed=0, chip_deg=120.0, offset_mm=8.0):
    rng = np.random.default_rng(seed)
    tx = rng.uniform(0.7, 1.3, 6) * np.exp(1j * rng.uniform(0, 2 * np.pi, 6))
    rx = rng.uniform(0.7, 1.3, 8) * np.exp(1j * rng.uniform(0, 2 * np.pi, 8))
    tx[3:] *= np.exp(1j * np.deg2rad(chip_deg))          # the second chip's TX ...
    rx[4:] *= np.exp(1j * np.deg2rad(chip_deg))          # ... and RX
    o_tx = rng.uniform(-offset_mm, offset_mm, 6) * 1e-3
    o_rx = rng.uniform(-offset_mm, offset_mm, 8) * 1e-3
    return tx, rx, o_tx, o_rx


def synth(params, targets, err, noise=1e-3, seed=1, moving=()):
    """
    Cube (frames, chirps, 8 RX, samples) of still point reflectors [(range m, az deg,
    amplitude)] through channels with errors (tx, rx, o_tx, o_rx), in the convention
    of the AoA views (cascade.calibration.steering) and with the paths' extra delay.
    """
    tx_e, rx_e, o_tx, o_rx = err
    masks = params["tx_order_masks"]
    loops = params["frameConfig"]["numLoops"]
    S = params["numAdcSamples"]
    fs = params["digOutSampleRate_MHz"] * 1e6
    slope = params["freqSlopeMHz_us"] * 1e12
    # the carrier phase of a delay is that of the frequency where the sampling starts; the
    # beat tone adds the rest of the sweep, so the range spectrum shows the centre's
    f0 = params["effectiveStartFreqGHz"] * 1e9
    lam = cc.C0 / f0
    n = np.arange(S)
    slot = np.zeros((len(masks), 8, S), complex)
    for s, m in enumerate(masks):
        t = cc.tx_index(m)
        for r in range(8):
            extra = 2 * (o_tx[t] + o_rx[r])                       # round trip of the longer paths
            for R, az, amp in targets:
                a = cc.steering([t], [r], R, az, lam)[0]
                tau = (2 * R + extra) / cc.C0                      # range from the beat
                slot[s, r] += (amp * a * tx_e[t] * rx_e[r] * np.exp(1j * 2 * np.pi * f0 * extra / cc.C0)
                               * np.exp(1j * 2 * np.pi * slope * tau * n / fs))
    rng = np.random.default_rng(seed)
    cube = np.broadcast_to(slot[None, None], (FRAMES, loops) + slot.shape).reshape(
        FRAMES, loops * len(masks), 8, S).copy()
    # moving points - a chest: a phase swinging by +-3 rad over the take
    for R, az, amp in moving:
        echo = synth(params, [(R, az, amp)], err, noise=0.0, seed=seed)[0, :len(masks)]   # one frame
        swing = np.exp(1j * 3.0 * np.sin(2 * np.pi * np.arange(FRAMES) / FRAMES * 3))
        cube += (echo[None, None] * swing[:, None, None, None, None]).reshape(
            FRAMES, 1, len(masks), 8, S).repeat(loops, axis=1).reshape(cube.shape)
    cube += noise * (rng.standard_normal(cube.shape) + 1j * rng.standard_normal(cube.shape))
    return cube.astype(np.complex64)


def phase_error_deg(cal, err, masks, fc_ghz):
    """How far the calibration's channel errors are from the true ones (common phase and
    gain removed), worst channel."""
    tx_e, rx_e, o_tx, o_rx = err
    true = []
    for m in masks:
        t = cc.tx_index(m)
        true.append(tx_e[t] * rx_e * np.exp(1j * 2 * np.pi * fc_ghz * 1e9 * 2 * (o_tx[t] + o_rx) / cc.C0))
    true = np.concatenate(true)
    g = cal.gains(masks, fc_ghz)
    d = np.angle(g / true)
    d = np.angle(np.exp(1j * (d - np.angle(np.mean(np.exp(1j * d))))))
    return float(np.rad2deg(np.max(np.abs(d))))


@pytest.fixture
def store(tmp_path, monkeypatch):
    d = tmp_path / "store"
    monkeypatch.setenv("DARTSCOPE_CASCADE_CALIBRATION_DIR", str(d))
    return d


def test_a_reflector_gives_every_channel_back_and_holds_at_another_angle(tmp_path, store):
    params, _ = write_take(tmp_path, "cal")
    err = errors()
    room = [(2.5, 0.0, 1.0), (4.2, -30.0, 0.05)]
    cal = cc.from_reflector(synth(params, room, err), params, range_m=2.5, az_deg=0.0)
    fc = cc.centre_frequency_ghz(params)
    assert phase_error_deg(cal, err, params["tx_order_masks"], fc) < 2.0
    assert cal.residual_deg < 2.0 and cal.coherence_after > 0.99 > 0.5 > cal.coherence_before
    assert abs(cal.reflector["az_after"]) < 0.3
    # the path offsets, measured from where each channel's peak lies
    assert np.allclose(cal.tx_offset_m - cal.tx_offset_m.mean(), err[2] - err[2].mean(), atol=1.5e-3)
    # verify: the reflector at +25 deg, without and with the calibration
    v = cc.verify(synth(params, [(2.5, 25.0, 1.0)], err), params, cal, range_m=2.5, az_deg=25.0)
    assert v["error_after_deg"] < 0.3
    assert v["coherence_before"] < 0.3 < 0.99 < v["coherence_after"]   # scattered, then one beam


def test_the_near_field_needs_the_exact_paths(tmp_path, store):
    """At 1.2 m, inside the 2.7 m far field of the aperture, a plane wave would be off."""
    params, _ = write_take(tmp_path, "near")
    err = errors(seed=2)
    cal = cc.from_reflector(synth(params, [(1.2, 0.0, 1.0)], err), params, range_m=1.2)
    assert phase_error_deg(cal, err, params["tx_order_masks"], cc.centre_frequency_ghz(params)) < 2.0
    lam = cc.C0 / (cc.centre_frequency_ghz(params) * 1e9)
    t = np.repeat(np.arange(6), 8)
    r = np.tile(np.arange(8), 6)
    plane = np.exp(1j * 2 * np.pi * cc.virtual_x(t, r) * np.sin(0.0))
    exact = cc.steering(t, r, 1.2, 0.0, lam)
    assert np.rad2deg(np.ptp(np.angle(exact / plane))) > 30      # what a plane wave misses


def test_a_calibration_moves_with_the_frequency(tmp_path, store):
    """Measured at 77 GHz, used at 76 GHz: 1 cm of extra path is 24 deg there."""
    p77, _ = write_take(tmp_path, "f77", start_freq_ghz=77.0)
    p76, _ = write_take(tmp_path, "f76", start_freq_ghz=76.0)
    err = errors(seed=3, offset_mm=10.0)
    cal = cc.from_reflector(synth(p77, [(2.5, 0.0, 1.0)], err), p77, range_m=2.5)
    fc76 = cc.centre_frequency_ghz(p76)
    assert phase_error_deg(cal, err, p76["tx_order_masks"], fc76) < 4.0
    v = cc.verify(synth(p76, [(2.5, -20.0, 1.0)], err), p76, cal, range_m=2.5, az_deg=-20.0)
    assert v["error_after_deg"] < 0.3


def test_one_calibration_serves_the_one_tx_preset(tmp_path, store):
    params, _ = write_take(tmp_path, "cal")
    err = errors(seed=4)
    cal = cc.from_reflector(synth(params, [(2.5, 0.0, 1.0)], err), params, range_m=2.5)
    p1, _ = write_take(tmp_path, "one", tx_chirps="1/0")
    c = cal.coefficients(p1["tx_order_masks"], cc.centre_frequency_ghz(p1))
    assert c.shape == (8,)
    g_true = err[0][0] * err[1] * np.exp(1j * 2 * np.pi * cc.centre_frequency_ghz(p1) * 1e9
                                         * 2 * (err[2][0] + err[3]) / cc.C0)
    d = np.angle(c * g_true)
    assert np.rad2deg(np.ptp(np.angle(np.exp(1j * (d - d[0]))))) < 3.0
    assert cal.coefficients([5]) is None                     # TX1 and TX3 at once: not a product


def test_autofocus_sharpens_a_room_without_a_reflector(tmp_path, store):
    params, _ = write_take(tmp_path, "room")
    err = errors(seed=5)
    room = [(1.4, -35.0, 0.6), (2.1, 12.0, 1.0), (2.9, -8.0, 0.8), (3.6, 40.0, 0.5),
            (4.4, 22.0, 0.4), (5.0, -50.0, 0.3), (5.6, 5.0, 0.3)]
    cal = cc.autofocus(synth(params, room, err), params, anchor_deg=0.0)
    assert cal.method == "autofocus" and cal.autofocus["targets"] >= 3
    assert cal.coherence_after > 0.97 and cal.coherence_before < 0.5      # on echoes not fitted
    # the reflectors at their angles - up to the one shift the autofocus cannot see, which
    # is a shift of sin(angle) (a phase growing evenly across the aperture)
    where = dict((round(r, 1), a) for r, a in cal.autofocus["where"])
    shift = [np.sin(np.deg2rad(where[round(R, 1)])) - np.sin(np.deg2rad(az))
             for R, az, _ in room if round(R, 1) in where]
    assert len(shift) >= 3 and np.ptp(shift) < 0.01
    assert where[2.1] == pytest.approx(0.0, abs=0.2)        # the strongest put at the anchor
    assert "relative" in cal.describe()


def test_autofocus_does_not_spoil_a_person_for_extended_objects(tmp_path, store):
    """
    The lab: still objects that are not points (here a table: 9 points over 20 deg at
    one range, a shelf likewise), a board whose channels are only a few tens of degrees
    off, and a person the raw channels already focus well. A calibration fitted to the
    objects spoilt the person (0.88 -> 0.42 on the real takes): it must be refused.
    """
    params, _ = write_take(tmp_path, "lab")
    rng = np.random.default_rng(8)
    tx = np.exp(1j * np.deg2rad(rng.normal(0, 20, 6)))
    rx = np.exp(1j * np.deg2rad(rng.normal(0, 20, 8)))
    err = (tx, rx, np.zeros(6), np.zeros(8))
    table = [(1.46, az, rng.uniform(0.3, 1.0)) for az in np.linspace(40, 60, 9)]
    shelf = [(3.25, az, rng.uniform(0.2, 0.6)) for az in np.linspace(-10, 10, 7)]
    take = synth(params, table + shelf, err, moving=[(2.0, 1.0, 0.3)])
    with pytest.raises(ValueError, match="spoil|improve|too few"):
        cc.autofocus(take, params)


def test_autofocus_does_not_force_one_echo_into_a_point(tmp_path, store):
    """
    The empty room: extended still objects and nothing moving. Nine takes gave calibrations
    that repeated to 1-2 deg - each from a single echo forced into a point - and were 85 deg
    off; repeating is not being right, and one echo is not a confirmation.
    """
    params, _ = write_take(tmp_path, "empty")
    rng = np.random.default_rng(12)
    tx = np.exp(1j * np.deg2rad(rng.normal(0, 30, 6)))
    rx = np.exp(1j * np.deg2rad(rng.normal(0, 30, 8)))
    err = (tx, rx, np.zeros(6), np.zeros(8))
    room = []
    for r0, a0, a1 in ((0.77, -55, -40), (0.70, 52, 70), (2.4, -48, -35), (3.4, 10, 25)):
        room += [(r0, az, rng.uniform(0.3, 1.0)) for az in np.linspace(a0, a1, 7)]
    with pytest.raises(ValueError, match="single echo|disagree|unfocused|improve|spoil|too few"):
        cc.autofocus(synth(params, room, err), params)


def test_a_person_calibrates_and_is_checked_on_what_else_moves(tmp_path, store):
    params, _ = write_take(tmp_path, "person")
    err = errors(seed=13, chip_deg=40.0, offset_mm=0.0)
    take = synth(params, [(3.0, -20.0, 0.2)], err, moving=[(2.0, 0.0, 1.0)])
    cal = cc.from_person(take, params, az_deg=0.0)
    assert cal.method == "person" and cal.coherence_after > 0.99 > 0.5 > cal.coherence_before
    assert phase_error_deg(cal, err, params["tx_order_masks"], cc.centre_frequency_ghz(params)) < 3.0
    with pytest.raises(ValueError, match="nothing moves|not one thing"):
        cc.from_person(synth(params, [(3.0, -20.0, 0.2)], err), params)


def test_saved_loaded_and_preferred_from_the_take(tmp_path, store):
    params, _ = write_take(tmp_path, "cal")
    err = errors(seed=6)
    cal = cc.from_reflector(synth(params, [(2.5, 0.0, 1.0)], err), params, range_m=2.5)
    assert cc.load_current() is None
    cc.save_current(cal)
    back = cc.load_current()
    assert back.describe() == cal.describe()
    assert np.allclose(back.coefficients([1, 2]), cal.coefficients([1, 2]))
    assert len(list((store / "cascade_calibrations").iterdir())) == 1
    older = cc.CascadeCalibration.from_dict(dict(cal.to_dict(), created="2026-01-01T00:00:00"))
    chosen, where = cc.for_take({"calibration": older.to_dict()})
    assert chosen.created.startswith("2026-01-01") and "take" in where
    chosen, where = cc.for_take({})
    assert chosen.created == cal.created and "current" in where
    cc.forget_current()
    assert cc.for_take({}) == (None, "")


def test_decoding_a_take_applies_the_calibration(tmp_path, store):
    params, summary = write_take(tmp_path, "plain")
    assert "channel_calibration" not in params and any("calibration: none" in s for s in summary)
    err = errors(seed=7)
    cal = cc.from_reflector(synth(params, [(2.5, 0.0, 1.0)], err), params, range_m=2.5)
    params, summary = write_take(tmp_path, "with", calibration=cal.to_dict())
    assert len(params["channel_calibration"]) == 48
    assert params["antenna_config"]["name"].startswith("AWR2243")
    assert params["cascade_calibration"]["method"] == "reflector"
    assert any("recorded with the take" in s for s in summary)


def test_the_measure_tab_shows_and_keeps_the_calibration(tk_root, tmp_path, store, monkeypatch):
    """Not calibrated -> a result saved -> calibrated in sight before a take, and verified."""
    import tkinter as tk
    import cascade_gui
    monkeypatch.setattr(cascade_gui, "_settings_path", lambda: tmp_path / "cascade.json")
    monkeypatch.setattr(cascade_gui, "check_dca1000", lambda: None)
    monkeypatch.setattr(cascade_gui.hardware, "xds110_present", lambda: False)
    params, _ = write_take(tmp_path, "cal")
    err = errors(seed=9)
    cal = cc.from_reflector(synth(params, [(2.5, 0.0, 1.0)], err), params, range_m=2.5)
    top = tk.Toplevel(tk_root)
    try:
        tab = cascade_gui.CascadeTab(top)
        assert "Not calibrated" in tab.cal_lbl.cget("text")
        tab._cal_finished(("reflector", cal), None)                 # a measurement came in
        assert "Not calibrated" in tab.cal_lbl.cget("text")          # ... not the board's yet
        assert "residual" in tab.cal_text.get("1.0", "end")
        tab._cal_save()
        assert tab.cal_lbl.cget("text").startswith("✓ Calibrated: reflector")
        assert cc.load_current().created == cal.created
        v = cc.verify(synth(params, [(2.5, 25.0, 1.0)], err), params, cal, range_m=2.5, az_deg=25.0)
        tab._cal_finished(("verify", v), None)
        assert "verified at +25 deg" in tab.cal_lbl.cget("text")
        assert cc.load_current().verify["az_deg"] == 25.0           # the check is kept with it
        # a new tab finds it by itself
        again = cascade_gui.CascadeTab(top)
        assert again.cal_lbl.cget("text") == tab.cal_lbl.cget("text")
        # and a take recorded now carries it (what start() hands the recorder)
        from cascade.recording import CascadeRecording
        rec = CascadeRecording(again.params(), tmp_path, "t", calibration=again.calibration.to_dict())
        assert rec.calibration["method"] == "reflector"
        tab._cal_forget = lambda: None
    finally:
        top.destroy()
