"""
Tests for the heart and breathing rates of the HR and AI tabs.

A chest is synthesised breathing and beating at known rates in one range bin of several
receivers (each with its own phase offset and noise), with a static reflection in it,
and both tabs have to recover the rates the way they run by default.

Run with:  python -m pytest test_hr_ai_tabs.py -q
"""
import json

import numpy as np
import pytest

import hr_analysis
from ai_vital_analysis import AIVitalAnalysis

BR_BPM, HR_BPM = 15.0, 72.0
LAM = 3e8 / 60.79e9


def chest_rx(fps=20.0, dur=90.0, rx=4, noise=0.02, static=3.0, seed=0):
    """(frames, rx) complex signals of one range bin: chest + static reflection + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fps * dur)) / fps
    disp = 4e-3 * np.sin(2 * np.pi * BR_BPM / 60 * t) + 0.3e-3 * np.sin(2 * np.pi * HR_BPM / 60 * t)
    chest = np.exp(1j * 4 * np.pi * disp / LAM)
    offs = np.exp(1j * rng.uniform(0, 2 * np.pi, rx))
    x = chest[:, None] * offs[None, :] + static * np.exp(1j * rng.uniform(0, 2 * np.pi, rx))[None, :]
    return x + noise * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)), t


def test_a_peak_at_the_bottom_of_the_band_is_found():
    """find_peaks on the band alone never returned its first bin: a 6 /min breathing peak
    at the bottom of the 5-40 /min band went missing and a weaker one higher up won."""
    an = AIVitalAnalysis()
    f = np.linspace(0, 1, 1001)
    p = 10.0 * np.exp(-0.5 * ((f - 0.1) / 0.01) ** 2) + 2.0 * np.exp(-0.5 * ((f - 0.5) / 0.01) ** 2)
    _, bpm, _ = an.find_best_peak(f, p, (0.095, 0.667))          # the band starts just below 6 /min
    assert bpm == pytest.approx(6.0, abs=0.2)
    k = hr_analysis.band_peak_index(f, p, (f >= 0.095) & (f <= 0.667))
    assert f[k] == pytest.approx(0.1, abs=0.002)


def test_the_ai_trends_follow_the_heart_and_sum_the_receivers_for_breathing():
    an = AIVitalAnalysis()
    x, t = chest_rx()
    d = an.centred_displacements(x, LAM)
    trends = an.estimate_trends(d.mean(axis=0), 20.0, 30.0, 90.0, rx_disps=d)
    hr = np.array([w['hr_bpm'] for w in trends if w['valid']])
    rr = np.array([w['rr_bpm'] for w in trends])
    assert hr.size >= 0.8 * len(trends)
    assert np.median(hr) == pytest.approx(HR_BPM, abs=1.5)
    assert np.median(rr) == pytest.approx(BR_BPM, abs=1.0)
    # one bin's trends straight from its receivers' signals (Use Selected Windows)
    tr = an.bin_trends(x, 20.0, 30.0, 90.0, LAM)
    assert np.median([w['hr_bpm'] for w in tr]) == pytest.approx(HR_BPM, abs=1.5)


def test_fused_displacement_centres_every_receiver():
    """With the static reflection left in, each receiver's phase is a short arc around it
    and the heart all but vanishes; the fusion centres every arc first."""
    an = AIVitalAnalysis()
    x, t = chest_rx(static=20.0)
    fused, kind = an.fuse_signals(x, 20.0, "weighted displacement")
    assert kind == "phase"
    d = fused * LAM / (4 * np.pi)
    d = d - d.mean()
    truth = 4e-3 * np.sin(2 * np.pi * BR_BPM / 60 * t)
    assert np.corrcoef(d, truth)[0, 1] > 0.95


@pytest.fixture
def quiet_dialogs(monkeypatch):
    """The message boxes the tabs open, recorded instead of shown."""
    from tkinter import messagebox
    shown = []
    for name in ("showinfo", "showwarning", "showerror"):
        monkeypatch.setattr(messagebox, name, lambda *a, _n=name, **k: shown.append((_n,) + a))
    return shown


def test_the_hr_tab_finds_the_person_and_follows_the_heart(tk_root, monkeypatch, tmp_path):
    """Bin 0 (the antenna coupling) is replaced by the range that breathes most, and the
    result is the heart track over the receivers' mean displacement - also in the export."""
    import tkinter as tk
    import hr_gui
    root = tk_root
    fps, R = 20.0, 32
    x, _ = chest_rx(fps)
    rng = np.random.default_rng(5)
    cube = 0.02 * (rng.standard_normal((x.shape[0], 1, x.shape[1], R))
                   + 1j * rng.standard_normal((x.shape[0], 1, x.shape[1], R)))
    cube[:, 0, :, 0] += 30.0                                        # the coupling at 0 m
    cube[:, 0, :, 20] += x                                          # the chest at 1.0 m
    r_axis = np.arange(R) * 0.05
    rp = np.mean(np.abs(cube), axis=(0, 1, 2))
    captured = {}
    frame = tk.Frame(root)
    try:
        tab = hr_gui.HRDetectionTab(frame, lambda: (cube, {"effectiveStartFreqGHz": 60.79}, fps,
                                                     r_axis, rp))
        orig = tab.update_main_plots

        def grab(*a, **k):
            captured["res"] = a[4]
            captured["trend"] = k.get("trend_data", a[9] if len(a) > 9 else None)
            return orig(*a, **k)
        monkeypatch.setattr(tab, "update_main_plots", grab)
        tab.run_detection()
        assert tab.selected_bin.get() == 20
        assert captured["res"]["hr_track_bpm"] == pytest.approx(HR_BPM, abs=1.5)
        valid = [w for w in captured["trend"]["trend"] if w["valid"]]
        assert len(valid) >= 0.8 * len(captured["trend"]["trend"])
        out = tmp_path / "hr.json"
        monkeypatch.setattr(hr_gui.filedialog, "asksaveasfilename", lambda **k: str(out))
        tab.export_json()
        saved = json.loads(out.read_text())
        assert saved["main_result"]["hr_track_bpm"] == pytest.approx(HR_BPM, abs=1.5)
        assert len(saved["heart_track"]["trend"]) == len(captured["trend"]["trend"])
        assert [r for r in saved["all_results"] if r["method"] == "track"]
    finally:
        frame.destroy()


def test_the_synthetic_tests_give_what_they_generate(tk_root, quiet_dialogs):
    """Run Synthetic Test in both tabs: a chest of real size (the phase of 0.1 m of breathing
    was aliased past unwrapping, 0.4 rad around the origin bent by the centring)."""
    import tkinter as tk
    import hr_gui
    import ai_vital_gui
    frame = tk.Frame(tk_root)
    try:
        hr = hr_gui.HRDetectionTab(frame, lambda: None)
        hr.run_synthetic_test()
        rows = {v[0]: float(v[1]) for v in (hr.rx_tree.item(i)["values"] for i in hr.rx_tree.get_children())}
        assert rows["RX1"] == pytest.approx(60, abs=1)
        assert rows["RX2"] == pytest.approx(80, abs=1)
        assert rows["RX3"] == pytest.approx(100, abs=1)
        assert hr.selected_bin.get() == 10
        ai = ai_vital_gui.AIVitalSignsTab(frame, lambda: None)
        ai.run_synthetic_test()
        res = ai.last_ai_result
        assert res["best_range_m"] == pytest.approx(1.5)
        assert res["hr_bpm"] == pytest.approx(72, abs=1)
        assert res["rr_bpm"] == pytest.approx(16, abs=0.5)
        assert all(w["valid"] for w in res["window_results"])
        # Use Selected Windows: the windows of the analysis, not the field changed since
        ai.win_len_var.set(10.0)
        ai.process_selected_windows()
        assert "HR: 72.0 BPM" in ai.selection_result_var.get()
        assert "RR: 16.0 BPM" in ai.selection_result_var.get()
    finally:
        frame.destroy()
