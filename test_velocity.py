"""
Tests for velocity - unfolding velocities beyond +-v_max (roadmap 1.4).

Run with:  python -m pytest test_velocity.py -q
"""
import numpy as np
import pytest

import velocity
from test_capon_rae import make_params, make_synthetic_cube

L, S, RX, R, R_BIN = 64, 3, 4, 64, 30


# --- multiple PRFs ------------------------------------------------------------------

@pytest.mark.parametrize("v_true", [3.7, -4.2, 0.3, 4.9, -0.95])
def test_two_prfs_unfold_a_velocity_neither_can_see(v_true):
    vmax = np.array([1.0, 1.25])                         # 4:5 -> unambiguous to +-5 m/s
    v_meas = velocity.wrap(v_true, vmax)
    v, res = velocity.unfold_multi_prf(v_meas, vmax, v_limit=5.0)
    assert v == pytest.approx(v_true, abs=1e-9)
    assert res < 1e-9


def test_unfolding_tolerates_measurement_noise():
    rng = np.random.default_rng(1)
    vmax = np.array([1.0, 1.25, 1.5])
    for v_true in rng.uniform(-4.5, 4.5, 50):
        v_meas = velocity.wrap(v_true + rng.normal(0, 0.02, 3), vmax)
        v, res = velocity.unfold_multi_prf(v_meas, vmax, v_limit=5.0)
        assert v == pytest.approx(v_true, abs=0.05)
        assert res < 0.1


def test_disagreeing_measurements_show_in_the_residual():
    v, res = velocity.unfold_multi_prf([0.5, -0.9], [1.0, 1.25], v_limit=5.0)
    good, res_good = velocity.unfold_multi_prf(velocity.wrap(3.7, np.array([1.0, 1.25])),
                                               [1.0, 1.25], v_limit=5.0)
    assert res > 10 * max(res_good, 1e-3)


def test_fold_candidates_cover_plus_minus_s_vmax_once_each():
    c = velocity.fold_candidates(0.5, 1.0, 3)
    assert sorted(v for _, v in c.values()) == [-1.5, 0.5, 2.5]
    c2 = velocity.fold_candidates(0.5, 1.0, 2)
    assert sorted(v for _, v in c2.values()) == [-1.5, 0.5]


# --- TDM hypotheses -----------------------------------------------------------------

def _frame(k_true, az=20.0, el=-10.0):
    """One range-domain frame (chirps, RX, range) with a target moving k_true loop bins."""
    params = make_params(R)
    cube = make_synthetic_cube([{"r_bin": R_BIN, "az_deg": az, "el_deg": el, "amp": 5.0,
                                 "static": True}], params, F=1, loops=L, R=R, noise=0.01)
    n = np.arange(L * S)
    cube[:, :, :, R_BIN] *= np.exp(2j * np.pi * k_true * n / (L * S))[None, :, None]
    return cube[0], params


def _rd_meta(v_max=3.0):
    dv = 2 * v_max / L
    return {"velocity_axis": (np.arange(L) - L // 2) * dv, "n_fft_doppler": L}


@pytest.mark.parametrize("k_true, fold", [(20.4, 0), (20.4 + L, 1), (-20.4 - L, -1),
                                          (-13.7, 0), (-13.7 + L, 1)])
def test_tdm_hypotheses_find_the_fold(k_true, fold):
    frame, params = _frame(k_true)
    meta = _rd_meta()
    dv = meta["velocity_axis"][1] - meta["velocity_axis"][0]
    k_meas = (k_true + L / 2) % L - L / 2                 # what the Doppler FFT shows
    peak = {"range_idx": R_BIN, "doppler_idx": int(round(k_meas + L // 2)),
            "doppler_pos": k_meas + L // 2, "velocity_ms": k_meas * dv}
    out = velocity.extend_detections([peak], frame, params, meta)[0]
    assert out["fold"] == fold
    assert out["velocity_unfolded_ms"] == pytest.approx(k_true * dv, abs=1e-6)
    # AOP array, RX-anchored score: 3.6 dB worst case without noise (velocity.py)
    assert out["fold_margin_db"] > 3.0


def test_non_tdm_captures_are_left_alone():
    frame, params = _frame(10.0)
    params = {**params, "tx_order_masks": [2, 5, 4]}      # a two-TX chirp: not TDM
    peak = {"range_idx": R_BIN, "doppler_idx": 42, "velocity_ms": 0.9}
    out = velocity.extend_detections([dict(peak)], frame, params, _rd_meta())[0]
    assert "velocity_unfolded_ms" not in out


def test_a_speed_limit_keeps_implausible_folds_out():
    frame, params = _frame(20.4 + L)
    meta = _rd_meta(v_max=3.0)
    dv = meta["velocity_axis"][1] - meta["velocity_axis"][0]
    peak = {"range_idx": R_BIN, "doppler_idx": int(20.4 + L // 2),
            "doppler_pos": 20.4 + L // 2, "velocity_ms": 20.4 * dv}
    out = velocity.extend_detections([dict(peak)], frame, params, meta, v_limit=5.0)[0]
    assert out["fold"] == 0 and not out["fold_decided"]           # 7.9 m/s > 5 m/s
    out = velocity.extend_detections([dict(peak)], frame, params, meta, v_limit=10.0)[0]
    assert out["fold"] == 1 and out["fold_decided"]
