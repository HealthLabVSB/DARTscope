"""
TDM-MIMO evidence: the 12-element virtual array of the IWR6843AOP against the 4 RX of
one transmitter, on the 3TX recordings (config C: 63 loops x TX masks 1, 2, 4).

    python -m tools.paper_array.tdm_gain [--stems ...]

Seated recordings (sitting_still_0deg_3, new_HR_seated_test):
  * range FFT (DARTscope's), chirp c of a frame -> TX slot c % 3, virtual channel
    slot * 4 + rx; inter-slot motion phase removed with AOA.compensate_tdm_doppler,
    then the mean over the 63 loops (the chain's own chirp mean);
  * chest bin: vitalsigns.auto_bin over the 12 channels;
  * Q_arc = vitalsigns.arc_snr (noise from the top 30 % of the slow-time band) and
    Q_th = the same arc numerator over the THERMAL floor, measured with the same weights
    on empty range bins (high-band power within 1 dB of the median bin);
  * combiners: one channel; the 4 RX of one TX; all 12 - each with the principal
    eigenvector of the 0-5 Hz covariance of the chest bin (max-output-power weights,
    optimal for white noise). Weights are fitted on one half of the record and Q is
    measured on the other half (and vice versa), so the selection of the weights cannot
    inflate the gain;
  * HR/BR: the default Vital Signs chain (vitalsigns.DEFAULT_CONFIG) fed with (a) the 4
    RX of each TX, (b) the 12 virtual channels as receivers, (c) the coherent 12-channel
    beam - against the SOMNOtouch reference of tools.vitals_check.

Walking recordings (HR_moving*): per frame the strongest moving bin; its Doppler-peak
vector across the 12 channels with and without the inter-slot phase; coherence of the
12-element Bartlett beam (peak |a^H x|^2 / (N |x|^2)) and the azimuth against the one
from the 4 RX of TX mask 1 alone (which has no inter-slot term).

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy import signal as sps

import AOA
import antenna_config
import spectral
import vital_bands
import vitalsigns
from tools import vitals_check
from tools.paper_array import common

SEATED = ["sitting_still_0deg_3", "new_HR_seated_test"]
WALKING = ["HR_moving", "HR_moving_3", "HR_moving_4"]
SETS = {"TX m1 (4 RX)": [0, 1, 2, 3], "TX m2 (4 RX)": [4, 5, 6, 7],
        "TX m4 (4 RX)": [8, 9, 10, 11], "12 virtual": list(range(12))}
LP_HZ = 5.0
NOISE_DB_TOL = 1.0


def aop_geometry():
    cfg = antenna_config.get_config(antenna_config.AOP_DEFAULT)
    gx, gy = antenna_config.build_geometry(cfg, masks=[1, 2, 4], num_virt_rx=12, num_phys_rx=4)
    fac = antenna_config.element_phase_factors(cfg, 12)
    return np.asarray(gx, float), np.asarray(gy, float), np.asarray(fac, complex)


def steering(gx, gy, fac, az_deg, el_deg):
    """(n_el, n_az, N) steering vectors incl. the preset's element phase."""
    a = np.radians(np.asarray(az_deg))[None, :, None]
    e = np.radians(np.asarray(el_deg))[:, None, None]
    ph = 2 * np.pi * (gx[None, None, :] * np.sin(a) * np.cos(e) + gy[None, None, :] * np.sin(e))
    return fac[None, None, :] * np.exp(1j * ph)


def noise_amp(z):
    """arc_snr's own noise estimate (amplitude per sample)."""
    zc = z - z.mean()
    p = np.abs(np.fft.fft(zc)) ** 2 / z.size
    hi = np.abs(np.fft.fftfreq(z.size)) > 0.35
    return float(np.sqrt(np.median(p[hi]) / np.log(2.0)))


def lowpass_cov(X, fps):
    Xc = X - X.mean(0)
    b, a = sps.butter(4, LP_HZ / (fps / 2))
    Xl = sps.filtfilt(b, a, Xc, axis=0)
    return Xl.T @ Xl.conj() / len(Xl)                    # E[x x^H]


def pe_weights(X, fps):
    w = np.linalg.eigh(lowpass_cov(X, fps))[1][:, -1]
    return w


def q_arc(X, w):
    return vitalsigns.arc_snr(X @ w.conj())


def q_th(X, XN, w):
    y = X @ w.conj()
    num = np.median(np.abs(y - y.mean())) / np.sqrt(np.log(2.0))
    den = np.median([noise_amp(XN[:, :, j] @ w.conj()) for j in range(XN.shape[2])])
    return float(num / den)


def db20(a, b):
    return float(20 * np.log10(a / b))


# --- seated ------------------------------------------------------------------------------------

def virtual_cube(cube, params, keep_loops=False):
    F, C, RX, S = cube.shape
    n_slot = 3
    L = C // n_slot
    kd = []

    def reduce(rf):
        f, c, rx, R = rf.shape
        v = rf.reshape(f, L, n_slot, rx, R).reshape(f, L, n_slot * rx, R)
        vc, k = AOA.compensate_tdm_doppler(v, n_slot, rx)
        kd.append(k)
        return np.stack([v.mean(1), vc.mean(1)], 1)         # (f, 2, 12, R)

    out, r = common.range_fft(cube, params, reduce=reduce)
    return out[:, 0], out[:, 1], np.concatenate(kd, 0), r, L


def seated(stem, rows_q, rows_hr, summary):
    cap = common.capture(stem)
    params, cube = common.load_cube(cap)
    fps = common.fps_of(params)
    damaged = vitalsigns.detect_damaged_frames(cube)
    x_raw, x_cmp, kdop, r, L = virtual_cube(cube, params)
    del cube
    F, V, R = x_cmp.shape
    bands = vital_bands.from_config({})
    b0 = vitalsigns.auto_bin(x_cmp[:, None], fps, bands.resp_hz, r, bands=bands)
    phi = np.degrees(np.abs(2 * np.pi * kdop[:, b0] / (L * 3)))
    # thermal floor: empty range bins
    hb = np.array([[noise_amp(x_cmp[:, i, b]) ** 2 for b in range(R)] for i in range(V)])
    med = np.median(hb, 0)
    floor = np.median(med[r > 0.3])
    nb = np.flatnonzero((r > max(1.6, r[b0] + 0.5)) & (np.abs(10 * np.log10(med / floor)) < NOISE_DB_TOL))
    X, XN = x_cmp[:, :, b0], x_cmp[:, :, nb]
    # spatial noise correlation, chest bin vs empty bin
    bh, ah = sps.butter(6, 0.7, "high")

    def mean_abs_corr(Z):
        Zh = sps.filtfilt(bh, ah, Z - Z.mean(0), axis=0)
        c = np.corrcoef(Zh.T)
        return float(np.abs(c[~np.eye(len(c), dtype=bool)]).mean())
    s = dict(stem=stem, fps=fps, frames=F, chest_bin=int(b0), chest_range_m=float(r[b0]),
             n_noise_bins=int(len(nb)),
             chest_floor_above_thermal_db=float(10 * np.log10(med[b0] / floor)),
             noise_corr_chest=mean_abs_corr(X), noise_corr_empty=mean_abs_corr(XN[:, :, len(nb) // 2]),
             tdm_phase_per_slot_deg_p50=float(np.percentile(phi, 50)),
             tdm_phase_per_slot_deg_p99=float(np.percentile(phi, 99)))
    # eigen-structure of the chest's dynamic (0-5 Hz) covariance
    for nm, idx in SETS.items():
        ev = np.sort(np.linalg.eigvalsh(lowpass_cov(X[:, idx], fps)))[::-1]
        s[f"eig1_share[{nm}]"] = float(ev[0] / ev.sum())
        s[f"eig1[{nm}]"] = float(ev[0])
    s["white_noise_gain_12_vs_4_db"] = {nm: float(10 * np.log10(s["eig1[12 virtual]"] / s[f"eig1[{nm}]"]))
                                        for nm in list(SETS)[:3]}
    # split-half Q
    h = F // 2
    halves = [(slice(0, h), slice(h, F)), (slice(h, F), slice(0, h))]
    for hi_, (tr, te) in enumerate(halves):
        for nm, idx in SETS.items():
            w = pe_weights(X[tr][:, idx], fps)
            Xt, XNt = X[te][:, idx], XN[te][:, idx]
            singles_arc = [vitalsigns.arc_snr(Xt[:, i]) for i in range(len(idx))]
            singles_th = [q_th(Xt, XNt, np.eye(len(idx))[i]) for i in range(len(idx))]
            rows_q.append(dict(stem=stem, test_half=hi_, set=nm, n=len(idx),
                               Q_arc_comb=q_arc(Xt, w), Q_th_comb=q_th(Xt, XNt, w),
                               Q_arc_single_median=float(np.median(singles_arc)),
                               Q_th_single_median=float(np.median(singles_th)),
                               Q_arc_single_max=float(np.max(singles_arc)),
                               Q_th_single_max=float(np.max(singles_th))))
    # full-record per-channel values
    s["Q_arc_channels"] = [float(vitalsigns.arc_snr(X[:, i])) for i in range(V)]
    s["Q_th_channels"] = [q_th(X, XN, np.eye(V)[i]) for i in range(V)]
    # Doppler compensation effect on the full-record numbers
    w12 = pe_weights(X, fps)
    w12r = pe_weights(x_raw[:, :, b0], fps)
    s["Q_th_12_full_compensated"] = q_th(X, XN, w12)
    s["Q_th_12_full_uncompensated"] = q_th(x_raw[:, :, b0], x_raw[:, :, nb], w12r)
    # HR / BR through the default chain
    ref = vitals_check.load_reference(cap, os.path.join(common.DATA_DIR, "somno_vysledky"))
    cfg = {**vitalsigns.DEFAULT_CONFIG, "damaged_frames": damaged}
    variants = {nm: x_cmp[:, None, idx, :] for nm, idx in list(SETS.items())[:3]}
    variants["12 virtual as receivers"] = x_cmp[:, None, :, :]
    variants["12 virtual coherent beam"] = np.einsum("fvr,v->fr", x_cmp, w12.conj())[:, None, None, :]
    for nm, rf in variants.items():
        with common.quiet():
            res = vitalsigns.process_vital_signs_logic(rf, params, fps, r, cfg)
        row = dict(stem=stem, variant=nm, dist_m=float(res["dist_m"]))
        if ref is not None:
            out = vitals_check.compare(res, ref, cfg["win_br"], cfg["win_hr"])
            for k in ("hr_pulse", "hr_beats", "br"):
                st = out[k]
                row[f"{k}_n"] = st.get("n", 0)
                row[f"{k}_mae"] = st.get("mae", np.nan)
                row[f"{k}_bias"] = st.get("bias", np.nan)
            row["hr_coverage"] = out["hr_coverage"]
            row["br_coverage"] = out["br_coverage"]
        rows_hr.append(row)
    summary[stem] = s
    return s


# --- walking -----------------------------------------------------------------------------------

def walking(stem, rows_w, summary, az_grid=np.arange(-80, 80.5, 1.0), el_grid=np.arange(-40, 40.5, 2.0)):
    cap = common.capture(stem)
    params, cube = common.load_cube(cap)
    fps = common.fps_of(params)
    gx, gy, fac = aop_geometry()
    A12 = steering(gx, gy, fac, az_grid, el_grid)            # (E, A, 12)
    A4 = A12[:, :, :4]
    F, C, RX, S = cube.shape
    L = C // 3
    t_slot = np.repeat(np.arange(3), 4)
    res = []

    def reduce(rf):
        f, c, rx, R = rf.shape
        v = rf.reshape(f, L, 3, rx, R).reshape(f, L, 12, R)
        dyn = v - v.mean(1, keepdims=True)                    # zero-Doppler (static) out
        en = np.sum(np.abs(dyn) ** 2, axis=(1, 2))           # (f, R)
        win = np.hanning(L + 2)[1:-1][None, :, None]
        for i in range(f):
            b = int(np.argmax(np.where(rmask, en[i], 0)))
            z = dyn[i, :, :, b] * win[0]                      # (L, 12)
            spec = np.fft.fft(z, axis=0)
            k = spectral.circular_peak(np.sum(np.abs(spec) ** 2, 1)[:, None], axis=0)[0]
            kk = int(np.round(k)) % L
            xv = spec[kk]                                    # Doppler-peak vector (12,)
            ks = k if k <= L / 2 else k - L
            phi = 2 * np.pi * ks / (L * 3)                   # inter-slot phase
            xc = xv * np.exp(-1j * t_slot * phi)
            out = []
            for x, Am in ((xv[:4], A4), (xv, A12), (xc, A12)):
                p = np.abs(np.einsum("ean,n->ea", Am.conj(), x)) ** 2
                e_i, a_i = np.unravel_index(np.argmax(p), p.shape)
                out.append((az_grid[a_i], el_grid[e_i], p[e_i, a_i] / (len(x) * np.sum(np.abs(x) ** 2))))
            v_ms = ks / L * (1 / (3 * Tc)) * lam / 2
            res.append((float(rax[b]), float(v_ms), float(np.degrees(phi)), *out[0], *out[1], *out[2],
                        float(en[i, b])))
        return np.zeros((f, 1), np.float32)

    lam = common.wavelength_m(params)
    Tc = float(params.get("chirpRepetitionTime_us")) * 1e-6 / 3    # one slot
    # the range axis first (one chunk), then the frames
    _, rax = common.range_fft(cube[:1], params)
    rmask = (rax > 0.5) & (rax < 3.4)
    common.range_fft(cube, params, reduce=reduce)
    del cube
    a = np.array(res)
    cols = ["range_m", "v_m_s", "slot_phase_deg", "az4", "el4", "coh4", "az12_raw", "el12_raw",
            "coh12_raw", "az12_comp", "el12_comp", "coh12_comp", "energy"]
    strong = a[:, 12] > np.percentile(a[:, 12], 50)
    fast = strong & (np.abs(a[:, 1]) > 0.3)
    s = dict(stem=stem, frames=len(a), frames_fast=int(fast.sum()),
             v_abs_p50=float(np.median(np.abs(a[strong, 1]))),
             v_abs_p90=float(np.percentile(np.abs(a[strong, 1]), 90)),
             slot_phase_abs_p90_deg=float(np.percentile(np.abs(a[strong, 2]), 90)))
    for tag, sel in (("strong", strong), ("fast", fast)):
        if sel.sum() < 5:
            continue
        s[f"coh12_raw_median[{tag}]"] = float(np.median(a[sel, 8]))
        s[f"coh12_comp_median[{tag}]"] = float(np.median(a[sel, 11]))
        s[f"az_err_raw_vs_4rx_median_deg[{tag}]"] = float(np.median(np.abs(a[sel, 6] - a[sel, 3])))
        s[f"az_err_comp_vs_4rx_median_deg[{tag}]"] = float(np.median(np.abs(a[sel, 9] - a[sel, 3])))
    for row in a:
        rows_w.append(dict(stem=stem, **{c: float(v) for c, v in zip(cols, row)}))
    summary[stem] = s
    return s


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seated", nargs="*", default=SEATED)
    ap.add_argument("--walking", nargs="*", default=WALKING)
    args = ap.parse_args(argv)
    os.makedirs(common.OUT_DIR, exist_ok=True)
    rows_q, rows_hr, rows_w, summary = [], [], [], {}
    for st in args.seated:
        s = seated(st, rows_q, rows_hr, summary)
        print(st, json.dumps({k: v for k, v in s.items() if not isinstance(v, list)}, default=float)[:900])
    for st in args.walking:
        s = walking(st, rows_w, summary)
        print(st, json.dumps(s, default=float))
    write_csv(os.path.join(common.OUT_DIR, "tdm_q_splithalf.csv"), rows_q)
    write_csv(os.path.join(common.OUT_DIR, "tdm_hr_br.csv"), rows_hr)
    write_csv(os.path.join(common.OUT_DIR, "tdm_walking_frames.csv"), rows_w)
    with open(os.path.join(common.OUT_DIR, "tdm_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
