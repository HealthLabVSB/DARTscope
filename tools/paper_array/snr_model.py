"""
Phase-arc SNR against distance and aspect angle - radar-equation model vs measurement -
and the localisation of the chest by range and 2x2 azimuth (config A and B positions of
6 and 7 October 2026; config C sitting_still_0deg_3 as a dwell-time check).

    python -m tools.paper_array.snr_model [--no-fig]

Per recording (the Vital Signs chain's own front end: chirp mean of the frame, range FFT
of max(256, next pow2) points, vitalsigns.auto_bin = the breathing-SNR bin):
  Q_arc   vitalsigns.arc_snr of each RX at the chest bin, median over RX (DARTscope's
          channel-quality number: noise from the top 30 % of the slow-time band);
  Q_th    the same arc numerator (median |z - mean z| / sqrt(ln 2)) over the THERMAL floor
          measured with the same estimator on empty range bins (high-band power within
          1 dB of the median bin, >= 0.5 m from the chest), median over RX;
  arc     the chest's phase swing (5-95 % of the unwrapped phase, best RX) and the radial
          displacement it means, swing * lambda / (4 pi);
  azimuth Bartlett scan (DARTscope convention: y = sum x_n f_n conj(exp(j 2 pi x_n sin az)),
          f_n the AOP preset's element phase) of the chest's dynamic spatial signature,
          the principal eigenvector of the 0-5 Hz covariance of the mean-removed chest bin;
          azimuth cut at elevation 0, and the 2-D (az, el) peak.

Model (amplitude, single channel):
  Q_th = K * sqrt(T_dwell / T_ref) * (R_ref / R)^2 * g(theta) * c(alpha)
  R slant range sqrt(d^2 + o^2); theta = atan(o / d) the azimuth off boresight;
  g(theta) = cos^2(theta): two-way amplitude of an element whose one-way power pattern is
  cos^2 (-6 dB at +-60 deg, TI's stated +-60 deg field of view - ASSUMED, the datasheet
  pattern is not tabulated); c(alpha) = cos(alpha), the radial projection of the chest
  motion, alpha the aspect angle (common.nominal_geometry); T_dwell = chirps of one TX per
  frame * ADC window, T_ref = 5 * 25.6 us (config A). K is fitted (least squares in dB)
  on the 0-deg on-axis recordings only; the other points are predictions.
  Variants reported: R^-2 only; + g(theta); + g(theta) c(alpha).

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np
from scipy import signal as sps

import antenna_config
import vital_bands
import vitalsigns
from radar_processing import range_fft_size
from tools.paper_array import common

T_REF = 5 * 25.6e-6
MISMATCH_M = 0.5                          # breathing bin this far from the nominal slant range
EXTRA = ["sitting_still_0deg_3"]          # config C, all 12 virtual channels


def noise_amp(z):
    zc = z - z.mean()
    p = np.abs(np.fft.fft(zc)) ** 2 / z.size
    hi = np.abs(np.fft.fftfreq(z.size)) > 0.35
    return float(np.sqrt(np.median(p[hi]) / np.log(2.0)))


def arc_num(z):
    return float(np.median(np.abs(z - z.mean())) / np.sqrt(np.log(2.0)))


def dyn_vector(X, fps, fc=5.0):
    Xc = X - X.mean(0)
    b, a = sps.butter(4, fc / (fps / 2))
    Xl = sps.filtfilt(b, a, Xc, axis=0)
    Rm = Xl.T @ Xl.conj() / len(Xl)
    ev, V = np.linalg.eigh(Rm)
    return V[:, -1], float(ev[-1] / ev.sum())


def geometry(params, n_virt):
    cfg = antenna_config.get_config(antenna_config.AOP_DEFAULT)
    masks = list(params.get("tx_order_masks") or [1])
    gx, gy = antenna_config.build_geometry(cfg, masks=masks, num_virt_rx=n_virt, num_phys_rx=4)
    fac = antenna_config.element_phase_factors(cfg, n_virt)
    return np.asarray(gx, float), np.asarray(gy, float), np.asarray(fac, complex)


def bartlett_angles(v, gx, gy, fac):
    az = np.arange(-90.0, 90.01, 0.5)
    el = np.arange(-60.0, 60.01, 1.0)
    a1 = fac[None, :] * np.exp(1j * 2 * np.pi * gx[None, :] * np.sin(np.radians(az))[:, None])
    p1 = np.abs(a1.conj() @ v) ** 2
    A, E = np.meshgrid(np.radians(az), np.radians(el))
    ph = 2 * np.pi * (gx[None, None, :] * (np.sin(A) * np.cos(E))[..., None] + gy[None, None, :] * np.sin(E)[..., None])
    p2 = np.abs(np.einsum("ean,n->ea", (fac[None, None, :] * np.exp(1j * ph)).conj(), v)) ** 2
    e_i, a_i = np.unravel_index(np.argmax(p2), p2.shape)
    return float(az[np.argmax(p1)]), float(az[a_i]), float(el[e_i])


def measure(cap):
    params, cube = common.load_cube(cap)
    fps = common.fps_of(params)
    cfg_lab = common.config_label(params)
    n = int(params.get("numAdcSamples"))
    if cfg_lab == "C":
        from tools.paper_array import tdm_gain
        _, x, _, r, L = tdm_gain.virtual_cube(cube, params)      # (F, 12, R), compensated
        rf = x[:, None]
        n_c_tx = L
    else:
        x = cube.mean(axis=1, keepdims=True)
        rf, r = common.range_fft(x, params, n_fft=max(256, range_fft_size(n, None)))
        n_c_tx = int(cube.shape[1])
    del cube
    F, _, V, R = rf.shape
    bands = vital_bands.from_config({})
    b0 = vitalsigns.auto_bin(rf, fps, bands.resp_hz, r, bands=bands)
    X = rf[:, 0, :, b0]
    hb = np.array([[noise_amp(rf[:, 0, i, b]) ** 2 for b in range(R)] for i in range(V)])
    med = np.median(hb, 0)
    floor = np.median(med[r > 0.3])
    nb = np.flatnonzero((r > 0.3) & (np.abs(r - r[b0]) > 0.5) & (np.abs(10 * np.log10(med / floor)) < 1.0))
    q_arc = [vitalsigns.arc_snr(X[:, i]) for i in range(V)]
    q_th = [arc_num(X[:, i]) / np.median([noise_amp(rf[:, 0, i, j]) for j in nb]) for i in range(V)]
    best = int(np.argmax(q_th))
    lam = common.wavelength_m(params)
    ph = np.unwrap(np.angle(vitalsigns.dc_offset_correction(X[:, best], method="mean")))
    ph = sps.detrend(ph)
    swing = float(np.percentile(ph, 95) - np.percentile(ph, 5))
    v, share = dyn_vector(X, fps)
    gx, gy, fac = geometry(params, V)
    az1, az2, el2 = bartlett_angles(v, gx, gy, fac)
    fs = float(params.get("digOutSampleRate_MHz") or 0) * 1e6
    t_adc = n / fs
    g = common.nominal_geometry(cap.stem)
    if g is None and cap.stem == "sitting_still_0deg_3":
        g = dict(orient=0, d=1.0, o=0.0, slant=1.0, az=0.0, alpha=0.0)
    row = dict(stem=cap.stem, day=cap.start.strftime("%Y-%m-%d"), config=cfg_lab, channels=V,
               rx_gain=params.get("rxGain"), tx_backoff=params.get("txOutPowerBackoffCode"),
               t_dwell_us=n_c_tx * t_adc * 1e6, chest_bin=int(b0), range_meas_m=float(r[b0]),
               bin_spacing_m=float(r[1] - r[0]), n_noise_bins=int(len(nb)),
               chest_floor_above_thermal_db=float(10 * np.log10(med[b0] / floor)),
               Q_arc=float(np.median(q_arc)), Q_th=float(np.median(q_th)),
               Q_th_max=float(np.max(q_th)), arc_swing_rad=swing,
               radial_disp_mm=swing * lam / (4 * np.pi) * 1e3, dyn_rank1_share=share,
               az_cut_deg=az1, az_2d_deg=az2, el_2d_deg=el2, lam_mm=lam * 1e3)
    if g:
        row.update(orient=g["orient"], d_m=g["d"], o_m=g["o"], slant_m=g["slant"],
                   az_nom_deg=g["az"], alpha_deg=g["alpha"])
    return row


def model_terms(slant, az_deg, alpha_deg, t_dwell_us):
    th = math.radians(az_deg)
    al = math.radians(alpha_deg)
    base = 20 * math.log10(1.0 / slant ** 2) + 10 * math.log10(t_dwell_us * 1e-6 / T_REF)
    g = 20 * math.log10(max(math.cos(th) ** 2, 1e-6))
    c = 20 * math.log10(max(math.cos(al), 1e-3))
    return base, g, c


def fit(rows):
    """K per model variant from the 0-deg on-axis config A/B recordings; residuals."""
    for r in rows:
        if "slant_m" not in r:
            continue
        r["range_mismatch"] = int(abs(r["range_meas_m"] - r["slant_m"]) > MISMATCH_M)
        b, g, c = model_terms(r["slant_m"], r["az_nom_deg"], r["alpha_deg"], r["t_dwell_us"])
        r["m_base_db"], r["m_g_db"], r["m_c_db"] = b, g, c
        r["Q_th_db"] = 20 * math.log10(r["Q_th"])
        r["Q_arc_db"] = 20 * math.log10(r["Q_arc"])
    cal = [r for r in rows if r.get("orient") == 0 and r.get("o_m") == 0 and r["config"] in ("A", "B")
           and not r.get("range_mismatch")]
    out = {}
    for name, terms in (("R^-2", ("m_base_db",)), ("R^-2 + pattern", ("m_base_db", "m_g_db")),
                        ("R^-2 + pattern + cos(alpha)", ("m_base_db", "m_g_db", "m_c_db"))):
        k = float(np.mean([r["Q_th_db"] - sum(r[t] for t in terms) for r in cal]))
        res = {}
        for r in rows:
            if "Q_th_db" in r:
                r[f"pred[{name}]"] = k + sum(r[t] for t in terms)
                r[f"resid[{name}]"] = r["Q_th_db"] - r[f"pred[{name}]"]
        for grp, sel in (("0deg on-axis (fit)", lambda r: r.get("orient") == 0 and r.get("o_m") == 0),
                         ("0deg offset", lambda r: r.get("orient") == 0 and r.get("o_m", 0) > 0),
                         ("90deg on-axis", lambda r: r.get("orient") == 90 and r.get("o_m") == 0),
                         ("90deg offset", lambda r: r.get("orient") == 90 and r.get("o_m", 0) > 0)):
            e = [r[f"resid[{name}]"] for r in rows if "Q_th_db" in r and r["config"] in ("A", "B") and sel(r)
                 and not r.get("range_mismatch")]
            if e:
                res[grp] = dict(n=len(e), bias_db=float(np.mean(e)), rms_db=float(np.sqrt(np.mean(np.square(e)))))
        out[name] = dict(K_db=k, n_fit=len(cal), groups=res)
    return out


def style():
    """DARTscope's plot style (theme.style_matplotlib) for print."""
    from tools.paper import dartscope_style
    dartscope_style.apply()


def figure(rows, fits, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    k = fits["R^-2 + pattern"]["K_db"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.165, top=0.98)
    rr = np.linspace(0.85, 2.4, 100)
    ax.plot(rr, [k + model_terms(x, 0, 0, T_REF * 1e6)[0] for x in rr], "-", color="black", lw=1.0,
            label=r"model, on axis ($R^{-2}$)")
    dd = np.linspace(0.8, 2.15, 60)
    g_only, g_cos = [], []
    for d in dd:
        b, g, c = model_terms(math.hypot(d, 1), math.degrees(math.atan(1 / d)),
                              math.degrees(math.atan(1 / d)), T_REF * 1e6)
        g_only.append(k + b + g)
        g_cos.append(k + b + g + c)
    ax.plot(np.hypot(dd, 1), g_only, "--", color="black", lw=1.0, label="model, 1 m offset")
    ax.plot(np.hypot(dd, 1), g_cos, ":", color="black", lw=1.2, label=r"model, 1 m offset, $\times\cos\alpha$")
    from tools.paper import dartscope_style as ds
    blue, orange = ds.BLUE, ds.ORANGE           # the interface's data colours

    def pts(sel, **kw):
        s = [r for r in rows if "Q_th_db" in r and sel(r)]
        if s:
            ax.plot([r["slant_m"] for r in s], [r["Q_th_db"] - 10 * math.log10(r["t_dwell_us"] * 1e-6 / T_REF)
                                               for r in s], ls="none", **kw)
    ok = lambda r: not r.get("range_mismatch")
    pts(lambda r: ok(r) and r["config"] == "A" and r["orient"] == 0 and r["o_m"] == 0, marker="o",
        color=blue, ms=4, label="0°, on axis")
    pts(lambda r: ok(r) and r["config"] == "A" and r["orient"] == 0 and r["o_m"] > 0, marker="s",
        color=blue, ms=4, mfc="white", label="0°, 1 m offset")
    pts(lambda r: ok(r) and r["config"] == "A" and r["orient"] == 90 and r["o_m"] == 0, marker="^",
        color=orange, ms=4.5, label="90°, on axis")
    pts(lambda r: ok(r) and r["config"] == "A" and r["orient"] == 90 and r["o_m"] > 0, marker="v",
        color=orange, ms=4.5, mfc="white", label="90°, 1 m offset")
    pts(lambda r: ok(r) and r["config"] == "B", marker="D", color="0.4", ms=3.5, label="0°, on axis, config B")
    pts(lambda r: r["config"] == "C", marker="*", color="0.1", ms=6.5, label="0°, config C (63 chirps)")
    pts(lambda r: not ok(r), marker="x", color="0.5", ms=4.5, mew=0.9, label="bin > 0.5 m off nominal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Slant range (m)")
    ax.set_ylabel(r"$Q_\mathrm{th}$ per 128-$\mu$s dwell (dB)")
    ax.legend(loc="upper right", ncol=2, handletextpad=0.3, labelspacing=0.2, columnspacing=0.8,
              borderaxespad=0.1, fontsize=6.5)
    ax.margins(x=0.04, y=0.06)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 9)
    fig.savefig(path)
    fig.savefig(os.path.join(common.OUT_DIR, "fig_snr_preview.png"), dpi=200)
    plt.close(fig)


def localization_summary(rows):
    """Per nominal position: measured breathing-bin range and 2x2 azimuth against nominal.
    DARTscope's azimuth is positive to the operator's left, so a subject 'right' is negative."""
    groups = {}
    for r in rows:
        if "slant_m" not in r:
            continue
        key = (r["orient"], r["d_m"], r["o_m"], r["config"])
        groups.setdefault(key, []).append(r)
    out = []
    for (orient, d, o, cfg), sel in sorted(groups.items()):
        good = [r for r in sel if not r["range_mismatch"]]
        az_nom = -math.degrees(math.atan2(o, d))
        row = dict(orient_deg=orient, d_m=d, o_m=o, config=cfg, n=len(sel), n_range_mismatch=len(sel) - len(good),
                   slant_nom_m=math.hypot(d, o), az_nom_deg=az_nom)
        if good:
            rm = np.array([r["range_meas_m"] for r in good])
            az = np.array([r["az_cut_deg"] for r in good])
            row.update(range_meas_mean_m=float(rm.mean()), range_err_mean_m=float(rm.mean() - math.hypot(d, o)),
                       range_meas_min_m=float(rm.min()), range_meas_max_m=float(rm.max()),
                       az_meas_mean_deg=float(az.mean()), az_err_mean_deg=float(az.mean() - az_nom),
                       az_meas_min_deg=float(az.min()), az_meas_max_deg=float(az.max()),
                       el_2d_mean_deg=float(np.mean([r["el_2d_deg"] for r in good])))
        row["mismatched"] = ";".join(f"{r['stem']} ({r['range_meas_m']:.2f} m, {r['az_cut_deg']:+.1f} deg)"
                                     for r in sel if r["range_mismatch"])
        out.append(row)
    return out


def dwell_check(rows):
    """Config C (63 chirps of one TX averaged) against config A at the same nominal
    position: realised vs predicted coherent-integration gain."""
    c = [r for r in rows if r["config"] == "C"]
    a = [r for r in rows if r["config"] == "A" and r.get("orient") == 0 and r.get("o_m") == 0
         and r.get("d_m") == 1.0 and not r.get("range_mismatch")]
    if not c or not a:
        return {}
    qa = float(np.median([r["Q_th"] for r in a]))
    return dict(Q_th_C=c[0]["Q_th"], Q_th_A_median=qa, n_A=len(a),
                dwell_C_us=c[0]["t_dwell_us"], dwell_A_us=a[0]["t_dwell_us"],
                gain_measured_db=float(20 * math.log10(c[0]["Q_th"] / qa)),
                gain_predicted_db=float(10 * math.log10(c[0]["t_dwell_us"] / a[0]["t_dwell_us"])))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args(argv)
    caps = [c for d in ("2026-10-06", "2026-10-07") for c in common.captures(d)
            if (common.nominal_geometry(c.stem) and common.config_label(c.params) in ("A", "B"))
            or c.stem in EXTRA]
    rows = []
    for c in caps:
        r = measure(c)
        rows.append(r)
        print(f"{c.stem:34s} {r['config']} R={r['range_meas_m']:.2f} (nom {r.get('slant_m', float('nan')):.2f}) "
              f"az={r['az_cut_deg']:+.1f}/{r['az_2d_deg']:+.1f} el={r['el_2d_deg']:+.0f} "
              f"(nom {r.get('az_nom_deg', float('nan')):.1f}) Qarc={r['Q_arc']:.0f} Qth={r['Q_th']:.0f} "
              f"swing={r['arc_swing_rad']:.1f} rad", flush=True)
    fits = fit(rows)
    os.makedirs(common.OUT_DIR, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(common.OUT_DIR, "snr_localization.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    import json
    with open(os.path.join(common.OUT_DIR, "snr_model_fit.json"), "w", encoding="utf-8") as f:
        json.dump(fits, f, indent=1)
    loc = localization_summary(rows)
    with open(os.path.join(common.OUT_DIR, "localization_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in loc for k in r)))
        w.writeheader()
        w.writerows(loc)
    dw = dwell_check(rows)
    fits["dwell_check"] = dw
    with open(os.path.join(common.OUT_DIR, "snr_model_fit.json"), "w", encoding="utf-8") as f:
        json.dump(fits, f, indent=1)
    print(json.dumps(fits, indent=1))
    for r in loc:
        print(r)
    if not args.no_fig:
        figure(rows, fits, os.path.join(common.FIG_DIR, "fig_snr.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
