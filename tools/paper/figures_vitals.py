"""
Figures of the vital-sign case study (T-MTT rules: ~9-10 pt at final size, units in
round brackets, black-and-white readable, vector PDF with rasterised maps).

    python -m tools.paper.figures_vitals TABLES_DIR FIG_DIR [--final-prewhiten 2]

fig_tracking  heart harmonic-sum spectra over the windows with the Viterbi path and the
              PPG reference, default chain vs prewhitened; mean heart-band spectrum
fig_coverage  MAE of the kept windows vs coverage (windows ranked by confidence)
fig_ba        Bland-Altman (repeated measurements), heart and breathing
fig_quality   (a) IQ arcs and displacement of a strong and a weak channel with Q_arc,
              (b) Q_arc of the candidates (range bin x receiver) over time, admitted ones
Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import moving_hr
import vital_bands
import vitalsigns
from tools.paper import chain as ch
from tools.paper import evaluate as ev
from tools.paper import dartscope_style as ds

# the data colours of the interface (theme.PLOT_COLORS)
BLUE = ds.BLUE            # final chain
GREY = ds.GREY            # default chain
ORANGE = ds.ORANGE        # reference
COL1, COL2 = ds.COL1, ds.COL2
# the two sets of recordings as the article names them
SET_NAME = {"October": "dev.", "July": "held-out"}


def style():
    """DARTscope's plot style (theme.style_matplotlib) for print."""
    ds.apply()


def letter(ax, s, dx=-0.16, dy=1.04):
    ds.letter(ax, s, dx, dy)


# --- tracking ------------------------------------------------------------------------------

def fig_tracking(stem, out, prewhiten=2):
    c = ch.load(stem)
    r0 = ch.run(c)
    r1 = ch.run(c, prewhiten=prewhiten)
    ref = np.array([c.ref.hr_beats(t - 15, t + 15) for t in r0.hr_t])
    bands = vital_bands.from_config(vitalsigns.DEFAULT_CONFIG)
    fig = plt.figure(figsize=(COL2, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9], wspace=0.32, left=0.07, right=0.99,
                          bottom=0.18, top=0.9)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, r, title, lab in ((axes[0], r0, "Default chain", "a"),
                              (axes[1], r1, "Prewhitened (acceleration)", "b")):
        spec = r.spec / r.spec.max(axis=1, keepdims=True)
        ax.imshow(spec.T, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1,
                  extent=[r.hr_t[0] - 0.5, r.hr_t[-1] + 0.5, r.grid[0], r.grid[-1]], rasterized=True)
        br = np.interp(r.hr_t, r.rr_t, r.rr_v)
        for m in range(6, 21):
            hm = m * br
            if hm.max() < 50 or hm.min() > 115:
                continue
            ax.plot(r.hr_t, hm, color="white", lw=0.4, ls=":", alpha=0.6, zorder=1)
        ax.plot(r.hr_t, ref, color=ORANGE, lw=1.6, label="PPG beats (reference)")
        # as the HR view: the tracked rate dotted in every window, solid where valid
        ok = r.hr_c >= ev.CONF_TH
        ax.plot(r.hr_t, r.hr_v, color="white", lw=0.8, ls=":")
        ax.plot(r.hr_t, np.where(ok, r.hr_v, np.nan), color="white", lw=1.8,
                label="heart rate (valid)")
        ax.set_ylim(50, 115)
        ax.set_xlabel("Time (s)")
        ax.set_title(title, loc="left")
        letter(ax, lab)
        e = r.hr_v - ref
        ax.text(0.98, 0.04, f"MAE {np.nanmean(np.abs(e)):.1f} bpm", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=ds.TEXT,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    axes[0].set_ylabel("Heart rate (bpm)")
    axes[1].legend(loc="upper right", fontsize=7, handlelength=1.8, borderaxespad=0.2, framealpha=0.6)
    # (c) mean heart-band spectrum of the displacement, raw and prewhitened
    ax = axes[2]
    d = r0.d_heart - r0.d_heart.mean()
    fs = c.fps
    x = moving_hr._bandpass(d, fs, max(0.5, bands.heart_hz[0] - 0.3), bands.heart_hz[1] + 1.0)
    wins = moving_hr._windows(len(d), fs, 30.0, 5.0)
    f = np.linspace(0.6, 2.6, 400)
    for k, (lab, ls, col) in enumerate((("displacement", "-", GREY),
                                        ("acceleration", "-", ds.RED))):
        y = x.copy()
        for _ in range(k * prewhiten):
            y = np.gradient(y) * fs
        P = np.mean([moving_hr._power(y[a:b], fs, f) for a, b in wins], axis=0)
        ax.plot(f * 60, 10 * np.log10(P / P.max()), ls=ls, color=col, label=lab)
    ax.grid(True, alpha=0.3)
    ax.axvline(np.nanmedian(ref), color=ORANGE, lw=1.2)
    ax.text(np.nanmedian(ref) - 2, -36, "reference", color=ORANGE, fontsize=7, va="bottom", ha="right")
    ax.axvspan(36, 50, color=ds.theme.GRID, zorder=0)
    ax.set_xlim(36, 156)
    ax.set_ylim(-42, 2)
    ax.set_xlabel("Frequency (bpm)")
    ax.set_ylabel("Power (dB)")
    ax.set_title("Heart-band spectrum", loc="left")
    ax.legend(loc="lower right", fontsize=7, borderaxespad=0.2)
    letter(ax, "c")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    return dict(stem=stem, mae_default=float(np.nanmean(np.abs(r0.hr_v - ref))),
                mae_fixed=float(np.nanmean(np.abs(r1.hr_v - ref))),
                cov_default=float(np.mean(r0.hr_c >= ev.CONF_TH)),
                cov_fixed=float(np.mean(r1.hr_c >= ev.CONF_TH)))


# --- coverage --------------------------------------------------------------------------------

def fig_coverage(curves, auroc, out, final_name, op_points=None):
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.88, wspace=0.28)
    for ax, kind, unit, lab in ((axes[0], "heart", "bpm", "a"), (axes[1], "breathing", "/min", "b")):
        for name, col, nm in (("default", GREY, "default"), (final_name, BLUE, "prewhitened")):
            if kind == "breathing" and name != "default":
                continue                   # the breathing path is the same in both chains
            for env, ls in (("October", "-"), ("July", "--")):
                shown = SET_NAME[env]
                cv = curves[(curves.variant == name) & (curves.kind == kind) & (curves.env == env)]
                if not len(cv):
                    continue
                a = auroc[(auroc.variant == name) & (auroc.kind == kind) & (auroc.env == env)
                          & (auroc["index"] == "conf")]
                au = f" ({a.auroc.iloc[0]:.2f})" if len(a) else ""
                label = (f"{nm}, {shown}" if kind == "heart" else shown) + au
                ax.plot(cv.coverage * 100, cv.mae, ls=ls, color=col, label=label)
        if op_points is not None and kind == "heart":
            for env, mk, shown in (("October LORO", "o", "dev."),
                                   ("July frozen", "^", "held-out")):
                p = op_points.get(env)
                if p:
                    ax.plot(p[0] * 100, p[1], mk, color=BLUE, mfc="white", ms=5, mew=1.1,
                            label=f"calibrated, {shown}")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Coverage (%)")
        ax.set_ylabel(f"MAE of kept windows ({unit})")
        ax.set_xlim(0, 101)
        ax.set_ylim(0, 15.5 if kind == "heart" else 4.2)
        ax.set_title("Heart rate" if kind == "heart" else "Respiration rate", loc="left")
        ax.legend(loc="upper left", fontsize=6.5, handlelength=1.8, borderaxespad=0.1,
                  ncol=2 if kind == "heart" else 1, columnspacing=0.8,
                  title="chain, set (AUROC)" if kind == "heart" else "set (AUROC)",
                  title_fontsize=6.5)
        letter(ax, lab, dx=-0.13)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)


# --- Bland-Altman ------------------------------------------------------------------------------

def fig_ba(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.4))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.19, top=0.88, wspace=0.3)
    res = {}
    for ax, kind, unit, lab in ((axes[0], "heart", "bpm", "a"), (axes[1], "breathing", "/min", "b")):
        ba = ev.bland_altman_repeated(df, kind, valid_only=True)
        res[kind] = {k: v for k, v in ba.items() if not isinstance(v, np.ndarray)}
        envs = np.array([df.loc[df.stem == s, "env"].iloc[0] for s in ba["stems"]])
        for env, mk, fc, nm in (("October", "o", BLUE, "development set"),
                                ("July", "^", "white", "held-out set")):
            m = envs == env
            ax.scatter(ba["mean_pair"][m], ba["diff"][m], s=10, marker=mk, facecolors=fc,
                       edgecolors=BLUE, linewidths=0.6, label=f"{nm} (n = {m.sum()})", zorder=3)
        for y, ls in ((ba["bias"], "-"), (ba["loa_lo"], "--"), (ba["loa_hi"], "--")):
            ax.axhline(y, color=ds.RED, ls=ls, lw=0.9, zorder=2)
        ax.margins(x=0.06, y=0.12)
        x0 = ax.get_xlim()[0]
        ax.text(x0, ba["loa_hi"], f" +1.96 SD {ba['loa_hi']:+.1f}", ha="left", va="bottom",
                fontsize=7, color=ds.RED)
        ax.text(x0, ba["loa_lo"], f" -1.96 SD {ba['loa_lo']:+.1f}", ha="left", va="top",
                fontsize=7, color=ds.RED)
        ax.text(x0, ba["bias"], f" bias {ba['bias']:+.1f}", ha="left", va="top", fontsize=7,
                color=ds.RED)
        ax.axhline(0, color="0.7", lw=0.5, zorder=1)
        ax.grid(True, alpha=0.3)
        what = "heart rate" if kind == "heart" else "respiration rate"
        ax.set_xlabel(f"Mean of radar and reference {what} ({unit})")
        ax.set_ylabel(f"Radar - reference ({unit})")
        ax.set_title("Heart rate vs PPG beats" if kind == "heart" else
                     "Respiration rate vs PPG-derived", loc="left")
        ax.legend(loc="lower right" if kind == "heart" else "upper left", fontsize=7,
                  handletextpad=0.2, borderaxespad=0.1)
        letter(ax, lab, dx=-0.13)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    return res


# --- quality -------------------------------------------------------------------------------------

def fig_quality(stem, out, span_bins=3, win_s=10.0, step_s=2.0):
    c = ch.load(stem)
    r = ch.run(c)
    b0 = r.bin_used
    RX = c.rf.shape[1]
    chans = [(o, i, vitalsigns.arc_snr(c.rf[:, i, b0 + o])) for o in range(-span_bins, span_bins + 1)
             for i in range(RX) if 0 <= b0 + o < c.rf.shape[-1]]
    adm = {(o, i) for o, i, q, a in r.channels if a}
    strong = max(chans, key=lambda x: x[2])
    weak = min(chans, key=lambda x: x[2])           # the weakest candidate of the map
    fig = plt.figure(figsize=(COL2, 2.75))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.55, 0.55, 1.6], height_ratios=[1, 0.8],
                          wspace=0.45, hspace=0.6, left=0.06, right=0.95, bottom=0.14, top=0.9)
    seg = slice(0, int(20 * c.fps))
    t = np.arange(c.rf.shape[0]) / c.fps
    for k, (o, i, q) in enumerate((strong, weak)):
        ax = fig.add_subplot(gs[0, k])
        z = c.rf[seg, i, b0 + o]
        zc = z - z.mean()
        s = np.max(np.abs(zc))
        ax.plot(zc.real / s, zc.imag / s, ".", ms=0.6, color=BLUE if k == 0 else GREY,
                rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_xlabel("I (norm.)")
        if k == 0:
            ax.set_ylabel("Q (norm.)")
            letter(ax, "a", dx=-0.5, dy=1.08)
        ax.set_title(f"{'strong' if k == 0 else 'weak'}: $Q_{{arc}}$ = {q:.1f}", fontsize=8, loc="left")
    ax = fig.add_subplot(gs[1, 0:2])
    for (o, i, q), col, ls, nm in ((strong, BLUE, "-", "strong"), (weak, GREY, "-", "weak")):
        d = ch.displacement(c.rf[:, i, b0 + o], c.wave_len)
        ax.plot(t[seg][::4], d[seg][::4] * 1e3, color=col, ls=ls, lw=0.8, label=nm)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displ. (mm)")
    ax.legend(loc="upper right", fontsize=7, ncol=2, borderaxespad=0.0, handlelength=1.2)
    # (b) Q_arc map
    axm = fig.add_subplot(gs[:, 2])
    wins = moving_hr._windows(c.rf.shape[0], c.fps, win_s, step_s)
    tc = np.array([(a + b) / 2 / c.fps for a, b in wins])
    M = np.array([[vitalsigns.arc_snr(c.rf[a:b, i, b0 + o]) for a, b in wins] for o, i, _ in chans])
    im = axm.imshow(np.log10(np.maximum(M, 0.5)), aspect="auto", origin="lower", cmap="viridis",
                    extent=[tc[0] - step_s / 2, tc[-1] + step_s / 2, -0.5, len(chans) - 0.5],
                    vmin=0, vmax=np.log10(max(2.0, np.percentile(M, 99))), rasterized=True)
    offs = sorted({o for o, _, _ in chans})
    axm.set_yticks([RX * k + (RX - 1) / 2 for k in range(len(offs))])
    axm.set_yticklabels([f"{c.r[b0 + o]:.2f}" for o in offs])
    axm.set_ylabel("Range bin (m); RX1-RX4 upwards")
    for j, (o, i, q) in enumerate(chans):
        if (o, i) in adm:
            axm.plot(tc[-1] + step_s * 0.5 + (tc[-1] - tc[0]) * 0.02, j, ">", color=ORANGE, ms=3.5,
                     clip_on=False)
    for j in range(RX, len(chans), RX):
        axm.axhline(j - 0.5, color="white", lw=0.4)
    axm.set_xlabel("Time (s)")
    axm.set_title("Candidates: range bin x receiver (arrow = admitted)", loc="left", fontsize=8)
    letter(axm, "b", dx=-0.2, dy=1.02)
    cb = fig.colorbar(im, ax=axm, pad=0.06, fraction=0.05)
    ticks = [1, 2, 5, 10, 20, 50, 100, 200]
    ticks = [x for x in ticks if np.log10(x) <= im.get_clim()[1] + 1e-9]
    cb.set_ticks(np.log10(ticks))
    cb.set_ticklabels([str(x) for x in ticks])
    cb.set_label("$Q_{arc}$ (10 s window)")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    return dict(stem=stem, bin_used=int(b0), range_m=float(c.r[b0]), strong=(int(strong[0]), int(strong[1]),
                float(strong[2])), weak=(int(weak[0]), int(weak[1]), float(weak[2])),
                admitted=len(adm), candidates=len(chans), arc_min=vitalsigns.ARC_SNR_MIN)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tables_dir")
    ap.add_argument("fig_dir")
    ap.add_argument("--final", default="prewhiten (acceleration)")
    ap.add_argument("--tracking-stem", default="sitting_still_90deg_1m_1mright")
    ap.add_argument("--quality-stem", default="sitting_still_90deg_2m_1mright_2")
    a = ap.parse_args(argv)
    style()
    info = {}
    info["tracking"] = fig_tracking(a.tracking_stem, os.path.join(a.fig_dir, "fig_tracking"))
    curves = pd.read_csv(os.path.join(a.tables_dir, "coverage_curves.csv"))
    auroc = pd.read_csv(os.path.join(a.tables_dir, "auroc.csv"))
    cal = pd.read_csv(os.path.join(a.tables_dir, "calibration_loro.csv"))
    ops = {}
    for s in ("October LORO", "July frozen"):
        x = cal[(cal.set == s) & (cal.kind == "heart")]
        if len(x):
            ops[s] = (float(x.coverage.iloc[0]), float(x.mae_valid.iloc[0]))
    fig_coverage(curves, auroc, os.path.join(a.fig_dir, "fig_coverage"), a.final, ops)
    both = pd.concat([pd.read_csv(os.path.join(a.tables_dir, "calibration_loro_october_windows.csv")),
                      pd.read_csv(os.path.join(a.tables_dir, "calibration_july_frozen_windows.csv"))],
                     ignore_index=True)
    info["ba"] = fig_ba(both, os.path.join(a.fig_dir, "fig_ba"))
    info["quality"] = fig_quality(a.quality_stem, os.path.join(a.fig_dir, "fig_quality"))
    with open(os.path.join(a.tables_dir, "figures_info.json"), "w") as f:
        json.dump(info, f, indent=1, default=float)
    print(json.dumps(info, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
