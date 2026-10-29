"""
Angle sensitivity of the IWR6843AOP arrays to random channel phase errors
(re-implementation; the original aop_phase_error_sim.py is not in the repository).

    python -m tools.paper_array.phase_error_sim [--trials 2000] [--no-fig]

Arrays (positions in wavelengths, from antenna_config's verified AOP preset):
  * 2x2   - the four physical RX of one TX (configs A and B);
  * 12    - the TDM virtual array (TX masks 1, 2, 4; config C), L-shaped.
A point target at azimuth 0, 26.6 and 45 deg (elevation 0: the chest at about radar
height), K = 64 snapshots (the loops of a frame), per-channel SNR 20 dB per snapshot,
i.i.d. channel phase errors e_n ~ N(0, sigma^2), sigma = 0..20 deg, a fresh draw per
trial. Azimuth found by a 0.1-deg scan of the azimuth cut (elevation 0) with
  Bartlett  P(theta) = a^H R a / N^2
  Capon     P(theta) = 1 / (a^H (R + 0.01 tr(R) I)^-1 a)   (DARTscope's loading)
Reported per (array, method, target, sigma): median |azimuth error| and the beam gain
loss at the true direction, -10 log10(|sum_n exp(j e_n)|^2 / N^2) (median), plus the
analytic small-error value 10 log10(e) sigma^2 [rad^2] ~ 4.34 sigma^2 for comparison.

Figure fig_array: (a) array geometry, physical RX and virtual elements; (b)-(d) median
azimuth error vs sigma per target angle.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

import antenna_config
from tools.paper_array import common

TARGETS_DEG = (0.0, 26.6, 45.0)
SIGMAS_DEG = np.arange(0, 20.1, 2.0)
K_SNAP = 64
SNR_DB = 20.0
LOADING = 0.01
SCAN = np.arange(-90.0, 90.0001, 0.1)


def arrays():
    cfg = antenna_config.get_config(antenna_config.AOP_DEFAULT)
    rx = np.asarray(cfg["rx_positions"], float)                        # (4, 2) lambda
    gx, gy = antenna_config.build_geometry(cfg, masks=[1, 2, 4], num_virt_rx=12, num_phys_rx=4)
    v = np.c_[np.asarray(gx), np.asarray(gy)]
    return {"2x2 RX": rx - rx.mean(0), "12 virtual": v - v.mean(0)}, cfg


def steer(pos, az_deg):
    a = np.radians(np.atleast_1d(az_deg))
    return np.exp(1j * 2 * np.pi * pos[:, 0][None, :] * np.sin(a)[:, None])   # (A, N)


def simulate(trials=2000, seed=1):
    rng = np.random.default_rng(seed)
    arrs, _ = arrays()
    rows = []
    for an, pos in arrs.items():
        N = len(pos)
        S = steer(pos, SCAN)                                            # (A, N)
        for th in TARGETS_DEG:
            a0 = steer(pos, th)[0]
            for sg in SIGMAS_DEG:
                err_b, err_c, loss = [], [], []
                for _ in range(trials):
                    e = rng.normal(0, np.radians(sg), N)
                    s = (rng.normal(size=K_SNAP) + 1j * rng.normal(size=K_SNAP)) / np.sqrt(2)
                    nse = (rng.normal(size=(N, K_SNAP)) + 1j * rng.normal(size=(N, K_SNAP))) / np.sqrt(2)
                    X = (a0 * np.exp(1j * e))[:, None] * s[None, :] * 10 ** (SNR_DB / 20) + nse
                    R = X @ X.conj().T / K_SNAP
                    pb = np.real(np.einsum("an,nm,am->a", S.conj(), R, S))
                    Ri = np.linalg.inv(R + np.eye(N) * LOADING * np.trace(R).real)
                    pc = 1.0 / np.real(np.einsum("an,nm,am->a", S.conj(), Ri, S))
                    err_b.append(abs(SCAN[np.argmax(pb)] - th))
                    err_c.append(abs(SCAN[np.argmax(pc)] - th))
                    loss.append(-10 * np.log10(np.abs(np.exp(1j * e).sum()) ** 2 / N ** 2))
                for m, er in (("Bartlett", err_b), ("Capon", err_c)):
                    rows.append(dict(array=an, N=N, method=m, target_deg=th, sigma_deg=float(sg),
                                     median_abs_err_deg=float(np.median(er)),
                                     p90_abs_err_deg=float(np.percentile(er, 90)),
                                     median_gain_loss_db=float(np.median(loss)),
                                     analytic_gain_loss_db=float(10 * np.log10(np.e) * np.radians(sg) ** 2
                                                                 * (1 - 1 / N)),
                                     trials=trials))
    return rows


def style():
    """DARTscope's plot style (theme.style_matplotlib) for print."""
    from tools.paper import dartscope_style
    dartscope_style.apply()


from tools.paper import dartscope_style as _ds
COL = {"2x2 RX": _ds.BLUE, "12 virtual": _ds.ORANGE}   # the interface's data colours
MK = {"2x2 RX": "o", "12 virtual": "s"}
LS = {"Bartlett": "-", "Capon": "--"}


def figure(rows, path_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    arrs, cfg = arrays()
    fig = plt.figure(figsize=(7.16, 2.2))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 3.25], wspace=0.24, left=0.075, right=0.99,
                             bottom=0.2, top=0.88)
    gs = outer[1].subgridspec(1, 3, wspace=0.08)
    ax0 = fig.add_subplot(outer[0])
    # (a) geometry: physical RX (2x2) and the 12 virtual elements, lambda units, uncentred
    rx = np.asarray(cfg["rx_positions"], float)
    tx = np.asarray(cfg["tx_offsets"], float)
    blocks = {1: "TX1", 2: "TX2", 4: "TX3"}  # transmit-enable masks 1, 2, 4
    vmk = {0: "s", 1: "D", 2: "^"}
    for t, (off, m) in enumerate(zip(tx, cfg["tx_masks"])):
        p = rx + off
        ax0.scatter(p[:, 0], p[:, 1], s=30, marker=vmk[t], facecolor="white", edgecolor=COL["12 virtual"],
                    linewidth=0.9, zorder=2, label=f"virt. {blocks[m]}")
    ax0.scatter(rx[:, 0], rx[:, 1], s=12, color=COL["2x2 RX"], zorder=3, label="physical RX")
    for i, (x, y) in enumerate(rx):
        ax0.annotate(f"RX{i + 1}", (x, y), xytext=(3, -8), textcoords="offset points", fontsize=7,
                     color=COL["2x2 RX"])
    ax0.set_xlabel(r"Azimuth position ($\lambda$)")
    ax0.set_ylabel(r"Elevation position ($\lambda$)")
    ax0.set_xlim(-0.35, 1.85)
    ax0.set_ylim(-0.35, 1.85)
    ax0.set_aspect("equal")
    ax0.set_xticks([0, 0.5, 1, 1.5])
    ax0.set_yticks([0, 0.5, 1, 1.5])
    ax0.legend(loc="upper right", fontsize=7, handletextpad=0.2, borderaxespad=0.0, labelspacing=0.25)
    ax0.text(-0.30, 1.0, "(a)", transform=ax0.transAxes, fontweight="bold", va="bottom")
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    ymax = max(r["median_abs_err_deg"] for r in rows) * 1.08
    for k, (ax, th) in enumerate(zip(axes, TARGETS_DEG)):
        for an in COL:
            for m in LS:
                sel = [r for r in rows if r["array"] == an and r["method"] == m and r["target_deg"] == th]
                ax.plot([r["sigma_deg"] for r in sel], [r["median_abs_err_deg"] for r in sel],
                        LS[m], color=COL[an], marker=MK[an], ms=3.2,
                        mfc=COL[an] if m == "Bartlett" else "white", mew=0.8,
                        label=f"{an}, {m}")
        ax.set_title(f"Target at {th:g}°", loc="left")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Phase error " + r"$\sigma$" + " (°)")
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.8, 20.8)
        ax.set_xticks([0, 5, 10, 15, 20])
        if k == 0:
            ax.set_ylabel("Median azimuth error (°)")
        else:
            ax.tick_params(labelleft=False)
        ax.text(-0.08 if k else -0.3, 1.0, "(" + "bcd"[k] + ")", transform=ax.transAxes,
                fontweight="bold", va="bottom", ha="right")
    axes[0].legend(loc="upper left", fontsize=7, handlelength=2.2, labelspacing=0.25)
    fig.savefig(path_pdf)
    fig.savefig(os.path.join(common.OUT_DIR, "fig_array_preview.png"), dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--from-csv", action="store_true", help="redraw the figure from the saved CSV")
    args = ap.parse_args(argv)
    os.makedirs(common.OUT_DIR, exist_ok=True)
    out = os.path.join(common.OUT_DIR, "phase_error_sim.csv")
    if args.from_csv:
        with open(out, encoding="utf-8") as f:
            rows = [{k: (v if k in ("array", "method") else float(v)) for k, v in r.items()}
                    for r in csv.DictReader(f)]
    else:
        rows = simulate(args.trials)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("written", out)
    for r in rows:
        if r["sigma_deg"] in (0.0, 10.0, 20.0):
            print(f"{r['array']:10s} {r['method']:8s} {r['target_deg']:5.1f} sigma={r['sigma_deg']:4.0f}  "
                  f"med|err|={r['median_abs_err_deg']:5.2f}  p90={r['p90_abs_err_deg']:5.2f}  "
                  f"loss={r['median_gain_loss_db']:.2f} dB (analytic {r['analytic_gain_loss_db']:.2f})")
    if not args.no_fig:
        figure(rows, os.path.join(common.FIG_DIR, "fig_array.pdf"))
        print("figure", os.path.join(common.FIG_DIR, "fig_array.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
