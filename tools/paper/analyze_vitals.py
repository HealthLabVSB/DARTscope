"""
Tables of the vital-sign case study from the per-window CSVs of tools.paper.run_variants.

    python -m tools.paper.analyze_vitals WINDOWS_DIR OUT_DIR --final NAME [--gates-prefix P]

Writes (OUT_DIR):
  ablation.csv          variant x (heart MAE/RMSE/bias/within/coverage, breathing ...),
                        all recordings and per environment, bootstrap CI of the heart MAE,
                        paired Wilcoxon over recordings against the default (Holm)
  per_recording_<v>.csv per recording of the default and of the final chain
  per_condition_<v>.csv n, MAE, RMSE, bias, LoA (BA 2007), coverage per condition
  per_environment.csv   October vs July, default and final chain, with CIs
  auroc.csv             AUROC of every quality index (heart: |e| > 5 bpm; breathing 2 /min)
  coverage_curves.csv   coverage vs MAE / share within, ranked by confidence
  calibration_loro.csv  leave-one-recording-out choice of (Q_arc gate, confidence threshold)
                        on October, and July evaluated once with the frozen choice
Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

from tools.paper import evaluate as ev

HEART_BAND = (50.0, 150.0)       # vital_bands "Adult", the default
Q_EDGE_GRID = (0.0, 0.5, 1.0, 2.0, 3.0)   # 0 = no Q_edge gate
CONF_GRID = np.round(np.arange(0.0, 20.01, 0.5), 2)
BAD_SHARE_TARGET = 0.10          # calibration: at most 10 % of the admitted windows > 5 bpm off
QUALITY = ["conf", "q_arc", "q_arc_max", "q_br", "q_edge", "q_band", "q_harm", "n_ch"]


def load_dir(d):
    """{variant: per-window table}; Q_edge recomputed from the windows' estimates
    (moving_hr.edge_index, two-sided band edge, 50-150 bpm, 30 s)."""
    import moving_hr
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(d, f))
            hr = df.kind == "heart"
            qe, qb, qh = moving_hr.edge_index(df.loc[hr, "radar"].to_numpy(), HEART_BAND,
                                              df.loc[hr, "br_radar"].to_numpy(), 30.0)
            df.loc[hr, "q_edge"], df.loc[hr, "q_band"], df.loc[hr, "q_harm"] = qe, qb, qh
            out[df.variant.iloc[0]] = df
    return out


def _row(df, kind, prefix):
    m = ev.metrics(df, kind)
    keys = ["n_rec", "n_win", "mae", "rmse", "bias", "within", "coverage", "mae_valid",
            "within_valid", "bad_valid_share"]
    return {f"{prefix}{k}": m.get(k, np.nan) for k in keys}


def per_recording(df):
    rows = []
    for s, g in df.groupby("stem"):
        r = dict(stem=s, env=g.env.iloc[0], date=g.date.iloc[0], condition=g.condition.iloc[0],
                 dist_m=g.dist_m.iloc[0])
        r.update(_row(g, "heart", "hr_"))
        r.update({f"hr_pulse_{k}": v for k, v in ev.metrics(g, "heart", "err_pulse").items()
                  if k in ("mae", "bias")})
        r.update(_row(g, "breathing", "br_"))
        rows.append(r)
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, float)
    out = np.full(p.size, np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


def ablation(tabs, order):
    base = per_recording(tabs["default"]).set_index("stem")
    rows = []
    for name in order:
        df = tabs[name]
        r = dict(variant=name)
        r.update(_row(df, "heart", "hr_"))
        r.update(_row(df, "breathing", "br_"))
        lo, hi = ev.cluster_bootstrap(df, "heart", "mae")
        r.update(hr_mae_ci_lo=lo, hr_mae_ci_hi=hi)
        for env in ("October", "July"):
            e = df[df.env == env]
            r[f"hr_mae_{env}"] = ev.metrics(e, "heart").get("mae", np.nan)
            r[f"hr_coverage_{env}"] = ev.metrics(e, "heart").get("coverage", np.nan)
            r[f"br_mae_{env}"] = ev.metrics(e, "breathing").get("mae", np.nan)
        pr = per_recording(df).set_index("stem")
        r["hr_failed_recordings"] = int((pr.hr_bias.abs() > 10).sum())
        if name != "default":
            j = base.index.intersection(pr.index)
            a, b = base.loc[j, "hr_mae"].to_numpy(), pr.loc[j, "hr_mae"].to_numpy()
            ok = np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) > 1e-9)
            r["hr_wilcoxon_p"] = float(sps.wilcoxon(a[ok], b[ok]).pvalue) if ok.sum() >= 5 else np.nan
            r["hr_recordings_better"] = int((b < a - 1e-9).sum())
            r["hr_recordings_worse"] = int((b > a + 1e-9).sum())
            a, b = base.loc[j, "br_mae"].to_numpy(), pr.loc[j, "br_mae"].to_numpy()
            ok = np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) > 1e-9)
            r["br_wilcoxon_p"] = float(sps.wilcoxon(a[ok], b[ok]).pvalue) if ok.sum() >= 5 else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    for k in ("hr_wilcoxon_p", "br_wilcoxon_p"):
        if k in out:
            out[k.replace("_p", "_p_holm")] = holm(out[k].to_numpy())
    return out


def per_condition(df):
    rows = []
    for (env, cond), g in df.groupby(["env", "condition"]):
        r = dict(env=env, condition=cond)
        for kind, pre in (("heart", "hr_"), ("breathing", "br_")):
            m = ev.metrics(g, kind)
            ba = ev.bland_altman_repeated(g, kind)
            r.update({f"{pre}n_rec": m.get("n_rec", 0), f"{pre}n_win": m.get("n_win", 0),
                      f"{pre}mae": m.get("mae", np.nan), f"{pre}rmse": m.get("rmse", np.nan),
                      f"{pre}bias": m.get("bias", np.nan), f"{pre}within": m.get("within", np.nan),
                      f"{pre}coverage": m.get("coverage", np.nan),
                      f"{pre}mae_valid": m.get("mae_valid", np.nan),
                      f"{pre}loa_lo": ba.get("loa_lo", np.nan), f"{pre}loa_hi": ba.get("loa_hi", np.nan),
                      f"{pre}ba_n_win": ba.get("n_win", 0)})
        rows.append(r)
    return pd.DataFrame(rows)


def per_environment(tabs, names):
    rows = []
    for name in names:
        df = tabs[name]
        for env in ("October", "July", "all"):
            g = df if env == "all" else df[df.env == env]
            for kind in ("heart", "breathing"):
                m = ev.metrics(g, kind)
                ba = ev.bland_altman_repeated(g, kind)
                r = dict(variant=name, env=env, kind=kind, **{k: m.get(k, np.nan) for k in
                         ("n_rec", "n_win", "mae", "rmse", "bias", "within", "coverage",
                          "mae_valid", "within_valid")})
                r["mae_ci_lo"], r["mae_ci_hi"] = ev.cluster_bootstrap(g, kind, "mae")
                r["within_ci_lo"], r["within_ci_hi"] = ev.cluster_bootstrap(g, kind, "within")
                r.update(loa_lo=ba.get("loa_lo", np.nan), loa_hi=ba.get("loa_hi", np.nan),
                         ba_bias=ba.get("bias", np.nan), ba_n_win=ba.get("n_win", 0))
                if kind == "heart":
                    mp = ev.metrics(g, kind, "err_pulse")
                    r.update(mae_vs_pulse=mp.get("mae", np.nan), within_vs_pulse=mp.get("within", np.nan))
                rows.append(r)
    return pd.DataFrame(rows)


def auroc_table(tabs, names):
    rows = []
    for name in names:
        df = tabs[name]
        for env in ("all", "October", "July"):
            g = df if env == "all" else df[df.env == env]
            for kind, lim in (("heart", ev.HR_WITHIN), ("breathing", ev.BR_WITHIN)):
                w = g[(g.kind == kind) & np.isfinite(g.err)]
                good = (w.err.abs() <= lim).to_numpy()
                for q in QUALITY:
                    if q not in w or not np.isfinite(w[q]).any():
                        continue
                    a = ev.auroc(w[q].to_numpy(), good)
                    # cluster bootstrap over recordings
                    rng = np.random.default_rng(0)
                    groups = [x for _, x in w.groupby("stem")]
                    bs = []
                    for _ in range(500):
                        b = pd.concat([groups[k] for k in rng.integers(0, len(groups), len(groups))])
                        bs.append(ev.auroc(b[q].to_numpy(), (b.err.abs() <= lim).to_numpy()))
                    bs = np.asarray(bs)[np.isfinite(bs)]
                    rows.append(dict(variant=name, env=env, kind=kind, index=q, auroc=a,
                                     ci_lo=float(np.percentile(bs, 2.5)) if bs.size else np.nan,
                                     ci_hi=float(np.percentile(bs, 97.5)) if bs.size else np.nan,
                                     n_win=len(w), n_bad=int((~good).sum())))
    return pd.DataFrame(rows)


def curves(tabs, names):
    rows = []
    for name in names:
        for kind in ("heart", "breathing"):
            for env in ("all", "October", "July"):
                df = tabs[name] if env == "all" else tabs[name][tabs[name].env == env]
                cov, mae, within = ev.coverage_curve(df, kind, "conf")
                if not cov.size:
                    continue
                # decimated to at most 200 points
                k = np.unique(np.linspace(0, cov.size - 1, min(200, cov.size)).round().astype(int))
                rows.append(pd.DataFrame(dict(variant=name, kind=kind, env=env, coverage=cov[k],
                                              mae=mae[k], within=within[k])))
    return pd.concat(rows, ignore_index=True)


def _admit(w, th, qe):
    return (w.conf.to_numpy() >= th) & (np.nan_to_num(w.q_edge.to_numpy(), nan=np.inf) >= qe)


def _choose(train: pd.DataFrame) -> tuple:
    """(threshold, q_edge_min, coverage, bad share): the pair whose admitted heart windows
    are at most BAD_SHARE_TARGET > 5 bpm off (pooled over train) with the largest
    coverage; if none meets the target, the pair with the smallest bad share."""
    w = train[(train.kind == "heart") & np.isfinite(train.err)]
    e = w.err.abs().to_numpy()
    best = fallback = None
    for qe in Q_EDGE_GRID:
        for th in CONF_GRID:
            a = _admit(w, th, qe)
            if a.sum() < 10:
                continue
            bad = float(np.mean(e[a] > ev.HR_WITHIN))
            cov = float(a.mean())
            if bad <= BAD_SHARE_TARGET and (best is None or cov > best[2]):
                best = (float(th), float(qe), cov, bad)
            if fallback is None or bad < fallback[3]:
                fallback = (float(th), float(qe), cov, bad)
    return best if best is not None else fallback


def calibrate(gate_tabs: dict):
    """Leave-one-recording-out over October: for each held-out recording the (Q_arc gate,
    confidence threshold, Q_edge minimum) chosen on the other October recordings - over
    the gates, the one whose best pair admits the most windows at the target (ties: the
    smaller MAE of the admitted windows). Then the triple chosen on all of October,
    applied once to July."""
    gates = sorted(gate_tabs)
    octo = {g: t[t.env == "October"] for g, t in gate_tabs.items()}
    stems = sorted(next(iter(octo.values())).stem.unique())

    def pick(train_by_gate):
        cands = []
        for g in gates:
            th, qe, cov, bad = _choose(train_by_gate[g])
            w = train_by_gate[g]
            w = w[(w.kind == "heart") & np.isfinite(w.err)]
            w = w[_admit(w, th, qe)]
            cands.append((bad > BAD_SHARE_TARGET, -cov, float(w.err.abs().mean()) if len(w) else 1e9,
                          g, th, qe))
        cands.sort()
        return cands[0][3], cands[0][4], cands[0][5]

    def apply(t, th, qe):
        t = t.copy()
        hr = t.kind == "heart"
        t.loc[hr, "valid"] = _admit(t[hr], th, qe)            # breathing keeps its own 6.0
        return t

    held, choices = [], []
    for s in stems:
        g, th, qe = pick({k: v[v.stem != s] for k, v in octo.items()})
        t = apply(octo[g][octo[g].stem == s], th, qe)
        t["chosen_gate"], t["chosen_th"], t["chosen_q_edge"] = g, th, qe
        held.append(t)
        choices.append(dict(held_out=s, gate=g, conf_th=th, q_edge_min=qe))
    held = pd.concat(held, ignore_index=True)
    g_all, th_all, qe_all = pick(octo)
    july = apply(gate_tabs[g_all][gate_tabs[g_all].env == "July"], th_all, qe_all)
    octo_frozen = apply(octo[g_all], th_all, qe_all)
    return held, july, octo_frozen, pd.DataFrame(choices), (g_all, th_all, qe_all)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("windows_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--final", required=True)
    ap.add_argument("--gates-dir", help="per-window CSVs of the Q_arc gate grid on the final chain")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    tabs = load_dir(a.windows_dir)
    order = ["default"] + [k for k in tabs if k not in ("default",)]
    ab = ablation(tabs, order)
    ab.to_csv(os.path.join(a.out_dir, "ablation.csv"), index=False)
    for name, tag in (("default", "default"), (a.final, "final")):
        per_recording(tabs[name]).to_csv(os.path.join(a.out_dir, f"per_recording_{tag}.csv"), index=False)
        per_condition(tabs[name]).to_csv(os.path.join(a.out_dir, f"per_condition_{tag}.csv"), index=False)
    per_environment(tabs, ["default", a.final]).to_csv(os.path.join(a.out_dir, "per_environment.csv"), index=False)
    q_names = [n for n in ("default", a.final) if "q_arc" in tabs[n]]
    auroc_table(tabs, q_names).to_csv(os.path.join(a.out_dir, "auroc.csv"), index=False)
    curves(tabs, ["default", a.final]).to_csv(os.path.join(a.out_dir, "coverage_curves.csv"), index=False)
    if a.gates_dir:
        gt = load_dir(a.gates_dir)
        held, july, octo_frozen, choices, (g_all, th_all, qe_all) = calibrate(gt)
        choices.to_csv(os.path.join(a.out_dir, "calibration_loro_choices.csv"), index=False)
        held.to_csv(os.path.join(a.out_dir, "calibration_loro_october_windows.csv"), index=False)
        july.to_csv(os.path.join(a.out_dir, "calibration_july_frozen_windows.csv"), index=False)
        rows = []
        for label, df in (("October LORO", held), ("October in-sample (frozen)", octo_frozen),
                          ("July frozen", july)):
            for kind in ("heart", "breathing"):
                m = ev.metrics(df, kind)
                lo, hi = ev.cluster_bootstrap(df, kind, "mae_valid")
                fold = label == "October LORO"
                rows.append(dict(set=label, kind=kind, gate="per fold" if fold else g_all,
                                 conf_th=np.nan if fold else th_all,
                                 q_edge_min=np.nan if fold else qe_all,
                                 mae_valid_ci_lo=lo, mae_valid_ci_hi=hi, **m))
        pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "calibration_loro.csv"), index=False)
        with open(os.path.join(a.out_dir, "calibration_frozen.json"), "w") as f:
            json.dump(dict(gate=g_all, conf_th=th_all, q_edge_min=qe_all,
                           bad_share_target=BAD_SHARE_TARGET, q_edge_grid=list(Q_EDGE_GRID),
                           conf_grid=[float(CONF_GRID[0]), float(CONF_GRID[-1]), 0.5]), f, indent=1)
    print("written to", a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
