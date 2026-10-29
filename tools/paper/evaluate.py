"""
Window-level evaluation of the Vital Signs chain against the SOMNOtouch reference, for
the paper (tools.paper.chain runs the chain; tools.paper.run_all writes the tables).

Definitions (they are the Methods of the paper):

  windows     heart and breathing: 30 s, centres every 1 s (vitalsigns.DEFAULT_CONFIG),
              window i spans [t_i - 15 s, t_i + 15 s] of the capture's own clock.
  reference   heart, primary: 60 / mean RR of the PPG (Pleth) beats inside the window
              (RR 0.33-1.6 s kept; vitals_check.Reference.hr_beats); secondary: the mean
              of the somnograph's Pulse over the window. Breathing: the mean of three
              Pleth surrogates (baseline, pulse amplitude, beat rate) when all three
              agree within 2 /min (smart fusion), else no reference for that window.
  error       radar - reference, per window with a reference.
  valid       confidence >= 6 (Min peak/avg, DEFAULT_CONF_THRESH); coverage = share of
              valid windows among the windows with a reference.
  metrics     MAE, RMSE, bias over ALL windows with a reference (as tools.vitals_check)
              and over the valid ones; share within 5 bpm / 2 /min.
  BA          Bland & Altman (2007), repeated measurements with the true value varying:
              recordings as the units, NON-overlapping windows (every 30th window),
              LoA = bias +- 1.96 sqrt(s_b^2 + s_w^2) from one-way ANOVA.
  CI          95 % percentile intervals of a cluster bootstrap over recordings (2000
              resamples of whole recordings, seed 0).
  AUROC       of a quality index (higher = better) for telling windows with
              |error| <= 5 bpm (heart) or <= 2 /min (breathing) from worse ones.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import moving_hr
import vitalsigns

HR_WITHIN = 5.0
BR_WITHIN = 2.0
CONF_TH = vitalsigns.DEFAULT_CONF_THRESH
N_BOOT = 2000


# --- conditions -----------------------------------------------------------------------------

def condition_of(stem: str, meta: dict | None = None) -> dict:
    """environment, session date, condition label and its parts, from the capture name."""
    date = (meta or {}).get("start", "")[:10]
    env = "July" if date.startswith("2026-07") else "October"
    d = dict(env=env, date=date, orientation="", distance_m=np.nan, offset_m=0.0,
             activity="still", config="")
    m = re.match(r"sitting_still_(0|90)deg(?:_(\d)m)?(?:_(\d)m(right))?", stem)
    if stem.startswith("sitting_swing"):
        d.update(activity="rocking", condition="rocking (seated)")
    elif stem.startswith("HR_moving"):
        d.update(activity="walking", condition="walking")
    elif stem.startswith("new_HR_seated"):
        mm = re.match(r"new_HR_seated_(\d)m(\d)?", stem)
        if mm:
            dist = float(mm.group(1)) + (0.5 if mm.group(2) else 0.0)
            d.update(orientation="0", distance_m=dist, condition=f"0°, {dist:g} m, 180 s (cfg B)")
        else:
            d.update(orientation="0", condition="seated, 3TX (cfg C)")
    elif m:
        o = m.group(1)
        dist = float(m.group(2) or (m.group(3) if m.group(4) else None) or 1.0)
        if m.group(2) and m.group(3):            # "_2m_1mright"
            dist = float(m.group(2))
        off = 1.0 if m.group(4) else 0.0
        if m.group(4) and not m.group(2):        # "_1m_1mright": the 1 m is the distance
            dist = 1.0
        d.update(orientation=o, distance_m=dist, offset_m=off,
                 condition=f"{o}°, {dist:g} m" + (", 1 m offset" if off else ""))
    elif env == "July":
        d.update(condition="July (other room)")
    else:
        d.update(condition="other")
    return d


# --- per-window table -----------------------------------------------------------------------

def _arc_window(z, a, b):
    return vitalsigns.arc_snr(z[a:b])


def window_table(c, res, variant: str = "default", with_quality: bool = True) -> pd.DataFrame:
    """One row per heart and breathing window of res (a chain.Result) on capture c."""
    fps = c.fps
    half = 15.0
    cond = condition_of(c.stem, c.meta)
    rows = []
    br_at = (lambda t: float(np.interp(t, res.rr_t, res.rr_v))) if len(res.rr_t) else (lambda t: np.nan)
    qbr_at = (lambda t: float(np.interp(t, res.rr_t, res.rr_c))) if len(res.rr_t) else (lambda t: np.nan)
    q_e, q_band, q_harm = (moving_hr.edge_index(res.hr_v, (50.0, 150.0) if res.grid is None else
                                                (float(res.grid[0]), float(res.grid[-1])),
                                                [br_at(t) for t in res.hr_t], 2 * half)
                           if len(res.hr_v) else (np.zeros(0),) * 3)
    # channels admitted to the heart mean, for the window's Q_arc
    adm = [(o, r) for o, r, q, a in res.channels if a]
    for i, (t, v, cf) in enumerate(zip(res.hr_t, res.hr_v, res.hr_c)):
        row = dict(stem=c.stem, variant=variant, kind="heart", t=float(t), radar=float(v),
                   conf=float(cf), valid=bool(cf >= CONF_TH),
                   ref=c.ref.hr_beats(t - half, t + half), ref_pulse=c.ref.hr_pulse(t - half, t + half),
                   br_radar=br_at(t), q_br=qbr_at(t), q_edge=float(q_e[i]), q_band=float(q_band[i]),
                   q_harm=float(q_harm[i]), n_ch=len(adm))
        if with_quality:
            a, b = int(round((t - half) * fps)), int(round((t + half) * fps))
            a, b = max(0, a), min(c.rf.shape[0], b)
            qs = [_arc_window(c.rf[:, r, res.bin_used + o], a, b) for o, r in adm]
            row["q_arc"] = float(np.median(qs)) if qs else np.nan
            row["q_arc_max"] = float(np.max(qs)) if qs else np.nan
        rows.append(row)
    for t, v, cf in zip(res.rr_t, res.rr_v, res.rr_c):
        rows.append(dict(stem=c.stem, variant=variant, kind="breathing", t=float(t), radar=float(v),
                         conf=float(cf), valid=bool(cf >= CONF_TH),
                         ref=c.ref.breathing(t - half, t + half)[0], ref_pulse=np.nan,
                         br_radar=float(v), q_br=float(cf)))
    df = pd.DataFrame(rows)
    for k, val in cond.items():
        df[k] = val
    df["dist_m"] = res.dist_m
    df["err"] = df["radar"] - df["ref"]
    df["err_pulse"] = df["radar"] - df["ref_pulse"]
    return df


# --- metrics ----------------------------------------------------------------------------------

def metrics(df: pd.DataFrame, kind: str, col: str = "err") -> dict:
    """MAE/RMSE/bias/within over all windows with a reference and over the valid ones,
    and the coverage."""
    w = df[(df.kind == kind) & np.isfinite(df[col])]
    lim = HR_WITHIN if kind == "heart" else BR_WITHIN
    out = dict(n_rec=int(w.stem.nunique()), n_win=int(len(w)))
    if not len(w):
        return out
    e = w[col].to_numpy()
    v = w.valid.to_numpy(bool)
    out.update(mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
               bias=float(np.mean(e)), within=float(np.mean(np.abs(e) <= lim)),
               coverage=float(np.mean(v)), n_valid=int(v.sum()))
    if v.any():
        ev = e[v]
        out.update(mae_valid=float(np.mean(np.abs(ev))), rmse_valid=float(np.sqrt(np.mean(ev ** 2))),
                   bias_valid=float(np.mean(ev)), within_valid=float(np.mean(np.abs(ev) <= lim)),
                   bad_valid_share=float(np.mean(np.abs(ev) > lim)))
    return out


def cluster_bootstrap(df, kind, stat, n_boot=N_BOOT, seed=0, col="err"):
    """95 % CI of stat(metrics dict) by resampling whole recordings."""
    w = df[(df.kind == kind) & np.isfinite(df[col])]
    groups = {s: g for s, g in w.groupby("stem")}
    keys = list(groups)
    if len(keys) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        b = pd.concat([groups[keys[k]] for k in pick], ignore_index=True)
        m = metrics(b, kind, col)
        if stat in m:
            vals.append(m[stat])
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def nonoverlapping(df: pd.DataFrame, win_s: float = 30.0) -> pd.DataFrame:
    """Windows of each recording and kind whose centres are win_s apart (the first, the
    one win_s later, ...)."""
    keep = []
    for (_, _), g in df.groupby(["stem", "kind"]):
        g = g.sort_values("t")
        last = -np.inf
        for idx, t in zip(g.index, g.t):
            if t - last >= win_s - 1e-6:
                keep.append(idx)
                last = t
    return df.loc[keep]


def bland_altman_repeated(df: pd.DataFrame, kind: str, col: str = "err", valid_only=False) -> dict:
    """Bland & Altman 2007 (true value varying): units = recordings."""
    w = df[(df.kind == kind) & np.isfinite(df[col])]
    if valid_only:
        w = w[w.valid]
    w = nonoverlapping(w)
    groups = [g[col].to_numpy() for _, g in w.groupby("stem")]
    groups = [g for g in groups if g.size]
    n = len(groups)
    N = int(sum(g.size for g in groups))
    if n < 2 or N <= n:
        return dict(n_rec=n, n_win=N)
    allv = np.concatenate(groups)
    grand = allv.mean()
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ss_b = sum(g.size * (g.mean() - grand) ** 2 for g in groups)
    ms_w = ss_w / (N - n)
    ms_b = ss_b / (n - 1)
    lam = (N ** 2 - sum(g.size ** 2 for g in groups)) / ((n - 1) * N)
    s_b2 = max(0.0, (ms_b - ms_w) / lam)
    sd = float(np.sqrt(s_b2 + ms_w))
    ref = w["ref"].to_numpy()
    return dict(n_rec=n, n_win=N, bias=float(grand), sd=sd, loa_lo=float(grand - 1.96 * sd),
                loa_hi=float(grand + 1.96 * sd), s_between=float(np.sqrt(s_b2)),
                s_within=float(np.sqrt(ms_w)), mean_pair=(w["radar"].to_numpy() + ref) / 2,
                diff=w[col].to_numpy(), stems=w.stem.to_numpy(), valid=w.valid.to_numpy())


def auroc(score, good) -> float:
    """P(score of a good window > score of a bad one), ties half (Mann-Whitney)."""
    s = np.asarray(score, float)
    g = np.asarray(good, bool)
    ok = np.isfinite(s)
    s, g = s[ok], g[ok]
    if g.all() or (~g).all():
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(s)
    n1, n0 = g.sum(), (~g).sum()
    return float((r[g].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def coverage_curve(df, kind, score="conf", col="err"):
    """(coverage, MAE of the kept windows, share within) as the score's threshold rises:
    the windows ranked by the score, the best kept."""
    w = df[(df.kind == kind) & np.isfinite(df[col]) & np.isfinite(df[score])]
    lim = HR_WITHIN if kind == "heart" else BR_WITHIN
    order = np.argsort(-w[score].to_numpy(), kind="stable")
    e = np.abs(w[col].to_numpy()[order])
    k = np.arange(1, e.size + 1)
    return k / e.size, np.cumsum(e) / k, np.cumsum(e <= lim) / k
