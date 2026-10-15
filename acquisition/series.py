"""
acquisition/series.py

A series of measurements over a range of settings - to find which chirp settings, which
transmitters and receivers suit a movement best - with the person in front of the radar
starting every take themselves.

    variables  a parameter and its values, e.g. loops = 32, 64, 128 and tx = 1 | 1,2,3
    plan()     every combination on top of the base settings, each checked (a
               combination the radar or the loader cannot do is listed and skipped),
               optionally every configuration more than once and in random order, so
               that tiring or getting better at the movement does not favour one setting
    Series     the recordings in <folder>/<name>/, a quick analysis of each
               (quality.py), the ranking for the chosen kind of movement, how much each
               parameter mattered (the mean score of each of its values), and what to
               try next - written after every take to <name>_series.json, .csv and
               _report.html, so an interrupted series keeps what it has

The ranking compares takes of the *same* movement: do it the same way every time.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import copy
import csv
import html
import itertools
import json
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime

from .config import ConfigError, RadarConfig
from .measurement import sanitize
from .params import MeasureParams, checks, derived, format_rx, format_tx, parse_rx, parse_tx
from .quality import OBJECTIVES, folded, score

SCHEMA = "dartscope-series/1"


@dataclass(frozen=True)
class Param:
    label: str
    unit: str
    kind: str                     # "float", "int", "tx", "rx"


PARAMS = {
    "tx": Param("TX pattern", "", "tx"),
    "rx": Param("RX", "", "rx"),
    "loops": Param("Loops (chirps per TX per frame)", "", "int"),
    "period_ms": Param("Frame period", "ms", "float"),
    "slope_mhz_us": Param("Slope", "MHz/us", "float"),
    "samples": Param("ADC samples", "", "int"),
    "fs_ksps": Param("Sample rate", "ksps", "int"),
    "idle_us": Param("Idle time", "us", "float"),
    "adc_start_us": Param("ADC start time", "us", "float"),
    "ramp_end_us": Param("Ramp end time (0 = auto)", "us", "float"),
    "rx_gain_db": Param("RX gain", "dB", "int"),
    "start_ghz": Param("Start frequency", "GHz", "float"),
    "duration_s": Param("Duration", "s", "float"),
}
RAMP_COUPLED = ("samples", "fs_ksps", "adc_start_us", "slope_mhz_us")


def parse_values(name: str, text: str) -> list:
    """
    The values of a parameter as typed: numbers '32, 64, 128' or 'from..to/step'
    ('50..200/50'); TX patterns and RX sets separated by '|' ('1 | 1,3 | 1,2,3', '1234 | 12').
    """
    if name not in PARAMS:
        raise ConfigError(f"unknown parameter {name!r}; one of {', '.join(PARAMS)}")
    kind = PARAMS[name].kind
    text = str(text).strip()
    if not text:
        raise ConfigError(f"{PARAMS[name].label}: no values")
    if kind in ("tx", "rx"):
        items = [s for s in re.split(r"[|;]", text) if s.strip()]
        return [parse_tx(s) if kind == "tx" else parse_rx(s) for s in items]
    out = []
    for tok in re.split(r"[,;\s]+", text):
        if not tok:
            continue
        m = re.fullmatch(r"([-\d.]+)\.\.([-\d.]+)(?:/([-\d.]+))?", tok)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            step = float(m.group(3) or 1)
            if step <= 0:
                raise ConfigError(f"{tok}: the step must be positive")
            n = int((b - a) / step + 1e-9) + 1
            if n > 50:
                raise ConfigError(f"{tok}: {n} values - at most 50")
            out += [a + i * step for i in range(n)]
        else:
            try:
                out.append(float(tok))
            except ValueError:
                raise ConfigError(f"{PARAMS[name].label}: {tok!r} is not a number") from None
    if kind == "int":
        out = [int(round(v)) for v in out]
    return list(dict.fromkeys(out))                       # unique, in order


def format_value(name: str, v) -> str:
    kind = PARAMS[name].kind if name in PARAMS else ""
    if kind == "tx":
        return format_tx(v)
    if kind == "rx":
        return format_rx(v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


@dataclass
class Step:
    index: int                    # 1-based, in the order of the plan
    values: dict                  # parameter -> value (only the varied ones)
    params: MeasureParams
    config: RadarConfig | None
    derived: dict
    problems: list                # (level, what, message)
    take: int = 1                 # 1.. when a configuration is recorded more than once

    @property
    def ok(self) -> bool:
        return self.config is not None and not any(p[0] == "error" for p in self.problems)

    @property
    def key(self) -> str:
        """The configuration, whatever the take: 'tx=1,2,3 loops=64'."""
        return " ".join(f"{k}={format_value(k, v)}" for k, v in self.values.items()) or "base"

    @property
    def label(self) -> str:
        return self.key + (f" (take {self.take})" if self.take > 1 else "")


def plan(base: MeasureParams, base_cfg: RadarConfig | None, variables: list,
         repeats: int = 1, shuffle: bool = False, seed: int | None = None,
         auto_ramp: bool = True) -> list[Step]:
    """
    Every combination of the variables [(name, [values]), ...] on top of `base`.
    auto_ramp: when the sampling changes (samples, sample rate, ADC start, slope) the ramp
    end follows it (ADC start + sampling + 1 us) unless it is itself a variable.
    """
    names = [n for n, _ in variables]
    if len(set(names)) != len(names):
        raise ConfigError("a parameter is listed twice")
    combos = list(itertools.product(*[vals for _, vals in variables])) if variables else [()]
    if len(combos) * max(1, repeats) > 500:
        raise ConfigError(f"{len(combos) * max(1, repeats)} takes - at most 500 in a series")
    follow_ramp = auto_ramp and "ramp_end_us" not in names and any(n in RAMP_COUPLED for n in names)
    configs = []
    for combo in combos:
        p = copy.deepcopy(base)
        if follow_ramp:
            p.ramp_end_us = 0.0
        values = {}
        for name, v in zip(names, combo):
            setattr(p, name, copy.deepcopy(v))
            values[name] = v
        try:
            cfg = p.to_config(base_cfg)
            probs = checks(cfg)
            der = derived(cfg)
        except Exception as e:
            cfg, probs, der = None, [("error", "configuration", str(e))], {}
        configs.append((values, p, cfg, der, probs))
    order = [(i, t) for t in range(1, max(1, repeats) + 1) for i in range(len(configs))]
    if repeats > 1 and not shuffle:
        order = [(i, t) for i in range(len(configs)) for t in range(1, repeats + 1)]
    if shuffle:
        random.Random(seed).shuffle(order)
    steps = []
    for n, (i, t) in enumerate(order, start=1):
        values, p, cfg, der, probs = configs[i]
        steps.append(Step(index=n, values=values, params=p, config=cfg, derived=der,
                          problems=probs, take=t))
    return steps


@dataclass
class TakeResult:
    step: int
    key: str
    take: int
    stem: str
    status: str                   # complete, incomplete, failed, cancelled, skipped
    values: dict
    derived: dict
    quality: dict | None = None
    score: float | None = None
    score_terms: list = field(default_factory=list)
    error: str | None = None
    time: str = ""

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["values"] = {k: format_value(k, v) for k, v in self.values.items()}
        return d


class Series:
    def __init__(self, folder: str, name: str, steps: list[Step], objective: str = "general",
                 notes: dict | None = None, base_params: MeasureParams | None = None,
                 variables: list | None = None):
        self.name = sanitize(name) or "series"
        self.folder = os.path.join(os.path.abspath(folder), self.name)
        self.steps = steps
        self.objective = objective if objective in OBJECTIVES else "general"
        self.notes = dict(notes or {})
        self.base_params = base_params
        self.variables = variables or []
        self.results: list[TakeResult] = []
        self.created = datetime.now().isoformat(timespec="seconds")

    # --- takes --------------------------------------------------------------------------------
    def stem_for(self, step: Step) -> str:
        return f"{self.name}_{step.index:02d}"

    def request(self, step: Step, **kw):
        """The CaptureRequest of a step (start_gate etc. through kw)."""
        from .capture import CaptureRequest
        notes = dict(self.notes)
        notes.update({"series": self.name, "series_step": step.index, "series_take": step.take,
                      "series_values": step.key, "objective": self.objective})
        return CaptureRequest(config=copy.deepcopy(step.config), folder=self.folder,
                              name=self.stem_for(step), auto_increment=True, notes=notes, **kw)

    def add(self, step: Step, capture_result=None, quality=None, status: str | None = None,
            error: str | None = None) -> TakeResult:
        stem = capture_result.measurement.stem if capture_result is not None else ""
        st = status or (capture_result.status if capture_result is not None else "skipped")
        r = TakeResult(step=step.index, key=step.key, take=step.take, stem=stem, status=st,
                       values=step.values, derived=step.derived,
                       error=error or (capture_result.error if capture_result is not None else None),
                       time=datetime.now().isoformat(timespec="seconds"))
        if quality is not None:
            r.quality = quality.to_dict()
            r.score, r.score_terms = score(quality.metrics, self.objective)
        self.results.append(r)
        return r

    def rescore(self, objective: str):
        self.objective = objective
        for r in self.results:
            if r.quality:
                r.score, r.score_terms = score(r.quality["metrics"], objective)

    def done_steps(self) -> set:
        return {r.step for r in self.results if r.status in ("complete", "incomplete", "skipped")}

    # --- evaluation ------------------------------------------------------------------------------
    def scored(self) -> list[TakeResult]:
        return [r for r in self.results if r.score is not None]

    def ranking(self) -> list[dict]:
        """One row per configuration: its mean score over its takes, best first."""
        groups: dict[str, list[TakeResult]] = {}
        for r in self.scored():
            groups.setdefault(r.key, []).append(r)
        rows = []
        for key, rs in groups.items():
            sc = [r.score for r in rs]
            best = max(rs, key=lambda r: r.score)
            rows.append({"key": key, "takes": len(rs), "score": sum(sc) / len(sc),
                         "spread": (max(sc) - min(sc)) if len(sc) > 1 else 0.0,
                         "best_stem": best.stem, "derived": best.derived,
                         "metrics": best.quality["metrics"] if best.quality else {},
                         "values": best.values})
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows

    def effects(self) -> dict:
        """parameter -> [(value, mean score, takes)], best value first."""
        out = {}
        names = []
        for r in self.results:
            for k in r.values:
                if k not in names:
                    names.append(k)
        for name in names:
            groups: dict[str, list[float]] = {}
            for r in self.scored():
                if name in r.values:
                    groups.setdefault(format_value(name, r.values[name]), []).append(r.score)
            if len(groups) >= 2:
                rows = [(v, sum(s) / len(s), len(s)) for v, s in groups.items()]
                rows.sort(key=lambda t: t[1], reverse=True)
                out[name] = rows
        return out

    def recommendations(self) -> list[str]:
        rs = self.scored()
        out = []
        if not rs:
            return ["no analysed take yet"]
        rank = self.ranking()
        best = rank[0]
        bm = best["metrics"]
        out.append(f"best for '{OBJECTIVES[self.objective][0]}': {best['key']} "
                   f"(score {best['score']:.1f} over {best['takes']} take(s), {best['best_stem']})")
        if not any((r.quality or {}).get("metrics", {}).get("motion_detected") for r in rs):
            out.append("no take shows clear motion: was the movement within the covered range "
                       "and in front of the radar? The ranking means little then.")
            return out
        lost = [r.stem for r in rs if (r.quality or {}).get("metrics", {}).get("zero_filled_packets")]
        if lost:
            out.append(f"{len(lost)} take(s) lost packets ({', '.join(lost[:4])}): lower the data "
                       f"rate (fewer samples, loops or RX, a longer frame period) or repeat them")
        def met(r):
            return (r.quality or {}).get("metrics", {})

        folded_takes = [r for r in rs if folded(met(r))]
        if folded_takes:
            fastest = max(max(met(r).get("v95_ms") or 0, met(r).get("v_range_p80_ms") or 0) for r in rs)
            vmax_ok = [r.derived.get("max_vel_ms", 0) for r in rs
                       if not folded(met(r)) and met(r).get("motion_detected")]
            msg = (f"{len(folded_takes)} take(s) show velocity folding (the movement is faster than "
                   f"their maximum velocity)")
            if vmax_ok:
                msg += f"; the takes without folding had v_max of {min(vmax_ok):.2f} m/s or more"
            else:
                msg += (f"; every configuration folded - the next series should reach a higher "
                        f"v_max (fewer TX in TDM, shorter idle or ramp): the motion reached at least "
                        f"{fastest:.2f} m/s")
            out.append(msg)
        clipped = [r.stem for r in rs if ((r.quality or {}).get("metrics", {}).get("clip_fraction") or 0) > 1e-4]
        if clipped:
            out.append(f"ADC clipping in {len(clipped)} take(s): lower the RX gain")
        for name, rows in self.effects().items():
            lab = PARAMS[name].label if name in PARAMS else name
            top, bottom = rows[0], rows[-1]
            if top[1] - bottom[1] < 1.5:
                out.append(f"{lab}: little difference ({bottom[1]:.1f} to {top[1]:.1f}) - take the "
                           f"value with the lower data rate")
            else:
                out.append(f"{lab}: {top[0]} best (mean {top[1]:.1f}), {bottom[0]} worst "
                           f"(mean {bottom[1]:.1f})")
        spreads = [r for r in rank if r["takes"] > 1 and r["spread"] > 6]
        if spreads:
            out.append("takes of the same configuration differ by more than 6 points ("
                       + ", ".join(r["key"] for r in spreads[:3]) + "): the movement varied - "
                       "more takes or a steadier movement before trusting small differences")
        if bm.get("target_near_max"):
            out.append("the target reaches the end of the covered range in the best configuration: "
                       "cover more range (a lower slope or a higher sample rate)")
        return out

    # --- files -------------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA, "name": self.name, "folder": self.folder, "created": self.created,
            "objective": self.objective, "notes": self.notes,
            "base_params": self.base_params.to_dict() if self.base_params else None,
            "variables": [[n, [format_value(n, v) for v in vals]] for n, vals in self.variables],
            "plan": [{"step": s.index, "key": s.key, "take": s.take, "ok": s.ok,
                      "problems": [list(p) for p in s.problems], "derived": s.derived}
                     for s in self.steps],
            "results": [r.to_dict() for r in self.results],
            "ranking": self.ranking(), "effects": self.effects(),
            "recommendations": self.recommendations(),
        }

    def write(self) -> dict:
        os.makedirs(self.folder, exist_ok=True)
        base = os.path.join(self.folder, self.name)
        paths = {"json": base + "_series.json", "csv": base + "_series.csv",
                 "html": base + "_report.html"}
        with open(paths["json"], "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        self._write_csv(paths["csv"])
        with open(paths["html"], "w", encoding="utf-8") as f:
            f.write(self.html())
        return paths

    METRIC_COLUMNS = ("motion_snr_db", "target_range_m", "range_span_m", "v95_ms", "edge_fraction",
                      "occupied_bins", "phase_step_p95_deg", "moving_fraction", "clip_fraction",
                      "zero_filled_packets")
    DERIVED_COLUMNS = ("range_res_cm", "max_range_m", "max_vel_ms", "vel_res_ms", "fps",
                       "rate_mb_s", "virtual_antennas")

    def _write_csv(self, path: str):
        names = []
        for r in self.results:
            for k in r.values:
                if k not in names:
                    names.append(k)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "take", "stem", "status", "score"] + names + list(self.DERIVED_COLUMNS)
                       + list(self.METRIC_COLUMNS) + ["verdict", "headline"])
            for r in self.results:
                m = (r.quality or {}).get("metrics", {})
                w.writerow([r.step, r.take, r.stem, r.status,
                            "" if r.score is None else f"{r.score:.2f}"]
                           + [format_value(k, r.values[k]) if k in r.values else "" for k in names]
                           + [_num(r.derived.get(k)) for k in self.DERIVED_COLUMNS]
                           + [_num(m.get(k)) for k in self.METRIC_COLUMNS]
                           + [(r.quality or {}).get("verdict", ""), (r.quality or {}).get("headline", "")])

    def html(self) -> str:
        e = html.escape
        obj, why = OBJECTIVES[self.objective]
        rows = self.ranking()
        best_key = rows[0]["key"] if rows else None
        parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{e(self.name)} - series report</title>",
                 "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#222}"
                 "table{border-collapse:collapse;margin:8px 0 20px}td,th{border:1px solid #ccc;"
                 "padding:4px 8px;font-size:13px;text-align:right}th{background:#f3f3f3}"
                 "td.l,th.l{text-align:left}tr.best td{background:#e8f5e9;font-weight:600}"
                 ".error{color:#c62828}.warning{color:#a15c00}.info{color:#1f5fa8}"
                 "h1{font-size:22px}h2{font-size:17px;margin-top:28px}</style></head><body>",
                 f"<h1>Series {e(self.name)}</h1>",
                 f"<p>{e(self.created)} &middot; ranked for <b>{e(obj)}</b> ({e(why)}) &middot; "
                 f"{len(self.results)} take(s) of {len(self.steps)} planned &middot; {e(self.folder)}</p>"]
        if self.notes:
            parts.append("<p>" + " &middot; ".join(f"{e(str(k))}: {e(str(v))}" for k, v in self.notes.items()) + "</p>")
        parts.append("<h2>What to take from it</h2><ul>")
        parts += [f"<li>{e(t)}</li>" for t in self.recommendations()]
        parts.append("</ul>")
        if rows:
            parts.append("<h2>Ranking of the configurations</h2><table><tr><th>#</th>"
                         "<th class='l'>configuration</th><th>score</th><th>takes</th><th>spread</th>"
                         "<th>range res. cm</th><th>max range m</th><th>v max m/s</th><th>v res. m/s</th>"
                         "<th>fps</th><th>MB/s</th><th>motion SNR dB</th><th>|v| 95 % m/s</th>"
                         "<th>folding %</th><th>vel. bins</th></tr>")
            for i, r in enumerate(rows, 1):
                d, m = r["derived"], r["metrics"]
                cls = " class='best'" if r["key"] == best_key else ""
                parts.append(
                    f"<tr{cls}><td>{i}</td><td class='l'>{e(r['key'])}</td><td>{r['score']:.1f}</td>"
                    f"<td>{r['takes']}</td><td>{r['spread']:.1f}</td>"
                    f"<td>{_num(d.get('range_res_cm'), 2)}</td><td>{_num(d.get('max_range_m'), 2)}</td>"
                    f"<td>{_num(d.get('max_vel_ms'), 2)}</td><td>{_num(d.get('vel_res_ms'), 3)}</td>"
                    f"<td>{_num(d.get('fps'), 1)}</td><td>{_num(d.get('rate_mb_s'), 1)}</td>"
                    f"<td>{_num(m.get('motion_snr_db'), 1)}</td><td>{_num(m.get('v95_ms'), 2)}</td>"
                    f"<td>{_num(100 * m['edge_fraction'] if m.get('edge_fraction') is not None else None, 0)}</td>"
                    f"<td>{_num(m.get('occupied_bins'))}</td></tr>")
            parts.append("</table>")
        eff = self.effects()
        if eff:
            parts.append("<h2>How much each parameter mattered</h2><table><tr><th class='l'>parameter</th>"
                         "<th class='l'>value: mean score (takes), best first</th></tr>")
            for name, vals in eff.items():
                lab = PARAMS[name].label if name in PARAMS else name
                parts.append(f"<tr><td class='l'>{e(lab)}</td><td class='l'>" + " &nbsp; ".join(
                    f"<b>{e(v)}</b>: {s:.1f} ({n})" for v, s, n in vals) + "</td></tr>")
            parts.append("</table>")
        parts.append("<h2>Every take</h2><table><tr><th>step</th><th>take</th><th class='l'>recording</th>"
                     "<th class='l'>configuration</th><th class='l'>status</th><th>score</th>"
                     "<th class='l'>report</th></tr>")
        for r in self.results:
            q = r.quality or {}
            finds = "".join(f"<div class='{e(f[0])}'>{e(f[0])}: {e(f[1])} - {e(f[2])}</div>"
                            for f in q.get("findings", []))
            terms = ", ".join(f"{e(n)} {t:+.1f}" for n, t in r.score_terms)
            parts.append(f"<tr><td>{r.step}</td><td>{r.take}</td><td class='l'>{e(r.stem)}</td>"
                         f"<td class='l'>{e(r.key)}</td><td class='l'>{e(r.status)}"
                         f"{(' - ' + e(r.error)) if r.error else ''}</td>"
                         f"<td>{'' if r.score is None else f'{r.score:.1f}'}</td>"
                         f"<td class='l'>{e(q.get('headline', ''))}<div style='color:#666'>{terms}</div>{finds}</td></tr>")
        parts.append("</table>")
        skipped = [s for s in self.steps if not s.ok]
        if skipped:
            parts.append("<h2>Combinations that cannot be recorded</h2><ul>")
            for s in skipped:
                errs = "; ".join(f"{p[1]}: {p[2]}" for p in s.problems if p[0] == "error")
                parts.append(f"<li>{e(s.label)}: {e(errs)}</li>")
            parts.append("</ul>")
        parts.append("<p style='color:#666;font-size:12px'>Scores are heuristics for comparing takes "
                     "of the same movement (motion SNR in dB, minus penalties for folding, lost data, "
                     "clipping, plus what the kind of movement needs). Open the recordings in "
                     "DARTscope to look at them properly.</p></body></html>")
        return "\n".join(parts)


def _num(v, digits: int = 3) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)
