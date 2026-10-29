"""
Run chain variants over every cached capture and write one per-window CSV per variant.

    python -m tools.paper.run_variants OUT_DIR [--set ablation|gates|NAME ...] [--quality]
                                        [--base KEY=VALUE ...] [--prefix P]

--base sets options every variant of the run starts from (e.g. prewhiten=True for the
gate grid on the fixed chain). Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
import time

import pandas as pd

from tools.paper import chain as ch
from tools.paper import evaluate as ev

ABLATION = {
    "default": {},
    "demod=DACM": dict(demod="dacm"),
    "demod=circle fit": dict(demod="circle"),
    "combine=centre bin (4 RX)": dict(combine="centre"),
    "combine=all 12, no gate": dict(combine="all"),
    "combine=best single channel": dict(combine="best"),
    "combine=Q_arc^2-weighted": dict(combine="qarc2"),
    "combine=SUM (coherent, centre bin)": dict(combine="sum"),
    "combine=principal eigenvector": dict(combine="eig"),
    "combine=MRC": dict(combine="mrc"),
    "heart spectrum=envelope only": dict(heart_method="envelope"),
    "heart spectrum=displacement only": dict(heart_method="displacement"),
    "notch=comb (all harmonics)": dict(notch="comb"),
    "prewhiten (velocity)": dict(prewhiten=True),
    "prewhiten (acceleration)": dict(prewhiten=2),
}
GATES = {f"gate Q_arc>={a:g}": dict(arc_min=a) for a in (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)}


def safe(name: str) -> str:
    return "".join(ch_ if ch_.isalnum() or ch_ in "-_.=" else "_" for ch_ in name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--set", nargs="*", default=["ablation"])
    ap.add_argument("--quality", action="store_true", help="per-window Q_arc (slower)")
    ap.add_argument("--base", nargs="*", default=[])
    ap.add_argument("--prefix", default="")
    a = ap.parse_args(argv)
    base = {}
    for kv in a.base:
        k, v = kv.split("=", 1)
        try:
            base[k] = ast.literal_eval(v)
        except Exception:
            base[k] = v
    todo = {}
    for s in a.set:
        if s == "ablation":
            todo.update(ABLATION)
        elif s == "gates":
            todo.update(GATES)
        elif s in ABLATION:
            todo[s] = ABLATION[s]
        elif s in GATES:
            todo[s] = GATES[s]
        else:
            raise SystemExit(f"unknown variant {s}")
    os.makedirs(a.out_dir, exist_ok=True)
    caps = {s: ch.load(s) for s in ch.stems()}
    t0 = time.time()
    for name, kw in todo.items():
        parts = []
        for s, c in caps.items():
            r = ch.run(c, keep_spectra=False, **{**base, **kw})
            parts.append(ev.window_table(c, r, a.prefix + name, with_quality=a.quality))
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(os.path.join(a.out_dir, safe(a.prefix + name) + ".csv"), index=False)
        print(f"{a.prefix + name}: {len(df)} windows, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
