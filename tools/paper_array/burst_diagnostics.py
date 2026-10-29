"""
Are the chirps interference_check flags foreign FMCW bursts or target motion?

    python -m tools.paper_array.burst_diagnostics [STEM ...]

For the hit chirps of the spectral-median rule (interference_check, mode 'specmed'):
  * fast-time burstiness: peak / median of the 4-sample-smoothed residual envelope
    (mean over RX) - a foreign chirp crossing the IF band is a short burst in fast time;
    a target that moved within the frame leaves a residual spread over the whole chirp.
    The same number for 200 random chirps of the recording as a reference;
  * timing: chirp index within the frame (histogram) and the intervals between hit
    frames - an asynchronous periodic radar gives a fixed beat period.
Writes burst_diagnostics.json in common.OUT_DIR.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from tools.paper_array import common
from tools.paper_array import interference_check as ic

DEFAULT = ["sitting_still_0deg_2m_1mright", "sitting_still_0deg_7", "sitting_still_90deg_2m_3",
           "sitting_swing", "HR_moving_4"]


def env_ratio(cube, f, c, pat):
    xs = cube[f, c % pat::pat]
    med = np.median(xs.real, 0) + 1j * np.median(xs.imag, 0)
    e = np.abs(cube[f, c] - med).mean(0)
    k = np.convolve(e, np.ones(4) / 4, "same")
    return float(k.max() / np.median(k))


def diagnose(stem, n_max=200, seed=0):
    cap = common.capture(stem)
    params, cube = common.load_cube(cap)
    pat = ic.slot_pattern(params)
    fps = common.fps_of(params)
    d = ic.detect(cube, pat, mode="specmed")
    fh, ch = np.nonzero(d["hit"])
    rng = np.random.default_rng(seed)
    nf = rng.integers(0, cube.shape[0], n_max)
    nc = rng.integers(0, cube.shape[1], n_max)
    hit_r = [env_ratio(cube, f, c, pat) for f, c in zip(fh[:n_max], ch[:n_max])]
    rnd_r = [env_ratio(cube, f, c, pat) for f, c in zip(nf, nc)]
    t = np.unique(fh) / fps
    dt = np.diff(t)
    ex = d["excess_db"][fh, ch]
    return dict(stem=stem, config=common.config_label(params), hit_chirps=int(len(fh)),
                hit_frames=int(len(np.unique(fh))),
                burstiness_hit_median=float(np.median(hit_r)) if hit_r else None,
                burstiness_random_median=float(np.median(rnd_r)),
                excess_db_median=float(np.median(np.sort(ex, 1)[:, -3])) if len(ex) else None,
                chirp_index_hist=np.bincount(ch, minlength=min(cube.shape[1], 16))[:16].tolist(),
                t_first_s=float(t.min()) if len(t) else None, t_last_s=float(t.max()) if len(t) else None,
                interval_median_s=float(np.median(dt)) if len(dt) else None,
                interval_p10_s=float(np.percentile(dt, 10)) if len(dt) > 2 else None,
                interval_p90_s=float(np.percentile(dt, 90)) if len(dt) > 2 else None)


def main(argv=None):
    stems = (argv if argv is not None else sys.argv[1:]) or DEFAULT
    out = []
    for st in stems:
        r = diagnose(st)
        out.append(r)
        print(json.dumps(r), flush=True)
    with open(os.path.join(common.OUT_DIR, "burst_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
