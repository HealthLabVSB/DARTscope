"""
Cache, per capture, what the vital-sign evaluation of the paper needs, so that the
algorithm variants (tools.paper.chain) need not decode the .bin again:

  * the chirp-mean range FFT exactly as tools.vitals_check.vitals_of computes it
    (n_fft_range = max(256, range_fft_size)), kept for the bins up to R_KEEP_M;
  * the frozen-default Vital Signs result on the FULL cube (window times, rates,
    confidences, bins) - the reference the re-implemented chain is checked against;
  * the SOMNOtouch reference around the capture (PPG beats, Pulse, breathing
    surrogates), via tools.vitals_check.load_reference.

    python -m tools.paper.cache_vitals <radar_dir> <out_dir> [--jobs N] [--only STEM ...]

Nothing in DARTscope is changed. Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

R_KEEP_M = 6.0


def _one(args):
    stem, radar_dir, out_dir = args
    import capture_params
    import vitalsigns
    from radar_processing import (compute_range_fft_cube, derive_capture_dimensions,
                                  load_raw_cube, range_fft_size)
    from tools.captures import find_captures
    from tools.pair_reference import DEFAULT_REFERENCE_DIRNAME
    from tools import vitals_check as vc
    t0 = time.time()
    cap = [c for c in find_captures(radar_dir) if c.stem == stem][0]
    ref = vc.load_reference(cap, os.path.join(radar_dir, DEFAULT_REFERENCE_DIRNAME))
    if ref is None:
        return stem, "no reference"
    if not cap.duration_s or cap.duration_s < 30.0:
        return stem, f"too short ({cap.duration_s or 0:.0f} s)"
    params = dict(cap.params)
    with contextlib.redirect_stdout(io.StringIO()):
        cube = load_raw_cube(cap.bin_paths[0], params, derive_capture_dimensions(dict(params)))
    damaged = vitalsigns.detect_damaged_frames(cube)
    n_chirps = int(cube.shape[1])
    x = cube.mean(axis=1, keepdims=True)
    del cube
    n = int(params.get("numAdcSamples") or x.shape[-1])
    with contextlib.redirect_stdout(io.StringIO()):
        rf, meta = compute_range_fft_cube(x, params, return_complex=True,
                                          n_fft_range=max(256, range_fft_size(n, None)))
        fps = capture_params.derive_fps(params)
    r = np.asarray(meta.get("range_m") if meta.get("range_m") is not None else meta["range_axis"])
    cfg = {**vitalsigns.DEFAULT_CONFIG, "damaged_frames": damaged}
    with contextlib.redirect_stdout(io.StringIO()):
        res = vitalsigns.process_vital_signs_logic(rf, params, fps, r, cfg)
    keep = int(np.searchsorted(r, R_KEEP_M)) + 1
    start_ghz = float(params.get('effectiveStartFreqGHz') or params.get('startFreqGHz')
                      or params.get('startFreq_GHz') or 77.0)
    resp = {f"resp_{k}_t": v[0] for k, v in ref.resp.items()}
    resp.update({f"resp_{k}_x": v[1] for k, v in ref.resp.items()})
    np.savez_compressed(
        os.path.join(out_dir, f"{stem}.npz"),
        rf=np.asarray(rf[:, 0, :, :keep], np.complex64), r=r[:keep], fps=float(fps),
        start_ghz=start_ghz, damaged=np.asarray(damaged, bool),
        d_hr_t=res["hr_times"], d_hr_v=res["hr_values"], d_hr_c=res["hr_confs"],
        d_rr_t=res["rr_times"], d_rr_v=res["rr_values"], d_rr_c=res["rr_confs"],
        d_best_bin=int(res["best_bin"]), d_bin_used=int(res["bin_used"]),
        d_dist_m=float(res["dist_m"]), d_heart_channels=int(res["heart_channels"]),
        beat_t=ref.beat_t, pulse_t=ref.pulse_t, pulse=ref.pulse, **resp)
    meta_out = dict(stem=stem, session=ref.session, start=cap.start.isoformat(),
                    duration_s=float(cap.duration_s), fps=float(fps), n_chirps=n_chirps,
                    n_adc=n, n_rx=int(rf.shape[2]), n_bins_full=int(rf.shape[-1]),
                    n_bins_kept=keep, slope_MHz_us=params.get("freqSlopeConst_MHz_usec")
                    or params.get("freqSlope"), start_ghz=start_ghz,
                    damaged_frames=int(np.sum(damaged)), seconds=round(time.time() - t0, 1))
    with open(os.path.join(out_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=1, default=str)
    return stem, f"ok {meta_out['seconds']} s"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("radar_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    from tools.captures import find_captures
    os.makedirs(a.out_dir, exist_ok=True)
    stems = [c.stem for c in find_captures(a.radar_dir) if not a.only or c.stem in a.only]
    jobs = [(s, a.radar_dir, a.out_dir) for s in stems]
    if a.jobs <= 1:                      # serial (sandboxes without named pipes)
        for j in jobs:
            try:
                stem, msg = _one(j)
            except Exception as e:       # one bad capture must not stop the rest
                stem, msg = j[0], f"FAILED {type(e).__name__}: {e}"
            print(f"{stem}: {msg}", flush=True)
        return 0
    with ProcessPoolExecutor(a.jobs) as ex:
        for stem, msg in ex.map(_one, jobs):
            print(f"{stem}: {msg}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
