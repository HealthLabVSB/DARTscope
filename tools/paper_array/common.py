"""
Shared helpers for the array/physics evidence of the T-MTT paper (tools/paper_array).

Nothing here changes DARTscope behaviour: the recordings are read with the same
loader (radar_processing.load_raw_cube) and range FFT (compute_range_fft_cube) the
GUI uses; only the bookkeeping around them (chunking, TDM slot split, nominal
geometry of the October 2026 positions) lives here.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import contextlib
import io
import math
import os
import re

import numpy as np

import capture_params
from radar_processing import compute_range_fft_cube, derive_capture_dimensions, load_raw_cube
from tools.captures import find_captures

# the recordings, the tables and the figures; set by environment variables
DATA_DIR = os.environ.get("DARTSCOPE_PAPER_DATA", "data")
OUT_DIR = os.environ.get("DARTSCOPE_PAPER_OUT", os.path.join("paper_results", "array"))
FIG_DIR = os.environ.get("DARTSCOPE_PAPER_FIGURES", "figures")
RADAR_HEIGHT_M = 1.165


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def captures(day: str | None = None):
    """Captures of DATA_DIR; day '2026-10-06' style filters by start date."""
    caps = find_captures(DATA_DIR)
    if day:
        caps = [c for c in caps if c.start is not None and c.start.strftime("%Y-%m-%d") == day]
    return caps


def capture(stem: str):
    for c in find_captures(DATA_DIR):
        if c.stem == stem:
            return c
    raise KeyError(stem)


def load_cube(cap):
    """(params, cube (F, C, RX, S) complex64) with the GUI's loader."""
    params = dict(cap.params)
    with quiet():
        dims = derive_capture_dimensions(dict(params))
        cube = load_raw_cube(cap.bin_paths[0], params, dims)
    return params, cube


def fps_of(params) -> float:
    with quiet():
        return float(capture_params.derive_fps(params))


def range_fft(cube, params, n_fft=256, rmax=3.5, chunk=200, reduce=None):
    """
    Complex range FFT (DARTscope's compute_range_fft_cube: DC removal, Hann, n_fft
    points) in chunks of frames, cropped to 0..rmax m. reduce(rf_chunk) -> array may
    shrink each chunk (e.g. a mean over loops) before it is kept.
    Returns (out, range_axis_m).
    """
    out, sel, rax = [], None, None
    for f0 in range(0, cube.shape[0], chunk):
        with quiet():
            rf, meta = compute_range_fft_cube(cube[f0:f0 + chunk], params, return_complex=True,
                                              n_fft_range=n_fft)
        if sel is None:
            r = np.asarray(meta.get("range_m") if meta.get("range_m") is not None
                           else meta["range_axis"])
            sel = np.flatnonzero((r >= 0) & (r <= rmax))
            rax = r[sel]
        rf = rf[..., sel].astype(np.complex64)
        out.append(reduce(rf) if reduce is not None else rf)
    return np.concatenate(out, 0), rax


def wavelength_m(params) -> float:
    f = float(params.get("effectiveStartFreqGHz") or params.get("startFreqGHz") or 60.25)
    f_stop = params.get("stopFreqGHz")
    fc = 0.5 * (f + float(f_stop)) if f_stop else f
    return 299_792_458.0 / (fc * 1e9)


# --- nominal geometry of the October 2026 positions --------------------------------------------

def nominal_geometry(stem: str) -> dict | None:
    """
    Nominal subject position from the recording name (Doplneni_zaznamu_TMTT.csv):
    distance d (radar chip -> chest, along the radar axis), lateral offset o, orientation
    (0 = facing the radar, 90 = sideways, radar at the subject's right hand).
    Slant range sqrt(d^2 + o^2), azimuth atan(o / d).
    Aspect angle alpha (chest normal vs line of sight): 0-deg positions alpha = atan(o/d);
    90-deg positions alpha = 90 - atan(o/d) (ASSUMED: chest turned towards the radar side
    when offset; the protocol does not say which way the subject faced).
    """
    m = re.match(r"sitting_still_(0|90)deg(?:_(\d)m)?(?:_(\d)mright)?(?:_\d+)?$", stem)
    if m:
        orient = int(m.group(1))
        d = float(m.group(2) or 1.0)
        o = float(m.group(3) or 0.0)
    else:
        m = re.match(r"new_HR_seated_(1m5|1m|2m)$", stem)
        if not m:
            return None
        orient, o = 0, 0.0
        d = {"1m": 1.0, "1m5": 1.5, "2m": 2.0}[m.group(1)]
    slant = math.hypot(d, o)
    az = math.degrees(math.atan2(o, d))
    alpha = az if orient == 0 else 90.0 - az
    return dict(orient=orient, d=d, o=o, slant=slant, az=az, alpha=alpha)


def config_label(params) -> str:
    masks = list(params.get("tx_order_masks") or [])
    n = int(params.get("numAdcSamples") or 0)
    if masks == [1, 2, 4]:
        return "C"
    if masks == [2] and n == 256:
        return "A"
    if masks == [1] and n == 80:
        return "B"
    return "other"
