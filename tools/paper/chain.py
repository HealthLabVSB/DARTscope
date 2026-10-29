"""
The Vital Signs chain of DARTscope (vitalsigns.process_vital_signs_logic with its
DEFAULT_CONFIG) re-run from the cached chirp-mean range FFT (tools.paper.cache_vitals),
with ONE stage swapped at a time for the ablation of the paper. With every option at
its default, run() reproduces the frozen default chain window by window
(tools.paper.evaluate checks this on every capture).

Stages and their options (default first):

  demod     "unwrap"  arc centred (mean removed), atan2, np.unwrap      (DEFAULT)
            "dacm"    differentiate-and-cross-multiply (vitalsigns.extract_phase_dacm)
            "circle"  Pratt algebraic circle fit, phase about the fitted centre
                      (ai_vital_analysis.AIVitalAnalysis._phase_circle_fit)
  combine   heart channels = receivers x the chosen bin and its two neighbours (12 for 4 RX)
            "gated"   mean displacement of the centre bin's receivers and of the
                      neighbours' receivers whose Q_arc >= arc_min (DEFAULT, arc_min 1.5)
            "centre"  mean displacement of the centre bin's receivers only
            "all"     mean displacement of all 12, no gate
            "best"    the single channel with the largest Q_arc
            "qarc2"   displacement mean weighted by Q_arc^2 (an SNR weighting)
            "sum"     coherent complex sum of the centre bin's centred receivers (SUM)
            "eig"     principal eigenvector of the 12 centred complex channels (beam)
            "mrc"     maximal-ratio combining: channel gains from the principal
                      eigenvector of the noise-whitened channels, weights gain/sigma^2
  notch     None      breathing harmonics 2-4 notched in the heart spectra (DEFAULT)
            "comb"    every breathing harmonic up to the top of the harmonic sum
  heart_method "auto" harmonic sums of the heart-band displacement AND of the envelope
                      of its 5-25 Hz content, added (DEFAULT); "envelope" / "displacement":
                      one of them (moving_hr.heart_spectra, an existing option)
  prewhiten False     opt-in: heart-band displacement differentiated before its spectra
  (the Q_edge gate acts on the windows afterwards: tools.paper.evaluate)

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

import fastcheck
import moving_hr
import vital_bands
import vitalsigns

CACHE_DIR = os.environ.get("DARTSCOPE_PAPER_CACHE", os.path.join("paper_results", "vitals", "cache"))
WIN_S = vitalsigns.DEFAULT_CONFIG["win_hr"]          # 30 s heart windows
WIN_BR_S = vitalsigns.DEFAULT_CONFIG["win_br"]       # 30 s breathing windows
STEP_S = vitalsigns.DEFAULT_CONFIG["win_step"]       # 1 s step
CONF_TH = vitalsigns.DEFAULT_CONFIG["conf_thresh"]   # 6.0 (Min peak/avg)


@dataclass
class Capture:
    stem: str
    meta: dict
    rf: np.ndarray            # (F, RX, R) chirp-mean range FFT, complex64
    r: np.ndarray
    fps: float
    wave_len: float
    damaged: np.ndarray
    default: dict             # the frozen default chain's result on the full cube
    ref: object               # tools.vitals_check.Reference


def load(stem: str, cache_dir: str = CACHE_DIR) -> Capture:
    from tools.vitals_check import Reference
    z = np.load(os.path.join(cache_dir, f"{stem}.npz"))
    with open(os.path.join(cache_dir, f"{stem}.json"), encoding="utf-8") as f:
        meta = json.load(f)
    resp = {}
    for k in ("riiv", "riav", "rifv"):
        if f"resp_{k}_t" in z.files:
            resp[k] = (z[f"resp_{k}_t"], z[f"resp_{k}_x"])
    ref = Reference(meta["session"], meta["duration_s"], z["beat_t"], z["pulse_t"], z["pulse"], resp)
    default = {k[2:]: z[k] for k in z.files if k.startswith("d_")}
    return Capture(stem, meta, z["rf"], z["r"], float(z["fps"]), 3e8 / (float(z["start_ghz"]) * 1e9),
                   z["damaged"], default, ref)


def stems(cache_dir: str = CACHE_DIR) -> list:
    return sorted(f[:-4] for f in os.listdir(cache_dir) if f.endswith(".npz"))


# --- demodulation -------------------------------------------------------------------------

def _circle_fit_phase(z):
    from ai_vital_analysis import AIVitalAnalysis
    with contextlib.redirect_stdout(io.StringIO()):
        return AIVitalAnalysis._phase_circle_fit(np.asarray(z))


def displacement(z, wave_len, demod="unwrap"):
    """Detrended chest displacement [m] of one complex slow-time channel z."""
    z = np.asarray(z, complex)
    zc = vitalsigns.dc_offset_correction(z, method="mean")
    if demod == "unwrap":
        ph = np.unwrap(np.angle(zc))
        d = ph * (wave_len / (4 * np.pi))
    elif demod == "dacm":
        d = vitalsigns.extract_phase_dacm(zc, wave_len)
    elif demod == "circle":
        d = _circle_fit_phase(z) * (wave_len / (4 * np.pi))
    else:
        raise ValueError(demod)
    return signal.detrend(d)


# --- the chain --------------------------------------------------------------------------------

@dataclass
class Result:
    stem: str
    best_bin: int
    bin_used: int
    dist_m: float
    rr_t: np.ndarray
    rr_v: np.ndarray
    rr_c: np.ndarray
    hr_t: np.ndarray
    hr_v: np.ndarray
    hr_c: np.ndarray
    grid: np.ndarray
    spec: np.ndarray
    channels: list             # (bin offset, rx, Q_arc whole capture, admitted)
    d_heart: np.ndarray
    extra: dict = field(default_factory=dict)


def choose_bins(c: Capture, bands):
    """(best_bin, bin_used) as the default chain picks them (Auto Bin, Best bin in ROI)."""
    rf4 = c.rf[:, None]                                              # (F, 1, RX, R)
    best = vitalsigns.auto_bin(rf4, c.fps, bands.resp_hz, c.r, bands=bands)
    R = c.rf.shape[-1]
    best = max(0, min(best, R - 1))
    roi = np.arange(max(0, best - 1), min(R, best + 2))
    k_best = int(np.flatnonzero(roi == best)[0])
    if c.rf.shape[0] >= 8 * c.fps and c.fps > 2 * bands.resp_hz[1]:
        with contextlib.redirect_stdout(io.StringIO()):
            k, _, _ = fastcheck.breathing_bin(c.rf[:, :, roi], c.fps, bands, np.asarray(c.r)[roi],
                                              None, min_range_m=0.0, nyquist_last=False)
        if k is not None:
            k_best = int(k)
    return int(best), int(roi[k_best])


def _noise_sigma(z):
    """The per-sample noise RMS arc_snr uses (median of the top 30 % of the band)."""
    zc = np.asarray(z, complex) - np.mean(z)
    p = np.abs(np.fft.fft(zc)) ** 2 / zc.size
    hi = np.abs(np.fft.fftfreq(zc.size)) > 0.35
    return float(np.sqrt(np.median(p[hi]) / np.log(2.0)))


def heart_signal(c: Capture, bin_used: int, combine="gated", demod="unwrap", arc_min=None,
                 hw: int = vitalsigns.HEART_BINS_HW):
    """(d_heart, channel list) for one combining option."""
    arc_min = vitalsigns.ARC_SNR_MIN if arc_min is None else float(arc_min)
    R = c.rf.shape[-1]
    lo, hi = max(0, bin_used - hw), min(R, bin_used + hw + 1)
    chans = []
    for b in range(lo, hi):
        for i in range(c.rf.shape[1]):
            z = c.rf[:, i, b]
            q = vitalsigns.arc_snr(z)
            chans.append(dict(off=b - bin_used, rx=i, z=z, q=q,
                              admitted=(b == bin_used) or q >= arc_min))
    info = [(ch["off"], ch["rx"], ch["q"], ch["admitted"]) for ch in chans]
    if combine in ("gated", "centre", "all", "best", "qarc2"):
        if combine == "gated":
            use = [ch for ch in chans if ch["admitted"]]
        elif combine == "centre":
            use = [ch for ch in chans if ch["off"] == 0]
        elif combine == "all":
            use = chans
        elif combine == "best":
            use = [max(chans, key=lambda ch: ch["q"])]
        else:
            use = chans
        D = np.array([displacement(ch["z"], c.wave_len, demod) for ch in use])
        if combine == "qarc2":
            w = np.array([ch["q"] ** 2 for ch in use])
            d = (w[:, None] * D).sum(0) / w.sum()
        else:
            d = D.mean(axis=0)
        info = [(o, r, q, any(u["off"] == o and u["rx"] == r for u in use)) for o, r, q, _ in info]
        return d, info
    # coherent combining in the complex domain, then one demodulation
    if combine == "sum":
        use = [ch for ch in chans if ch["off"] == 0]
        Y = np.array([ch["z"] - ch["z"].mean() for ch in use])
        zs = Y.sum(axis=0)
    elif combine in ("eig", "mrc"):
        use = chans
        Y = np.array([ch["z"] - ch["z"].mean() for ch in use])
        if combine == "mrc":
            sig = np.array([max(_noise_sigma(ch["z"]), 1e-30) for ch in use])
            Yw = Y / sig[:, None]
            _, V = np.linalg.eigh(Yw @ Yw.conj().T / Y.shape[1])
            w = V[:, -1] / sig                       # gains/sigma^2 = (u*sigma)/sigma^2
        else:
            _, V = np.linalg.eigh(Y @ Y.conj().T / Y.shape[1])
            w = V[:, -1]
        zs = w.conj() @ Y
    else:
        raise ValueError(combine)
    info = [(o, r, q, any(u["off"] == o and u["rx"] == r for u in use)) for o, r, q, _ in info]
    return displacement(zs + 0.0, c.wave_len, demod), info


def run(c: Capture, *, demod="unwrap", combine="gated", arc_min=None, notch=None,
        heart_method="auto", prewhiten=False, keep_spectra=True) -> Result:
    """The chain with one option changed (see the module notes)."""
    cfg = dict(vitalsigns.DEFAULT_CONFIG)
    bands = vital_bands.from_config(cfg)
    fps = c.fps
    best, bin_used = choose_bins(c, bands)
    # breathing: the centre bin's receivers, their spectra summed (default path)
    rx_disps = np.array([displacement(c.rf[:, i, bin_used], c.wave_len, demod)
                         for i in range(c.rf.shape[1])])
    sos_br = signal.butter(4, list(bands.resp_hz), "bandpass", fs=fps, output="sos")
    br_in = signal.sosfiltfilt(sos_br, rx_disps, axis=-1)
    rr_t, rr_v, rr_c = vitalsigns.sliding(br_in, fps, WIN_BR_S, STEP_S, bands.resp_hz)
    # heart (step 1 s = the tracker's own step, so heart_rate_track decimates nothing)
    d_heart, info = heart_signal(c, bin_used, combine, demod, arc_min)
    hr_t, hr_v, hr_c, grid, spec = vitalsigns.heart_rate_track(
        d_heart, fps, bands, WIN_S, STEP_S, rr_t, rr_v, breath_notch=notch,
        method=heart_method, prewhiten=prewhiten)
    return Result(c.stem, best, bin_used, float(c.r[bin_used]), rr_t, rr_v, rr_c, hr_t, hr_v,
                  hr_c, grid, spec if keep_spectra else None, info, d_heart)
