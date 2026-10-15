"""
acquisition/scene.py

Synthetic radar data with known content, for the DCA1000 simulator (the Measure tab's
"simulator" mode) and the tests of quality.py: static reflectors and one moving point
target, as an FMCW radar with the configuration's chirps would sample them.

For a target at range r(t) the de-chirped sample n of a chirp starting at t is
    A exp(j (2 pi f_b n / fs + 4 pi r(t) / lambda)),   f_b = 2 slope r(t) / c
with t running over frames, loops and the chirps of a loop, so range, Doppler (the phase
from chirp to chirp, TDM timing included) and slow motion across frames are all right.
Complex formats only; real formats take the real part.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import RadarConfig

C0 = 299_792_458.0


@dataclass
class Target:
    range_m: float = 1.5            # centre of the motion
    amplitude_m: float = 0.2        # of a sinusoidal motion along the line of sight
    freq_hz: float = 0.5            # of that motion
    velocity_ms: float = 0.0        # plus a steady drift (away > 0)
    level: float = 3000.0           # sample amplitude, full scale 32767


@dataclass
class Scene:
    targets: list = field(default_factory=lambda: [Target()])
    static: list = field(default_factory=lambda: [(0.8, 1500.0), (3.3, 2500.0)])   # (m, level)
    noise: float = 40.0              # standard deviation per I and Q
    seed: int = 0


def target_range(t: Target, time_s: np.ndarray) -> np.ndarray:
    return (t.range_m + t.amplitude_m * np.sin(2 * np.pi * t.freq_hz * time_s)
            + t.velocity_ms * time_s)


def scene_cube(cfg: RadarConfig, frames: int, scene: Scene | None = None) -> np.ndarray:
    """(frames, chirps, RX, samples) of whole numbers within int16, for layout.encode_for."""
    sc = scene or Scene()
    ph = cfg.physical()
    slope = ph["slope_mhz_us"] * 1e12
    fs = cfg.profile.dig_out_sample_rate * 1e3
    lam = C0 / (ph["start_ghz"] * 1e9)
    ns, rx, c = cfg.profile.num_adc_samples, cfg.channel.n_rx, cfg.chirps_per_frame
    tc = cfg.chirp_time_us * 1e-6
    period = cfg.frame.period_ms * 1e-3
    n = np.arange(ns) / fs
    rng = np.random.default_rng(sc.seed)
    out = np.empty((int(frames), c, rx, ns), dtype=np.complex64)
    chirp_t = np.arange(c) * tc
    rx_phase = np.exp(1j * np.pi * 0.25 * np.arange(rx))[:, None]       # some array phase
    for f in range(int(frames)):
        t = f * period + chirp_t                                         # (C,)
        sig = np.zeros((c, ns), dtype=np.complex128)
        for r, level in sc.static:
            fb = 2 * slope * r / C0
            sig += level * np.exp(1j * (2 * np.pi * fb * n + 4 * np.pi * r / lam))[None, :]
        for tg in sc.targets:
            r = target_range(tg, t)[:, None]                             # (C, 1)
            fb = 2 * slope * r / C0
            sig += tg.level * np.exp(1j * (2 * np.pi * fb * n[None, :] + 4 * np.pi * r / lam))
        x = sig[:, None, :] * rx_phase[None]                             # (C, RX, Ns)
        x = x + sc.noise * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
        out[f] = x
    out.real[...] = np.clip(np.rint(out.real), -32768, 32767)       # in place: no copies
    if not cfg.channel.is_complex:
        return out.real.copy()
    out.imag[...] = np.clip(np.rint(out.imag), -32768, 32767)
    return out
