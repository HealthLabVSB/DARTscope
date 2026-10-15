"""
acquisition/layout.py

A radar cube written in the byte layout of an mmWave Studio + DCA1000 recording - the
inverse of radar_processing.lvds_decode_concat - so a simulated DCA1000 can stream
samples whose every value is known, and the test can require the unchanged loader to
give exactly those values back.

The layout (docs/ACQUISITION.md, section 3), complex, non-interleaved:
    cube [frames, chirps, RX, samples] read in that order (C order) is one sample stream;
    each group of `lanes` consecutive samples becomes `lanes` I values then `lanes` Q values
    (2 lanes: I(n) I(n+1) Q(n) Q(n+1)); QI order (iqSwap) puts Q first;
    int16, little-endian.
Real: the I values in stream order.

This only fixes what the loader reads; which physical layout the hardware produces was
established separately, on real data (pcap vs. recordings, docs/ACQUISITION.md).

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import numpy as np

from .config import RadarConfig


def encode_cube(cube: np.ndarray, lanes: int = 2, complex_: bool = True,
                iq_swap: bool = False) -> bytes:
    """(F, C, RX, Ns) integer-valued samples -> the bytes of a recording."""
    x = np.asarray(cube)
    flat = x.reshape(-1)
    if not complex_:
        re = np.asarray(flat.real if np.iscomplexobj(flat) else flat)
        return _int16(re).astype("<i2").tobytes()
    if flat.size % lanes:
        raise ValueError(f"{flat.size} samples do not fill whole groups of {lanes} lanes")
    i = _int16(flat.real).reshape(-1, lanes)
    q = _int16(flat.imag).reshape(-1, lanes)
    first, second = (q, i) if iq_swap else (i, q)
    return np.concatenate([first, second], axis=1).astype("<i2").tobytes()


def encode_for(cfg: RadarConfig, cube: np.ndarray) -> bytes:
    """encode_cube with the lanes, format and IQ order of a configuration."""
    lanes = cfg.dca.container_lanes or cfg.lvds.n_lanes
    return encode_cube(cube, lanes=lanes, complex_=cfg.channel.is_complex,
                       iq_swap=bool(cfg.channel.iq_swap))


def _int16(a) -> np.ndarray:
    a = np.asarray(a)
    r = np.rint(a)
    if np.any(r != a) or r.min(initial=0) < -32768 or r.max(initial=0) > 32767:
        raise ValueError("samples must be whole numbers within int16")
    return r.astype(np.int16)


def random_cube(cfg: RadarConfig, frames: int, seed: int = 0, amplitude: int = 2000) -> np.ndarray:
    """
    Known samples for a configuration: noise plus a tone per RX and chirp, so that every
    frame, chirp, RX, sample and the I/Q assignment are distinguishable - a transposition
    anywhere changes the values the loader returns.
    """
    rng = np.random.default_rng(seed)
    c, rx, ns = cfg.chirps_per_frame, cfg.channel.n_rx, cfg.profile.num_adc_samples
    shape = (int(frames), c, rx, ns)
    freq = 0.05 + 0.01 * np.arange(rx)                           # one beat tone per RX
    tone = amplitude * np.exp(2j * np.pi * freq[:, None] * np.arange(ns)[None, :])
    ramp = 1 + 0.1 * np.arange(c) / max(c, 1)                    # and a level per chirp
    base = np.broadcast_to(tone[None, None] * ramp[None, :, None, None], shape)
    re = np.rint(base.real) + rng.integers(-300, 300, shape)
    if not cfg.channel.is_complex:
        return re
    return re + 1j * (np.rint(base.imag) + rng.integers(-300, 300, shape))
