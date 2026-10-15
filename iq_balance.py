"""
iq_balance.py

IQ imbalance: estimate and correction (roadmap 1.7).

A complex receiver whose Q branch has a slightly different gain g, or is not exactly
90 degrees from I (phase error phi), turns a tone e^{j theta} into
cos(theta) + j g sin(theta - phi) - in general y = mu x + nu conj(x) instead of x, with
mu = (1 + g e^{-j phi}) / 2 and nu = (1 - g e^{j phi}) / 2. Every target at +f gets a
mirror image at -f, weaker by the image rejection ratio |mu|^2 / |nu|^2. Any such
error is undone by a linear map of the I-Q plane, y_c = a y + b conj(y); two ways of
estimating it are offered.

  gram-schmidt   The textbook one: make Q orthogonal to I and give it I's power,
                 rho = E[IQ] / E[I^2], Q' = (Q - rho I) sqrt(E[I^2] / E[(Q - rho I)^2]).
                 It needs signal content whose I and Q average out, and a radar scene
                 is not that: strong targets at a few beat cycles per chirp correlate
                 I and Q for reasons unrelated to the receiver. On the 26 real
                 captures with a clear target it reported phase errors up to 40 deg
                 where the measured image rejection (31-42 dB) allows about 3 at most,
                 and correcting with it lowered the image rejection on every one of
                 them, mostly to 10-20 dB. Kept because it is the standard method and
                 works on noise-like content; not the default.

  image          From the targets themselves: for a target at +k the image bin holds
                 nu conj(X(k)) while the target bin holds mu X(k), so
                 r = sum Y(-k) Y(k) / sum |Y(k)|^2 over the clear targets and all chirps
                 estimates nu / conj(mu), and y - r conj(y) removes the image exactly
                 (up to a common scale). Noise and anything not tied to the
                 target average out over the chirps.

Measured on the same captures (estimate on 40 frames, judge on 40 others): the image
method moved the image rejection by -1 to +3 dB on most channels - these chips are
already well balanced - and by up to -11 dB where a moving target polluted the estimate.
correct() therefore keeps the correction only on the RX channels where the image
rejection of the held-out frames does not get worse.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

METHODS = ("image", "gram-schmidt")
EXCLUDE_DC_BINS = 4
TARGET_SNR_DB = 20.0


@dataclass
class Imbalance:
    a: np.ndarray               # (RX,) complex: y_c = a y + b conj(y)
    b: np.ndarray
    method: str
    gain_db: np.ndarray         # (RX,) Q branch relative to I
    phase_deg: np.ndarray       # (RX,) departure from 90 deg
    use: np.ndarray = field(default=None)   # (RX,) bool, channels to correct

    def __post_init__(self):
        if self.use is None:
            self.use = np.ones(self.a.size, dtype=bool)

    @property
    def irr_limit_db(self) -> np.ndarray:
        """Image rejection this imbalance alone would allow."""
        g = 10 ** (self.gain_db / 20)
        c = np.cos(np.deg2rad(self.phase_deg))
        return 10 * np.log10((1 + 2 * g * c + g * g) / np.maximum(1 - 2 * g * c + g * g, 1e-12))

    def summary(self) -> str:
        return "\n".join(
            f"  RX{r}: gain {self.gain_db[r]:+.2f} dB, phase {self.phase_deg[r]:+.2f} deg "
            f"(image rejection limit {self.irr_limit_db[r]:.1f} dB)"
            f"{'' if self.use[r] else ' - not corrected'}"
            for r in range(self.a.size))


def _chirps(x: np.ndarray) -> np.ndarray:
    s = x.reshape(-1, x.shape[-1])
    return s - s.mean(axis=-1, keepdims=True)


def _estimate_gram_schmidt(s: np.ndarray, exclude: int):
    n = s.shape[-1]
    spec = np.fft.fft(s, axis=-1)
    k = np.fft.fftfreq(n) * n
    spec[:, np.abs(k) < exclude] = 0.0
    h = np.fft.ifft(spec, axis=-1)
    i, q = h.real.ravel(), h.imag.ravel()
    pi, pq = np.mean(i * i), np.mean(q * q)
    rho = np.mean(i * q) / (pi + 1e-30)
    s_q = np.sqrt(pi / max(pq - rho ** 2 * pi, 1e-30))
    a = (1 + s_q - 1j * s_q * rho) / 2
    b = (1 - s_q - 1j * s_q * rho) / 2
    g = np.sqrt(pq / (pi + 1e-30))
    phi = np.arcsin(np.clip(-rho / max(g, 1e-12), -1, 1))   # Q = g sin(theta - phi)
    return a, b, 20 * np.log10(max(g, 1e-12)), np.rad2deg(phi)


def _target_bins(p: np.ndarray, min_snr_db: float, n_targets: int = 8):
    """Clear targets (local maxima >= min_snr_db over the floor), away from DC/Nyquist."""
    ns = p.size
    half = ns // 2
    noise = float(np.median(p)) + 1e-30
    band = np.r_[EXCLUDE_DC_BINS:half - 4, half + 4:ns - EXCLUDE_DC_BINS]
    ks = []
    for k in band[np.argsort(p[band])[::-1]]:
        if 10 * np.log10(p[k] / noise) < min_snr_db:
            break
        if p[k] < p[k - 1] or p[k] < p[k + 1]:
            continue
        if all(abs(k - q) > 3 and abs((ns - k) - q) > 3 for q in ks):
            ks.append(int(k))
        if len(ks) == n_targets:
            break
    return ks


def _estimate_image(s: np.ndarray, min_snr_db: float):
    ns = s.shape[-1]
    S = np.fft.fft(s * np.hanning(ns), axis=-1)
    ks = _target_bins((np.abs(S) ** 2).mean(axis=0), min_snr_db)
    if not ks:
        return 1.0 + 0j, 0j, 0.0, 0.0
    num = sum(np.sum(S[:, ns - k] * S[:, k]) for k in ks)
    den = sum(np.sum(np.abs(S[:, k]) ** 2) for k in ks)
    r = num / den                                  # nu / conj(mu) = (1 - G) / (1 + G)
    ge = (1 - r) / (1 + r)                         # G = g e^{j phi}
    return 1.0 + 0j, -r, 20 * np.log10(max(abs(ge), 1e-12)), float(np.rad2deg(np.angle(ge)))


def estimate(cube: np.ndarray, method: str = "image", rx_axis: int = 2,
             frames: int | None = 40, min_snr_db: float = TARGET_SNR_DB) -> Imbalance:
    """
    IQ imbalance of each RX channel.

    cube: complex ADC samples [frames, chirps, rx, samples] (samples last, RX on
    `rx_axis`); the first `frames` along axis 0 are used.
    """
    if method not in METHODS:
        raise ValueError(f"unknown IQ method {method!r}; choose from {METHODS}")
    x = np.asarray(cube)
    if not np.iscomplexobj(x):
        raise ValueError("IQ imbalance needs complex samples")
    if frames:
        x = x[:frames]
    x = np.moveaxis(x, rx_axis, 0)
    out = []
    for r in range(x.shape[0]):
        s = _chirps(x[r])
        out.append(_estimate_gram_schmidt(s, EXCLUDE_DC_BINS) if method == "gram-schmidt"
                   else _estimate_image(s, min_snr_db))
    a, b, g, p = (np.array(v) for v in zip(*out))
    return Imbalance(a=a.astype(complex), b=b.astype(complex), method=method,
                     gain_db=g.astype(float), phase_deg=p.astype(float))


def apply(cube: np.ndarray, imb: Imbalance, rx_axis: int = 2) -> np.ndarray:
    """Corrected copy; channels with use=False are left as they are."""
    x = np.asarray(cube)
    shape = [1] * x.ndim
    shape[rx_axis] = imb.a.size
    a = np.where(imb.use, imb.a, 1.0).reshape(shape)
    b = np.where(imb.use, imb.b, 0.0).reshape(shape)
    return (a * x + b * np.conj(x)).astype(x.dtype, copy=False)


def correct(cube: np.ndarray, method: str = "image", rx_axis: int = 2, frames: int = 80,
            tolerance_db: float = 0.5):
    """
    Estimate on the first half of `frames`, judge on the second half, correct.

    A channel is corrected only when the image rejection of the held-out frames does
    not drop by more than `tolerance_db`; a channel without a clear target cannot be
    judged and is left alone. Returns (cube, imbalance, irr_before, irr_after) with
    irr_* measured on the held-out frames (NaN where there was no clear target).
    """
    from capture_health import image_rejection_db

    x = np.asarray(cube)
    n = min(frames, x.shape[0])
    half = max(1, n // 2)
    imb = estimate(x[:half], method, rx_axis, frames=None)
    held = x[half:n] if n - half >= 1 else x[:half]
    fixed = apply(held, imb, rx_axis)

    def irr(c):
        c = np.moveaxis(c, rx_axis, 0)
        return np.array([image_rejection_db(c[r]) for r in range(c.shape[0])])

    before, after = irr(held), irr(fixed)
    imb.use = np.isfinite(before) & np.isfinite(after) & (after >= before - tolerance_db)
    return apply(x, imb, rx_axis), imb, before, after
