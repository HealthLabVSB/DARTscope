"""
cascade/calibration.py

The phase and gain of every transmitter and receiver of the AWR2243 cascade, so that its
48 virtual channels add up into one beam.

Why it is needed. An angle is read off the phase differences of one echo between the
antennas: across the cascade's 18.5-wavelength aperture one degree of angle is 116
degrees of phase at the far end (9 on the IWR6843AOP's 1.5 wavelengths). The radar
measures that phase plus the phase of the electronics on the way - feed lines (at
77 GHz a tenth of a millimetre is tens of degrees), each TX and RX channel of a chip,
and the second chip with its own front end and its own path for the LO. Small random
errors average out over 48 channels; an offset between the chips moves the beam and
splits it (180 degrees: one person at two angles), unknown phases scatter it entirely.
Breathing and heart rate do not need it - they are the change of a channel's phase
over time, and a constant offset drops out - but where a person is, and telling two
apart at one range, do.

Model. Channel (t, r) - TX t, RX r - carries tx[t] * rx[r] times the echo: every TX
and every RX has its own complex gain. 14 unknowns (less one common phase) against 48
measurements: what the product does not explain is reported as a residual, and a large
one says something else is wrong (a reflection near the reflector, coupling, a reflector
in the near field placed somewhere else than entered). A calibration from all six TX
then serves any TX pattern, the one-TX vital-signs preset included.

The echo's phase is computed from the exact path TX -> target -> RX with the element
positions of the antenna configuration, not as a plane wave: the far field of this
aperture begins at 2 D^2 / lambda = 2.7 m, and at 2 m a plane wave is 30 degrees off at
the ends of the array.

The phases hold at the centre frequency they were measured at. A channel whose path is
longer by dL arrives later, which shows as a range offset of the channel and turns its
phase by 2 pi df 2 dL / c when the frequency changes by df (1 cm and 1 GHz: 24 degrees),
so the range offset of every TX and RX is measured too and the phases are moved to the
frequency of each take.

Two ways to measure it:

  reflector  a corner reflector at a known range and angle (boresight is easiest), the
             rest of the room still. The echo of a known point: absolute angles, and a
             second position checks it (verify).
  autofocus  no reflector: the strongest still point-like reflectors of the room, each at
             an angle found with the calibration of the moment, which is then refitted to
             all of them (alternating, as Weiss and Friedlander's self-calibration). It
             makes the beam sharp, but a phase growing evenly across the aperture looks
             exactly like every target being a little further to one side (sin of every
             angle shifted alike): autofocus cannot see that turn of the whole picture.
             It puts the strongest still reflector at an angle given to it (0 deg by
             default) - angles are relative to it until something at a known angle
             confirms them.

Where it lives: the board's current calibration in the configuration folder
(current_path; DARTSCOPE_CASCADE_CALIBRATION_DIR overrides it), an archive of every
saved one next to it, and a copy in the sidecar of every take recorded with it. Decoding
a take uses the take's own copy, else the board's current one (cascade/decode.py).

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

C0 = 299_792_458.0
N_TX, N_RX = 6, 8
FORMAT = "dartscope-cascade-calibration/1"
AUTOFOCUS_R_MIN_M = 0.6          # nearer is the antenna coupling and the radar's own housing
AUTOFOCUS_TARGETS = 12           # point-like reflectors the autofocus fits at most
AUTOFOCUS_MIN_COHERENCE = 0.7    # ... and how plane they have to be once focused
GAIN_LIMITS = (0.25, 4.0)        # a channel weaker or stronger than this is not a gain to undo


# --- geometry ----------------------------------------------------------------------------

def geometry():
    """(tx (6, 2), rx (8, 2)) element positions in wavelengths: TX1-6 (masks 1 << t), RX1-8."""
    import antenna_config as ac
    cfg = ac.get_config(ac.CASCADE_CONFIG)
    rx = np.asarray(cfg["rx_positions"], float)
    offs = {int(m): np.asarray(o, float) for m, o in zip(cfg["tx_masks"], cfg["tx_offsets"])}
    return np.stack([offs[1 << t] for t in range(N_TX)]), rx


def tx_index(mask) -> int | None:
    """The TX a chirp mask switches on (0-5), None for none or several."""
    m = int(mask)
    if m <= 0 or m & (m - 1) or m >= (1 << N_TX):
        return None
    return m.bit_length() - 1


def steering(tx_idx, rx_idx, range_m: float, az_deg, wavelength_m: float, el_deg: float = 0.0):
    """
    The echo of a point at (range, az, el) on channels (tx_idx[v], rx_idx[v]): (V,) or
    (V, K) for K angles. The exact paths, in the convention of the AoA views: for a far
    target it is their plane-wave steering exp(j 2 pi (x_t + x_r) sin az).
    """
    tx, rx = geometry()
    lam = float(wavelength_m)
    R = float(range_m) / lam                                        # in wavelengths
    az = np.deg2rad(np.atleast_1d(np.asarray(az_deg, float)))
    el = np.deg2rad(float(el_deg))
    P = R * np.stack([np.sin(az) * np.cos(el), np.full(az.shape, np.sin(el)),
                      np.cos(az) * np.cos(el)], axis=1)                # (K, 3)
    T = np.column_stack([tx[np.asarray(tx_idx)], np.zeros(len(tx_idx))])
    Rr = np.column_stack([rx[np.asarray(rx_idx)], np.zeros(len(rx_idx))])
    d_t = np.linalg.norm(P[None, :, :] - T[:, None, :], axis=2)       # (V, K)
    d_r = np.linalg.norm(P[None, :, :] - Rr[:, None, :], axis=2)
    a = np.exp(1j * 2 * np.pi * (2 * R - d_t - d_r))
    return a[:, 0] if np.ndim(az_deg) == 0 else a


def virtual_x(tx_idx, rx_idx) -> np.ndarray:
    tx, rx = geometry()
    return tx[np.asarray(tx_idx), 0] + rx[np.asarray(rx_idx), 0]


def centre_frequency_ghz(params: dict) -> float:
    """The frequency in the middle of the sampled part of the chirp."""
    f0 = float(params.get("effectiveStartFreqGHz") or params.get("startFreqGHz") or 77.0)
    slope = float(params.get("freqSlopeMHz_us") or 0.0)
    adc_us = float(params.get("adcSampleTime_us") or 0.0)
    return f0 + slope * adc_us / 2.0 / 1000.0


# --- the result ----------------------------------------------------------------------------

@dataclass
class CascadeCalibration:
    method: str                                  # "reflector" or "autofocus"
    tx: np.ndarray                               # (6,) complex, TX1-6; nan where not measured
    rx: np.ndarray                               # (8,) complex
    tx_offset_m: np.ndarray                      # (6,) range offset of every TX path
    rx_offset_m: np.ndarray                      # (8,)
    f_center_ghz: float
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    residual_deg: float = float("nan")           # rms phase the TX x RX model leaves
    coherence_before: float = float("nan")       # how plane the echo(es) were, 1 = a perfect beam
    coherence_after: float = float("nan")
    reflector: dict = field(default_factory=dict)
    autofocus: dict = field(default_factory=dict)
    verify: dict = field(default_factory=dict)
    source: str = ""
    notes: list = field(default_factory=list)

    # --- applying it -----------------------------------------------------------------
    def gains(self, masks, f_center_ghz: float | None = None) -> np.ndarray | None:
        """The channel errors g (V,) of a take with these chirp masks (one slot each, 8 RX
        per slot), at its centre frequency; None when a slot sends on several TX or on
        one this calibration has not measured."""
        idx = [tx_index(m) for m in masks]
        if any(i is None or not np.isfinite(self.tx[i]) for i in idx):
            return None
        df = 0.0 if f_center_ghz is None else (float(f_center_ghz) - self.f_center_ghz) * 1e9
        g = []
        for t in idx:
            delay = 2.0 * (self.tx_offset_m[t] + self.rx_offset_m) / C0         # (8,)
            g.append(self.tx[t] * self.rx * np.exp(1j * 2 * np.pi * df * delay))
        return np.concatenate(g)

    def coefficients(self, masks, f_center_ghz: float | None = None) -> np.ndarray | None:
        """What the data is multiplied by (params['channel_calibration']): 1 / g, mean gain
        one, the first channel at zero phase."""
        g = self.gains(masks, f_center_ghz)
        if g is None:
            return None
        g = g / (g[0] / abs(g[0]))
        mag = np.clip(np.abs(g) / np.mean(np.abs(g)), *GAIN_LIMITS)
        return (1.0 / mag) * np.exp(-1j * np.angle(g))

    def describe(self) -> str:
        when = self.created.replace("T", " ")[:16]
        txt = f"{self.method}, {when}"
        if np.isfinite(self.residual_deg):
            txt += f", residual {self.residual_deg:.1f} deg"
        if self.verify.get("error_after_deg") is not None:
            txt += (f", verified at {self.verify['az_deg']:+.0f} deg: "
                    f"{self.verify['az_after']:+.1f} deg")
        if self.method == "autofocus" or (self.method == "person" and self.notes
                                          and any("relative" in n for n in self.notes)):
            txt += " (angles relative)"
        return txt

    # --- files -------------------------------------------------------------------------
    def to_dict(self) -> dict:
        c = lambda z: [[float(np.real(v)), float(np.imag(v))] if np.isfinite(v) else None for v in z]
        return {"format": FORMAT, "method": self.method, "created": self.created,
                "f_center_ghz": self.f_center_ghz, "tx": c(self.tx), "rx": c(self.rx),
                "tx_offset_m": [float(v) for v in self.tx_offset_m],
                "rx_offset_m": [float(v) for v in self.rx_offset_m],
                "residual_deg": self.residual_deg, "coherence_before": self.coherence_before,
                "coherence_after": self.coherence_after, "reflector": self.reflector,
                "autofocus": self.autofocus, "verify": self.verify, "source": self.source,
                "notes": list(self.notes)}

    @classmethod
    def from_dict(cls, d: dict) -> "CascadeCalibration":
        if not str(d.get("format", "")).startswith("dartscope-cascade-calibration/"):
            raise ValueError("not a cascade calibration")
        z = lambda a: np.array([complex(*v) if v is not None else np.nan for v in a])
        nanf = lambda v: float("nan") if v is None else float(v)
        return cls(method=d["method"], tx=z(d["tx"]), rx=z(d["rx"]),
                   tx_offset_m=np.asarray(d.get("tx_offset_m") or [0.0] * N_TX, float),
                   rx_offset_m=np.asarray(d.get("rx_offset_m") or [0.0] * N_RX, float),
                   f_center_ghz=float(d["f_center_ghz"]), created=d.get("created", ""),
                   residual_deg=nanf(d.get("residual_deg")),
                   coherence_before=nanf(d.get("coherence_before")),
                   coherence_after=nanf(d.get("coherence_after")),
                   reflector=d.get("reflector") or {}, autofocus=d.get("autofocus") or {},
                   verify=d.get("verify") or {}, source=d.get("source", ""),
                   notes=list(d.get("notes") or []))

    def summary(self) -> str:
        lines = [f"Cascade calibration ({self.describe()})",
                 f"  measured at {self.f_center_ghz:.2f} GHz from {self.source or '?'}"]
        if self.reflector and self.reflector.get("person"):
            r = self.reflector
            lines.append(f"  person at {r['range_m']:.2f} m, taken as a point at {r['az_deg']:+.1f} deg "
                         f"(raw channels: {r['az_before']:+.1f}); {r['dominance']:.2f} of the motion there is one thing")
        elif self.reflector:
            r = self.reflector
            lines.append(f"  reflector at {r.get('range_m', float('nan')):.2f} m, entered angle "
                         f"{r.get('az_deg', 0):+.1f} deg, {r.get('margin_db', float('nan')):.0f} dB above "
                         f"the cells around; its angle {r.get('az_before', float('nan')):+.1f} -> "
                         f"{r.get('az_after', float('nan')):+.1f} deg")
        if self.autofocus:
            a = self.autofocus
            lines.append(f"  {a.get('targets', 0)} still reflectors used, at "
                         + ", ".join(f"{r:.2f} m {x:+.1f} deg" for r, x in a.get('where', [])[:6]))
        what = {"autofocus": "the echoes left out of the fit", "person": "the person's echo"}.get(self.method, "the reflector")
        lines.append(f"  how plane {what} {'are' if what.endswith('s') else 'is'} (1 = one clean beam): {self.coherence_before:.2f} -> "
                     f"{self.coherence_after:.2f}; TX x RX model residual {self.residual_deg:.1f} deg")
        ph = lambda z: " ".join(f"{np.rad2deg(np.angle(v)):+5.0f}" if np.isfinite(v) else "    -" for v in z)
        lines.append(f"  TX phase [deg]: {ph(self.tx)}")
        lines.append(f"  RX phase [deg]: {ph(self.rx)}")
        lines.append("  path offsets [mm]: TX " + " ".join(f"{v * 1e3:+.1f}" for v in self.tx_offset_m)
                     + " | RX " + " ".join(f"{v * 1e3:+.1f}" for v in self.rx_offset_m))
        if self.verify:
            v = self.verify
            lines.append(f"  verified with the reflector at {v['az_deg']:+.1f} deg: {v['az_before']:+.1f} "
                         f"without, {v['az_after']:+.1f} deg with the calibration")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


# --- the data --------------------------------------------------------------------------------

@dataclass
class _Static:
    rc: np.ndarray            # (frames, V, R) range spectra of every channel, loops averaged
    range_m: np.ndarray
    tx_idx: np.ndarray        # (V,)
    rx_idx: np.ndarray
    wavelength_m: float
    f_center_ghz: float


def static_spectra(cube: np.ndarray, params: dict, frames: int = 64, oversample: int = 4) -> _Static:
    """
    The range spectra of every virtual channel of a TDM take - each TX on its own chirp -
    averaged over the loops of a frame (the still part), in `frames` frames spread over the
    take, zero-padded `oversample` times for the peaks between bins.
    """
    import vitalsigns
    from radar_processing import compute_range_fft_cube, range_fft_size

    cube = np.asarray(cube)
    F, C, RX, S = cube.shape
    if RX != N_RX:
        raise ValueError(f"a cascade take has {N_RX} RX, this one {RX}")
    eff = int(params.get("uniqueChirps_effective") or 1)
    stride = max(1, int(params.get("unique_chirps_frame") or eff))
    masks = [int(m) for m in (params.get("tx_order_masks") or [])][:eff]
    idx = [tx_index(m) for m in masks]
    if not masks or C % stride or any(i is None for i in idx) or len(set(idx)) != len(idx):
        raise ValueError("calibration needs a take with every TX on its own chirp "
                         "(TDM, e.g. the multi-person presets)")
    bad = vitalsigns.detect_damaged_frames(cube)
    good = np.flatnonzero(~bad)
    if good.size == 0:
        raise ValueError("every frame of the take is damaged")
    pick = good[np.unique(np.linspace(0, good.size - 1, min(frames, good.size)).round().astype(int))]
    L = C // stride
    x = cube[pick].reshape(pick.size, L, stride, RX, S)[:, :, :eff].mean(axis=1)   # (n, U, RX, S)
    nfft = range_fft_size(S, None) * int(oversample)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        rc, meta = compute_range_fft_cube(x, params, n_fft_range=nfft, use_full_spectrum=False,
                                          return_complex=True)
    rc = rc.reshape(pick.size, eff * RX, rc.shape[-1])
    fc = centre_frequency_ghz(params)
    return _Static(rc=rc, range_m=np.asarray(meta["range_m"], float),
                   tx_idx=np.repeat(idx, RX), rx_idx=np.tile(np.arange(RX), eff),
                   wavelength_m=C0 / (fc * 1e9), f_center_ghz=fc)


def _principal(snaps: np.ndarray) -> np.ndarray:
    """The still echo of one cell over the frames: principal eigenvector, phase of channel 0
    at zero, scaled to the mean amplitude."""
    X = np.asarray(snaps)
    w, U = np.linalg.eigh(X.T @ X.conj() / max(1, X.shape[0]))
    u = U[:, -1] * np.sqrt(max(w[-1], 0.0))
    return u * np.exp(-1j * np.angle(u[0]))


def _factor(G: np.ndarray, st: _Static, weights=None):
    """
    Rank one TX x RX fit of channel errors G (V,): (tx (6,), rx (8,), residual deg).
    TX that are not in the take stay nan.
    """
    slots = sorted(set(st.tx_idx.tolist()))
    M = np.zeros((len(slots), N_RX), complex)
    for v, (t, r) in enumerate(zip(st.tx_idx, st.rx_idx)):
        M[slots.index(t), r] = G[v]
    U, s, Vh = np.linalg.svd(M)
    a, b = U[:, 0] * np.sqrt(s[0]), Vh[0] * np.sqrt(s[0])
    ph = np.exp(-1j * np.angle(b[0]))                    # RX1 at zero phase
    a, b = a / ph, b * ph
    fit = np.outer(a, b)
    res = np.angle(M / np.where(fit == 0, 1, fit))
    w = np.abs(M) ** 2 if weights is None else weights
    residual = float(np.rad2deg(np.sqrt(np.sum(w * res ** 2) / np.sum(w))))
    tx = np.full(N_TX, np.nan + 0j)
    for i, t in enumerate(slots):
        tx[t] = a[i]
    return tx, b, residual


def _offsets(st: _Static, bins) -> tuple[np.ndarray, np.ndarray]:
    """
    Range offset of every TX and RX path [m]: the peak of each channel's spectrum around
    the target bins, between bins, its mean taken away, fitted as offset(TX) + offset(RX).
    """
    mag = np.abs(st.rc.mean(axis=0))                                 # (V, R)
    dr = float(np.median(np.diff(st.range_m)))
    per = []
    for b in bins:
        b = int(b)
        if b < 2 or b > mag.shape[1] - 3:
            continue
        pk = []
        for v in range(mag.shape[0]):
            k = b - 2 + int(np.argmax(mag[v, b - 2:b + 3]))
            y0, y1, y2 = mag[v, k - 1], mag[v, k], mag[v, k + 1]
            den = y0 - 2 * y1 + y2
            pk.append(st.range_m[k] + (0.5 * (y0 - y2) / den * dr if den < 0 else 0.0))
        pk = np.asarray(pk)
        per.append(pk - pk.mean())
    if not per:
        return np.zeros(N_TX), np.zeros(N_RX)
    d = np.median(np.stack(per), axis=0)                               # (V,)
    # least squares d[v] = o_tx[t] + o_rx[r]: alternate means (exact for this model)
    o_tx, o_rx = np.zeros(N_TX), np.zeros(N_RX)
    for _ in range(20):
        for t in set(st.tx_idx.tolist()):
            m = st.tx_idx == t
            o_tx[t] = np.mean(d[m] - o_rx[st.rx_idx[m]])
        for r in range(N_RX):
            m = st.rx_idx == r
            o_rx[r] = np.mean(d[m] - o_tx[st.tx_idx[m]])
    shift = np.mean(o_rx)
    return o_tx + shift, o_rx - shift


def _coherence(u: np.ndarray, a: np.ndarray) -> float:
    return float(np.abs(np.vdot(a, u)) ** 2 / (np.vdot(a, a).real * np.vdot(u, u).real + 1e-30))


def _peak_angle(u: np.ndarray, st: _Static, range_m: float, grid=None) -> float:
    grid = np.arange(-80.0, 80.01, 0.1) if grid is None else grid
    A = steering(st.tx_idx, st.rx_idx, range_m, grid, st.wavelength_m)
    return float(grid[int(np.argmax(np.abs(A.conj().T @ u)))])


def _refine_range(st: _Static, b: int) -> float:
    p = (np.abs(st.rc) ** 2).mean(axis=(0, 1))
    if 0 < b < p.size - 1:
        y0, y1, y2 = p[b - 1], p[b], p[b + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            return float(st.range_m[b] + 0.5 * (y0 - y2) / den * np.median(np.diff(st.range_m)))
    return float(st.range_m[b])


def _model_gains(tx, rx, st: _Static) -> np.ndarray:
    return tx[st.tx_idx] * rx[st.rx_idx]


# --- a corner reflector --------------------------------------------------------------------------

def from_reflector(cube: np.ndarray, params: dict, range_m: float, az_deg: float = 0.0,
                   window_m: float = 0.3, source: str = "") -> CascadeCalibration:
    """
    Calibration from a take of a corner reflector at about range_m (the strongest still
    cell within window_m) and at az_deg.
    """
    st = static_spectra(cube, params)
    p = (np.abs(st.rc.mean(axis=0)) ** 2).mean(axis=0)
    sel = np.flatnonzero(np.abs(st.range_m - range_m) <= window_m)
    if sel.size == 0:
        raise ValueError(f"no range bin within {window_m} m of {range_m} m")
    b = int(sel[np.argmax(p[sel])])
    dr = float(np.median(np.diff(st.range_m)))
    near = np.flatnonzero((np.abs(st.range_m - st.range_m[b]) <= 1.0)
                          & (np.abs(st.range_m - st.range_m[b]) > 0.15 + 3 * dr))
    margin = float(10 * np.log10(p[b] / (np.median(p[near]) + 1e-30))) if near.size else float("nan")
    r_ref = _refine_range(st, b)
    u = _principal(st.rc[:, :, b])
    a = steering(st.tx_idx, st.rx_idx, r_ref, az_deg, st.wavelength_m)
    tx, rx, residual = _factor(u / a, st)
    g = _model_gains(tx, rx, st)
    o_tx, o_rx = _offsets(st, [b])
    cal = CascadeCalibration(
        method="reflector", tx=tx, rx=rx, tx_offset_m=o_tx, rx_offset_m=o_rx,
        f_center_ghz=st.f_center_ghz, residual_deg=residual,
        coherence_before=_coherence(u, a), coherence_after=_coherence(u / g, a),
        reflector={"range_m": r_ref, "range_entered_m": float(range_m), "az_deg": float(az_deg),
                   "margin_db": margin, "az_before": _peak_angle(u, st, r_ref),
                   "az_after": _peak_angle(u / g, st, r_ref)},
        source=source)
    if margin < 15:
        cal.notes.append(f"the reflector stands only {margin:.0f} dB above the cells around it - "
                         "something else may share its range")
    if residual > 20:
        cal.notes.append(f"the TX x RX model leaves {residual:.0f} deg - check the reflector's "
                         "placement and that nothing else is near it")
    return cal


def verify(cube: np.ndarray, params: dict, cal: CascadeCalibration, range_m: float,
           az_deg: float, window_m: float = 0.3) -> dict:
    """The reflector somewhere else at a known angle: its angle without and with the
    calibration. The fit on its own take always looks right; this says whether it holds."""
    st = static_spectra(cube, params)
    p = (np.abs(st.rc.mean(axis=0)) ** 2).mean(axis=0)
    sel = np.flatnonzero(np.abs(st.range_m - range_m) <= window_m)
    if sel.size == 0:
        raise ValueError(f"no range bin within {window_m} m of {range_m} m")
    b = int(sel[np.argmax(p[sel])])
    r_ref = _refine_range(st, b)
    u = _principal(st.rc[:, :, b])
    masks = [1 << t for t in sorted(set(st.tx_idx.tolist()), key=list(st.tx_idx).index)]
    g = cal.gains(masks, st.f_center_ghz)
    if g is None:
        raise ValueError("the calibration does not cover the TX of this take")
    a = steering(st.tx_idx, st.rx_idx, r_ref, az_deg, st.wavelength_m)
    before, after = _peak_angle(u, st, r_ref), _peak_angle(u / g, st, r_ref)
    return {"range_m": r_ref, "az_deg": float(az_deg), "az_before": before, "az_after": after,
            "error_before_deg": abs(before - az_deg), "error_after_deg": abs(after - az_deg),
            "coherence_before": _coherence(u, a), "coherence_after": _coherence(u / g, a)}


# --- autofocus -------------------------------------------------------------------------------------

AUTOFOCUS_MIN_GAIN = 0.02        # held-out echoes must focus at least this much better ...
AUTOFOCUS_MAX_LOSS = 0.05        # ... and none that the raw channels focus well (GOOD) worse
AUTOFOCUS_GOOD = 0.6             # than this
AUTOFOCUS_AGREE_DEG = 20.0       # the fitted echoes, each taken alone, give calibrations this close
AUTOFOCUS_FOCUSED = 0.6          # and the held-out ones end up at least this focused


def _candidates(st: _Static, r_min: float) -> list[dict]:
    """
    Echoes that may be points: the still reflectors (the frames agree with their mean)
    and the moving ones (the principal part of what changes in a range cell - a person),
    strongest first, 15 cm apart in range at least.
    """
    S = st.rc.mean(axis=0)                                            # (V, R)
    p = (np.abs(S) ** 2).sum(axis=0)
    steady = np.abs((st.rc * S.conj()[None]).sum(axis=1)).mean(axis=0) / (
        np.sqrt((np.abs(st.rc) ** 2).sum(axis=1).mean(axis=0) * p) + 1e-30)
    M = st.rc - S[None]
    pm = (np.abs(M) ** 2).sum(axis=1).mean(axis=0)
    dr = float(np.median(np.diff(st.range_m)))
    sep = max(2, int(round(0.15 / dr)))
    out = []

    def peaks(power, ok):
        return [b for b in range(1, power.size - 1)
                if ok[b] and power[b] >= power[b - 1] and power[b] >= power[b + 1]]

    far = st.range_m >= r_min
    for b in peaks(p, far & (steady > 0.9)):
        out.append({"kind": "still", "bin": b, "vec": S[:, b], "power": float(p[b])})
    floor = np.median(pm[far]) if far.any() else 0.0
    for b in peaks(pm, far & (pm > 10 * floor)):
        X = M[:, :, b]
        w, U = np.linalg.eigh(X.T @ X.conj() / X.shape[0])
        if w[-1] / max(w.sum(), 1e-30) >= 0.5:                        # one thing moves there
            out.append({"kind": "moving", "bin": b, "vec": U[:, -1] * np.sqrt(w[-1]),
                        "power": float(w[-1])})
    out.sort(key=lambda c: -c["power"])
    chosen = []
    for c in out:
        if all(abs(c["bin"] - d["bin"]) >= sep or c["kind"] != d["kind"] for d in chosen):
            chosen.append(c)
        if len(chosen) >= 3 * AUTOFOCUS_TARGETS:
            break
    for c in chosen:
        c["range_m"] = _refine_range(st, c["bin"])
    return chosen


def autofocus(cube: np.ndarray, params: dict, iterations: int = 30, source: str = "",
              r_min: float = AUTOFOCUS_R_MIN_M, anchor_deg: float | None = None) -> CascadeCalibration:
    """
    Calibration from what is in the room - still reflectors and moving people - with no
    reflector placed for it.

    Each echo is placed at the angle where it focuses best with the calibration of the
    moment, the calibration is refitted to all of them (TX x RX model), and again. Two
    starts: the raw channels, and the most point-like echo taken as a point at its angle
    (from equal phases the alternation got lost when every channel had a random phase).

    It is only kept when echoes that took no part in the fit - every moving one (a person
    is what the calibration is for) and every third still one - focus better with it than
    without (AUTOFOCUS_MIN_GAIN), and none that the raw channels already focus well gets
    worse (AUTOFOCUS_MAX_LOSS): fitted to the extended still objects of the lab, a
    calibration sharpened them and took the person in the same takes, focused at +0.7
    deg with coherence 0.88 by the raw channels, down to 0.42. People are never fitted:
    a chest is a patch, not a point. A rotation of the whole picture stays
    unknown (see the module's description): without anchor_deg the result keeps the raw
    channels' pointing, with it the most point-like echo is put there.
    """
    st = static_spectra(cube, params)
    cands = _candidates(st, r_min)
    if len(cands) < 3:
        raise ValueError("fewer than three echoes stand out in this take - record a room with some "
                         "furniture or a person in it, or use a corner reflector")
    grid = np.arange(-75.0, 75.01, 0.5)
    fine = np.arange(-0.5, 0.501, 0.05)
    A = [steering(st.tx_idx, st.rx_idx, c["range_m"], grid, st.wavelength_m) for c in cands]
    x_v = virtual_x(st.tx_idx, st.rx_idx)
    V = st.rc.shape[1]
    ones = np.ones(V, complex)

    def focus(gains, k):
        """(angle, coherence, implied gains, amplitude) of echo k."""
        y = cands[k]["vec"] / gains
        i = int(np.argmax(np.abs(A[k].conj().T @ y)))
        az = grid[i] + fine
        Af = steering(st.tx_idx, st.rx_idx, cands[k]["range_m"], az, st.wavelength_m)
        j = int(np.argmax(np.abs(Af.conj().T @ y)))
        a = Af[:, j]
        beta = np.vdot(a, y) / V
        return float(az[j]), _coherence(y, a), cands[k]["vec"] / (a * beta), abs(beta)

    def tilt_to(gains, k, az):
        """Turn the picture so echo k sits at az."""
        now = focus(gains, k)[0]
        g = gains * np.exp(1j * 2 * np.pi * x_v * (np.sin(np.deg2rad(now)) - np.sin(np.deg2rad(az))))
        return g / (g[0] / abs(g[0]))

    def fit(g, use, keep_pointing):
        ref = max(use, key=lambda k: raw[k][1])
        ref_az = raw[ref][0] if keep_pointing else (anchor_deg or 0.0)
        for _ in range(iterations):
            f = {k: focus(g, k) for k in use}
            order = sorted(use, key=lambda k: -f[k][1] * f[k][3])[:AUTOFOCUS_TARGETS]
            w = np.array([f[k][1] ** 2 * f[k][3] ** 2 for k in order])
            G = sum(wk * f[k][2] / np.mean(np.abs(f[k][2])) for wk, k in zip(w, order)) / w.sum()
            tx, rx, _ = _factor(G, st)
            g_new = tilt_to(_model_gains(tx, rx, st), ref, ref_az)
            g_new = g_new / np.mean(np.abs(g_new))
            change = float(np.max(np.abs(np.angle(g_new / g))))
            g = g_new
            if change < np.deg2rad(0.2):
                break
        return g

    raw = [focus(ones, k) for k in range(len(cands))]
    # the moving echoes and every third still one (by strength) judge the fit
    still = [k for k in range(len(cands)) if cands[k]["kind"] == "still"]
    held = [k for k in range(len(cands)) if cands[k]["kind"] == "moving"] + still[2::3]
    use = [k for k in still if k not in held]
    if len(use) < 2 or not held:
        raise ValueError("too few still point-like reflectors in this take for an autofocus - "
                         "use a corner reflector")
    score = lambda g: float(np.mean([focus(g, k)[1] for k in held]))
    base = score(ones)

    def harms(g) -> list:
        """Held-out echoes the raw channels focus well that this calibration spoils."""
        out = []
        for k in held:
            if raw[k][1] >= AUTOFOCUS_GOOD:
                c = focus(g, k)[1]
                if c < raw[k][1] - AUTOFOCUS_MAX_LOSS:
                    out.append((cands[k], raw[k][1], c))
        return out
    keep = anchor_deg is None
    starts = [fit(ones, use, keep)]
    best_raw = max(use, key=lambda k: raw[k][1] * raw[k][3])
    a0 = steering(st.tx_idx, st.rx_idx, cands[best_raw]["range_m"],
                  raw[best_raw][0] if keep else anchor_deg, st.wavelength_m)
    tx, rx, _ = _factor(cands[best_raw]["vec"] / a0, st)
    g0 = _model_gains(tx, rx, st)
    starts.append(fit(g0 / np.mean(np.abs(g0)), use, keep))
    good = [g for g in starts if not harms(g)]
    if not good:
        spoilt = harms(max(starts, key=score))[0]
        raise ValueError(f"autofocus would spoil an echo the raw channels focus well ({spoilt[0]['kind']} "
                         f"at {spoilt[0]['range_m']:.2f} m: {spoilt[1]:.2f} -> {spoilt[2]:.2f}) - the still "
                         "reflectors of this room are not points; use a corner reflector")
    scores = [score(g) for g in good]
    g = good[int(np.argmax(scores))]
    after = max(scores)
    if after < base + AUTOFOCUS_MIN_GAIN:
        raise ValueError(f"autofocus does not improve on the raw channels: the echoes it did not "
                         f"fit focus {base:.2f} raw and {after:.2f} with it (1 = a perfect beam) - "
                         "this scene cannot tell more; use a corner reflector")
    f = [focus(g, k) for k in range(len(cands))]
    used = [k for k in sorted(use, key=lambda k: -f[k][1] * f[k][3])[:AUTOFOCUS_TARGETS]
            if f[k][1] >= AUTOFOCUS_MIN_COHERENCE]
    # a calibration resting on one echo is that echo forced into a point: on the empty room
    # (2026-10-11) nine takes gave two such families, each repeating to 1-2 deg, 122 deg
    # apart and 85 deg from the calibration a seated person gave, and both took that person
    # from 0.89 to 0.23-0.31. Two echoes or more must agree, each taken alone ...
    if len(used) < 2:
        raise ValueError("autofocus rests on a single echo - nothing in this scene confirms it is a point; "
                         "use a person or a corner reflector")
    singles = []
    for k in used[:6]:
        a = steering(st.tx_idx, st.rx_idx, cands[k]["range_m"], f[k][0], st.wavelength_m)
        t1, r1, _ = _factor(cands[k]["vec"] / a, st)
        gk = _model_gains(t1, r1, st)
        singles.append(gk / np.abs(gk))
    spread = []
    for i in range(len(singles)):
        for j in range(i + 1, len(singles)):
            q = singles[i] / singles[j]
            beta = np.linspace(-0.6, 0.6, 1201)
            k = int(np.argmax(np.abs(np.exp(-1j * 2 * np.pi * np.outer(beta, x_v)) @ q)))
            q = q * np.exp(-1j * 2 * np.pi * beta[k] * x_v)
            q = q / np.exp(1j * np.angle(q.sum()))
            spread.append(float(np.rad2deg(np.sqrt(np.mean(np.angle(q) ** 2)))))
    if min(spread) > AUTOFOCUS_AGREE_DEG:
        raise ValueError(f"the echoes autofocus could use disagree ({min(spread):.0f} deg rms at best, each "
                         "taken as a point) - they are not points; use a person or a corner reflector")
    # ... and the echoes kept out end up focused, not just a little less blurred
    if after < AUTOFOCUS_FOCUSED:
        raise ValueError(f"the echoes autofocus did not fit stay unfocused ({base:.2f} raw, {after:.2f} with "
                         "it) - this scene has no points to calibrate on; use a person or a corner reflector")
    tx, rx, residual = _factor(g, st)
    o_tx, o_rx = _offsets(st, [cands[k]["bin"] for k in used[:4]] or [cands[0]["bin"]])
    cal = CascadeCalibration(
        method="autofocus", tx=tx, rx=rx, tx_offset_m=o_tx, rx_offset_m=o_rx,
        f_center_ghz=st.f_center_ghz, residual_deg=residual,
        coherence_before=base, coherence_after=after,
        autofocus={"targets": len(used), "held_out": len(held),
                   "moving": sum(cands[k]["kind"] == "moving" for k in used),
                   "anchor_deg": anchor_deg,
                   "where": [(float(cands[k]["range_m"]), float(f[k][0])) for k in used]},
        source=source)
    cal.notes.append(f"checked on {len(held)} echoes left out of the fit: they focus {base:.2f} raw, "
                     f"{after:.2f} with it")
    cal.notes.append("angles are relative: a turn of the whole picture is invisible to the autofocus - "
                     + ("it keeps the raw channels' pointing" if keep else
                        f"the most point-like echo is put at {anchor_deg:+.0f} deg")
                     + "; check one target at a known angle")
    return cal


# --- a person -------------------------------------------------------------------------------------------

PERSON_MIN_DOMINANCE = 0.4       # one thing must make most of the motion in the person's cell


def from_person(cube: np.ndarray, params: dict, az_deg: float | None = None,
                r_min: float = AUTOFOCUS_R_MIN_M, source: str = "") -> CascadeCalibration:
    """
    Calibration from someone sitting still - breathing - in front of the radar: the
    principal part of what moves in the range cell that moves most, taken as a point at
    az_deg (where they sat; None keeps the raw channels' pointing, angles relative).

    A chest is a patch, not a point - but a small one, and the only thing that moves: on
    the takes of 2026-10-09 (someone seated about 2 m in front, three takes with different
    profiles) a calibration from the person in one take took the person in the other two
    from 0.87-0.88 to 0.92-0.98, and the moving part 35 cm further back, which it was not
    fitted to, from 0.56-0.59 to 0.59-0.68, the angles staying within 0.5 deg. The other
    moving cells of the take are kept out and reported (validation): a calibration that
    makes them worse is refused.
    """
    st = static_spectra(cube, params, frames=256)
    M = st.rc - st.rc.mean(axis=0, keepdims=True)
    pm = (np.abs(M) ** 2).sum(axis=1).mean(axis=0)
    far = st.range_m >= r_min
    peaks = [b for b in range(1, pm.size - 1) if far[b] and pm[b] >= pm[b - 1] and pm[b] >= pm[b + 1]]
    if not peaks:
        raise ValueError("nothing moves in this take - someone has to sit (and breathe) in front of the radar")
    peaks.sort(key=lambda b: -pm[b])
    dr = float(np.median(np.diff(st.range_m)))
    others = [b for b in peaks[1:] if abs(b - peaks[0]) * dr >= 0.15 and pm[b] >= 0.05 * pm[peaks[0]]][:4]
    b = peaks[0]
    X = M[:, :, b]
    w, U = np.linalg.eigh(X.T @ X.conj() / X.shape[0])
    dominance = float(w[-1] / max(w.sum(), 1e-30))
    if dominance < PERSON_MIN_DOMINANCE:
        raise ValueError(f"the motion in the strongest moving cell ({st.range_m[b]:.2f} m) is not one thing "
                         f"({dominance:.2f} of it) - someone sitting still, alone in front of the radar")
    u = U[:, -1]
    R = _refine_range(st, b)
    grid = np.arange(-75.0, 75.01, 0.1)

    def best(vec, r, gains):
        y = vec / gains
        A = steering(st.tx_idx, st.rx_idx, r, grid, st.wavelength_m)
        coh = np.abs(A.conj().T @ y) ** 2 / (y.size * np.vdot(y, y).real)
        k = int(np.argmax(coh))
        return float(grid[k]), float(coh[k])

    ones = np.ones(u.size, complex)
    az_raw, coh_raw = best(u, R, ones)
    az_t = az_raw if az_deg is None else float(az_deg)
    a = steering(st.tx_idx, st.rx_idx, R, az_t, st.wavelength_m)
    tx, rx, residual = _factor(u / a, st)
    g = _model_gains(tx, rx, st)
    checks = []
    for o in others:
        Xo = M[:, :, o]
        wo, Uo = np.linalg.eigh(Xo.T @ Xo.conj() / Xo.shape[0])
        ro = _refine_range(st, o)
        checks.append((float(ro), best(Uo[:, -1], ro, ones), best(Uo[:, -1], ro, g)))
    worse = [c for c in checks if c[1][1] >= AUTOFOCUS_GOOD and c[2][1] < c[1][1] - AUTOFOCUS_MAX_LOSS]
    if worse:
        r0, (_, c0), (_, c1) = worse[0]
        raise ValueError(f"the calibration from the person would spoil another moving echo ({r0:.2f} m: "
                         f"{c0:.2f} -> {c1:.2f}) - the motion there is not the person alone")
    cal = CascadeCalibration(
        method="person", tx=tx, rx=rx, tx_offset_m=np.zeros(N_TX), rx_offset_m=np.zeros(N_RX),
        f_center_ghz=st.f_center_ghz, residual_deg=residual,
        coherence_before=coh_raw, coherence_after=best(u, R, g)[1],
        reflector={"range_m": R, "az_deg": az_t, "az_before": az_raw, "az_after": best(u, R, g)[0],
                   "person": True, "dominance": dominance,
                   "checks": [{"range_m": r0, "before": c0, "after": c1} for r0, c0, c1 in checks]},
        source=source)
    if az_deg is None:
        cal.notes.append(f"angles relative: the person is kept where the raw channels put them ({az_raw:+.1f} deg)")
    for r0, (a0, c0), (a1, c1) in checks:
        cal.notes.append(f"another moving echo at {r0:.2f} m (not fitted): {c0:.2f} at {a0:+.1f} deg raw -> "
                         f"{c1:.2f} at {a1:+.1f} deg calibrated")
    cal.notes.append("path offsets not measured from a person: other chirp profiles get the phases as measured")
    return cal


# --- where it is kept --------------------------------------------------------------------------------

def store_dir() -> Path:
    env = os.environ.get("DARTSCOPE_CASCADE_CALIBRATION_DIR")
    if env:
        d = Path(env)
    else:
        try:
            from platformdirs import user_config_dir  # type: ignore
            d = Path(user_config_dir("bme_radar_gui", appauthor=False))
        except Exception:
            d = Path.home() / ".bme_radar_gui"
    d.mkdir(parents=True, exist_ok=True)
    return d


def current_path() -> Path:
    return store_dir() / "cascade_calibration.json"


def save_current(cal: CascadeCalibration) -> Path:
    """Make this the board's calibration; every saved one is also kept in the archive."""
    doc = json.dumps(cal.to_dict(), indent=2)
    arch = store_dir() / "cascade_calibrations"
    arch.mkdir(exist_ok=True)
    (arch / (re.sub(r"[^0-9A-Za-z]", "", cal.created) + f"_{cal.method}.json")).write_text(doc, encoding="utf-8")
    current_path().write_text(doc, encoding="utf-8")
    return current_path()


def load_current() -> CascadeCalibration | None:
    p = current_path()
    if not p.is_file():
        return None
    try:
        return CascadeCalibration.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def forget_current() -> None:
    """The board has no calibration any more (the archive keeps the old ones)."""
    p = current_path()
    if p.is_file():
        p.unlink()


def for_take(meta: dict) -> tuple[CascadeCalibration | None, str]:
    """(calibration, where from) for a take's sidecar: its own copy, else the board's."""
    d = meta.get("calibration")
    if d:
        try:
            return CascadeCalibration.from_dict(d), "recorded with the take"
        except (ValueError, KeyError, TypeError):
            pass
    cal = load_current()
    if cal is None:
        return None, ""
    return cal, "the board's current calibration"
