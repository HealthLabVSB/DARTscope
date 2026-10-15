"""
calibration.py

Per-channel antenna calibration from a corner reflector (roadmap 1.6).

Every virtual channel (TX t, RX r) has its own gain and phase: feed lines, the TX
phase shifters and the package differ by tens of degrees. The beamformer assumes they
are all equal, so an uncalibrated array puts targets at a biased angle and leaks
energy into sidelobes.

Model. After the per-element phase correction of the antenna configuration, a single
point target at angle (az, el) gives the virtual snapshot

    x_v = g_v * a_v(az, el) * s + noise

with a_v the steering vector the beamformer uses and g_v the unknown channel error.
The reflector is static, so its snapshots over frames are one vector up to noise and
a common phase; their principal eigenvector u is proportional to g * a. Knowing the
angle, g = u / a. The coefficient applied to the data is c = 1 / g, normalised so the
mean gain is one and channel 0 has zero phase.

What the cell really holds has to be checked on the range profile, not on the
snapshots: several static scatterers in one range cell also give one fixed vector
over time, so a rank-1 test cannot tell them apart. The estimate reports the
reflector's margin over the other range cells nearby, and an empty-scene recording
(same setup, reflector removed) can be subtracted to leave the reflector alone.
The honest check is a second reflector position: `verify` applies a calibration to
another recording and reports the angle error before and after.

Procedure in short:
  - trihedral corner reflector, edge 5-10 cm at 60 GHz, on a non-metallic stand at
    radar height, 1-3 m away (far field of this array is centimetres), nothing else
    within +-0.5 m in range;
  - boresight (az 0, el 0) is the easiest known angle; tilt the radar, not the stand;
  - record 5-10 s with the same chirp and TX configuration as the measurements;
  - optionally record the same scene without the reflector;
  - the coefficients hold for that board, RF configuration and roughly that
    temperature - not needed for every measurement.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class Calibration:
    coefficients: np.ndarray            # (V,) complex, multiply the data by these
    num_tx: int
    num_rx: int
    range_bin: int = -1
    range_m: float = float("nan")
    az_deg: float = 0.0
    el_deg: float = 0.0
    reflector_margin_db: float = float("nan")
    angle_before: tuple = (float("nan"), float("nan"))
    angle_after: tuple = (float("nan"), float("nan"))
    coherence_before: float = float("nan")
    coherence_after: float = float("nan")
    range_bias_m: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def phase_deg(self) -> np.ndarray:
        return np.rad2deg(np.angle(1.0 / self.coefficients))

    @property
    def gain(self) -> np.ndarray:
        return 1.0 / np.abs(self.coefficients)

    def to_params(self) -> dict:
        """What AOA._apply_calibration reads: params['channel_calibration']."""
        return {"channel_calibration": [[float(c.real), float(c.imag)] for c in self.coefficients]}

    def summary(self) -> str:
        lines = [f"Calibration of {self.num_tx} TX x {self.num_rx} RX = "
                 f"{self.coefficients.size} virtual channels",
                 f"  reflector at {self.range_m:.3f} m (bin {self.range_bin}), "
                 f"assumed az {self.az_deg:+.1f} deg, el {self.el_deg:+.1f} deg",
                 f"  reflector margin over nearby range cells: {self.reflector_margin_db:.1f} dB"]
        if self.range_bias_m is not None:
            lines.append(f"  range bias: {self.range_bias_m * 1000:+.1f} mm")
        lines.append("  channel   phase [deg]   gain")
        for v, (p, g) in enumerate(zip(self.phase_deg, self.gain)):
            lines.append(f"  TX{v // self.num_rx} RX{v % self.num_rx}   {p:+8.1f}      {g:5.2f}")
        lines.append(f"  angle of the reflector: az {self.angle_before[0]:+.1f}, "
                     f"el {self.angle_before[1]:+.1f} deg before -> az {self.angle_after[0]:+.1f}, "
                     f"el {self.angle_after[1]:+.1f} deg after")
        lines.append(f"  array coherence at that angle: {self.coherence_before:.3f} -> "
                     f"{self.coherence_after:.3f}")
        return "\n".join(lines)


# --- building blocks -----------------------------------------------------------------

def _layout(params: dict, num_chirps: int, num_rx: int):
    """(num_tx, stride, loops) for plain TDM-MIMO, else (1, 1, chirps) - RX only."""
    import AOA

    eff = int(params.get("uniqueChirps_effective") or 1)
    stride = max(1, int(params.get("unique_chirps_frame") or eff))
    masks = params.get("tx_order_masks")
    if eff >= 2 and num_chirps % stride == 0 and AOA._is_safe_tdm(masks[:eff] if masks else None):
        return eff, stride, num_chirps // stride
    return 1, 1, num_chirps


def static_snapshots(range_cube: np.ndarray, params: dict, range_bin: int) -> np.ndarray:
    """
    One virtual-array snapshot per frame at `range_bin`: the mean over the loops, i.e.
    the zero-Doppler (static) part, with the configuration's element phases applied.

    range_cube: (frames, chirps, RX, range bins), complex.
    Returns (frames, V).
    """
    import AOA

    F, C, RX, R = range_cube.shape
    ntx, stride, L = _layout(params, C, RX)
    x = range_cube[:, :L * stride, :, range_bin].reshape(F, L, stride, RX)[:, :, :ntx]
    x = x.reshape(F, L, ntx * RX).mean(axis=1)
    return AOA._apply_aop_rx_phase_correction(x[:, None, :, None], params, ntx * RX, RX)[:, 0, :, 0]


def steering(params: dict, num_virt: int, num_rx: int, az_deg, el_deg) -> np.ndarray:
    """Steering vector(s) the AoA code uses, (V,) or (V, K) for arrays of angles."""
    import AOA

    x, y, _ = AOA._get_antenna_geometry(params, num_virt, num_rx)
    az, el = np.deg2rad(np.asarray(az_deg, float)), np.deg2rad(np.asarray(el_deg, float))
    kx, ky = np.sin(az) * np.cos(el), np.sin(el)
    return np.exp(1j * 2.0 * np.pi * (np.multiply.outer(x, kx) + np.multiply.outer(y, ky)))


def _angle_grid(params, V, RX):
    import AOA

    _, y, _ = AOA._get_antenna_geometry(params, V, RX)
    az = np.arange(-80.0, 80.25, 0.25)
    el = np.arange(-50.0, 50.25, 0.25) if np.any(np.asarray(y) != 0) else np.zeros(1)
    AZ, EL = np.meshgrid(az, el)
    return AZ.ravel(), EL.ravel()


def peak_angle(u: np.ndarray, params: dict, num_rx: int):
    """(az, el) in degrees where the Bartlett beam of snapshot u peaks."""
    V = u.size
    az, el = _angle_grid(params, V, num_rx)
    SV = steering(params, V, num_rx, az, el)
    i = int(np.argmax(np.abs(SV.conj().T @ u)))
    return float(az[i]), float(el[i])


def coherence(u: np.ndarray, a: np.ndarray) -> float:
    """|a^H u|^2 / (|a|^2 |u|^2): 1.0 when u is exactly the plane wave a."""
    return float(np.abs(np.vdot(a, u)) ** 2 / (np.vdot(a, a).real * np.vdot(u, u).real + 1e-30))


def principal(snapshots: np.ndarray) -> np.ndarray:
    """Principal eigenvector of the covariance E[x x^H] of snapshots shaped (N, V)."""
    X = np.asarray(snapshots)
    w, U = np.linalg.eigh(X.T @ X.conj() / max(1, X.shape[0]))
    return U[:, -1]


def coefficients_from(u: np.ndarray, a: np.ndarray, amplitude: bool = True,
                      gain_limits=(0.5, 2.0)) -> np.ndarray:
    """c = 1 / (u / a), mean gain one, channel 0 at zero phase."""
    g = u / a
    g = g / (g[0] / abs(g[0]))
    g = g / np.mean(np.abs(g))
    if amplitude:
        mag = np.clip(np.abs(g), *gain_limits)
    else:
        mag = np.ones(g.size)
    return (1.0 / mag) * np.exp(-1j * np.angle(g))


def apply(data: np.ndarray, coefficients, axis: int = -1) -> np.ndarray:
    c = np.asarray(coefficients)
    shape = [1] * np.ndim(data)
    shape[axis] = c.size
    return data * c.reshape(shape)


# --- estimation and verification -------------------------------------------------------

def _range_axis(params: dict, R: int, range_axis=None) -> np.ndarray:
    if range_axis is not None:
        return np.asarray(range_axis, float)
    if params.get("range_m") is not None and len(params["range_m"]) == R:
        return np.asarray(params["range_m"], float)
    raise ValueError("no range axis: pass range_axis or params['range_m']")


def _find_reflector(range_cube, r_axis, range_m, window_m):
    power = np.mean(np.abs(range_cube.mean(axis=1)) ** 2, axis=(0, 1))    # static power per bin
    sel = np.flatnonzero(np.abs(r_axis - range_m) <= window_m)
    if sel.size == 0:
        raise ValueError(f"no range bin within {window_m} m of {range_m} m")
    b = int(sel[np.argmax(power[sel])])
    near = np.flatnonzero((np.abs(r_axis - r_axis[b]) <= 1.0) & (np.abs(np.arange(r_axis.size) - b) > 2))
    margin = 10 * np.log10(power[b] / (np.median(power[near]) + 1e-30)) if near.size else float("nan")
    return b, float(margin), power


def estimate(range_cube: np.ndarray, params: dict, range_m: float, az_deg: float = 0.0,
             el_deg: float = 0.0, window_m: float = 0.15, background: np.ndarray | None = None,
             true_range_m: float | None = None, range_axis=None,
             amplitude: bool = True) -> Calibration:
    """
    Calibration from a recording of a corner reflector at a known angle.

    range_cube: (frames, chirps, RX, range bins) after the range FFT.
    range_m, window_m: where to look for the reflector - the strongest static bin within
    the window is taken.
    background: the same scene without the reflector, same shape except frames; its
    mean is subtracted so only the reflector is left.
    true_range_m: the measured distance to the reflector, to report the range bias.
    A range bias already in params (an active calibration) is added back, so the bias
    reported is always the whole offset of this board, not what is left of it.
    """
    import spectral

    cube = np.asarray(range_cube)
    if background is not None:
        cube = cube - np.asarray(background).mean(axis=0, keepdims=True)
    F, C, RX, R = cube.shape
    r_axis = _range_axis(params, R, range_axis)
    b, margin, power = _find_reflector(cube, r_axis, range_m, window_m)
    snaps = static_snapshots(cube, params, b)
    V = snaps.shape[1]
    ntx = V // RX
    u = principal(snaps)
    a = steering(params, V, RX, az_deg, el_deg)
    c = coefficients_from(u, a, amplitude)
    bias = None
    r_meas = float(r_axis[b])
    if true_range_m is not None:
        from radar_processing import range_bias_m
        r_meas, _ = spectral.refine_peak(r_axis, power, b)
        bias = float(r_meas - true_range_m) + range_bias_m(params)
    return Calibration(
        coefficients=c, num_tx=ntx, num_rx=RX, range_bin=b, range_m=r_meas,
        az_deg=float(az_deg), el_deg=float(el_deg), reflector_margin_db=margin,
        angle_before=peak_angle(u, params, RX), angle_after=peak_angle(u * c, params, RX),
        coherence_before=coherence(u, a), coherence_after=coherence(u * c, a),
        range_bias_m=bias,
        meta={"created": datetime.now().isoformat(timespec="seconds"),
              "antenna_config": (params.get("antenna_config") or {}).get("name"),
              "tx_order_masks": params.get("tx_order_masks"),
              "background_subtracted": background is not None,
              "amplitude": bool(amplitude), "frames": int(F)})


def verify(range_cube: np.ndarray, params: dict, cal, range_m: float, az_deg: float,
           el_deg: float = 0.0, window_m: float = 0.15, range_axis=None) -> dict:
    """
    Apply a calibration to another recording of a reflector at a known angle.

    Returns the peak angle and the coherence at the known angle before and after.
    This, not the fit on the recording the calibration came from, says whether it
    holds.
    """
    c = cal.coefficients if isinstance(cal, Calibration) else np.asarray(cal)
    cube = np.asarray(range_cube)
    F, C, RX, R = cube.shape
    b, margin, _ = _find_reflector(cube, _range_axis(params, R, range_axis), range_m, window_m)
    u = principal(static_snapshots(cube, params, b))
    if u.size != c.size:
        raise ValueError(f"calibration has {c.size} channels, this recording {u.size}")
    a = steering(params, u.size, RX, az_deg, el_deg)
    before, after = peak_angle(u, params, RX), peak_angle(u * c, params, RX)
    return {"range_bin": b, "reflector_margin_db": margin,
            "angle_before": before, "angle_after": after,
            "error_before_deg": float(np.hypot(before[0] - az_deg, before[1] - el_deg)),
            "error_after_deg": float(np.hypot(after[0] - az_deg, after[1] - el_deg)),
            "coherence_before": coherence(u, a), "coherence_after": coherence(u * c, a)}


# --- files ---------------------------------------------------------------------------

def save(path: str, cal: Calibration) -> None:
    doc = {
        "format": "dartscope-channel-calibration/1",
        "num_tx": cal.num_tx, "num_rx": cal.num_rx,
        "coefficients": [[float(z.real), float(z.imag)] for z in cal.coefficients],
        "reflector": {"range_m": cal.range_m, "range_bin": cal.range_bin,
                      "az_deg": cal.az_deg, "el_deg": cal.el_deg,
                      "margin_db": cal.reflector_margin_db},
        "range_bias_m": cal.range_bias_m,
        "check": {"angle_before": list(cal.angle_before), "angle_after": list(cal.angle_after),
                  "coherence_before": cal.coherence_before,
                  "coherence_after": cal.coherence_after},
        "meta": cal.meta,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def load(path: str) -> Calibration:
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if not str(doc.get("format", "")).startswith("dartscope-channel-calibration/"):
        raise ValueError(f"{os.path.basename(path)} is not a DARTscope channel calibration")
    c = np.array([complex(re, im) for re, im in doc["coefficients"]])
    ref = doc.get("reflector") or {}
    chk = doc.get("check") or {}
    return Calibration(
        coefficients=c, num_tx=int(doc["num_tx"]), num_rx=int(doc["num_rx"]),
        range_bin=int(ref.get("range_bin", -1)), range_m=float(ref.get("range_m", np.nan)),
        az_deg=float(ref.get("az_deg", 0.0)), el_deg=float(ref.get("el_deg", 0.0)),
        reflector_margin_db=float(ref.get("margin_db", np.nan)),
        angle_before=tuple(chk.get("angle_before", (np.nan, np.nan))),
        angle_after=tuple(chk.get("angle_after", (np.nan, np.nan))),
        coherence_before=float(chk.get("coherence_before", np.nan)),
        coherence_after=float(chk.get("coherence_after", np.nan)),
        range_bias_m=doc.get("range_bias_m"), meta=doc.get("meta") or {})
