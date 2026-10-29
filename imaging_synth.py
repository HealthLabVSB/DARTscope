"""
imaging_synth.py

Synthetic 4D scenes for the Imaging view (radar_imaging.py): point reflectors at 3D
positions moving at 3D velocities, seen by a TDM-MIMO radar with any antenna
configuration - a chirp sent on several TX at once is the sum of their responses - every
chirp computed exactly (its time, the reflector's position then, range tone, carrier
phase and the phase of every virtual element from its exact path TX -> point -> RX - a
plane wave far away, in the AoA views' convention).

    p = params(ntx=3, antenna=antenna_config.AOP_DEFAULT)
    cube = capture([Point(0.5, 2.0, 0.3, vy=1.0)], p, frames=20)

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import antenna_config
import limb_synth

C0 = 299_792_458.0


@dataclass
class Point:
    x: float                       # across [m] (sign as the AoA views' angles)
    y: float                       # away from the radar [m]
    z: float = 0.0                 # up [m]
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    amp: float = 1.0               # echo amplitude at 1 m (falls off as 1/r^2)
    swing: tuple = (0.0, 0.0, 0.0)  # a limb: back and forth by this much [m] ...
    swing_hz: float = 0.0          # ... this often ...
    swing_phase: float = 0.0       # ... from this phase [rad]


def params(ntx: int = 3, loops: int = 32, n_samples: int = 128, fs_msps: float = 10.0,
           slope_mhz_us: float = 60.0, f0_ghz: float = 60.25, idle_us: float = 10.0,
           ramp_us: float = 40.0, frame_ms: float = 50.0, antenna: str | None = antenna_config.AOP_DEFAULT,
           masks=None) -> dict:
    """Capture parameters; masks overrides the TX of every chirp of the loop (e.g. [2, 5])."""
    p = limb_synth.params(ntx=ntx, loops=loops, idle_us=idle_us, ramp_us=ramp_us, n_samples=n_samples,
                          fs_msps=fs_msps, slope_mhz_us=slope_mhz_us, f0_ghz=f0_ghz, frame_ms=frame_ms,
                          antenna=antenna)
    p["numAdcSamples"] = n_samples
    if masks is not None:
        masks = [int(m) for m in masks]
        p["tx_order_masks"] = masks
        p["uniqueChirps_effective"] = p["unique_chirps_frame"] = len(masks)
        p["frameConfig"]["chirpEndIdx"] = len(masks) - 1
        p["chirps"] = [{"chirpStartIdx": i, "chirpEndIdx": i} for i in range(len(masks))]
    if antenna and "cascade" in antenna.lower():
        p.pop("platform", None)
        p["chip_hint"] = "AWR2243 cascade"
    return p


def body(x: float, y: float, vx: float, vy: float, n: int = 25, seed: int = 0, amp: float = 0.3,
         height: float = 1.75, radar_height: float = 1.0) -> list:
    """
    A walking person: n scatterers in a body (torso, head, arms and legs) moving at
    (vx, vy); arms and legs swing along the walking direction at the step rate (2 Hz
    for legs at 1 m/s), opposite on either side.
    """
    rng = np.random.default_rng(seed)
    speed = float(np.hypot(vx, vy))
    ux, uy = (vx / speed, vy / speed) if speed > 0 else (0.0, 1.0)
    step_hz = 1.8 * max(speed, 0.3)
    pts = []
    for i in range(n):
        h = rng.uniform(0.0, height)                       # height above the floor
        side = 1 if i % 2 else -1
        limb = h < 0.85 or (h > 0.9 and h < 1.4 and abs(rng.normal()) > 1.0)
        a = (0.25 if h < 0.85 else 0.15) * (h / 0.85 if h < 0.85 else 1.0) if limb else 0.0
        pts.append(Point(x + rng.normal(0, 0.12), y + rng.normal(0, 0.08), h - radar_height,
                         vx=vx, vy=vy, amp=amp, swing=(a * ux, a * uy, 0.0), swing_hz=step_hz / 2,
                         swing_phase=0.0 if side > 0 else np.pi))
    return pts


def _bits(mask: int) -> list[int]:
    return [1 << k for k in range(32) if (int(mask) >> k) & 1]


def capture(points, p: dict, frames: int, noise: float = 0.01, seed: int = 0) -> np.ndarray:
    """Raw cube (frames, chirps, RX, samples), complex64."""
    cfg = p.get("antenna_config")
    rx = np.asarray(cfg["rx_positions"], float)
    offs = {int(m): np.asarray(o, float) for m, o in zip(cfg["tx_masks"], cfg["tx_offsets"])}
    masks = [int(m) for m in p["tx_order_masks"]][: int(p["uniqueChirps_effective"])]
    U, n_rx = len(masks), rx.shape[0]
    hw = antenna_config.element_phase_factors(cfg, U * n_rx)
    hw = np.conj(hw) if hw is not None else np.ones(U * n_rx)
    loops = int(p["frameConfig"]["numLoops"])
    C = loops * U
    S = int(p["numAdcSamples"])
    tc = (p["idleTime_us"] + p["rampEndTime_us"]) * 1e-6
    period = p["framePeriodicity_ms_selected"] * 1e-3
    fs = p["digOutSampleRate_MHz"] * 1e6
    slope = p["freqSlopeMHz_us"] * 1e12
    f0 = p["effectiveStartFreqGHz"] * 1e9
    t = np.arange(frames)[:, None] * period + np.arange(C)[None, :] * tc            # (F, C)
    slot = np.arange(C) % U
    n = np.arange(S)
    cube = np.zeros((frames, C, n_rx, S), np.complex64)
    for pt in points:
        osc = np.sin(2 * np.pi * pt.swing_hz * t + pt.swing_phase)
        x = pt.x + pt.vx * t + pt.swing[0] * osc
        y = pt.y + pt.vy * t + pt.swing[1] * osc
        z = pt.z + pt.vz * t + pt.swing[2] * osc
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        lam = C0 / f0
        # the point in wavelengths, in the array's frame: across, up, away
        P = np.stack([x, z, y], axis=-1) / lam                                       # (F, C, 3)
        R = r / lam
        rx3 = np.column_stack([rx, np.zeros(n_rx)])
        tau = 2 * r / C0
        car = pt.amp / np.maximum(r, 0.3) ** 2 * np.exp(1j * 2 * np.pi * ((f0 * tau) % 1.0))     # (F, C)
        tone = np.exp(1j * 2 * np.pi * (slope * tau)[:, :, None] * n / fs)                     # (F, C, S)
        spatial = np.zeros((frames, C, n_rx), complex)
        for s_i, m in enumerate(masks):
            on = slot == s_i
            Pn = P[:, on]                                                             # (F, Con, 3)
            d_r = np.linalg.norm(Pn[:, :, None, :] - rx3[None, None], axis=-1)        # (F, Con, RX)
            resp = 0
            for b in _bits(m):
                d_t = np.linalg.norm(Pn - np.array([offs[b][0], offs[b][1], 0.0]), axis=-1)
                resp = resp + np.exp(1j * 2 * np.pi * ((2 * R[:, on] - d_t)[:, :, None] - d_r))
            spatial[:, on] = resp * hw[s_i * n_rx:(s_i + 1) * n_rx]
        cube += (car[:, :, None, None] * spatial[:, :, :, None] * tone[:, :, None, :]).astype(np.complex64)
    rng = np.random.default_rng(seed)
    cube += (noise * (rng.standard_normal(cube.shape, dtype=np.float32)
                      + 1j * rng.standard_normal(cube.shape, dtype=np.float32)) / np.float32(np.sqrt(2))
             ).astype(np.complex64)
    return cube
