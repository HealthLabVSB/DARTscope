"""
hr_synth.py

Synthetic raw radar captures of a person whose heart rate is known (roadmap phase 12),
for testing moving_hr.py: a chest that breathes and moves with every heartbeat, on a
body that may sit still, sway, walk in place or walk towards and away from the radar,
among static objects. limb_synth.capture turns the scatterers into the ADC samples a TI
FMCW radar records, so the whole chain from the range FFT on is exercised.

The chest wall moves 0.1-0.5 mm with each beat (here a sharp outward push and a smaller
recoil, about 0.15 s long - sharp enough to carry energy up to 25 Hz, as the mechanical
events of a real heart do), 4-12 mm with each breath, and walking adds a bounce of the
trunk at the step frequency with its second harmonic and a short jolt at every footfall.
Real bodies add much more (arms, rotation, clothing); results on these captures are an
upper bound of what the pipeline can do, not a measurement of it.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import numpy as np

import limb_synth


def beat_times(duration_s: float, hr_bpm, hrv_ms: float = 20.0, seed: int = 0) -> np.ndarray:
    """Times of the heartbeats; hr_bpm is a number or a function of time (bpm)."""
    rng = np.random.default_rng(seed)
    rate = hr_bpm if callable(hr_bpm) else (lambda t, v=float(hr_bpm): v)
    out, t = [], float(rng.uniform(0, 0.5))
    while t < duration_s + 1.0:
        out.append(t)
        t += 60.0 / float(rate(t)) + rng.normal(0.0, hrv_ms * 1e-3)
    return np.array(out)


def heart_displacement(t, beats, amp_m: float = 0.3e-3) -> np.ndarray:
    """Chest wall displacement of the heartbeats: an outward push, then a smaller recoil."""
    t = np.asarray(t, float)
    d = np.zeros_like(t)
    for tb in beats:
        dt = t - tb
        near = (dt > -0.1) & (dt < 0.6)
        if near.any():
            x = dt[near]
            d[near] += amp_m * (np.exp(-((x - 0.06) / 0.025) ** 2)
                                - 0.45 * np.exp(-((x - 0.2) / 0.05) ** 2))
    return d


def footfall_jolts(t, steps, amp_m: float = 0.15e-3, freq_hz: float = 14.0,
                   decay_s: float = 0.04) -> np.ndarray:
    """A short damped ring of the trunk at every footfall."""
    t = np.asarray(t, float)
    d = np.zeros_like(t)
    for ts in steps:
        dt = t - ts
        on = (dt >= 0) & (dt < 8 * decay_s)
        if on.any():
            x = dt[on]
            d[on] += amp_m * np.exp(-x / decay_s) * np.sin(2 * np.pi * freq_hz * x)
    return d


class Person:
    """
    A chest at range r0 and what moves it.

    hr_bpm: number or function of time; breath: amplitude [m] and rate [Hz];
    motion: "rest", "sway" (slow random drift of a few cm), "walk_in_place" (trunk
    bounce at the cadence, footfall jolts) or "walk" (the same while walking between
    r0 and r0 + walk_span_m at walk_speed).
    """

    def __init__(self, r0: float = 1.2, hr_bpm=72.0, heart_amp_m: float = 0.3e-3,
                 breath_amp_m: float = 5e-3, breath_hz: float = 0.25, motion: str = "rest",
                 cadence_hz: float = 1.8, bounce_amp_m: float = 10e-3,
                 walk_speed: float = 0.8, walk_span_m: float = 2.0, sway_amp_m: float = 0.03,
                 chest_amp: float = 1.0, clutter=((2.8, 2.0), (4.5, 1.5)), legs: bool = False,
                 duration_s: float = 60.0, seed: int = 0):
        self.r0, self.heart_amp, self.breath_amp, self.breath_hz = r0, heart_amp_m, breath_amp_m, breath_hz
        self.motion, self.cadence = motion, cadence_hz
        self.bounce, self.walk_speed, self.walk_span, self.sway = bounce_amp_m, walk_speed, walk_span_m, sway_amp_m
        self.chest_amp, self.clutter, self.legs = chest_amp, list(clutter), legs
        self.seed = seed
        self.beats = beat_times(duration_s, hr_bpm, seed=seed)
        self.hr_bpm = hr_bpm
        moving = motion in ("walk", "walk_in_place")
        self.steps = (np.arange(0.2, duration_s + 1.0, 1.0 / cadence_hz) if moving else np.zeros(0))
        rng = np.random.default_rng(seed + 1)
        self._sway_f = rng.uniform(0.05, 0.4, 4)
        self._sway_p = rng.uniform(0, 2 * np.pi, 4)

    def true_hr(self, t) -> np.ndarray:
        """The heart rate at times t [bpm], from the beat intervals."""
        mids = 0.5 * (self.beats[1:] + self.beats[:-1])
        return np.interp(np.asarray(t, float), mids, 60.0 / np.diff(self.beats))

    def body_range(self, t) -> np.ndarray:
        """Range of the trunk without breathing and heart [m]."""
        t = np.asarray(t, float)
        r = np.full_like(t, self.r0)
        if self.motion == "sway":
            for f, p in zip(self._sway_f, self._sway_p):
                r += 0.5 * self.sway * np.sin(2 * np.pi * f * t + p)
        if self.motion in ("walk", "walk_in_place"):
            w = 2 * np.pi * self.cadence
            r += self.bounce * np.sin(w * t) + 0.3 * self.bounce * np.sin(2 * w * t + 0.6)
            r += footfall_jolts(t, self.steps)
        if self.motion == "walk":
            # there and back with smooth turns: a sinusoid whose mean speed is walk_speed
            period = 2 * self.walk_span / self.walk_speed
            r += 0.5 * self.walk_span * (1 - np.cos(2 * np.pi * t / period))
        return r

    def chest_range(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        return (self.body_range(t) + self.breath_amp * np.sin(2 * np.pi * self.breath_hz * t)
                + heart_displacement(t, self.beats, self.heart_amp))

    def scatterers(self, t):
        out = [(self.chest_range(t), self.chest_amp, 0.0, 0.0)]
        for r, a in self.clutter:                                  # static objects
            out.append((np.full_like(np.asarray(t, float), r), a, 20.0, 0.0))
        if self.legs and self.motion in ("walk", "walk_in_place"):
            body = self.body_range(t)
            for side in (0.0, 0.5):                                # the two legs
                ph = 2 * np.pi * (self.cadence / 2.0) * np.asarray(t, float) + 2 * np.pi * side
                for k in range(1, 9):
                    out.append((body + 0.25 * (k / 8.0) * np.sin(ph), 0.08, 0.0, -20.0))
        return out


def capture(person: Person, duration_s: float = 60.0, fps: float = 100.0, loops: int = 4,
            noise: float = 0.02, n_rx: int = 4, seed: int = 0):
    """(cube, params): 1 TX, `loops` chirps of 50 us per frame, frames 1/fps apart."""
    p = limb_synth.params(ntx=1, loops=loops, idle_us=10.0, ramp_us=40.0, n_samples=64,
                          fs_msps=10.0, slope_mhz_us=100.0, frame_ms=1000.0 / fps)
    p['numAdcSamples'] = 64
    cube = limb_synth.capture(person, p, duration_s, n_rx=n_rx, noise=noise, seed=seed)
    return cube, p


class Occupant:
    """
    One person of a crowd_capture scene: a chest at (x, y) [m] - x to the right, y
    away from the radar - or walking along path(t) -> (x, y), breathing and beating.
    """

    def __init__(self, x: float = 0.0, y: float = 1.5, breath_hz: float = 0.25,
                 breath_amp_m: float = 4e-3, hr_bpm=72.0, heart_amp_m: float = 0.3e-3,
                 amp: float = 1.0, path=None, duration_s: float = 60.0, seed: int = 0):
        self.x, self.y, self.path = x, y, path
        self.breath_hz, self.breath_amp = breath_hz, breath_amp_m
        self.heart_amp, self.amp, self.hr_bpm = heart_amp_m, amp, hr_bpm
        self.beats = beat_times(duration_s, hr_bpm, seed=seed)
        self.phase = np.random.default_rng(seed + 7).uniform(0, 2 * np.pi)

    def position(self, t):
        t = np.asarray(t, float)
        if self.path is None:
            return np.full_like(t, self.x), np.full_like(t, self.y)
        x, y = self.path(t)
        return np.broadcast_to(np.asarray(x, float), t.shape), np.broadcast_to(np.asarray(y, float), t.shape)

    def chest(self, t):
        """(range m, azimuth deg) of the chest at times t."""
        x, y = self.position(t)
        r = np.hypot(x, y) + self.breath_amp * np.sin(2 * np.pi * self.breath_hz * np.asarray(t) + self.phase) \
            + heart_displacement(t, self.beats, self.heart_amp)
        return r, np.rad2deg(np.arctan2(x, y))


def crowd_capture(people, duration_s: float = 60.0, fps: float = 20.0, ntx: int = 3,
                  loops: int = 4, antenna: str | None = "IWR6843AOP (3TX x 4RX, 2D)",
                  clutter=((0.0, 3.5, 3.0), (-1.5, 2.5, 2.0)), noise: float = 0.02,
                  n_samples: int = 64, seed: int = 0):
    """
    (cube, params): several Occupants and static reflectors [(x, y, amplitude)] seen by
    a TDM-MIMO radar with ntx TX (masks 1, 2, 4) and the given antenna configuration,
    echoes falling off as 1/r^2 (normalised to 1 m). Every scatterer keeps its azimuth
    for a frame; the carrier phase is exact for every chirp, the range tone carried to
    the chirps of a frame to first order (as limb_synth.capture).
    """
    import AOA
    import antenna_config as ac
    p = limb_synth.params(ntx=ntx, loops=loops, idle_us=10.0, ramp_us=40.0, n_samples=n_samples,
                          fs_msps=10.0, slope_mhz_us=100.0, frame_ms=1000.0 / fps, antenna=antenna)
    p['numAdcSamples'] = n_samples
    rng = np.random.default_rng(seed)
    n_rx = 4
    tc = (p['idleTime_us'] + p['rampEndTime_us']) * 1e-6
    period = p['framePeriodicity_ms_selected'] * 1e-3
    F = int(duration_s / period)
    p['frameConfig']['numFrames'] = F
    C = loops * ntx
    fs = p['digOutSampleRate_MHz'] * 1e6
    slope = p['freqSlopeMHz_us'] * 1e12
    f_first = p['effectiveStartFreqGHz'] * 1e9
    t_chirp = np.arange(F)[:, None] * period + np.arange(C)[None, :] * tc
    slot = np.arange(C) % ntx
    V = ntx * n_rx
    if antenna:
        x_el, y_el, _ = AOA._get_antenna_geometry(p, V, n_rx)
        hw = ac.element_phase_factors(p['antenna_config'], V)
        hw = np.conj(hw) if hw is not None else np.ones(V)
    else:
        x_el, hw = 0.5 * np.arange(V), np.ones(V)
    x_el = np.asarray(x_el, float)
    nn = np.arange(n_samples, dtype=np.float32)
    cube = np.zeros((F, C, n_rx, n_samples), np.complex64)

    def add(r_fc, az_f, amp):
        tau = 2.0 * r_fc / limb_synth.C0
        tau_f = tau.mean(axis=1)
        car = (amp * np.exp(1j * 2 * np.pi * ((f_first * tau) % 1.0))).astype(np.complex64)
        tone = np.exp(1j * 2 * np.pi * ((slope * tau_f / fs)[:, None] * nn)).astype(np.complex64)
        dx = (2 * np.pi * slope * (tau - tau_f[:, None]) / fs).astype(np.float32)
        base = (car[:, :, None] * tone[:, None, :]) * (1 + 1j * dx[:, :, None] * nn)   # (F, C, S)
        a = np.exp(1j * 2 * np.pi * np.sin(np.deg2rad(az_f))[:, None] * x_el[None, :]) * hw   # (F, V)
        a = a.reshape(F, ntx, n_rx)[:, slot]                                                # (F, C, RX)
        cube[:] += base[:, :, None, :] * a.astype(np.complex64)[:, :, :, None]

    for occ in people:
        r, az = occ.chest(t_chirp)
        add(r, az[:, 0], occ.amp / np.maximum(r, 0.3) ** 2)
    for cx, cy, a in clutter:
        r = np.full((F, C), np.hypot(cx, cy))
        add(r, np.full(F, np.rad2deg(np.arctan2(cx, cy))), a / max(np.hypot(cx, cy), 0.3) ** 2)
    cube += (noise * (rng.standard_normal(cube.shape, dtype=np.float32)
                      + 1j * rng.standard_normal(cube.shape, dtype=np.float32))
             / np.float32(np.sqrt(2))).astype(np.complex64)
    return cube, p
