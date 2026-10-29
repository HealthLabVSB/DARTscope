"""
limb_synth.py

Synthetic raw radar captures of a moving person for testing limb_motion.py (roadmap
phase 10): a torso and legs or arms made of many point scatterers, turned into the
ADC samples a TI FMCW radar would record - beat signal per chirp, per receiver and, for
TDM-MIMO, per transmitter - so the whole chain from the range FFT on is exercised.

A limb is a line of scatterers from the joint to its tip whose velocity relative to
the torso grows linearly along it, as in a pendulum; that gives the continuous band
from the torso line to the tip that real limbs leave in a micro-Doppler spectrogram.
A single point would leave a lone streak instead, which no real leg does.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import numpy as np

C0 = 299_792_458.0


def params(ntx: int = 1, loops: int = 32, idle_us: float = 10.0, ramp_us: float = 40.0,
           n_samples: int = 64, fs_msps: float = 10.0, slope_mhz_us: float = 100.0,
           f0_ghz: float = 60.25, frame_ms: float | None = None, n_frames: int = 100,
           antenna: str | None = None) -> dict:
    """Capture parameters as parse_logfile_full would give them. frame_ms=None: frames back to back."""
    tc_ms = (idle_us + ramp_us) * 1e-3
    frame_ms = loops * ntx * tc_ms if frame_ms is None else float(frame_ms)
    p = {
        'startFreqGHz': f0_ghz, 'effectiveStartFreqGHz': f0_ghz,
        'freqSlopeMHz_us': slope_mhz_us, 'idleTime_us': idle_us, 'rampEndTime_us': ramp_us,
        'digOutSampleRate_MHz': fs_msps, 'numAdcSamples': n_samples,
        'dataFmt_adcFmt': 'Complex1X',
        'frameConfig': {'numLoops': loops, 'chirpStartIdx': 0, 'chirpEndIdx': ntx - 1,
                        'framePeriodicity_ms': frame_ms, 'numFrames': n_frames},
        'framePeriodicity_ms_selected': frame_ms,
        'chirps': [{'chirpStartIdx': i, 'chirpEndIdx': i} for i in range(ntx)],
        'tx_order_masks': [1, 2, 4][:ntx],
        'uniqueChirps_effective': ntx,
        'unique_chirps_frame': ntx,
    }
    if antenna:
        import antenna_config as ac
        p['antenna_config'] = ac.get_config(antenna)
        p['chip_hint'] = 'IWR6843AOP'
        p['platform'] = 'AOP'
    return p


def swing_profile(t, t_start: float, duration: float, v_peak: float):
    """Velocity of a swing: 0 outside [t_start, t_start + duration], a sin^2 bump inside."""
    t = np.asarray(t, float)
    x = (t - t_start) / duration
    return np.where((x >= 0) & (x <= 1), v_peak * np.sin(np.pi * np.clip(x, 0, 1)) ** 2, 0.0)


def swing_displacement(t, t_start: float, duration: float, v_peak: float):
    """Integral of swing_profile from -inf to t (the tip's extra path)."""
    t = np.asarray(t, float)
    x = np.clip((t - t_start) / duration, 0.0, 1.0)
    return v_peak * duration * (x / 2.0 - np.sin(2.0 * np.pi * x) / (4.0 * np.pi))


class Body:
    """
    A torso at range r0 moving with v_torso, and limbs that swing.

    swings: list of dicts {t_start, duration, v_peak (tip, relative to the torso,
    signed), amp (tip), az_deg, el_deg, n (scatterers along the limb)}.
    """

    def __init__(self, r0: float = 2.0, v_torso: float = 0.0, torso_amp: float = 1.0,
                 torso_az_deg: float = 0.0, torso_el_deg: float = 0.0, swings=(),
                 seed: int = 0):
        self.r0, self.v_torso, self.torso_amp = r0, v_torso, torso_amp
        self.torso_az, self.torso_el = torso_az_deg, torso_el_deg
        self.swings = [dict(s) for s in swings]
        self.seed = seed

    def scatterers(self, t):
        """
        [(range(t) array, complex amplitude, az_deg, el_deg)] for chirp times t.

        Points along a limb sit at slightly irregular places and each reflects with its
        own phase, like a rough surface: evenly spaced points with one phase would add
        up like an antenna array and cancel over whole stretches of the band.
        """
        rng = np.random.default_rng(self.seed)
        torso = self.r0 + self.v_torso * t
        out = [(torso, self.torso_amp, self.torso_az, self.torso_el)]
        for s in self.swings:
            n = int(s.get('n', 20))
            disp = swing_displacement(t, s['t_start'], s['duration'], s['v_peak'])
            for k in range(1, n + 1):
                frac = min(1.0, (k + rng.uniform(-0.4, 0.4)) / n)
                amp = s.get('amp', 0.3) * (1.5 - 0.5 * frac)      # stronger near the joint
                amp = amp / np.sqrt(n / 4.0) * np.exp(2j * np.pi * rng.random())
                out.append((torso + frac * disp, amp, s.get('az_deg', 0.0), s.get('el_deg', 0.0)))
        return out


def capture(body: Body, p: dict, duration_s: float, n_rx: int = 4, noise: float = 0.05,
            seed: int = 0) -> np.ndarray:
    """Raw ADC cube (frames, chirps, RX, samples), complex64, for `body` over `duration_s`."""
    rng = np.random.default_rng(seed)
    ntx = len(p['tx_order_masks'])
    loops = p['frameConfig']['numLoops']
    tc = (p['idleTime_us'] + p['rampEndTime_us']) * 1e-6
    period = p['framePeriodicity_ms_selected'] * 1e-3
    F = int(duration_s / period)
    p['frameConfig']['numFrames'] = F
    C = loops * ntx
    ns = p['numAdcSamples']
    fs = p['digOutSampleRate_MHz'] * 1e6
    slope = p['freqSlopeMHz_us'] * 1e12
    f_first = p['effectiveStartFreqGHz'] * 1e9
    t_chirp = np.arange(F)[:, None] * period + np.arange(C)[None, :] * tc     # (F, C)
    slot = np.arange(C) % ntx

    geo = None
    if p.get('antenna_config') is not None:
        import AOA
        import antenna_config as ac
        V = ntx * n_rx
        x, y, _ = AOA._get_antenna_geometry(p, V, n_rx)
        hw = ac.element_phase_factors(p['antenna_config'], V)
        geo = (np.asarray(x), np.asarray(y), np.conj(hw) if hw is not None else np.ones(V))

    # Beat signal of a point at delay tau: exp(j 2 pi (f tau + slope tau n / fs)). The
    # first term - the carrier phase that carries the Doppler - is exact for every chirp.
    # The second, the range tone, is computed once per frame and carried to each chirp
    # to first order, exp(j x) ~ 1 + j x (x stays below 0.1 rad here): a tone that only
    # changed from frame to frame would modulate every moving point at the frame rate
    # and draw sidebands 1/T_frame apart in the spectrogram. Scatterers seen from the
    # same angle share one sum.
    groups = {}
    nn = np.arange(ns, dtype=np.float32)
    for rng_t, amp, az, el in body.scatterers(t_chirp):
        tau = 2.0 * rng_t / C0                                             # (F, C)
        tau_f = tau.mean(axis=1)
        car = (amp * np.exp(1j * 2.0 * np.pi * ((f_first * tau) % 1.0))).astype(np.complex64)
        tone = np.exp(1j * 2.0 * np.pi * ((slope * tau_f / fs)[:, None]
                                           * np.arange(ns))).astype(np.complex64)
        dx = (2.0 * np.pi * slope * (tau - tau_f[:, None]) / fs).astype(np.float32)  # (F, C)
        key = (float(az), float(el))
        if key not in groups:
            groups[key] = np.zeros((F, C, ns), dtype=np.complex64)
        groups[key] += (car[:, :, None] * tone[:, None, :]) * (1 + 1j * dx[:, :, None] * nn)

    cube = np.zeros((F, C, n_rx, ns), dtype=np.complex64)
    for (az, el), base in groups.items():
        azr, elr = np.deg2rad(az), np.deg2rad(el)
        if geo is None:
            a = np.exp(1j * np.pi * np.arange(n_rx) * np.sin(azr))            # ULA, lambda/2
            a = np.broadcast_to(a, (C, n_rx))
        else:
            x, y, hw = geo
            a = np.exp(1j * 2 * np.pi * (x * np.sin(azr) * np.cos(elr) + y * np.sin(elr))) * hw
            a = a.reshape(ntx, n_rx)[slot]                                   # (C, RX)
        cube += base[:, :, None, :] * a.astype(np.complex64)[None, :, :, None]
    cube += (noise * (rng.standard_normal(cube.shape, dtype=np.float32)
                      + 1j * rng.standard_normal(cube.shape, dtype=np.float32))
             / np.float32(np.sqrt(2))).astype(np.complex64)
    return cube
