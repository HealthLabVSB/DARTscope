"""
moving_hr.py

Heart rate of a person who may be moving (roadmap phase 12).

The heartbeat moves the chest wall by 0.1-0.5 mm, breathing by 4-12 mm, and walking
bounces the trunk by about 10 mm at the step frequency - a heart some 30 dB weaker than
what surrounds it, with the cadence (1.6-2 Hz) and its harmonics right in its band.
The pipeline:

    chest_signal   the trunk followed in range (the bin with the most moving energy,
                   smoothed), the range FFT phase-centred (spectral.centre_phase,
                   roadmap 1.10) so moving to the next bin keeps the phase, the static
                   scene of every bin removed, the bins around the trunk and the
                   receivers combined coherently - the beam steered to the trunk by the
                   principal eigenvector of the receivers (5.1, 12.3).
    track_phase    the phase of every chirp followed by a Kalman filter (phase, rate,
                   acceleration). Inside a frame the chirps are microseconds apart;
                   across the gap to the next frame the prediction from rate and
                   acceleration unwraps it (12.4). A walker moves the phase by tens of
                   radians from frame to frame - a plain unwrap at the frame rate
                   cannot follow that, the filter can while the acceleration is smooth.
    displacement   the phase at the middle of each frame, in metres with the
                   wavelength at the middle of the ADC window (1.9).
    motion         every window rest, light or strong motion from the speed of the
                   trunk (12.7); strong motion gives no estimate.
    cadence        the step frequency while moving: the bounce of the trunk (12.5c) -
                   notched out of the heart band with its harmonics, like the harmonics
                   of breathing (5.4).
    heart spectra  per window, from the displacement in the heart band and from the
                   envelope of its 5-25 Hz content - the mechanical events of the heart
                   are short, so a beat shows there where breathing and slow body
                   motion carry no energy (12.6); a harmonic sum f + 2f + 3f (12.8).
    viterbi        the heart rate followed through the windows with a limited change
                   per second (12.8); windows without information are carried over.

Nothing here touches Tk.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import contextlib
import io
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, signal

import capture_params
import radar_processing as rp
import spectral
import vital_bands
import vitalsigns

# the trunk in range
R_MIN_M = 0.3
HOLD_S = 0.5                   # median over this for the trunk bin
BLOCK_S = 2.0                  # receiver weights re-estimated this often
# the phase tracker
MAX_CHIRPS = 4                 # chirps per frame fed to the tracker
JERK_M_S3 = 20.0               # process noise: RMS jerk of the chest [m/s^3 per sqrt(Hz)]
GAP_RESET_S = 0.03             # after a longer gap without a measurement: start afresh
V_LIMIT_M_S = 5.0              # the chest of a walking or running person, at most
RATE_FLOOR_M_S = 0.02          # the uncertainty of a frame's own rate, at least
A_LIMIT_M_S2 = 50.0
# motion classes [m/s, RMS speed of the trunk in a window]
REST_MS = 0.02
STRONG_MS = 0.3
MOTION_WIN_S = 2.0
# heart
HR_WIN_S = 15.0
REST_WIN_S = 30.0               # the heart's window while the person stays still that long
HR_STEP_S = 1.0
HR_GRID_BPM = 0.5
ENVELOPE_BAND_HZ = (5.0, 25.0)
MAX_CHANGE_BPM_S = 3.0          # a heart rate hardly changes faster than this at rest
CONF_MIN = 3.0                  # harmonic-sum peak over its average: an estimate
CADENCE_BAND_HZ = (1.2, 2.8)
CADENCE_MIN_M = 1e-3            # a bounce this large is a step, not a heartbeat
BREATH_NOTCH = True             # breathing harmonics 2-4 out of the heart band
MOTION_NAMES = ("rest", "light", "strong")


@dataclass
class ChestSignal:
    """The chest: one complex sample per chirp used, and what it was taken from."""
    z: np.ndarray                  # (frames, chirps) combined signal
    t: np.ndarray                  # (frames, chirps) time of each sample [s]
    fps: float
    range_m: np.ndarray            # (frames,) range of the trunk bin
    damaged: np.ndarray            # (frames,) zero-filled frames
    lam_m: float                   # wavelength at the middle of the ADC window
    noise: float                   # RMS of the noise in z
    bins: np.ndarray | None = None  # (frames, RX, 3) the trunk bins of every receiver, the
    #                                chirps of a frame averaged (the heart at rest: rest_heart)


@dataclass
class MovingHR:
    """What the pipeline found. Per-frame arrays: t_s; per-window arrays: win_t."""
    fps: float
    t_s: np.ndarray
    displacement_m: np.ndarray     # (frames,) chest displacement
    velocity_ms: np.ndarray        # (frames,) from the tracker
    range_m: np.ndarray            # (frames,) trunk range
    damaged: np.ndarray
    slips: np.ndarray              # (frames,) large innovations of the tracker
    win_t: np.ndarray              # window centres [s]
    motion: np.ndarray             # 0 rest, 1 light, 2 strong
    speed_ms: np.ndarray           # RMS trunk speed per window
    br_bpm: np.ndarray             # breathing per window (nan when unclear)
    cadence_hz: np.ndarray         # step frequency per window (nan when not walking)
    hr_grid_bpm: np.ndarray
    spectra: np.ndarray            # (windows, grid) normalised harmonic sums
    hr_bpm: np.ndarray             # Viterbi path
    hr_argmax_bpm: np.ndarray      # the largest harmonic sum of each window alone
    confidence: np.ndarray         # peak of the path over the window's average
    valid: np.ndarray
    method: str = "auto"
    notes: dict = field(default_factory=dict)
    heart_win_s: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # (windows, 2):
    #                                the start and end of the stretch each heart rate is from [s]

    def coverage(self) -> float:
        return float(np.mean(self.valid)) if self.valid.size else 0.0

    WINDOW_FIELDS = ("time_s", "motion", "speed_m_s", "breathing_per_min", "steps_per_min",
                     "hr_bpm", "hr_argmax_bpm", "confidence", "valid")

    def window_table(self):
        """(array, header) of the windows, for a CSV."""
        cols = [self.win_t, self.motion.astype(float), self.speed_ms, self.br_bpm,
                self.cadence_hz * 60.0, self.hr_bpm, self.hr_argmax_bpm, self.confidence,
                self.valid.astype(float)]
        return np.column_stack(cols) if self.win_t.size else np.zeros((0, len(cols))), \
            ",".join(self.WINDOW_FIELDS)


# --- the chest signal ----------------------------------------------------------------------

def _track_bins(moving: np.ndarray, fps: float, hold_s: float, dr_m: float = 0.05,
                v_max_m_s: float = 2.0, penalty: float = 1.0) -> np.ndarray:
    """
    Bin of the trunk in every frame: the path through the moving energy that follows it
    best, by dynamic programming - it may move by at most what a body at v_max covers
    in a frame, and every bin it moves costs `penalty` (natural log of energy). A
    per-frame maximum toggled between the chest and a second reflection of the body a
    few bins away, and every toggle broke the phase; the path moves only when the
    energy elsewhere stays higher for long enough to pay for the move.
    """
    F, G = moving.shape
    e = ndimage.uniform_filter1d(moving, size=3, axis=1, mode="nearest")
    e = ndimage.uniform_filter1d(e, size=max(1, int(round(hold_s * fps))), axis=0, mode="nearest")
    score_f = np.log(e + 1e-30)
    score_f -= score_f.max(axis=1, keepdims=True)
    score_f = np.maximum(score_f, -20.0)               # an empty frame should not decide
    step = max(1, int(np.ceil(v_max_m_s / max(fps, 1e-9) / max(dr_m, 1e-9))))
    shifts = range(-step, step + 1)
    total = score_f[0].copy()
    back = np.zeros((F, G), np.int16)
    for f in range(1, F):
        best = np.full(G, -np.inf)
        arg = np.zeros(G, np.int16)
        for s_ in shifts:
            cand = np.full(G, -np.inf)
            if s_ >= 0:
                cand[s_:] = total[:G - s_] - penalty * s_
            else:
                cand[:G + s_] = total[-s_:] + penalty * s_
            better = cand > best
            best[better] = cand[better]
            arg[better] = s_
        back[f] = arg
        total = best + score_f[f]
    path = np.zeros(F, int)
    path[-1] = int(np.argmax(total))
    for f in range(F - 1, 0, -1):
        path[f - 1] = path[f] - back[f, path[f]]
    return path


REST_SHARE = 0.9                # the trunk path within a bin of its median this often: still


def _rest_anchor(k: np.ndarray, xbar: np.ndarray, fps: float, range_m, damaged) -> np.ndarray:
    """
    The trunk path k, moved onto the neighbouring bin that breathes most when the person
    sits still. The moving energy peaks on the strongest moving reflection, which for a
    seated person was a bin in front of the chest: the heart there was 21 instead of 1 bpm
    off a pulse oximeter (fastcheck.breathing_bin, the Vital Signs view's Auto Bin, on
    the bins around the path).
    """
    if k.size == 0 or xbar.shape[0] < 8 * fps:
        return k
    kmed = int(np.median(k))
    if np.mean(np.abs(k - kmed) <= 1) < REST_SHARE:
        return k
    import fastcheck
    lo, hi = max(0, kmed - 2), min(xbar.shape[-1], kmed + 3)
    if hi - lo < 2:
        return k
    with contextlib.redirect_stdout(io.StringIO()):
        b, _, _ = fastcheck.breathing_bin(xbar[:, :, lo:hi], fps, vital_bands.current(),
                                          np.asarray(range_m)[lo:hi], damaged,
                                          min_range_m=0.0, nyquist_last=False)
    return k if b is None else k + (lo + int(b) - kmed)


def chest_signal(cube: np.ndarray, params: dict, *, r_min: float = R_MIN_M,
                 r_max: float | None = None, n_fft_range: int | None = None,
                 fps: float | None = None, t0_s: float = 0.0, progress_cb=None) -> ChestSignal:
    """
    The chest of the person in cube (frames, chirps, RX, samples): see the module notes.
    Two passes over the cube: the trunk is found on the chirp means, then only the
    three bins around it are kept for every chirp.
    """
    cube = np.asarray(cube)
    F, C, RXn, Ns = cube.shape
    if fps is None:
        with contextlib.redirect_stdout(io.StringIO()):
            fps = capture_params.derive_fps(params)
    fps = float(fps)
    nfft = rp.range_fft_size(Ns, n_fft_range)
    chunk = max(1, int(8e6 // max(1, C * RXn * nfft)))
    quiet = contextlib.redirect_stdout(io.StringIO())

    def spectra(f0):
        with contextlib.redirect_stdout(io.StringIO()):
            spec, meta = rp.compute_range_fft_cube(cube[f0:f0 + chunk], params, window="hann",
                                                   remove_dc=True, n_fft_range=nfft,
                                                   use_full_spectrum=False, return_complex=True)
        return spec, meta

    # pass 1: chirp means of the gate, the static scene, zero-filled frames
    xbar = damaged = None
    static_l = None             # the static scene per chirp of the frame (sum over frames)
    n_good = 0
    for f0 in range(0, F, chunk):
        spec, meta = spectra(f0)
        if xbar is None:
            range_m = np.asarray(meta["range_m"], float)
            b0 = int(np.searchsorted(range_m, r_min))
            b1 = len(range_m) - 1 if r_max is None else int(np.searchsorted(range_m, r_max))
            b1 = max(b0 + 2, min(b1, len(range_m) - 1))
            xbar = np.zeros((F, RXn, b1 - b0 + 1), np.complex64)
            damaged = np.zeros(F, bool)
        g = spectral.centre_phase(spec[..., b0:b1 + 1], Ns, nfft, first_bin=b0)
        with quiet:
            sel, interval, info = rp.doppler_slow_time(g, params)
        n = sel.shape[0]
        xbar[f0:f0 + n] = sel.mean(axis=-1)
        damaged[f0:f0 + n] = vitalsigns.detect_damaged_frames(cube[f0:f0 + n])
        ok = ~damaged[f0:f0 + n]
        part = sel[ok].sum(axis=0) if ok.any() else np.zeros(sel.shape[1:], sel.dtype)
        static_l = part if static_l is None else static_l + part
        n_good += int(ok.sum())
        if progress_cb is not None:
            progress_cb(f0 + n, 2 * F)
    good = ~damaged
    static = xbar[good].mean(axis=0) if good.any() else xbar.mean(axis=0)     # (RX, G)
    moving = (np.abs(xbar - static) ** 2).sum(axis=1)
    moving[damaged] = 0.0
    dr = float(range_m[1] - range_m[0]) if len(range_m) > 1 else 0.05
    k = _track_bins(moving, fps, HOLD_S, dr_m=dr)                               # (F,)
    G = xbar.shape[-1]
    k = _rest_anchor(k, xbar, fps, range_m[b0:b1 + 1], damaged)
    k = np.clip(k, 1, G - 2)

    # pass 2: every chirp (up to MAX_CHIRPS) of the three bins around the trunk
    L = None
    y = None
    for f0 in range(0, F, chunk):
        spec, meta = spectra(f0)
        g = spectral.centre_phase(spec[..., b0:b1 + 1], Ns, nfft, first_bin=b0)
        with quiet:
            sel, interval, info = rp.doppler_slow_time(g, params)             # (n, RX, G, Lall)
        n, Lall = sel.shape[0], sel.shape[-1]
        if y is None:
            use = np.unique(np.linspace(0, Lall - 1, min(Lall, MAX_CHIRPS)).round().astype(int))
            L = use.size
            y = np.zeros((F, RXn, 3, L), np.complex64)
        idx = k[f0:f0 + n, None] + np.arange(-1, 2)[None, :]                    # (n, 3)
        s = sel[..., use]                                                       # (n, RX, G, L)
        picked = np.take_along_axis(s, idx[:, None, :, None], axis=2)          # (n, RX, 3, L)
        # the static scene of every chirp position of the frame on its own: the chirps
        # right after the idle time differ a little in phase (the synthesiser settling),
        # and with one static scene for all of them that difference read as a velocity
        # inside every frame, 0.05 m/s on test_D1_3min - enough to unwrap a turn wrong
        st = (static_l[..., use] / max(1, n_good))[None]                       # (1, RX, G, L)
        st = np.take_along_axis(np.broadcast_to(st, (n, RXn, G, L)), idx[:, None, :, None], axis=2)
        y[f0:f0 + n] = picked - st
        if progress_cb is not None:
            progress_cb(F + f0 + n, 2 * F)

    # the beam on the trunk, block by block; the range bins weighted by their strength
    offset = float(np.mean(info["positions"])) * capture_params.chirp_period_s(params)
    with quiet:
        t_all = capture_params.slow_time_axis(params, F, Lall, interval, offset, t0_s)
    t = t_all[:, use]
    blk = max(1, int(round(BLOCK_S * fps)))
    ref = None
    z = np.zeros((F, L), np.complex64)
    for s0 in range(0, F, blk):
        yb = y[s0:s0 + blk]
        ok = ~damaged[s0:s0 + blk]
        v = yb[ok] if ok.any() else yb
        m = v.transpose(1, 0, 2, 3).reshape(RXn, -1)                          # RX x samples
        cov = m @ m.conj().T
        _, vecs = np.linalg.eigh(cov)
        w = vecs[:, -1]
        if ref is None:
            ref = int(np.argmax(np.abs(w)))
        w = w * np.exp(-1j * np.angle(w[ref]))
        a = np.sqrt((np.abs(v) ** 2).mean(axis=(0, 1, 3)))                    # (3,) bins
        a = a / (a.sum() + 1e-30)
        z[s0:s0 + blk] = np.einsum("r,frbl,b->fl", w.conj(), yb, a)
    if L >= 2:
        noise = float(np.std(np.diff(z[~damaged], axis=1)) / np.sqrt(2)) if (~damaged).any() else 0.0
    else:
        noise = float(np.std(np.diff(z[~damaged, 0])) / np.sqrt(2)) if (~damaged).sum() > 2 else 0.0
    return ChestSignal(z=z, t=t, fps=fps, range_m=range_m[b0 + k], damaged=damaged,
                       lam_m=rp.wavelength_mid_m(params), noise=max(noise, 1e-12),
                       bins=y.mean(axis=-1))


# --- the phase of every chirp ---------------------------------------------------------------

def track_phase(z: np.ndarray, t: np.ndarray, noise: float, lam_m: float,
                damaged: np.ndarray | None = None, jerk_m_s3: float = JERK_M_S3):
    """
    Unwrapped phase of z (frames, chirps) at times t, followed by a Kalman filter on
    (phase, rate, acceleration) with white jerk. Returns (phase, rate [rad/s], slips):
    slips flags the frames where a measurement fell more than 2.5 rad off the
    prediction - a likely wrong unwrap.

    Between frames the phase only tells the rate modulo 2 pi / T_frame: a walker's
    chest turns the phase by tens of radians per frame, and a filter whose rate is one
    such step off would follow the measurements just as well, losing a turn every
    frame. The rate each frame measures from its own chirps tells them apart - unless
    an arm or a leg shares the cell: its fast phase spoils that rate, and on synthetic
    walks with legs the filter then locks onto an alias (see the ROADMAP, 12.4). The
    trunk's range rate was tried as the arbiter and is too coarse (0.08 m/s RMS against
    0.125 m/s between the aliases at 100 fps). Frames without gaps (continuous
    chirping) remove the ambiguity altogether.
    """
    F, L = z.shape
    q = (4.0 * np.pi / lam_m * jerk_m_s3) ** 2
    zf = z.reshape(-1)
    tf = t.reshape(-1)
    amp = np.abs(zf)
    meas = np.angle(zf)
    usable = amp > 3.0 * noise
    if damaged is not None:
        usable &= ~np.repeat(np.asarray(damaged, bool), L)
    out = np.empty(zf.size)
    rate = np.empty(zf.size)
    slip = np.zeros(zf.size, bool)
    # the rate every frame measures itself from its chirps (unambiguous up to the chirp
    # spacing's own v_max): when the filter drifts off it - after a turn, a jolt or a
    # dropout it could not follow - it is taken over, or the filter would stay lost
    frame_rate = np.full(F, np.nan)
    frame_var = np.full(F, np.inf)
    if L >= 2:
        ph = np.unwrap(np.angle(z), axis=1)
        tc = t - t.mean(axis=1, keepdims=True)
        den = (tc ** 2).sum(axis=1)
        okf = den > 0
        frame_rate[okf] = ((ph - ph.mean(axis=1, keepdims=True)) * tc).sum(axis=1)[okf] / den[okf]
        amp_f = np.maximum(np.abs(z).min(axis=1), 1e-30)
        frame_var[okf] = (noise ** 2 / (2 * amp_f[okf] ** 2)) / den[okf]
    # state and covariance as plain floats: 30 000+ steps of 3x3 numpy would take long
    first = int(np.argmax(usable)) if usable.any() else 0
    p0, w0, a0 = float(meas[first]), 0.0, 0.0
    var_w0 = (4 * np.pi / lam_m * 2.0) ** 2           # 2 m/s when nothing better is known
    if L >= 2 and usable.any():
        # the start: the mean rate of the first few frames - one frame's rate is too
        # rough to unwrap the first gaps with
        f = first // L
        rates = frame_rate[f:f + 5]
        rates = rates[np.isfinite(rates)]
        if rates.size:
            w0 = float(np.median(rates))
            fv = frame_var[f:f + 5]
            fv = fv[np.isfinite(fv)]
            if fv.size:
                var_w0 = 1.6 * float(np.mean(fv)) / fv.size
    P = [[1.0, 0.0, 0.0], [0.0, var_w0, 0.0], [0.0, 0.0, (4 * np.pi / lam_m * 20.0) ** 2]]
    x = [p0, w0, a0]
    t_prev = float(tf[first])
    t_meas = t_prev
    two_pi = 2.0 * math.pi
    disagree = 0                 # frames in a row whose own rate disagrees with the filter
    recent = []                  # their rates
    k_ph = 4.0 * np.pi / lam_m
    w_lim, a_lim = k_ph * V_LIMIT_M_S, k_ph * A_LIMIT_M_S2
    for i in range(zf.size):
        dt = float(tf[i]) - t_prev
        t_prev = float(tf[i])
        if dt != 0.0:
            d2 = dt * dt
            # x = F x
            x0 = x[0] + dt * x[1] + 0.5 * d2 * x[2]
            x1 = x[1] + dt * x[2]
            x = [x0, x1, x[2]]
            # P = F P F^T + Q
            p00, p01, p02 = P[0]
            p11, p12 = P[1][1], P[1][2]
            p22 = P[2][2]
            a00 = p00 + 2 * dt * p01 + d2 * p02 + d2 * p11 + dt * d2 * p12 + 0.25 * d2 * d2 * p22
            a01 = p01 + dt * p02 + dt * p11 + 1.5 * d2 * p12 + 0.5 * dt * d2 * p22
            a02 = p02 + dt * p12 + 0.5 * d2 * p22
            a11 = p11 + 2 * dt * p12 + d2 * p22
            a12 = p12 + dt * p22
            a22 = p22
            d3, d4, d5 = d2 * dt, d2 * d2, d2 * d2 * dt
            a00 += q * d5 / 20.0
            a01 += q * d4 / 8.0
            a02 += q * d3 / 6.0
            a11 += q * d3 / 3.0
            a12 += q * d2 / 2.0
            a22 += q * dt
            P = [[a00, a01, a02], [a01, a11, a12], [a02, a12, a22]]
        if usable[i] and float(tf[i]) - t_meas > GAP_RESET_S:
            # After a dropout the prediction has run on the rate and acceleration alone
            # for a tenth of a second or more; a jump of a few millimetres is more than
            # it can bridge. Start afresh on the frame's own rate; the offset across
            # the gap is left to stitch_steps, which fits it rather than guessing turns.
            f = i // L
            w_new = float(frame_rate[f]) if np.isfinite(frame_rate[f]) else x[1]
            x = [x[0] + ((float(meas[i]) - x[0] + math.pi) % two_pi - math.pi), w_new, 0.0]
            P = [[1e-4, 0.0, 0.0], [0.0, float(frame_var[f]) if np.isfinite(frame_var[f])
                                     else (4 * np.pi / lam_m * 0.05) ** 2, 0.0],
                 [0.0, 0.0, (4 * np.pi / lam_m * 20.0) ** 2]]
            slip[i] = True
        if usable[i]:
            t_meas = float(tf[i])
            r = max(noise * noise / (2.0 * float(amp[i]) ** 2), 1e-6)
            nu = (float(meas[i]) - x[0] + math.pi) % two_pi - math.pi
            if abs(nu) > 2.5:
                slip[i] = True
            s = P[0][0] + r
            k0, k1, k2 = P[0][0] / s, P[0][1] / s, P[0][2] / s
            out[i] = x[0] + nu
            x = [x[0] + k0 * nu, x[1] + k1 * nu, x[2] + k2 * nu]
            p00, p01, p02 = P[0]
            p11, p12, p22 = P[1][1], P[1][2], P[2][2]
            P = [[p00 - k0 * p00, p01 - k0 * p01, p02 - k0 * p02],
                 [p01 - k1 * p00, p11 - k1 * p01, p12 - k1 * p02],
                 [p02 - k2 * p00, p12 - k2 * p01, p22 - k2 * p02]]
        else:
            out[i] = x[0]
        f = i // L
        if i % L == L - 1 and np.isfinite(frame_rate[f]) and usable[i]:
            # the frame's own rate is a measurement of the filter's rate, weighed as four
            # times noisier than it is (it comes from the same samples as the phases);
            # one far off is left out - a leg or an arm in the same cell spoils a frame's
            # rate now and then - and only a disagreement that persists means the filter
            # lost the chest
            # never better than 0.05 m/s: another moving part in the cell (a hand) bends
            # the phase inside a frame far beyond what the noise alone would
            rv = max(4.0 * float(frame_var[f]), (k_ph * RATE_FLOOR_M_S) ** 2)
            nu_v = float(frame_rate[f]) - x[1]
            s_v = P[1][1] + rv
            if nu_v * nu_v <= 9.0 * s_v:
                disagree = 0
                recent = []
                g0, g1, g2 = P[0][1] / s_v, P[1][1] / s_v, P[2][1] / s_v
                x = [x[0] + g0 * nu_v, x[1] + g1 * nu_v, x[2] + g2 * nu_v]
                p01, p11, p12 = P[0][1], P[1][1], P[1][2]
                P = [[P[0][0] - g0 * p01, P[0][1] - g0 * p11, P[0][2] - g0 * p12],
                     [P[1][0] - g1 * p01, P[1][1] - g1 * p11, P[1][2] - g1 * p12],
                     [P[2][0] - g2 * p01, P[2][1] - g2 * p11, P[2][2] - g2 * p12]]
            else:
                disagree += 1
                recent.append(float(frame_rate[f]))
            if disagree >= 3:
                x = [x[0], float(np.median(recent)), 0.0]
                P = [[P[0][0], 0.0, 0.0], [0.0, float(frame_var[f]), 0.0],
                     [0.0, 0.0, (4 * np.pi / lam_m * 20.0) ** 2]]
                slip[i] = True
                disagree = 0
                recent = []
        # guards: the covariance spans ten orders of magnitude and plain floats can
        # lose its positiveness; a filter that did would push the rate anywhere
        if not (P[0][0] > 0 and P[1][1] > 0 and P[2][2] > 0):
            P = [[1e-4, 0.0, 0.0], [0.0, (k_ph * 0.05) ** 2, 0.0], [0.0, 0.0, (k_ph * 20.0) ** 2]]
            slip[i] = True
        if abs(x[1]) > w_lim or abs(x[2]) > a_lim:
            x = [x[0], max(-w_lim, min(w_lim, x[1])), max(-a_lim, min(a_lim, x[2]))]
        rate[i] = x[1]
    return out.reshape(F, L), rate.reshape(F, L), slip.reshape(F, L).any(axis=1)


def stitch_steps(d: np.ndarray, fps: float, bad: np.ndarray, slips: np.ndarray,
                 context_s: float = 0.5) -> tuple:
    """
    d with the steps taken out that a lost stretch leaves behind.

    Over a dropout the tracker can only predict; when the chest moved more than a
    quarter wavelength meanwhile (breathing does in 0.2 s), the unwrap after it lands a
    whole turn off - a step of k * lambda / 2 in the displacement. Every boundary - a
    run of zero-filled frames, or a frame where the tracker started afresh - is bridged
    in turn from the left: a smooth curve (quadratic, linear for short pieces) is fitted
    to up to context_s of good frames before it and after it, up to the next boundary,
    together with an unknown step, and the step is taken off everything after it.
    Returns (d, number of steps removed).
    """
    d = np.asarray(d, float).copy()
    F = d.size
    bad = np.asarray(bad, bool)
    w = max(3, int(round(context_s * fps)))
    # boundaries (a, b): frames a..b-1 are lost, the new piece starts at b
    bounds = []
    run = np.diff(np.concatenate(([0], bad.view(np.int8), [0])))
    for a, b in zip(np.flatnonzero(run == 1), np.flatnonzero(run == -1)):
        bounds.append((int(a), int(b)))
    for f in np.flatnonzero(np.asarray(slips, bool) & ~bad):
        if not any(a <= f <= b for a, b in bounds):
            bounds.append((int(f), int(f)))
    bounds.sort()
    good = ~bad
    n_steps = 0
    for i, (a, b) in enumerate(bounds):
        nxt = bounds[i + 1][0] if i + 1 < len(bounds) else F
        pre = np.arange(max(0, a - w), a)
        post = np.arange(b, min(nxt, b + w))
        pre, post = pre[good[pre]], post[good[post]]
        if pre.size < 3 or post.size < 2:
            continue
        tt = (np.concatenate([pre, post]).astype(float) - 0.5 * (a + b)) / fps
        deg = 2 if min(pre.size, post.size) >= 10 else 1
        A = np.column_stack([tt ** p for p in range(deg + 1)]
                            + [np.concatenate([np.zeros(pre.size), np.ones(post.size)])])
        coef, *_ = np.linalg.lstsq(A, np.concatenate([d[pre], d[post]]), rcond=None)
        if abs(coef[-1]) > 1e-9:
            d[b:] -= coef[-1]
            n_steps += 1
    return d, n_steps


# --- windows -----------------------------------------------------------------------------------

def _bandpass(x, fs, lo, hi, order=4):
    hi = min(hi, 0.45 * fs)
    if lo <= 0 or hi <= lo or x.size < 3 * (order * 2 + 1) * 3:
        return x - x.mean()
    sos = signal.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x - x.mean())


def _power(x, fs, freqs_hz, res_hz: float = 0.005):
    """Periodogram of x (Hann) at the frequencies asked for, from a zero-padded FFT
    fine enough (res_hz) to interpolate."""
    n = x.size
    if n < 4:
        return np.zeros(len(freqs_hz))
    nfft = 1 << int(np.ceil(np.log2(max(n * 4, fs / res_hz))))
    spec = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(n), nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    return np.interp(freqs_hz, f, spec)


def _windows(n_frames, fps, win_s, step_s):
    win = max(4, int(round(win_s * fps)))
    step = max(1, int(round(step_s * fps)))
    if n_frames < win:
        return [(0, n_frames)]
    return [(s, s + win) for s in range(0, n_frames - win + 1, step)]


def _rate(x, fs, band_hz, min_q=4.0):
    """(rate per minute, peak over band average) of the strongest line in the band."""
    f = np.linspace(band_hz[0], band_hz[1], 200)
    p = _power(x, fs, f)
    k = int(np.argmax(p))
    q = float(p[k] / (p.mean() + 1e-30))
    if q < min_q or k in (0, len(f) - 1):
        return float("nan"), q
    return float(f[k] * 60.0), q


def viterbi(logp: np.ndarray, grid_bpm: np.ndarray, step_s: float,
            max_change_bpm_s: float = MAX_CHANGE_BPM_S, jump_prob: float = 1e-3) -> np.ndarray:
    """
    Most likely path through logp (windows x grid) when the rate changes by a
    Gaussian step of sigma = max_change * step per window, with a small chance of a
    jump anywhere (a missed stretch). Returns grid indices.
    """
    T, K = logp.shape
    d = grid_bpm[:, None] - grid_bpm[None, :]
    sigma = max(max_change_bpm_s * step_s, 0.5)
    trans = np.log((1 - jump_prob) * np.exp(-0.5 * (d / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
                   * (grid_bpm[1] - grid_bpm[0]) + jump_prob / K)
    score = logp[0].copy()
    back = np.zeros((T, K), int)
    for i in range(1, T):
        cand = score[None, :] + trans            # [to, from]
        back[i] = np.argmax(cand, axis=1)
        score = cand[np.arange(K), back[i]] + logp[i]
    path = np.zeros(T, int)
    path[-1] = int(np.argmax(score))
    for i in range(T - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def _notch(freq_hz, centres_hz, width_hz, hard_hz: float = 0.0):
    """1 away from the centres; a Gaussian dip of width_hz at each, and 0 within
    hard_hz of it (the whole main lobe of a line far stronger than the heart)."""
    g = np.ones_like(freq_hz)
    for c in centres_hz:
        g *= 1.0 - np.exp(-0.5 * ((freq_hz - c) / width_hz) ** 2)
        if hard_hz > 0:
            g[np.abs(freq_hz - c) < hard_hz] = 0.0
    return g


# --- the heart --------------------------------------------------------------------------------

def heart_spectra(d: np.ndarray, fs: float, bands, wins, *, br_bpm=None, cadence_hz=None,
                  rest=None, method: str = "auto"):
    """
    (grid [bpm], spectra (windows x grid)) of the heart in the chest displacement d,
    one normalised harmonic sum per window (s0, s1) of wins.

    br_bpm, cadence_hz: breathing rate and step frequency per window (nan: unknown) -
    their harmonics are notched out of each spectrum before the sum (after it, a step
    harmonic at 3f would still raise f through the sum). rest: per window whether the
    person is still; with method "auto" the displacement is used only there - in
    motion the bounce of the steps sits 50 dB above the heart and the sidelobes of its
    line alone reach it - and the envelope of the 5-25 Hz content always.
    """
    n_w = len(wins)
    br_bpm = np.full(n_w, np.nan) if br_bpm is None else np.asarray(br_bpm, float)
    cadence_hz = np.full(n_w, np.nan) if cadence_hz is None else np.asarray(cadence_hz, float)
    rest = np.ones(n_w, bool) if rest is None else np.asarray(rest, bool)
    hr_lo, hr_hi = bands.heart_hz
    top = min(hr_hi + 1.0, 0.45 * fs)
    disp_band = _bandpass(d, fs, max(0.5, hr_lo - 0.3), top)
    env = None
    if method in ("envelope", "both", "auto") and fs > 2 * ENVELOPE_BAND_HZ[0] + 2:
        a = _bandpass(d, fs, ENVELOPE_BAND_HZ[0], ENVELOPE_BAND_HZ[1])
        env = _bandpass(np.abs(signal.hilbert(a)), fs, max(0.5, hr_lo - 0.3), top)
    grid = np.arange(bands.heart_bpm[0], bands.heart_bpm[1] + 1e-9, HR_GRID_BPM)
    gf = grid / 60.0
    allf = np.concatenate([gf, 2 * gf, 3 * gf])
    spectra = np.zeros((n_w, grid.size))
    for i, (s0, s1) in enumerate(wins):
        notch = np.ones(allf.size)                           # the steps: in both parts
        breath = np.ones(allf.size)                          # breathing: the displacement only
        df = 0.6 / ((s1 - s0) / fs)                          # about 3 bpm in 12 s windows
        if np.isfinite(br_bpm[i]) and BREATH_NOTCH:
            breath = _notch(allf, [m * br_bpm[i] / 60.0 for m in range(2, 5)], df)
        if np.isfinite(cadence_hz[i]):
            notch *= _notch(allf, [m * cadence_hz[i] for m in (0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6)], df)
        # The envelope of the 5-25 Hz content holds no breathing harmonics, and notched
        # with the displacement it lost the heart wherever that sat on one of them: a heart
        # of 60 bpm under breathing of 15 /min was followed at 120.
        parts = []
        if method in ("displacement", "both") or (method == "auto" and rest[i]):
            parts.append((disp_band[s0:s1], notch * breath))
        if env is not None and method != "displacement":
            parts.append((env[s0:s1], notch))
        hs = np.zeros(grid.size)
        for x, g in parts:
            p = _power(x, fs, allf) * g
            h = p[:grid.size] + 0.5 * p[grid.size:2 * grid.size] + 0.25 * p[2 * grid.size:]
            hs += h / (h.sum() + 1e-30)
        spectra[i] = hs / (hs.sum() + 1e-30)
    return grid, spectra


def rest_heart(chest: ChestSignal, n_frames: int):
    """
    The displacement the heart is followed in while the person is still: every receiver
    in each of the trunk's three bins on its own - its arc centred, its phase unwrapped -
    and their mean (vitalsigns.heart_channels), as the Vital Signs view takes it. The beam
    over the receivers and bins, right for a moving trunk, was 2.2 bpm off a pulse
    oximeter at rest on five seated recordings, this 1.3. None without the bins.
    """
    if chest.bins is None or chest.bins.shape[0] < n_frames:
        return None
    ch = vitalsigns.heart_channels(np.asarray(chest.bins[:n_frames]), 1, chest.lam_m, half=1)
    return ch.mean(axis=0) if len(ch) else None


def rest_windows(wins, motion, win_s: float, fs: float):
    """
    The heart's windows: wins, widened to REST_WIN_S about the same centre - shifted
    inside the recording at its ends - where every window over that stretch is at rest.
    A seated person's heart line needs about 30 s: on six seated recordings the 15 s
    windows were 7.0 bpm off a pulse oximeter with 76 % of them valid, these 4.6 bpm
    with 94 % (2.3-4.1 bpm on five of them). In motion (the trunk moving, the heart rate
    changing) they stay as they are.
    """
    full = int(round(REST_WIN_S * fs))
    if REST_WIN_S <= win_s or len(wins) == 0:
        return list(wins)
    first, last = wins[0][0], wins[-1][1]
    if last - first < full:
        return list(wins)
    starts = np.array([w[0] for w in wins])
    ends = np.array([w[1] for w in wins])
    still = np.asarray(motion) == 0
    out = []
    for s0, s1 in wins:
        a = min(max((s0 + s1) // 2 - full // 2, first), last - full)
        b = a + full
        out.append((a, b) if still[(starts < b) & (ends > a)].all() else (s0, s1))
    return out


def heart_track(spectra: np.ndarray, grid: np.ndarray, step_s: float,
                max_change_bpm_s: float = MAX_CHANGE_BPM_S, skip=None):
    """
    (heart rate, confidence, largest peak) per window: the Viterbi path through the
    spectra, windows in skip carrying no information; confidence is the path's value
    over the window's average (CONF_MIN and above: an estimate).
    """
    n_w = spectra.shape[0]
    if n_w == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    logp = np.log(spectra + 0.02 / grid.size)
    if skip is not None:
        logp[np.asarray(skip, bool)] = 0.0
    path = viterbi(logp, grid, step_s, max_change_bpm_s)
    conf = spectra[np.arange(n_w), path] * grid.size
    return grid[path], conf, grid[np.argmax(spectra, axis=1)]


# --- everything ----------------------------------------------------------------------------------

def analyse(cube: np.ndarray, params: dict, *, fps: float | None = None,
            bands: vital_bands.Bands | None = None, method: str = "auto",
            win_s: float = HR_WIN_S, step_s: float = HR_STEP_S,
            max_change_bpm_s: float = MAX_CHANGE_BPM_S, r_min: float = R_MIN_M,
            r_max: float | None = None, n_fft_range: int | None = None, t0_s: float = 0.0,
            allow_strong: bool = False, chest: ChestSignal | None = None,
            progress_cb=None) -> MovingHR:
    """
    Heart rate of the person in cube, window by window. method: "displacement" (the
    heart band of the chest displacement), "envelope" (the envelope of its 5-25 Hz
    content), "both" (their normalised spectra added) or "auto" (both at rest, the
    envelope alone in motion). allow_strong: estimate in strong motion (walking) too -
    untested on real recordings, see the module notes.
    """
    bands = bands or vital_bands.current()
    if chest is None:
        chest = chest_signal(cube, params, r_min=r_min, r_max=r_max, n_fft_range=n_fft_range,
                             fps=fps, t0_s=t0_s, progress_cb=progress_cb)
    fs = chest.fps
    phase, rate, slips = track_phase(chest.z, chest.t, chest.noise, chest.lam_m, chest.damaged)
    k = 4.0 * np.pi / chest.lam_m
    d = phase.mean(axis=1) / k                                   # the middle of every frame
    v = rate.mean(axis=1) / k
    F = d.size
    bad = chest.damaged
    d, n_steps = stitch_steps(d, chest.fps, bad, slips)
    if bad.any() and (~bad).sum() > 2:
        idx = np.arange(F)
        d = d.copy()
        d[bad] = np.interp(idx[bad], idx[~bad], d[~bad])
    t_s = chest.t[:, 0] - (chest.t[0, 0] - t0_s)

    # motion, breathing and cadence per window of the heart analysis
    wins = _windows(F, fs, win_s, step_s)
    body = _bandpass(d, fs, 0.02, 3.0) if F > 50 else d - d.mean()
    speed_all = np.gradient(body) * fs
    n_w = len(wins)
    win_t = np.zeros(n_w)
    motion = np.zeros(n_w, int)
    speed = np.zeros(n_w)
    br = np.full(n_w, np.nan)
    cad = np.full(n_w, np.nan)
    for i, (s0, s1) in enumerate(wins):
        win_t[i] = t_s[(s0 + s1 - 1) // 2]
        # motion from 2 s pieces: the worst piece decides
        pieces = _windows(s1 - s0, fs, MOTION_WIN_S, MOTION_WIN_S)
        rms = [float(np.sqrt(np.mean(speed_all[s0 + a:s0 + b] ** 2))) for a, b in pieces]
        speed[i] = max(rms)
        motion[i] = 0 if speed[i] < REST_MS else (1 if speed[i] < STRONG_MS else 2)
        if bad[s0:s1].mean() > 0.2 or slips[s0:s1].mean() > 0.05:
            motion[i] = 2
        br[i], _ = _rate(_bandpass(d[s0:s1], fs, *bands.resp_hz), fs, bands.resp_hz)
        if motion[i] >= 1:
            seg = _bandpass(d[s0:s1], fs, CADENCE_BAND_HZ[0] - 0.2, CADENCE_BAND_HZ[1] + 0.5)
            c, q = _rate(seg, fs, CADENCE_BAND_HZ, min_q=4.0)
            if np.isfinite(c) and np.sqrt(2) * np.std(seg) > CADENCE_MIN_M:
                cad[i] = c / 60.0
    hwins = rest_windows(wins, motion, win_s, fs)
    grid, spectra = heart_spectra(d, fs, bands, hwins,
                                  br_bpm=br, cadence_hz=cad, rest=motion == 0, method=method)
    still = motion == 0
    d_rest = rest_heart(chest, F) if still.any() else None
    if d_rest is not None:
        d_rest, _ = stitch_steps(d_rest, chest.fps, bad, np.zeros(F, bool))   # as d above
        if bad.any() and (~bad).sum() > 2:
            idx = np.arange(F)
            d_rest = d_rest.copy()
            d_rest[bad] = np.interp(idx[bad], idx[~bad], d_rest[~bad])
        _, spec_rest = heart_spectra(d_rest, fs, bands, [w for w, s_ in zip(hwins, still) if s_],
                                     br_bpm=br[still], cadence_hz=cad[still], method=method)
        spectra[still] = spec_rest
    hr, conf, hr_arg = heart_track(spectra, grid, step_s, max_change_bpm_s,
                                   skip=None if allow_strong else motion == 2)
    valid = ((motion < 2) | allow_strong) & (conf >= CONF_MIN)
    return MovingHR(fps=fs, t_s=t_s, displacement_m=d, velocity_ms=v, range_m=chest.range_m,
                    damaged=chest.damaged, slips=slips, win_t=win_t, motion=motion,
                    speed_ms=speed, br_bpm=br, cadence_hz=cad, hr_grid_bpm=grid,
                    spectra=spectra, hr_bpm=hr, hr_argmax_bpm=hr_arg, confidence=conf,
                    valid=valid, method=method, notes={"steps_removed": n_steps},
                    heart_win_s=np.array([[t_s[a], t_s[min(b, F) - 1]] for a, b in hwins]).reshape(-1, 2))


def describe(res: MovingHR) -> str:
    """A few lines for the Log."""
    n = res.win_t.size
    if not n:
        return "no windows"
    share = [float(np.mean(res.motion == m)) * 100 for m in range(3)]
    ok = res.valid
    txt = [f"windows {n}: rest {share[0]:.0f} %, light motion {share[1]:.0f} %, "
           f"strong {share[2]:.0f} % (no estimate); coverage {res.coverage() * 100:.0f} %"]
    if ok.any():
        txt.append(f"heart rate median {np.median(res.hr_bpm[ok]):.0f} bpm "
                   f"({np.min(res.hr_bpm[ok]):.0f}-{np.max(res.hr_bpm[ok]):.0f})")
    if np.isfinite(res.cadence_hz).any():
        txt.append(f"step frequency {np.nanmedian(res.cadence_hz) * 60:.0f} /min "
                   f"(notched out of the heart band)")
    if res.slips.any() or res.notes.get("steps_removed"):
        txt.append(f"phase tracker: {int(res.slips.sum())} frame(s) with a doubtful unwrap, "
                   f"{res.notes.get('steps_removed', 0)} step(s) bridged")
    return "; ".join(txt)


def compare_with_reference(res: MovingHR, ref_t_s, ref_bpm, offset_s: float = 0.0,
                           only_valid: bool = True) -> dict:
    """
    The windows against a reference heart rate (ECG belt, pulse oximeter) given at
    times ref_t_s (reference time + offset_s = radar time), each window compared with
    the mean reference over the window (roadmap 12.10).

    Returns {"all": stats, "rest": ..., "light": ..., "strong": ...}; stats: n, MAE,
    bias and the Bland-Altman limits of agreement (bias +- 1.96 SD) in bpm, the share
    within 5 and 10 bpm, and coverage - the share of the class's windows with a valid
    estimate.
    """
    rt = np.asarray(ref_t_s, float) + float(offset_s)
    rv = np.asarray(ref_bpm, float)
    ok = np.isfinite(rt) & np.isfinite(rv) & (rv > 20) & (rv < 250)
    rt, rv = rt[ok], rv[ok]
    half = 0.5 * HR_WIN_S
    spans = np.asarray(getattr(res, "heart_win_s", np.zeros((0, 2))), float)
    ref = np.full(res.win_t.size, np.nan)
    for i, tc in enumerate(res.win_t):
        # over the stretch the estimate is from (a 30 s one at rest), else the default window
        lo, hi = spans[i] if len(spans) == res.win_t.size else (tc - half, tc + half)
        m = (rt >= lo) & (rt <= hi)
        if m.any():
            ref[i] = rv[m].mean()
    out = {}
    for name, cls in (("all", None), ("rest", 0), ("light", 1), ("strong", 2)):
        in_cls = np.ones(res.win_t.size, bool) if cls is None else res.motion == cls
        have = in_cls & np.isfinite(ref)
        use = have & res.valid if only_valid else have
        e = res.hr_bpm[use] - ref[use]
        stats = {"n": int(use.sum()),
                 "coverage": float(use.sum() / have.sum()) if have.any() else float("nan")}
        if e.size:
            sd = float(np.std(e, ddof=1)) if e.size > 1 else 0.0
            stats.update(mae=float(np.mean(np.abs(e))), bias=float(np.mean(e)),
                         loa=(float(np.mean(e) - 1.96 * sd), float(np.mean(e) + 1.96 * sd)),
                         within5=float(np.mean(np.abs(e) <= 5)),
                         within10=float(np.mean(np.abs(e) <= 10)))
        out[name] = stats
    out["reference_per_window"] = ref
    return out
