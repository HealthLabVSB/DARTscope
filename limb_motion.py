"""
limb_motion.py

Fast movements of legs and arms in the micro-Doppler spectrogram (roadmap phase 10):
when a limb swings, how fast it went, and whether it was a leg or an arm.

A person is one strong, slow return - the torso - with weaker and faster ones around
it: arms, legs, feet. Along slow time each of them adds its own Doppler, so a
spectrogram of the person's range cells (velocity against time) shows the torso as a
bright line and every swing as an excursion from it. The outer edge of an excursion is
the fastest radial speed that limb reached.

Pipeline - one function per step; the Micro-Doppler tab and tools/limb_speed.py chain
them through analyse():

  spectrogram()   range FFT; the chirps of one transmitter, exactly as the
                  Range-Doppler map takes them (radar_processing.doppler_slow_time);
                  clutter suppression from the Analysis bar; a range gate on the person
                  (fixed bins, or following the moving energy frame by frame); STFT
                  along slow time, with times in seconds. Three layouts:
                    continuous  frames follow each other without a gap, so windows
                                run across frame boundaries;
                    frame       windows slide inside each frame (enough chirps);
                    per-frame   one spectrum per frame (too few chirps for a window).
  noise_floor()   a low percentile of every time slice, smoothed over time.
  envelopes()     the torso line and the upper and lower envelope: the fastest
                  velocity above the noise that is still connected to the body.
  detect_swings() excursions from the torso beyond a speed, with hysteresis.
  limb_height()   the elevation of the fastest point (MIMO with a vertical array
                  extent) and so its height: a foot is low, a hand is high.
  radial_velocity_from_positions(), compare_with_reference()
                  the same radial speed from motion-capture joint positions (Captury,
                  roadmap 10.8), to validate against.

Aliasing. A spectrum only knows velocity modulo 2 v_max. The envelopes are therefore
measured on the circular velocity axis, as the connected arc around the torso, and the
torso line itself is unwrapped over time. A foot at 4 m/s in front of a torso at 1 m/s
is read correctly with v_max = 2.94 m/s, because the whole body still spans less than
2 v_max. When it does not, the arc closes on itself; such slices are flagged
`saturated` instead of giving a wrong number.

Sign: positive velocity is motion away from the radar, as in every other view. The
radar measures only the radial component v cos(alpha): a limb moving across the line
of sight looks slower than it is.

The velocity scale follows radar_processing.wavelength_mid_m like the Range-Doppler map:
the wavelength in the middle of the ADC window (roadmap 1.9).

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy import signal as sps

import capture_params
import clutter
import radar_processing as rp

DEFAULT_WINDOW_S = 0.02          # STFT window: 10-30 ms resolves a swing in time
DEFAULT_OVERLAP = 0.75
DEFAULT_THRESHOLD_DB = 8.0       # above the noise floor
DEFAULT_SWING_M_S = 1.5          # speed relative to the torso that makes a swing
DEFAULT_GATE_WIDTH_M = 1.6       # automatic range gate, total width around the person
DEFAULT_MIN_DURATION_S = 0.08
DEFAULT_MAX_DURATION_S = 1.0
DEFAULT_LEG_MAX_HEIGHT_M = 0.6   # fastest point below this (above the floor): a leg
NOISE_PERCENTILE = 20.0
MIN_WINDOW = 8
MIN_NFFT = 64
_TINY = 1e-30


# --- the spectrogram ---------------------------------------------------------------

@dataclass
class Spectrogram:
    times_s: np.ndarray          # (T,) centre of every window, recording time
    velocity_m_s: np.ndarray     # (D,) Doppler axis, -v_max ... +v_max
    power: np.ndarray            # (T, D) linear power, summed over the gate and RX
    v_max: float                 # unambiguous velocity of this slow time
    interval_s: float            # time between two slow-time samples (one TX)
    window: int                  # samples per STFT window
    hop: int                     # samples between windows
    layout: str                  # 'continuous', 'frame' or 'per-frame'
    frame_of: np.ndarray         # (T,) frame (of the cube) each slice's centre lies in
    gate_bins: np.ndarray        # (T, 2) first and last range bin of the gate
    gate_range_m: np.ndarray     # (T,) centre of the gate
    range_m: np.ndarray          # range axis of all bins
    meta: dict = field(default_factory=dict)

    @property
    def velocity_resolution_m_s(self) -> float:
        """Doppler bin width of the window itself, lambda / (2 W interval) (not of the padded FFT)."""
        return float(self.velocity_m_s[1] - self.velocity_m_s[0]) * len(self.velocity_m_s) / self.window


def _next_pow2(n: int) -> int:
    return 1 << max(0, int(n) - 1).bit_length()


def _odd(n: float) -> int:
    k = max(1, int(round(n)))
    return k if k % 2 else k + 1


def _gate_centres(moving: np.ndarray, fps: float, hold_s: float = 0.5) -> np.ndarray:
    """
    Range bin of the person in every frame: the bin with the most moving energy,
    smoothed over range and time. Frames without clear motion keep their neighbours'.
    """
    F, G = moving.shape
    e = ndimage.uniform_filter1d(moving, size=3, axis=1, mode="nearest")
    centre = np.argmax(e, axis=1).astype(float)
    peak = e[np.arange(F), centre.astype(int)]
    clear = peak > 4.0 * np.median(e, axis=1)
    if not clear.any():
        return np.full(F, float(np.argmax(e.sum(axis=0))))
    idx = np.arange(F)
    centre = np.interp(idx, idx[clear], centre[clear])
    k = _odd(hold_s * fps) if fps else 1
    if k >= 3 and F >= k:
        centre = ndimage.median_filter(centre, size=k, mode="nearest")
    return np.round(centre)


def _stream_clutter(x: np.ndarray, params: dict, method: str, interval: float,
                    period: float | None, window: int):
    """
    Clutter suppression along the slow time of back-to-back frames: (x, window_mean).

    The per-frame methods of the Range-Doppler map would leave a step at every frame
    boundary, and a window running across it turns the steps into sidebands at
    multiples of the frame rate - lines over the whole velocity axis. So "mean" is
    taken out of every STFT window instead (window_mean=True), MTI and EMA run along
    the whole stream (EMA with the same time constant, per sample), and SVD removes
    the strongest range x receiver components of the whole stream.
    """
    if method in ("none", ""):
        return x, False
    if method == "mean":
        return x, True
    F, RX, G, L = x.shape
    s = np.transpose(x, (1, 2, 0, 3)).reshape(RX, G, F * L)
    if method in ("mti2", "mti3"):
        d = np.zeros_like(s)
        if method == "mti2":
            d[..., 1:] = s[..., 1:] - s[..., :-1]
        else:
            d[..., 2:] = s[..., 2:] - 2.0 * s[..., 1:-1] + s[..., :-2]
    elif method == "ema":
        a_frame = float(params.get("clutter_alpha", clutter.DEFAULT_ALPHA))
        a = 1.0 - (1.0 - a_frame) ** (interval / period) if period else a_frame
        d = clutter.suppress(s, "ema", slow_axis=-1, frame_axis=None, alpha=a)
    elif method == "svd":
        # The strongest components over range bins x receivers, taken out of the whole
        # stream at once (projection on the leading eigenvectors of their covariance):
        # the same thing clutter.py does per frame, without the steps that per-block
        # removal leaves where the blocks meet.
        rank = int(params.get("clutter_rank", clutter.DEFAULT_RANK))
        M = s.reshape(RX * G, F * L)
        cov = np.zeros((RX * G, RX * G), dtype=np.complex128)
        step = max(1, int(2e6 // max(1, RX * G)))
        for b0 in range(0, M.shape[1], step):
            blk = M[:, b0:b0 + step]
            cov += blk @ blk.conj().T
        _, U = np.linalg.eigh(cov)
        Uk = U[:, ::-1][:, :max(0, rank)].astype(np.complex64)
        d = np.empty_like(M)
        for b0 in range(0, M.shape[1], step):
            blk = M[:, b0:b0 + step]
            d[:, b0:b0 + step] = blk - Uk @ (Uk.conj().T @ blk)
        d = d.reshape(RX, G, F * L)
    else:
        raise ValueError(f"unknown clutter method {method!r}; choose from {clutter.METHODS}")
    d = np.asarray(d).astype(np.complex64, copy=False)
    return np.transpose(d.reshape(RX, G, F, L), (2, 0, 1, 3)), False


def _stft_power(x: np.ndarray, starts: np.ndarray, W: int, nfft: int,
                block: int = 256, keep_first: bool = False,
                demean: bool = False, gate: np.ndarray | None = None) -> np.ndarray:
    """
    Power spectra of windows of the last axis of x (..., N) starting at `starts`,
    fftshifted and summed over the leading axes: (len(starts), nfft), or
    (x.shape[0], len(starts), nfft) with keep_first. demean takes the mean of every
    window out first (static clutter of that window).

    gate (len(starts), G): x is (..., G, N) and every window sums only the range bins
    its row of `gate` selects. Gating the spectra and not the signal matters when the
    gate moves while a window runs: switching a bin on or off mid-window is a step,
    and the step of a strong torso's range sidelobe splatters over the whole
    velocity axis.

    The window is a 4-term Blackman-Harris (sidelobes -92 dB), not a Hann: the torso is
    40-60 dB above the noise, and the Hann skirt (-31 dB, falling 18 dB per octave)
    drew it +-1 m/s wide at the envelope threshold on a synthetic walker - wider than
    the arm swings being looked for.
    """
    win = sps.windows.blackmanharris(W, sym=True).astype(np.float32)
    view = np.lib.stride_tricks.sliding_window_view(x, W, axis=-1)
    axes = tuple(range(1 if keep_first else 0, x.ndim - 1))
    shape = ((x.shape[0],) if keep_first else ()) + (len(starts), nfft)
    out = np.empty(shape, dtype=np.float32)
    for i in range(0, len(starts), block):
        sub = view[..., starts[i:i + block], :]
        if demean:
            # The mean weighted by the window: that is what makes the DC bin of the
            # windowed spectrum exactly zero. A plain mean leaves a moving torso's
            # residual at 0 m/s (~20 dB above the noise on a synthetic walker).
            sub = sub - (sub @ win)[..., None] / win.sum()
        sub = sub * win
        spec = np.fft.fft(sub, n=nfft, axis=-1)
        pw = spec.real ** 2 + spec.imag ** 2
        if gate is not None:
            pw = pw.sum(axis=tuple(range(pw.ndim - 3)))              # (G, nb, nfft)
            p = np.einsum("gbf,bg->bf", pw, gate[i:i + block].astype(np.float32))
        else:
            p = pw.sum(axis=axes)
        out[..., i:i + block, :] = np.fft.fftshift(p, axes=-1)
    return out


def spectrogram(cube: np.ndarray, params: dict, *, bins=None, gate: str = "auto",
                gate_width_m: float = DEFAULT_GATE_WIDTH_M,
                window_s: float = DEFAULT_WINDOW_S, overlap: float = DEFAULT_OVERLAP,
                rx_mode: str = "MRC", n_fft_range: int | None = None,
                t0_s: float = 0.0) -> Spectrogram:
    """
    Micro-Doppler spectrogram of the range gate on the person.

    cube: raw (frames, chirps, RX, samples) as decoded.
    bins: (first, last) range bins searched for the person (gate="auto", within the
          positive ranges) or taken as they are (gate="fixed"); default: from 0.25 m to
          the end of positive range.
    gate_width_m: total width of the automatic gate around the person.
    window_s, overlap: STFT window length and overlap; the window is shortened to the
          chirps of a frame when frames have gaps and too few chirps.
    rx_mode: MRC (power sum), SUM (coherent sum) or RXk, like the Antenna selector.
    t0_s: recording time of frame 0 (the start of a segment).
    Clutter suppression is params['clutter_method'] (default mean), as in the
    Range-Doppler map; MTI runs across frame boundaries when frames are back to back.
    """
    cube = np.asarray(cube)
    F, C, RXn, Ns = cube.shape
    if F < 1 or C < 1:
        raise ValueError("empty cube")
    per_frame_bytes = C * RXn * rp.range_fft_size(Ns, n_fft_range) * 8
    chunk = max(1, int(64e6 // max(1, per_frame_bytes)))

    X = None
    range_m = None
    for f0 in range(0, F, chunk):
        spec, meta = rp.compute_range_fft_cube(cube[f0:f0 + chunk], params, window="hann",
                                               remove_dc=True, n_fft_range=n_fft_range,
                                               return_complex=True)
        if range_m is None:
            range_m = np.asarray(meta["range_m"], dtype=float)
            R = spec.shape[-1]
            positive = R // 2 if meta.get("use_full_spectrum") else R
            if bins is None:
                dr = range_m[1] - range_m[0] if R > 1 else 1.0
                b0 = int(np.clip(np.ceil((0.25 + rp.range_bias_m(params)) / dr), 1, R - 1))
                bins = (b0, max(b0, positive - 1))
            b0 = int(np.clip(min(bins), 0, R - 1))
            b1 = int(np.clip(max(bins), b0, R - 1))
            if str(gate).lower().startswith("auto") and b0 < positive:
                b1 = min(b1, positive - 1)        # do not follow a mirror image
        sel, interval, info = rp.doppler_slow_time(spec[..., b0:b1 + 1], params)
        if X is None:
            X = np.empty((F,) + sel.shape[1:], dtype=np.complex64)
        X[f0:f0 + sel.shape[0]] = sel
        del spec, sel
    _, RX, G, L = X.shape
    dr = float(range_m[1] - range_m[0]) if len(range_m) > 1 else 1.0

    # Time of every slow-time sample, and whether frames follow without a gap.
    offset = float(np.mean(info["positions"])) * capture_params.chirp_period_s(params)
    t = capture_params.slow_time_axis(params, F, L, interval, offset, t0_s)
    period = capture_params.frame_period_s(params)
    contiguous = F > 1 and (period is None or abs(period - L * interval) <= 0.5 * interval)
    fps = 1.0 / period if period else 1.0 / (L * interval)

    # The gate: every frame's person bin +- half the width, or the fixed bins.
    if str(gate).lower().startswith("auto"):
        moving = np.empty((F, G))
        for f0 in range(0, F, chunk):
            xb = X[f0:f0 + chunk]
            moving[f0:f0 + chunk] = np.sum(np.abs(xb - xb.mean(axis=-1, keepdims=True)) ** 2,
                                           axis=(1, 3))
        centre = _gate_centres(moving, fps)
        half = max(1, int(round(0.5 * gate_width_m / dr)))
        lo = np.clip(centre - half, 0, G - 1).astype(int)
        hi = np.clip(centre + half, 0, G - 1).astype(int)
    else:
        centre = np.full(F, (G - 1) / 2.0)
        lo, hi = np.zeros(F, int), np.full(F, G - 1)
    mask = (np.arange(G)[None, :] >= lo[:, None]) & (np.arange(G)[None, :] <= hi[:, None])

    method = str(params.get("clutter_method") or "mean").lower()
    W_req = max(MIN_WINDOW, int(round(window_s / interval)))
    window_mean = False
    if contiguous:
        # gated per window on the spectra below, never by switching the signal
        X, window_mean = _stream_clutter(X, params, method, interval, period, W_req)
    else:
        if method == "mean":
            window_mean = True                      # per window, weighted: see _stft_power
        else:
            X = clutter.from_params(X, params, slow_axis=-1, frame_axis=0).astype(np.complex64,
                                                                                  copy=False)
        X *= mask[:, None, :, None]                 # windows never cross a frame here

    mode = (rx_mode or "MRC").strip().upper()
    if mode == "SUM":
        X = X.sum(axis=1, keepdims=True)
    elif mode.startswith("RX"):
        try:
            k = int(mode[2:])
        except ValueError:
            k = 0
        k = int(np.clip(k, 0, RX - 1))
        X = X[:, k:k + 1]

    warnings = []
    if contiguous:
        layout, W = "continuous", min(W_req, F * L)
    elif L >= W_req:
        layout, W = "frame", W_req
    else:
        layout, W = "per-frame", L
        warnings.append(f"{L} samples per frame are shorter than the {window_s * 1e3:.0f} ms "
                        f"window and frames have gaps: one spectrum per frame.")
    H = W if layout == "per-frame" else max(1, int(round(W * (1.0 - overlap))))
    nfft = max(MIN_NFFT, _next_pow2(W))

    if layout == "continuous":
        s = np.transpose(X, (1, 2, 0, 3)).reshape(X.shape[1], G, F * L)
        starts = np.arange(0, F * L - W + 1, H)
        frame_of = (starts + (W - 1) // 2) // L
        power = _stft_power(s, starts, W, nfft, demean=window_mean, gate=mask[frame_of])
        tf = t.reshape(-1)
        times = 0.5 * (tf[starts] + tf[starts + W - 1])
    else:
        starts_in = np.arange(0, L - W + 1, H)
        Tw = len(starts_in)
        power = np.empty((F * Tw, nfft), dtype=np.float32)
        step = max(1, int(1e7 // max(1, X.shape[1] * G * Tw * max(W, nfft) * 8)))
        for f0 in range(0, F, step):
            blk = X[f0:f0 + step]                         # (frames, RX, G, L)
            p = _stft_power(blk, starts_in, W, nfft, keep_first=True, demean=window_mean)
            power[f0 * Tw:(f0 + blk.shape[0]) * Tw] = p.reshape(-1, nfft)
        times = (0.5 * (t[:, starts_in] + t[:, starts_in + W - 1])).reshape(-1)
        frame_of = np.repeat(np.arange(F), Tw)

    lam = rp.wavelength_mid_m(params)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=interval))
    v = freqs * lam / 2.0
    v_max = lam / (4.0 * interval) if interval > 0 else float(np.max(np.abs(v)))
    if method in ("mti2", "mti3"):
        # An MTI filter colours the spectrum: |H|^2 = 4 sin^2(pi f T) (two-pulse) or its
        # square (three-pulse), +6 / +12 dB at +-v_max and a deep null at 0. A noise
        # floor read from a low percentile then sits in the attenuated part and the
        # amplified noise near +-v_max passes the threshold as "body" (false swings at
        # 40-90 m/s on a synthetic walker). Dividing by |H|^2 - limited to 30 dB near
        # the null - makes the noise flat again and gives moving targets their level.
        h2 = (2.0 * np.sin(np.pi * freqs * interval)) ** (2 if method == "mti2" else 4)
        power = power / np.maximum(h2, 1e-3)[None, :].astype(np.float32)
    if L < MIN_WINDOW:
        warnings.append(f"Only {L} slow-time samples per frame: velocity resolution "
                        f"{lam / (2.0 * max(1, W) * interval):.2f} m/s.")
    gate_bins = np.stack([b0 + lo[frame_of], b0 + hi[frame_of]], axis=1)
    gate_range = range_m[np.clip(b0 + np.round(centre[frame_of]).astype(int), 0, len(range_m) - 1)]
    meta = {"clutter_method": method, "rx_mode": mode, "gate": "auto" if str(gate).lower().startswith("auto") else "fixed",
            "gate_width_m": float(gate_width_m), "search_bins": (b0, b1), "nfft": nfft,
            "window_s": W * interval, "hop_s": H * interval, "positions": info["positions"],
            "pattern_len": info["pattern_len"], "frames": F, "samples_per_frame": L,
            "frame_period_s": period, "warnings": warnings, "t0_s": float(t0_s)}
    return Spectrogram(times_s=np.asarray(times, float), velocity_m_s=v, power=power,
                       v_max=float(v_max), interval_s=float(interval), window=int(W), hop=int(H),
                       layout=layout, frame_of=np.asarray(frame_of, int), gate_bins=gate_bins,
                       gate_range_m=gate_range, range_m=range_m, meta=meta)


def display_image(spec: Spectrogram, max_columns: int = 4000):
    """
    (image in dB shaped (D, N), extent) on an even time grid for imshow.

    Slices are not evenly spaced when frames have gaps; each column takes the slice
    nearest to it, so the gaps do not squeeze the picture.
    """
    t = spec.times_s
    db = 10.0 * np.log10(spec.power + _TINY)
    if len(t) < 2:
        return db.T, [t[0] - 0.5, t[0] + 0.5, spec.velocity_m_s[0], spec.velocity_m_s[-1]]
    step = max(float(np.median(np.diff(t))), (t[-1] - t[0]) / max_columns)
    grid = np.arange(t[0], t[-1] + 0.5 * step, step)
    idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    idx -= (grid - t[idx - 1]) < (t[idx] - grid)
    dv = spec.velocity_m_s[1] - spec.velocity_m_s[0]
    extent = [grid[0] - 0.5 * step, grid[-1] + 0.5 * step,
              spec.velocity_m_s[0] - 0.5 * dv, spec.velocity_m_s[-1] + 0.5 * dv]
    return db[idx].T, extent


# --- noise, torso and envelopes ----------------------------------------------------

def noise_floor(power: np.ndarray, percentile: float = NOISE_PERCENTILE,
                smooth: int = 9, long_slices: int = 0, margin_db: float = 3.0) -> np.ndarray:
    """
    Noise power of every time slice: a low percentile over velocity, median-smoothed
    over `smooth` slices. A low percentile rather than the median, because a walking
    person can fill more than half of the Doppler bins of a slice.

    long_slices > 0 caps it at `margin_db` above its own 10th percentile over that many
    slices. When the body fills the whole velocity axis the percentile of the slice is
    the body, and the floor would jump up by 30-40 dB with it; the real noise does not
    change from one moment to the next, so the cap keeps such slices visible (and
    flagged as saturated) instead of empty.
    """
    floor = np.percentile(np.asarray(power, float), percentile, axis=1)
    if smooth > 1 and floor.size >= smooth:
        floor = ndimage.median_filter(floor, size=_odd(smooth), mode="nearest")
    if long_slices and floor.size >= 3:
        k = _odd(min(long_slices, floor.size))
        k = k - 2 if k > floor.size else k
        if k >= 3:
            low = ndimage.percentile_filter(floor, 10, size=k, mode="nearest")
            floor = np.minimum(floor, low * 10.0 ** (margin_db / 10.0))
    return np.maximum(floor, _TINY)


def _fill_gaps(occ: np.ndarray, gap: int) -> np.ndarray:
    """Fill runs of at most `gap` empty bins between occupied ones, circularly."""
    if gap <= 0:
        return occ
    left = np.zeros_like(occ)
    right = np.zeros_like(occ)
    for s in range(1, gap + 1):
        left |= np.roll(occ, s, axis=1)
        right |= np.roll(occ, -s, axis=1)
    return occ | (left & right)


@dataclass
class Envelopes:
    times_s: np.ndarray
    torso_m_s: np.ndarray        # torso line, unwrapped over time
    upper_m_s: np.ndarray        # fastest velocity away from the radar still on the body
    lower_m_s: np.ndarray        # fastest velocity towards the radar
    saturated: np.ndarray        # the body filled the whole velocity axis
    floor: np.ndarray            # noise power per slice
    threshold_db: float
    method: str
    v_max: float

    @property
    def relative_up(self) -> np.ndarray:
        return self.upper_m_s - self.torso_m_s

    @property
    def relative_down(self) -> np.ndarray:
        return self.torso_m_s - self.lower_m_s


def _torso_line(P, occ, v, v_max, times, torso_db, smooth_s):
    """
    Circular centroid of the strongest bins, unwrapped over time and median-smoothed
    over `smooth_s`. A slice with nothing above the noise counts as 0 m/s: clutter
    suppression takes out a body standing still, and what is left then is only the
    limbs - their velocity is not the body's. The median over about a second keeps a
    swing (a few tenths of a second) from dragging the line along with it.
    """
    peak = P.max(axis=1, keepdims=True)
    strong = occ & (P >= peak * 10.0 ** (-torso_db / 10.0))
    z = np.sum(np.where(strong, P, 0.0) * np.exp(1j * np.pi * v / v_max)[None, :], axis=1)
    valid = strong.any(axis=1) & (np.abs(z) > 0)
    # A slice whose body covers most of the axis has no meaningful centre; it is
    # bridged from its neighbours instead (and would otherwise let the unwrapping
    # slip by a whole 2 v_max).
    crowded = occ.mean(axis=1) > 0.6
    torso = np.zeros(len(times))
    if valid.any():
        theta = np.angle(z)
        theta[~valid] = 0.0
        use = ~crowded
        if use.any():
            un = np.unwrap(theta[use])
            torso = np.interp(times, times[use], un) * v_max / np.pi
    if len(times) > 2:
        k = _odd(smooth_s / max(np.median(np.diff(times)), 1e-9))
        if k >= 3 and len(times) >= k:
            torso = ndimage.median_filter(torso, size=k, mode="nearest")
    return torso, valid


def _arc_edge(db_seq: np.ndarray, length: np.ndarray, thr_db: np.ndarray, method: str,
              k_local: int, drop_db: float = 3.0) -> np.ndarray:
    """
    Where the arc ends along one walk direction, in bins from the anchor (T,).

    db_seq: (T, D) power in dB along the walk, index 0 = the anchor; length: occupied
    bins from the anchor. "threshold": the crossing of the noise threshold after the
    last occupied bin. "half-power": the outermost point where the power is still
    within `drop_db` of the strongest of the outermost `k_local` bins. A band of
    scatterers ending at v_tip is a step smoothed by the window, and a symmetric window
    puts the half-power point of that step at v_tip whatever the SNR - unlike the
    threshold crossing, which the window's skirt pushes further out the stronger the
    band is. It is looked for from the outside in, so a fade inside the band does not
    stop it early. Where the band does not stand `drop_db` above the noise threshold,
    the threshold crossing is used.
    """
    T, D = db_seq.shape
    rows = np.arange(T)
    last = np.clip(length - 1, 0, D - 1)
    nxt = np.clip(length, 0, D - 1)
    span = db_seq[rows, last] - db_seq[rows, nxt]
    frac = np.clip((db_seq[rows, last] - thr_db) / np.maximum(span, 1e-9), 0.0, 1.0)
    edge = last + frac
    if method != "half-power":
        return edge
    m = np.arange(D)[None, :]
    outer = (m >= np.clip(length - k_local, 0, None)[:, None]) & (m <= last[:, None])
    target = np.where(outer, db_seq, -np.inf).max(axis=1) - drop_db
    inside = (m <= last[:, None]) & (db_seq >= target[:, None])
    mo = np.where(inside.any(axis=1), D - 1 - np.argmax(inside[:, ::-1], axis=1), 0)
    after = np.clip(mo + 1, 0, D - 1)
    s2 = db_seq[rows, mo] - db_seq[rows, after]
    f2 = np.clip((db_seq[rows, mo] - target) / np.maximum(s2, 1e-9), 0.0, 1.0)
    return np.where(target > thr_db, mo + f2, edge)


def envelopes(spec: Spectrogram, threshold_db: float = DEFAULT_THRESHOLD_DB,
              method: str = "half-power", max_gap_m_s: float = 0.25, torso_db: float = 6.0,
              torso_smooth_s: float = 1.0, percentile: float = 1.0,
              noise_percentile: float = NOISE_PERCENTILE, smooth_slices: int = 3,
              smooth: tuple = (3, 3), floor: np.ndarray | None = None) -> Envelopes:
    """
    Torso line and the upper and lower envelope of the spectrogram.

    The body is the connected arc of bins more than `threshold_db` above the noise
    around the torso line; gaps up to `max_gap_m_s` wide do not break it. Its edge:
    method="half-power" (default): where the outer part of the band falls 3 dB below
        its own level - independent of how strong the limb is;
    method="threshold": where it falls below the noise threshold - the furthest
        extent, biased outwards by the window's skirt the more the stronger the limb;
    method="percentile": the velocities below which `percentile` % and above which
        `percentile` % of the power above the noise lie, around the torso.
    All are measured on the circular velocity axis around the unwrapped torso line,
    so they may exceed +-v_max (see the module notes on aliasing).

    The power is first averaged over `smooth` = (slices, velocity bins). A limb is many
    scatterers at nearly the same velocity; within one Doppler bin they add with
    random phases and a bin can fade by 10-20 dB (speckle), which otherwise cuts a
    band in two from one slice to the next.
    """
    P = np.asarray(spec.power, float)
    T, D = P.shape
    if smooth and T >= smooth[0]:
        P = ndimage.uniform_filter(P, size=smooth, mode=("nearest", "wrap"))
    v = spec.velocity_m_s
    dv = float(v[1] - v[0])
    vmax = spec.v_max
    if floor is None:
        dt = float(np.median(np.diff(spec.times_s))) if T > 1 else 1.0
        floor = noise_floor(P, noise_percentile, long_slices=int(2.0 / max(dt, 1e-9)))
    floor = np.asarray(floor, float)
    thr = floor * 10.0 ** (threshold_db / 10.0)
    occ_raw = P > thr[:, None]
    gap = max(1, int(round(max_gap_m_s / dv)))
    occ = _fill_gaps(occ_raw, gap)
    # Clutter suppression leaves a notch around 0 m/s. A limb passing through zero
    # velocity (an arm swinging back while the body walks forward) would be cut there,
    # so the notch is bridged - but only where the body continues on its other side,
    # otherwise the envelope would reach out to the notch's edge by itself.
    method_c = str(spec.meta.get("clutter_method", "none"))
    if method_c != "none":
        notch = 2.0 * spec.velocity_resolution_m_s
        if method_c in ("mti2", "mti3"):            # a high-pass: slow speeds fade gradually
            notch = max(notch, 0.05 * vmax)
        in_notch = np.abs(v) <= notch
        if in_notch.any():
            k = int(in_notch.sum()) + gap
            left = np.zeros_like(occ)
            right = np.zeros_like(occ)
            for s in range(1, k + 1):
                left |= np.roll(occ, s, axis=1)
                right |= np.roll(occ, -s, axis=1)
            occ |= in_notch[None, :] & left & right
    torso, _ = _torso_line(P, occ_raw, v, vmax, spec.times_s, torso_db, torso_smooth_s)

    # Anchor: the bin of the torso line, or the nearest occupied bin close to it.
    wrapped = (torso + vmax) % (2.0 * vmax) - vmax
    a = np.round((wrapped - v[0]) / dv).astype(int) % D
    rows = np.arange(T)
    found = occ[rows, a].copy()
    search = max(2, int(round(0.1 * D)))
    for o in [s * d for s in range(1, search + 1) for d in (1, -1)]:
        cand = (a + o) % D
        take = ~found & occ[rows, cand]
        a = np.where(take, cand, a)
        found |= take
    v_anchor = v[a] + 2.0 * vmax * np.round((torso - v[a]) / (2.0 * vmax))

    upper = np.full(T, np.nan)
    lower = np.full(T, np.nan)
    ar = np.arange(D)[None, :]
    up_seq = occ[rows[:, None], (a[:, None] + ar) % D]
    dn_seq = occ[rows[:, None], (a[:, None] - ar) % D]
    up_len = np.where(up_seq.all(axis=1), D, np.argmin(up_seq, axis=1))
    dn_len = np.where(dn_seq.all(axis=1), D, np.argmin(dn_seq, axis=1))
    # The arc (nearly) closes on itself: the body spans ~2 v_max and an edge could be
    # read on the wrong side. A few free bins are needed to tell the sides apart.
    free = D - (up_len + dn_len - 1)
    saturated = found & (free < max(3, D // 16))
    ok = found & ~saturated
    if method == "percentile":
        # Power above the threshold on the arc only, laid out with the anchor in the
        # middle; the velocities that cut `percentile` % off either end.
        rel_bins = np.arange(D) - D // 2
        on_arc = (rel_bins[None, :] <= (up_len - 1)[:, None]) & \
                 (rel_bins[None, :] >= -(dn_len - 1)[:, None])
        order = (a[:, None] + rel_bins[None, :]) % D
        e = np.where(on_arc, np.clip(P[rows[:, None], order] - thr[:, None], 0.0, None), 0.0)
        tot = e.sum(axis=1)
        ok &= tot > 0
        cum = np.cumsum(e, axis=1) / np.maximum(tot, _TINY)[:, None]
        q = percentile / 100.0
        for level, out in ((1.0 - q, upper), (q, lower)):
            k = np.argmax(cum >= level, axis=1)
            k0 = np.clip(k - 1, 0, D - 1)
            c0, c1 = cum[rows, k0], cum[rows, k]
            frac = np.where(c1 > c0, (level - c0) / np.maximum(c1 - c0, _TINY), 0.0)
            vel = v_anchor + (rel_bins[k0] + np.where(k > 0, frac, 0.0)) * dv
            out[ok] = vel[ok]
    else:
        db = 10.0 * np.log10(P + _TINY)
        thr_db = 10.0 * np.log10(thr)
        k_local = max(2, int(round(4.0 * D / max(1, spec.window))))   # ~4 resolution bins
        for sign, length, out in ((1, up_len, upper), (-1, dn_len, lower)):
            seq = db[rows[:, None], (a[:, None] + sign * ar) % D]
            edge = _arc_edge(seq, length, thr_db, method, k_local)
            out[ok] = (v_anchor + sign * edge * dv)[ok]

    if smooth_slices and smooth_slices > 1 and T >= smooth_slices:
        for out in (upper, lower):
            nan = np.isnan(out)
            filled = np.where(nan, torso, out)
            sm = ndimage.median_filter(filled, size=_odd(smooth_slices), mode="nearest")
            out[:] = np.where(nan, np.nan, sm)
    return Envelopes(times_s=spec.times_s, torso_m_s=torso, upper_m_s=upper, lower_m_s=lower,
                     saturated=saturated, floor=floor, threshold_db=float(threshold_db),
                     method=method, v_max=float(vmax))


# --- swings ------------------------------------------------------------------------

@dataclass
class Swing:
    direction: str               # 'away' (positive velocity) or 'towards'
    index: int                   # time slice of the peak
    t_start_s: float
    t_peak_s: float
    t_end_s: float
    duration_s: float
    v_peak_m_s: float            # radial velocity at the peak, signed
    v_rel_m_s: float             # speed relative to the torso at the peak
    torso_m_s: float
    range_m: float               # centre of the range gate at the peak
    saturated: bool              # the body filled the velocity axis somewhere in it
    unfolded: bool               # |v_peak| > v_max: read from the connected envelope
    elevation_deg: float = float("nan")
    height_m: float = float("nan")   # above the floor (or relative to the radar)
    limb: str = ""               # 'leg', 'arm' or '' when it cannot be told

    @property
    def speed_m_s(self) -> float:
        return abs(self.v_peak_m_s)


def _runs(mask: np.ndarray):
    """[start, end] index pairs (inclusive) of the True runs of a boolean array."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def detect_swings(env: Envelopes, spec: Spectrogram | None = None,
                  min_speed: float = DEFAULT_SWING_M_S, release: float | None = None,
                  min_duration: float = DEFAULT_MIN_DURATION_S,
                  max_duration: float = DEFAULT_MAX_DURATION_S,
                  merge_gap_s: float = 0.04, baseline_s: float = 2.0) -> list[Swing]:
    """
    Swings: stretches where an envelope rises more than `min_speed` above its usual
    distance from the torso and until it falls back within `release` of it (default
    half of min_speed) - hysteresis, so one noisy slice does not cut a swing in two.

    "Usual distance" is the 20th percentile of the envelope's distance from the torso
    over `baseline_s`: even a still body has a Doppler width (breathing, sway, the
    window's skirt around a strong torso), and a swing is what stands out of it.
    Stretches closer than `merge_gap_s` are joined; the rest must last
    `min_duration`..`max_duration`. The reported v_rel is from the torso line itself.
    """
    t = env.times_s
    if len(t) == 0:
        return []
    release = 0.5 * min_speed if release is None else release
    hop = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    n_base = _odd(baseline_s / hop) if hop > 0 else 1
    out = []
    for direction, rel, vel in (("away", env.relative_up, env.upper_m_s),
                                ("towards", env.relative_down, env.lower_m_s)):
        r = np.nan_to_num(rel, nan=0.0)
        if n_base >= 3 and len(r) >= n_base:
            base = ndimage.percentile_filter(r, 20, size=n_base, mode="nearest")
        else:
            base = np.full_like(r, np.percentile(r, 20))
        x = r - np.maximum(base, 0.0)
        merged = []
        for s, e in _runs(x >= release):
            if merged and t[s] - t[merged[-1][1]] <= merge_gap_s:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        for s, e in merged:
            if not np.any(x[s:e + 1] >= min_speed):
                continue
            dur = float(t[e] - t[s] + hop)
            if dur < min_duration or dur > max_duration:
                continue
            k = int(s + np.argmax(x[s:e + 1]))
            v_pk = float(vel[k])
            near = (t >= t[s] - 0.1) & (t <= t[e] + 0.1)
            out.append(Swing(
                direction=direction, index=k, t_start_s=float(t[s]), t_peak_s=float(t[k]),
                t_end_s=float(t[e]), duration_s=dur, v_peak_m_s=v_pk, v_rel_m_s=float(r[k]),
                torso_m_s=float(env.torso_m_s[k]),
                range_m=float(spec.gate_range_m[k]) if spec is not None else float("nan"),
                saturated=bool(env.saturated[near].any()),
                unfolded=bool(abs(v_pk) > env.v_max)))
    return sorted(out, key=lambda sw: sw.t_peak_s)


def summary(env: Envelopes, swings: list[Swing]) -> dict:
    """Maximum speeds: of the swings, and of the envelopes where they are not saturated."""
    ok = ~env.saturated
    up = env.upper_m_s[ok]
    dn = env.lower_m_s[ok]
    away = [s for s in swings if s.direction == "away"]
    towards = [s for s in swings if s.direction == "towards"]
    return {
        "swings": len(swings),
        "swings_away": len(away),
        "swings_towards": len(towards),
        "max_swing_away_m_s": max((s.v_peak_m_s for s in away), default=float("nan")),
        "max_swing_towards_m_s": min((s.v_peak_m_s for s in towards), default=float("nan")),
        "max_swing_speed_m_s": max((s.speed_m_s for s in swings), default=float("nan")),
        "max_envelope_away_m_s": float(np.nanmax(up)) if np.isfinite(up).any() else float("nan"),
        "max_envelope_towards_m_s": float(np.nanmin(dn)) if np.isfinite(dn).any() else float("nan"),
        "saturated_fraction": float(np.mean(env.saturated)) if env.saturated.size else 0.0,
        "v_max_m_s": env.v_max,
    }


# --- leg or arm: height of the fastest point -------------------------------------------

def limb_height(cube: np.ndarray, params: dict, spec: Spectrogram, swings: list[Swing],
                radar_height_m: float | None = None, tilt_deg: float = 0.0,
                leg_max_height_m: float = DEFAULT_LEG_MAX_HEIGHT_M,
                n_fft_range: int | None = None, fraction: float = 0.85) -> list[Swing]:
    """
    Elevation and height of every swing's fastest point, from one frame of the
    virtual array (needs an array with vertical extent, e.g. the IWR6843AOP).

    The cell is taken at `fraction` of the way from the torso to the peak velocity -
    the envelope itself lies at the noise threshold, a little inside it the limb is
    strong enough for an angle. TDM phases between the transmitters are compensated
    with that (unfolded) velocity, so a swing faster than v_max still gets its angle.
    Height = radar height + range * sin(elevation + tilt); with a radar height given,
    a fastest point below `leg_max_height_m` is a leg, above it an arm. Without a
    radar height the height is relative to the radar and `limb` stays empty.
    """
    import AOA
    import calibration

    if not swings:
        return swings
    F, C, RX, _ = cube.shape
    ntx, stride, L = calibration._layout(params, C, RX)
    V = ntx * RX
    _, y, _ = AOA._get_antenna_geometry(params, V, RX)
    if not np.any(np.asarray(y) != 0) or L < 2:
        return swings                                   # no elevation in this array
    az_grid, el_grid = calibration._angle_grid(params, V, RX)
    SV = calibration.steering(params, V, RX, az_grid, el_grid).conj().T   # (K, V)
    lam = rp.wavelength_mid_m(params)
    tc = capture_params.chirp_period_s(params)
    t_rep = stride * tc
    v_max_loop = lam / (4.0 * t_rep) if t_rep > 0 else spec.v_max
    win = np.hanning(L).astype(np.float32)
    for sw in swings:
        f = int(np.clip(spec.frame_of[sw.index], 0, F - 1))
        rc, meta = rp.compute_range_fft_cube(cube[f:f + 1], params, window="hann", remove_dc=True,
                                             n_fft_range=n_fft_range, return_complex=True)
        x = rc[0, :L * stride].reshape(L, stride, RX, -1)[:, :ntx].reshape(L, V, -1)
        v_sel = sw.torso_m_s + fraction * (sw.v_peak_m_s - sw.torso_m_s)
        slot_phase = 4.0 * np.pi * v_sel * tc / lam
        comp = np.exp(-1j * np.repeat(np.arange(ntx), RX) * slot_phase).astype(np.complex64)
        x = (x - x.mean(axis=0, keepdims=True)) * comp[None, :, None]
        dop = np.fft.fftshift(np.fft.fft(x * win[:, None, None], axis=0), axes=0)  # (L, V, R)
        v_loop = np.fft.fftshift(np.fft.fftfreq(L, d=t_rep)) * lam / 2.0
        vw = (v_sel + v_max_loop) % (2.0 * v_max_loop) - v_max_loop
        k0 = int(np.argmin(np.abs(v_loop - vw)))
        ks = [(k0 + d) % L for d in (-1, 0, 1)]
        g0, g1 = spec.gate_bins[sw.index]
        cells = np.sum(np.abs(dop[ks][:, :, g0:g1 + 1]) ** 2, axis=1)       # (3, G)
        kk, rr = np.unravel_index(int(np.argmax(cells)), cells.shape)
        u = dop[ks[kk], :, g0 + rr]
        u = AOA._apply_aop_rx_phase_correction(u[None, None, :, None], params, V, RX)[0, 0, :, 0]
        u = AOA._apply_calibration(u, params, V)
        i = int(np.argmax(np.abs(SV @ u)))
        el = float(el_grid[i])
        r = float(np.asarray(meta["range_m"])[g0 + rr])
        z = r * np.sin(np.deg2rad(el + tilt_deg))
        sw.elevation_deg = el
        if radar_height_m is not None:
            sw.height_m = float(radar_height_m + z)
            sw.limb = "leg" if sw.height_m < leg_max_height_m else "arm"
        else:
            sw.height_m = float(z)
    return swings


# --- validation against motion capture ---------------------------------------------------

def radial_velocity_from_positions(t_s, xyz_m, radar_xyz_m, smooth_s: float = 0.05):
    """
    Radial velocity of a tracked point (a joint from motion capture) as the radar sees
    it: the rate of change of its distance to the radar, positive away from it.

    Only the radar's position in the capture's coordinates is needed - a distance does
    not depend on how the radar is turned. The derivative is a Savitzky-Golay one over
    about `smooth_s` when the samples are evenly spaced, np.gradient otherwise.
    """
    t = np.asarray(t_s, float)
    p = np.asarray(xyz_m, float).reshape(len(t), 3)
    d = np.linalg.norm(p - np.asarray(radar_xyz_m, float)[None, :], axis=1)
    if len(t) < 3:
        return np.zeros_like(d)
    dt = np.diff(t)
    step = float(np.median(dt))
    even = step > 0 and np.all(np.abs(dt - step) <= 0.05 * step)
    win = _odd(smooth_s / step) if step > 0 else 0
    if even and win >= 5 and len(d) >= win:
        return sps.savgol_filter(d, win, 2, deriv=1, delta=step)
    return np.gradient(d, t)


def compare_with_reference(swings: list[Swing], ref_t_s, ref_v_m_s, offset_s: float = 0.0,
                           margin_s: float = 0.05) -> dict:
    """
    Every swing's peak against the reference's extreme radial velocity (same sign)
    within the swing, reference time + offset_s = radar time.
    """
    rt = np.asarray(ref_t_s, float) + float(offset_s)
    rv = np.asarray(ref_v_m_s, float)
    pairs = []
    for sw in swings:
        m = (rt >= sw.t_start_s - margin_s) & (rt <= sw.t_end_s + margin_s)
        if not m.any():
            continue
        ref = float(np.max(rv[m])) if sw.direction == "away" else float(np.min(rv[m]))
        pairs.append((sw, ref))
    if not pairs:
        return {"pairs": [], "n": 0}
    err = np.array([sw.v_peak_m_s - ref for sw, ref in pairs])
    radar = np.array([sw.v_peak_m_s for sw, _ in pairs])
    ref = np.array([r for _, r in pairs])
    corr = float(np.corrcoef(radar, ref)[0, 1]) if len(pairs) > 2 else float("nan")
    return {"pairs": pairs, "n": len(pairs), "bias_m_s": float(err.mean()),
            "mae_m_s": float(np.abs(err).mean()), "rmse_m_s": float(np.sqrt(np.mean(err ** 2))),
            "correlation": corr}


# --- everything at once, and exports ---------------------------------------------------

@dataclass
class LimbAnalysis:
    spec: Spectrogram
    env: Envelopes
    swings: list
    summary: dict


def analyse(cube: np.ndarray, params: dict, *, bins=None, gate: str = "auto",
            gate_width_m: float = DEFAULT_GATE_WIDTH_M, window_s: float = DEFAULT_WINDOW_S,
            overlap: float = DEFAULT_OVERLAP, rx_mode: str = "MRC",
            n_fft_range: int | None = None, t0_s: float = 0.0,
            threshold_db: float = DEFAULT_THRESHOLD_DB, method: str = "half-power",
            swing_m_s: float = DEFAULT_SWING_M_S,
            min_duration: float = DEFAULT_MIN_DURATION_S,
            max_duration: float = DEFAULT_MAX_DURATION_S,
            radar_height_m: float | None = None, tilt_deg: float = 0.0,
            heights: bool = True) -> LimbAnalysis:
    """spectrogram() -> envelopes() -> detect_swings() -> limb_height() -> summary()."""
    spec = spectrogram(cube, params, bins=bins, gate=gate, gate_width_m=gate_width_m,
                       window_s=window_s, overlap=overlap, rx_mode=rx_mode,
                       n_fft_range=n_fft_range, t0_s=t0_s)
    env = envelopes(spec, threshold_db=threshold_db, method=method)
    swings = detect_swings(env, spec, min_speed=swing_m_s, min_duration=min_duration,
                           max_duration=max_duration)
    if heights and swings:
        try:
            limb_height(cube, params, spec, swings, radar_height_m=radar_height_m,
                        tilt_deg=tilt_deg, n_fft_range=n_fft_range)
        except Exception as e:                      # heights are extra information only
            print(f"[limb] height of the swings not available: {e}")
    return LimbAnalysis(spec=spec, env=env, swings=swings, summary=summary(env, swings))


SWING_FIELDS = ("direction", "t_start_s", "t_peak_s", "t_end_s", "duration_s", "v_peak_m_s",
                "v_rel_m_s", "torso_m_s", "range_m", "elevation_deg", "height_m", "limb",
                "unfolded", "saturated")


def swing_rows(swings: list[Swing]) -> list[dict]:
    """One dict per swing with SWING_FIELDS, for CSV."""
    return [{k: getattr(sw, k) for k in SWING_FIELDS} for sw in swings]


def envelope_table(env: Envelopes):
    """(array, header): time, torso, upper and lower envelope, saturated - one row per slice."""
    arr = np.column_stack([env.times_s, env.torso_m_s, env.upper_m_s, env.lower_m_s,
                           env.saturated.astype(float)])
    return arr, "time_s,torso_m_s,upper_m_s,lower_m_s,saturated"


def describe(res: LimbAnalysis) -> str:
    """Text report of an analysis: settings, maximum speeds and the swings."""
    s, sm = res.spec, res.summary
    m = s.meta
    lines = [f"Micro-Doppler: {s.layout} layout, window {m['window_s'] * 1e3:.1f} ms, "
             f"hop {m['hop_s'] * 1e3:.1f} ms, {len(s.times_s)} slices; v_max {s.v_max:.2f} m/s, "
             f"resolution {s.velocity_resolution_m_s:.3f} m/s",
             f"Gate: {m['gate']} (bins {m['search_bins'][0]}-{m['search_bins'][1]}"
             + (f", {m['gate_width_m']:.2f} m around the person" if m['gate'] == 'auto' else "")
             + f"), clutter {m['clutter_method']}, antenna {m['rx_mode']}"]
    lines += [f"Note: {w}" for w in m.get("warnings", [])]

    def _f(x):
        return "-" if not np.isfinite(x) else f"{x:+.2f}"

    lines.append(f"Max speed of swings: away {_f(sm['max_swing_away_m_s'])} m/s, towards "
                 f"{_f(sm['max_swing_towards_m_s'])} m/s ({sm['swings']} swings: "
                 f"{sm['swings_away']} away, {sm['swings_towards']} towards)")
    lines.append(f"Envelope extremes: {_f(sm['max_envelope_away_m_s'])} / "
                 f"{_f(sm['max_envelope_towards_m_s'])} m/s; saturated "
                 f"{100 * sm['saturated_fraction']:.1f} % of the time")
    if res.swings:
        lines.append(f"{'peak [s]':>9}{'dur [s]':>9}{'dir':>9}{'v [m/s]':>9}{'rel':>7}"
                     f"{'range':>7}{'height':>8}  limb")
        for sw in res.swings:
            h = "" if not np.isfinite(sw.height_m) else f"{sw.height_m:.2f}"
            flag = (" unfolded" if sw.unfolded else "") + (" saturated" if sw.saturated else "")
            lines.append(f"{sw.t_peak_s:>9.2f}{sw.duration_s:>9.2f}{sw.direction:>9}"
                         f"{sw.v_peak_m_s:>+9.2f}{sw.v_rel_m_s:>7.2f}{sw.range_m:>7.2f}"
                         f"{h:>8}  {sw.limb}{flag}")
    return "\n".join(lines)
