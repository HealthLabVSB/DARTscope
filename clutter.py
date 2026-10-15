"""
clutter.py

Static clutter suppression along slow time.

Everything here works on complex data (after the range FFT) with a slow-time axis - the
chirps (loops) of one frame - and, optionally, a frame axis. The methods differ in what
they call "static":

    mean   subtract the mean over the chirps of each frame. Removes everything at zero
           Doppler in that frame - a wall, but also a person sitting still, whose chest
           moves far too little within one frame to leave the zero-Doppler bin.
    ema    subtract a background tracked over FRAMES by exponential averaging,
           b[f] = (1 - alpha) b[f-1] + alpha * mean(frame f). A wall stays put and is
           removed; a breathing person keeps changing phase from frame to frame (a few
           millimetres of chest travel is a large fraction of a 5 mm wavelength), so
           the background never catches up with them and they stay in the map.
           alpha = 1 - exp(-frame_period / tau): tau is how long the scene has to stay
           unchanged before it counts as background.
    mti2   two-pulse canceller x[n] - x[n-1]: |H| = 2 |sin(pi f / PRF)|, a null at zero
           Doppler, 6 dB of gain at the Nyquist velocity and slow targets attenuated.
    mti3   three-pulse canceller x[n] - 2x[n-1] + x[n-2]: |H| = 4 sin^2(pi f / PRF),
           a wider null (clutter that sways a little) at the cost of more loss on slow
           targets.
    svd    remove the `rank` strongest singular components of the slow-time x (range,
           channel) matrix of each frame. The dominant component is the static scene
           even when its amplitude drifts a little, which a plain mean does not track.

The MTI filters are applied circularly (the first sample is differenced against the
last) so the chirp count, and with it every Doppler axis, stays the same. The one
wrapped sample is where a Hann window is zero; without a window it leaks a little.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import numpy as np

METHODS = ("mean", "ema", "mti2", "mti3", "svd", "none")
DEFAULT_ALPHA = 0.05
DEFAULT_RANK = 1
FEW_LOOPS = 8           # fewer chirps per frame: "mean" would take a seated person away too
SHORT_FRAME_S = 0.003   # ... as would a frame whose chirps of one TX span less than this: under
                        # 3 ms a bin is wider than 0.6 m/s and a walker hardly leaves 0 m/s; at
                        # 20 ms the people preset's 8.3 ms frames lost the frame mean too and
                        # the still scene's remains filled 0 m/s at every range


def alpha_from_time_constant(tau_s: float, frame_period_s: float) -> float:
    """EMA weight for a background time constant `tau_s` at one update per frame."""
    if tau_s <= 0 or frame_period_s <= 0:
        return 1.0
    return float(1.0 - np.exp(-frame_period_s / tau_s))


def _ema_background(m: np.ndarray, alpha: float, axis: int, warm: bool = False) -> np.ndarray:
    """
    The background each sample is compared with: the average of the ones BEFORE it.

    Using the previous background rather than one that already includes the current
    sample keeps a target that appears suddenly at full strength. The first sample has
    no past, so it is its own background - or, with `warm`, the mean of the first time
    constant (1/alpha samples), so the first frames are not blank either.
    """
    from scipy.signal import lfilter

    alpha = float(np.clip(alpha, 1e-6, 1.0))
    m = np.moveaxis(m, axis, 0)
    if warm:
        n0 = int(min(m.shape[0], max(1, round(1.0 / alpha))))
        m0 = m[:n0].mean(axis=0, keepdims=True)
    else:
        m0 = m[:1]
    b = lfilter([alpha], [1.0, -(1.0 - alpha)], m, axis=0, zi=(1.0 - alpha) * m0)[0]
    prev = np.concatenate([m0, b[:-1]], axis=0)
    return np.moveaxis(prev.astype(m.dtype, copy=False), 0, axis)


def _svd_remove(x: np.ndarray, slow_axis: int, batch_axis: int | None, rank: int) -> np.ndarray:
    x = np.asarray(x)
    if batch_axis is None:
        xs = np.moveaxis(x, slow_axis, 0)
        shape = xs.shape
        mat = xs.reshape(1, shape[0], -1)
    else:
        xs = np.moveaxis(x, (batch_axis, slow_axis), (0, 1))
        shape = xs.shape
        mat = xs.reshape(shape[0], shape[1], -1)
    rank = int(max(0, min(rank, min(mat.shape[1:]))))
    if rank == 0:
        return x
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    low = (u[..., :rank] * s[..., None, :rank]) @ vh[..., :rank, :]
    out = (mat - low).reshape(shape)
    if batch_axis is None:
        return np.moveaxis(out, 0, slow_axis)
    return np.moveaxis(out, (0, 1), (batch_axis, slow_axis))


def suppress(x: np.ndarray, method: str = "mean", slow_axis: int = 1,
             frame_axis: int | None = 0, alpha: float = DEFAULT_ALPHA,
             rank: int = DEFAULT_RANK, warm: bool = False) -> np.ndarray:
    """
    Clutter-suppressed copy of `x`.

    slow_axis:  chirps (loops) within a frame - the axis the Doppler FFT runs along.
    frame_axis: frames, for "ema" (background over frames) and to run "svd" per frame.
                None when the data has no separate frame axis, e.g. a slow-time series
                with one sample per frame; "ema" then runs along `slow_axis`.
    warm:       "ema" starts from the mean of its first time constant instead of the
                first frame alone.
    """
    method = (method or "none").lower()
    x = np.asarray(x)
    if method == "none":
        return x
    if method == "mean":
        return x - x.mean(axis=slow_axis, keepdims=True)
    if method == "ema":
        if frame_axis is None:
            return x - _ema_background(x, alpha, slow_axis, warm)
        m = x.mean(axis=slow_axis, keepdims=True)
        return x - _ema_background(m, alpha, frame_axis, warm)
    if method == "mti2":
        return x - np.roll(x, 1, axis=slow_axis)
    if method == "mti3":
        return x - 2.0 * np.roll(x, 1, axis=slow_axis) + np.roll(x, 2, axis=slow_axis)
    if method == "svd":
        return _svd_remove(x, slow_axis, frame_axis, rank)
    raise ValueError(f"unknown clutter method {method!r}; choose from {METHODS}")


def from_params(x: np.ndarray, params: dict | None, slow_axis: int = 1,
                frame_axis: int | None = 0, default: str = "mean") -> np.ndarray:
    """
    suppress() with the method chosen in the capture parameters.

    params['clutter_method'] (default `default`), params['clutter_alpha'] and
    params['clutter_rank'] - how the GUI passes its choice down to AOA.py and the
    Range-Doppler map without changing every signature in between.

    With fewer than FEW_LOOPS chirps per frame along `slow_axis`, "mean" becomes "ema"
    from its first time constant on (few_loops_method): the mean of 4 chirps removes a
    seated person with the wall, which left the Movement Map of the vital-signs settings
    a flat field.
    """
    p = params or {}
    method, warm = few_loops_method(p.get("clutter_method") or default,
                                    x.shape[slow_axis] if frame_axis is not None else None)
    return suppress(x, method, slow_axis, frame_axis,
                    float(p.get("clutter_alpha", DEFAULT_ALPHA)),
                    int(p.get("clutter_rank", DEFAULT_RANK)),
                    warm or bool(p.get("clutter_ema_warm", False)))


def few_loops_method(method: str, loops: int | None, span_s: float | None = None) -> tuple[str, bool]:
    """
    (method, warm start) for frames of `loops` chirps spanning span_s: "mean" over fewer
    than FEW_LOOPS chirps, or over less than SHORT_FRAME_S, is "ema" with a warm start -
    the background over frames - since a chest moves far too little within a frame to
    survive the frame mean (the cascade's multi-person preset: 8 chirps of a TX in
    1.9 ms, and the Range-Doppler map of someone seated 2 m away was noise). None: no
    frame axis.
    """
    if (method or "").lower() == "mean" and loops is not None and (
            loops < FEW_LOOPS or (span_s is not None and 0 < span_s < SHORT_FRAME_S)):
        return "ema", True
    return method, False
