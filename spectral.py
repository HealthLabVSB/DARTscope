"""
spectral.py

Sub-bin peak interpolation and zoom FFT.

A spectrum is sampled on a grid, and a peak between two grid points is reported at
the nearer one: up to half a bin off. For vital signs that is the whole measurement -
at 100 fps with a 20 s window a Welch bin is 3 bpm wide, so a heart at 72.4 bpm reads
as 72 or 75. Fitting a parabola through the peak and its two neighbours places it
between the bins.

Which parabola matters, and it was measured rather than taken on trust. On a tone
at a known fractional frequency, Hann window as the range FFT and the vital-sign
spectra use, worst-case error over offsets -0.5 .. +0.5 bin:

                               no padding    4x zero padding
    parabola on magnitude       0.052 bin       0.0031 bin
    parabola on log (Gaussian)  0.016 bin       0.0008 bin

The Gaussian fit is the default. It works unchanged on power spectra (Welch): the log
of power is twice the log of magnitude, and the vertex of the parabola does not move.
A "window-corrected Jacobsen" estimator was tried as well; with a Hann window its error
reached 1.4 bins, so it is not offered.

Zoom FFT evaluates the same spectrum on a finer grid over a narrow band only. It is
the zero-padded FFT restricted to that band, computed without padding the whole
record: it places a peak more finely, but it does not separate two targets closer than
the bandwidth or the observation time allows - nothing short of more bandwidth or a
longer record does that.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import numpy as np

METHODS = ("gaussian", "parabolic", "none")
_default_method = "gaussian"


def set_default_method(method: str) -> None:
    """Choose the interpolation every caller gets unless it asks for one explicitly."""
    global _default_method
    if method not in METHODS:
        raise ValueError(f"unknown peak interpolation {method!r}; choose from {METHODS}")
    _default_method = method


def default_method() -> str:
    return _default_method


def _vertex(a: float, b: float, c: float) -> float:
    den = a - 2.0 * b + c
    if den == 0 or not np.isfinite(den):
        return 0.0
    return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))


def peak_offset(values, k: int, method: str | None = None, scale: str = "linear") -> float:
    """
    Offset of the true peak from bin `k`, in bins, within [-0.5, 0.5].

    values: magnitude or power (scale="linear"), or already in dB (scale="db").
    A parabola through dB values is the Gaussian fit, so "gaussian" on a dB spectrum
    and on the same spectrum in linear units give the same answer.
    Returns 0 at the edges, where there is no neighbour on one side.
    """
    method = method or _default_method
    if method == "none":
        return 0.0
    if method not in METHODS:
        raise ValueError(f"unknown peak interpolation {method!r}")
    v = np.asarray(values)
    k = int(k)
    if k <= 0 or k >= v.shape[0] - 1:
        return 0.0
    a, b, c = (float(np.real(v[k - 1])), float(np.real(v[k])), float(np.real(v[k + 1])))
    if scale == "db":
        return _vertex(a, b, c) if method == "gaussian" else _vertex(
            *(10 ** (np.array([a, b, c]) / 20.0)))
    if method == "gaussian":
        if min(a, b, c) <= 0:
            return 0.0
        return _vertex(*np.log([a, b, c]))
    return _vertex(a, b, c)


def vertex_offsets(a, b, c, method: str | None = None, scale: str = "linear"):
    """
    peak_offset for many peaks at once: a, b, c are the values left of, at and right
    of each peak (arrays of one shape). Returns offsets in bins within [-0.5, 0.5].
    """
    method = method or _default_method
    if method not in METHODS:
        raise ValueError(f"unknown peak interpolation {method!r}")
    a, b, c = (np.real(np.asarray(v)).astype(float) for v in (a, b, c))
    if method == "none":
        return np.zeros(np.broadcast(a, b, c).shape)
    if scale == "db":
        if method != "gaussian":
            a, b, c = (10 ** (v / 20.0) for v in (a, b, c))
    elif method == "gaussian":
        tiny = np.finfo(float).tiny
        ok = (a > 0) & (b > 0) & (c > 0)
        a, b, c = (np.log(np.maximum(v, tiny)) for v in (a, b, c))
    den = a - 2.0 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(den != 0, 0.5 * (a - c) / den, 0.0)
    d = np.where(np.isfinite(d), d, 0.0)
    if scale != "db" and method == "gaussian":
        d = np.where(ok, d, 0.0)
    return np.clip(d, -0.5, 0.5)


def circular_peak(power, axis: int = -1, method: str | None = None):
    """
    Fractional index of the maximum along a circular axis (an unshifted FFT), signed:
    within [-L/2, L/2). Neighbours wrap around, as the spectrum does.
    """
    p = np.asarray(power)
    n = p.shape[axis]
    k = np.argmax(p, axis=axis)
    ke = np.expand_dims(k, axis)
    if n >= 3:
        d = vertex_offsets(np.take_along_axis(p, (ke - 1) % n, axis),
                           np.take_along_axis(p, ke, axis),
                           np.take_along_axis(p, (ke + 1) % n, axis), method)
        kf = k + np.squeeze(d, axis)
    else:
        kf = k.astype(float)
    return np.where(kf >= n / 2.0, kf - n, kf)


def refine_peak(axis, values, k: int, method: str | None = None, scale: str = "linear"):
    """
    Interpolated position and height of the peak at bin `k`.

    axis: the physical axis (Hz, m, m/s, degrees), locally uniform around k.
    Returns (position, height); height is in the units of `values`.
    """
    axis = np.asarray(axis, dtype=float)
    v = np.asarray(values)
    k = int(k)
    d = peak_offset(v, k, method, scale)
    if d == 0.0 or k <= 0 or k >= axis.size - 1:
        return float(axis[k]), float(np.real(v[k]))
    step = 0.5 * (axis[k + 1] - axis[k - 1])
    a, b, c = float(np.real(v[k - 1])), float(np.real(v[k])), float(np.real(v[k + 1]))
    method = method or _default_method
    if method == "gaussian" and scale == "linear" and min(a, b, c) > 0:
        la, lb, lc = np.log([a, b, c])
        height = float(np.exp(lb - 0.25 * (la - lc) * d))
    else:
        height = b - 0.25 * (a - c) * d
    return float(axis[k] + d * step), height


def refine_peak_nd(values, index, method: str | None = None, scale: str = "linear"):
    """
    Fractional index of a peak in an N-dimensional map, one axis at a time.

    Separable fits are what radar processing chains use for range-Doppler and
    range-angle maps; each axis gets its own three-point parabola through the peak.
    Returns a tuple of floats of the same length as `index`.
    """
    v = np.asarray(values)
    out = []
    for ax, i in enumerate(index):
        sl = [int(j) for j in index]
        sl[ax] = slice(None)
        out.append(float(i) + peak_offset(v[tuple(sl)], int(i), method, scale))
    return tuple(out)


def interp_axis(axis, fractional_index: float) -> float:
    """Physical value at a fractional index of a uniformly sampled axis."""
    axis = np.asarray(axis, dtype=float)
    if axis.size == 0:
        return float("nan")
    return float(np.interp(fractional_index, np.arange(axis.size), axis))


# --- zoom FFT ----------------------------------------------------------------------

def zoom_spectrum(x, fs: float, f_lo: float, f_hi: float, n: int = 512,
                  window: str | None = "hann", axis: int = -1):
    """
    Spectrum of `x` on `n` points from f_lo to f_hi, via the chirp-z transform.

    Numerically the same values a zero-padded FFT gives at those frequencies (see the
    module docstring for what that does and does not buy). Complex input may use
    negative frequencies; real input is symmetric, so ask for the positive band.
    Returns (frequencies, complex spectrum along `axis`).
    """
    from scipy.signal import czt

    x = np.asarray(x)
    m = x.shape[axis]
    if window:
        from scipy.signal import get_window
        shape = [1] * x.ndim
        shape[axis] = m
        x = x * get_window(window, m, fftbins=True).reshape(shape)
    step = (f_hi - f_lo) / n
    w = np.exp(-2j * np.pi * step / fs)
    a = np.exp(2j * np.pi * f_lo / fs)
    spec = czt(x, m=n, w=w, a=a, axis=axis)
    freqs = f_lo + step * np.arange(n)
    return freqs, spec


def zoom_range_profile(samples, params: dict, r_lo: float, r_hi: float, n: int = 512,
                       window: str | None = "hann"):
    """
    Range profile over [r_lo, r_hi] metres on `n` points, from fast-time samples.

    samples: ADC samples with fast time on the last axis, e.g. one chirp or a whole
    cube [frames, chirps, rx, samples]. The beat frequency of a target at range R is
    2 R S / c for a chirp slope S, which is how the range band becomes a frequency band.
    Returns (range_m, complex spectrum).
    """
    from radar_processing import C0, _fs_hz, _slope_hz_per_s, range_bias_m

    fs, slope = _fs_hz(params), _slope_hz_per_s(params)
    if fs <= 0 or slope <= 0:
        raise ValueError("the capture parameters give no sample rate or chirp slope")
    to_f = 2.0 * slope / C0
    bias = range_bias_m(params)          # a calibrated offset, as on every range axis
    freqs, spec = zoom_spectrum(samples, fs, (r_lo + bias) * to_f, (r_hi + bias) * to_f,
                                n, window, axis=-1)
    return freqs / to_f - bias, spec


def centre_phase(spec, n_samples: int, n_fft: int, axis: int = -1, first_bin: int = 0,
                 full_spectrum: bool = False):
    """
    Range spectra with the phase of every bin referred to the middle of the chirp
    (roadmap 1.10), so a target moving into the next bin keeps its phase.

    The FFT refers the phase of bin k to the first sample of the chirp; for a target
    between bins it is then the target's phase at the middle of the chirp plus
    pi * (f - k / n_fft) * (N - 1), which jumps as soon as the peak moves to the next
    bin. Multiplying bin k by exp(+j pi k (N - 1) / n_fft) leaves the phase at the
    middle of the chirp in the whole main lobe - exactly for a window symmetric about
    the middle (Hann is), whatever the zero padding.

    spec: spectra along `axis`, holding bins first_bin, first_bin + 1, ... of an
    n_fft-point FFT of n_samples samples; full_spectrum: natural FFT order of a complex
    FFT, where the upper half are the negative bins.
    """
    spec = np.asarray(spec)
    nb = spec.shape[axis]
    k = np.arange(first_bin, first_bin + nb, dtype=float)
    if full_spectrum:
        k = np.where(k >= n_fft / 2.0, k - n_fft, k)
    shape = [1] * spec.ndim
    shape[axis] = nb
    rot = np.exp(1j * np.pi * k * (int(n_samples) - 1) / float(n_fft)).reshape(shape)
    return spec * rot.astype(np.complex64 if spec.dtype == np.complex64 else complex)
