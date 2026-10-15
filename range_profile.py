import numpy as np
from typing import Tuple

C0 = 299_792_458.0  # m/s


def _profile_complex(x: np.ndarray, Nfft: int, remove_dc: bool = True):
    """
    Chirp-averaged, normalised complex spectrum of one channel: every bin for complex
    samples, the positive half for real ones.

    Complex 1x sampling spans 0..fs, so every bin is a range, as the RTI and the Range
    FFT show it; the positive half alone ended the profile at half the maximum range.

    x: (chirps, samples) or (samples,). Returns (spectrum, Nfft) with |spectrum| on
    the dBFS scale range_profile_dbfs reports.
    """
    if x.ndim > 2:
        raise ValueError("Input x must be 1D or 2D.")
    if x.ndim == 1:
        x = x[np.newaxis, :]
    num_samples = x.shape[1]
    if Nfft < num_samples:                 # never cut the chirp short (roadmap 9.2)
        from radar_processing import range_fft_size
        Nfft = range_fft_size(num_samples, Nfft)

    # 1. DC removal
    if remove_dc:
        x = x - x.mean(axis=1, keepdims=True)

    # 2. Hann window
    hann_win = np.hanning(num_samples).astype(x.dtype, copy=False)
    x_win = x * hann_win

    # 3. FFT (and average if multiple chirps)
    spec = np.fft.fft(x_win, n=Nfft, axis=1)
    spec_avg = spec.mean(axis=0)  # Coherent averaging over chirps

    # 4. Complex: every bin; real: the positive half (including Nyquist bin to match rfft)
    n_bins = Nfft if np.iscomplexobj(x) else Nfft // 2 + 1
    spec_half = spec_avg[:n_bins]

    # 5. Normalization: Nfft * coherent_gain (input assumed normalized in loader)
    coherent_gain = np.sum(hann_win) / num_samples if num_samples > 0 else 1.0
    norm_factor = max(1e-12, Nfft * coherent_gain)
    out = spec_half / norm_factor
    # Compensate for one-sided spectrum (multiply by 2) for REAL data
    # to match levels of analytic complex signals.
    if not np.iscomplexobj(x):
        out = out * 2.0
    return out, Nfft


def _range_axis(Fs: float, slope: float, Nfft: int, range_bias_m: float = 0.0,
                n_bins: int | None = None) -> np.ndarray:
    """Range of each of the first n_bins bins (default: the positive half)."""
    if n_bins is None:
        n_bins = Nfft // 2 + 1
    df = Fs / float(Nfft) if Nfft > 0 else 0.0
    dR = C0 * df / (2.0 * slope) if slope > 0 else 1.0
    return np.arange(n_bins, dtype=float) * dR - float(range_bias_m or 0.0)


def range_profile_dbfs(
    x: np.ndarray,
    Fs: float,
    slope: float,
    Nfft: int,
    adcBits: int,
    remove_dc: bool = True,
    range_bias_m: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates a single range profile in dBFS (decibels relative to full scale).
    Can average over multiple chirps if x is 2D.

    Args:
        x: 1D or 2D numpy array of complex or real samples for one RX.
           If 2D, shape is (num_chirps, num_samples).
        Fs: Sampling frequency in Hz.
        slope: Chirp slope in Hz/s.
        Nfft: FFT size (raised to hold the whole chirp if shorter).
        adcBits: ADC bit depth for full-scale normalization (kept for API compatibility).
        remove_dc: If True, subtracts the mean from the input signal.
        range_bias_m: measured range offset taken off the axis (calibration.py).

    Returns:
        A tuple of (range_axis_m, magnitude_dBFS).
    """
    spec, Nfft = _profile_complex(np.asarray(x), Nfft, remove_dc)
    return _range_axis(Fs, slope, Nfft, range_bias_m, spec.size), 20 * np.log10(np.abs(spec) + 1e-9)


def range_profile_rx(
    x: np.ndarray,
    Fs: float,
    slope: float,
    Nfft: int,
    adcBits: int,
    mode: str = "MRC",
    remove_dc: bool = True,
    range_bias_m: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Range profile of one frame over several RX channels, x shaped (chirps, rx, samples).

    mode, as the Analysis bar's antenna selector:
        "MRC"  root mean square over RX of each channel's spectrum - power combining,
               which needs no phase alignment between channels;
        "SUM"  coherent mean over RX - gains only where the channels are in phase
               (a target at boresight on a calibrated array);
        "RXk"  channel k alone.
    Returns (range_axis_m, magnitude_dBFS) on the scale of range_profile_dbfs.
    """
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[:, np.newaxis, :]
    n_rx = x.shape[1]
    mode = (mode or "MRC").strip().upper()
    if mode.startswith("RX"):
        try:
            k = int(mode[2:])
        except ValueError:
            k = 0
        return range_profile_dbfs(x[:, max(0, min(n_rx - 1, k)), :], Fs, slope, Nfft,
                                  adcBits, remove_dc, range_bias_m)
    specs = []
    for r in range(n_rx):
        s, n_used = _profile_complex(x[:, r, :], Nfft, remove_dc)
        specs.append(s)
    specs = np.array(specs)
    if mode == "SUM":
        mag = np.abs(specs.mean(axis=0))
    else:
        mag = np.sqrt(np.mean(np.abs(specs) ** 2, axis=0))
    return _range_axis(Fs, slope, n_used, range_bias_m, mag.size), 20 * np.log10(mag + 1e-9)
