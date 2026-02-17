import numpy as np
from typing import Tuple

C0 = 299_792_458.0  # m/s

def range_profile_dbfs(
    x: np.ndarray,
    Fs: float,
    slope: float,
    Nfft: int,
    adcBits: int,
    remove_dc: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates a single range profile in dBFS (decibels relative to full scale).
    Can average over multiple chirps if x is 2D.

    Args:
        x: 1D or 2D numpy array of complex or real samples for one RX.
           If 2D, shape is (num_chirps, num_samples).
        Fs: Sampling frequency in Hz.
        slope: Chirp slope in Hz/s.
        Nfft: FFT size.
        adcBits: ADC bit depth for full-scale normalization (kept for API compatibility).
        remove_dc: If True, subtracts the mean from the input signal.

    Returns:
        A tuple of (range_axis_m, magnitude_dBFS).
    """
    if x.ndim > 2:
        raise ValueError("Input x must be 1D or 2D.")

    is_2d = x.ndim == 2
    if not is_2d:
        x = x[np.newaxis, :]  # Make it 2D for consistent processing

    num_samples = x.shape[1]

    # 1. DC removal
    if remove_dc:
        x = x - x.mean(axis=1, keepdims=True)

    # 2. Hann window
    hann_win = np.hanning(num_samples).astype(x.dtype, copy=False)
    x_win = x * hann_win

    # 3. FFT (and average if multiple chirps)
    spec = np.fft.fft(x_win, n=Nfft, axis=1)
    spec_avg = spec.mean(axis=0)  # Coherent averaging over chirps

    # 4. Take positive half (including Nyquist bin to match rfft)
    n_bins = Nfft // 2 + 1
    spec_half = spec_avg[:n_bins]

    # 5. Normalization: Nfft * coherent_gain (input assumed normalized in loader)
    coherent_gain = np.sum(hann_win) / num_samples if num_samples > 0 else 1.0
    norm_factor = max(1e-12, Nfft * coherent_gain)
    
    # NEW: Compensate for one-sided spectrum (multiply by 2) for REAL data
    # to match levels of analytic complex signals.
    mag_linear = np.abs(spec_half) / norm_factor
    if not np.iscomplexobj(x):
        mag_linear *= 2.0
        
    mag_dbfs = 20 * np.log10(mag_linear + 1e-9)

    # 6. Range axis
    df = Fs / float(Nfft) if Nfft > 0 else 0.0
    dR = C0 * df / (2.0 * slope) if slope > 0 else 1.0
    range_axis = np.arange(n_bins, dtype=float) * dR

    return range_axis, mag_dbfs

