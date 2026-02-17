import numpy as np

def compute_range_fft_cube(
    cube: np.ndarray,
    params: dict,
    window: str = 'hann',
    pad_pow2: bool = True,
    remove_dc: bool = True,
    n_fft_range: int = None,
    use_full_spectrum: bool = False,
    return_complex: bool = False,
):
    """
    1D Range FFT over samples (last axis).
    - DC removal per (frame, chirp, RX) before windowing.
    - Hann window by default.
    - For complex input: use FFT and optionally show full spectrum (no fftshift).
    - For real input: use rFFT (positive half only) or standard FFT + half cut.
    Returns magnitude or complex spectrum of the FFT.
    """
    is_complex = np.iscomplexobj(cube)
    # NEW: Complex2X / PseudoReal -> half-spectrum i pro komplexní vstup
    try:
        fmt_str = str(params.get('dataFmt_adcFmt', '')).upper()
        is_c2x = ('COMPLEX2X' in fmt_str)
        is_pseudo = ('PSEUDOREAL' in fmt_str)
    except Exception:
        is_c2x = False
        is_pseudo = False
    if is_complex and (is_c2x or is_pseudo):
        # Complex2X is analytic, and PseudoReal is real data in complex container.
        # Often we only care about positive half for these.
        # We can force use_full_spectrum=False if not explicitly requested otherwise,
        # but here we'll follow radar_processing logic to avoid mirroring.
        use_full_spectrum = False
    if not is_complex:
        use_full_spectrum = False

    sig = cube.astype(np.complex64 if is_complex else np.float32, copy=False)

    if remove_dc:
        sig = sig - sig.mean(axis=-1, keepdims=True)

    n_samples = sig.shape[-1]
    if window == 'hann':
        w = np.hanning(n_samples)
    elif window == 'hamming':
        w = np.hamming(n_samples)
    else:
        w = np.ones(n_samples)
    sig = sig * w.reshape([1] * (sig.ndim - 1) + [n_samples])

    if n_fft_range is not None and n_fft_range > 0:
        n_fft = int(n_fft_range)
    else:
        n_fft = int(1 << (n_samples - 1).bit_length()) if pad_pow2 else n_samples

    if is_complex:
        spec = np.fft.fft(sig, n=n_fft, axis=-1)
        if use_full_spectrum:
            n_bins = n_fft
            transform = 'fft_full'
        else:
            # Take positive half including Nyquist bin to match rfft behavior
            n_bins = n_fft // 2 + 1
            spec = spec[..., :n_bins]
            transform = 'fft_half'
        is_real_input = False
    else:
        # Use rFFT for real data to avoid mirrored spectra
        spec = np.fft.rfft(sig, n=n_fft, axis=-1)
        n_bins = spec.shape[-1]
        transform = 'rfft'
        is_real_input = True
        use_full_spectrum = False

    if return_complex:
        out = spec.astype(np.complex64, copy=False)
    else:
        out = np.abs(spec) + 1e-9
        if not use_full_spectrum and is_real_input:
            # Compensate for one-sided spectrum magnitude for REAL input
            # This makes peaks comparable to analytic Complex1X/2X signals
            out = out * 2.0

    # Range axis calculation (simplified, for GUI use)
    fs = float(params.get('digOutSampleRate_MHz', 0)) * 1e6
    slope = float(params.get('freqSlopeMHz_us', 0)) * 1e12
    if fs > 0 and slope > 0 and n_fft > 0:
        df = fs / float(n_fft)
        dR = 299792458.0 * df / (2.0 * slope)
        range_m = np.arange(n_bins, dtype=float) * dR
    else:
        range_m = np.arange(n_bins, dtype=float)

    meta = {
        'range_bins': n_bins,
        'range_m': range_m,
        'window': window,
        'n_input_samples': n_samples,
        'n_fft_range': n_fft,
        'transform': transform,
        'is_complex_input': is_complex,
        'use_full_spectrum': use_full_spectrum,
        'return_complex': return_complex
    }
    return out, meta
