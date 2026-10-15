"""
radar_processing.py

Core FMCW radar signal processing for TI mmWave + DCA1000 captures.
Implements robust LVDS demux with auto selection (IQ/QI, interleave), cube loading and normalization, and
range/RTI/range‑Doppler computations with consistent full‑spectrum handling and notch/cleanup steps.

Includes:
- derive_capture_dimensions: dimensions from parsed params (unique chirps × loops).
- load_raw_cube: LVDS decode (2/4 lanes) → complex cube (F, C, RX, S).
- compute_range_fft_cube: windowing, Range FFT, physical range axis (full spectrum).
- compute_rti: time unfolding, RX averaging, DC/edge suppression.
- compute_range_doppler_map: TX separation (TDM), Doppler FFT (two‑sided), velocity axis.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
import os
import re
import numpy as np
import scipy.fft as _sfft
from typing import Dict, Tuple
from functools import lru_cache
import json

import capture_params
import clutter

C0 = 299_792_458.0  # m/s

def _pretty_print_dict(d: dict, title: str):
    """Helper to print dictionaries for debugging."""
    print(f"\n--- {title} ---")
    # A simple json-based pretty printer that handles non-serializable items
    try:
        # Create a shallow copy to avoid modifying the original dict
        d_copy = d.copy()
        # Remove non-serializable items before printing
        for key, value in d.items():
            if isinstance(value, (np.ndarray, np.generic)):
                d_copy[key] = f"<np.ndarray shape={value.shape}>"
            elif not isinstance(value, (dict, list, tuple, str, int, float, bool, type(None))):
                 d_copy[key] = str(value)
        print(json.dumps(d_copy, indent=2))
    except Exception as e:
        # Fallback for any other serialization errors
        for k, v in d.items():
            if isinstance(v, (np.ndarray, np.generic)):
                print(f"  {k}: <np.ndarray shape={v.shape}>")
            else:
                print(f"  {k}: {v}")
    print("-" * (len(title) + 8) + "\n")


def _effective_unique_chirps(params: Dict, frame_cfg: Dict) -> int:
    """
    Chirps the frame actually transmits: indices inside FrameConfig's chirp range that
    a ChirpConfig defines, each counted once.

    Summing the length of every ChirpConfig in the log over-counted in two ways. A
    session log keeps every definition ever sent, including chirps the frame never
    uses, and a later definition overwrites an earlier one on the same index.
    vitals/data_test1 defines chirps 0 and 1 but its frame sends only 0, and came out at
    400 chirps per frame instead of 200; kleky_dataset1 defines 0..2 as (0,0), (0,2),
    (1,1), (2,2) and came out at 6 instead of 3. Each decoded frame then held two real
    ones, so the file seemed to cover half the time the recorder ran - 93 of the 311
    captures with a recorder log looked like that.

    The intersection keeps the case this was written for: a FrameConfig range wider
    than the chirps actually defined counts only the defined ones.
    """
    defined = set()
    for ch in params.get('chirps') or []:
        cs, ce = ch.get('chirpStartIdx'), ch.get('chirpEndIdx')
        if cs is None or ce is None or ce < cs:
            continue
        defined.update(range(int(cs), int(ce) + 1))
    fs, fe = frame_cfg.get('chirpStartIdx'), frame_cfg.get('chirpEndIdx')
    has_range = fs is not None and fe is not None and fe >= fs
    if defined and has_range:
        used = defined & set(range(int(fs), int(fe) + 1))
        if used:
            return len(used)
    if defined and not has_range:
        return len(defined)
    # Fallback
    return frame_cfg.get('uniqueChirps', 1) or 1

def derive_capture_dimensions(params: Dict) -> Dict[str, int]:
    """
    Derive capture dimensions from decoded params.
    Now uses effective chirp count from actual ChirpConfig definitions to
    avoid overestimation when FrameConfig covers a large index range.
    """
    num_samples = int(params.get('numAdcSamples') or 0)
    rx_mask = params.get('dataFmt_rxChanEnMask')
    if rx_mask is not None:
        # 4 bits on a single chip, 8 on the AWR2243 cascade (cascade/decode.py)
        num_rx = sum((rx_mask >> i) & 1 for i in range(16))
    else:
        # fallback from string like "1 1 1 1"
        rx_str = params.get('dataFmt_rxChannels', "")
        num_rx = sum(int(x) for x in rx_str.split() if x.isdigit()) or 1

    frame_cfg = params.get('frameConfig', {}) or {}
    loops = frame_cfg.get('numLoops') or 1
    # Effective unique chirps from actual definitions (what DCA1000 captures)
    eff_unique_chirps = _effective_unique_chirps(params, frame_cfg)
    
    # Dimensioning must follow captured data (eff_unique * loops)
    chirps_per_frame = eff_unique_chirps * loops
    num_frames_declared = frame_cfg.get('numFrames') or 1
    adc_fmt = params.get('dataFmt_adcFmt', 'Complex1X')
    is_complex = 'Complex' in str(adc_fmt)

    return {
        'num_rx': num_rx,
        'num_samples': num_samples,
        'chirps_per_frame': chirps_per_frame,
        'num_frames': num_frames_declared,
        'is_complex': is_complex,
        'adc_fmt': adc_fmt,
        'unique_chirps_effective': eff_unique_chirps,
        'unique_chirps_frame': eff_unique_chirps,  # For backward compatibility / stride
        'loops': loops
    }

def _expected_int16_per_frame(dims: Dict) -> int:
    per_chirp = dims['num_rx'] * dims['num_samples'] * (2 if dims['is_complex'] else 1)
    return dims['chirps_per_frame'] * per_chirp

def _expected_int16_total(dims: Dict) -> int:
    return dims['num_frames'] * _expected_int16_per_frame(dims)

def load_raw_cube(bin_path: str, params: Dict, dims: Dict, chirp_selection: Dict | None = None, lane_order: list[int] | None = None) -> np.ndarray:
    """
    Load raw data and reshape to (frames, chirps, rx, samples).
    Deterministic based on log (interleave + IQ), with optional chirp selection within unique set.
    chirp_selection: None or {'positions': [int...], 'pattern_len': eff_unique}
    """
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(bin_path)

    if params.get('cascade'):
        # an AWR2243 cascade take is written in chirp order already - no LVDS demux
        from cascade.decode import load_cascade_cube
        if lane_order:
            print("[load_raw_cube] Lane swap does not apply to a cascade take; ignored.")
        return _apply_chirp_selection(load_cascade_cube(bin_path, params, dims), params,
                                      chirp_selection)

    print(f"[load_raw_cube] Using LVDS decode processing. lane_order={lane_order}")
    _pretty_print_dict(params, "Full Parameters for Processing")
    _pretty_print_dict(dims, "Full Dimensions for Processing")

    Ns = dims['num_samples']
    RX = dims['num_rx']
    lanes = params.get('numLanes')
    if lanes not in (1, 2, 4):
        raise ValueError(
            f"numLanes missing/invalid in log: {lanes}. An LVDS container is 1, 2 or 4 "
            f"lanes wide. Verify the recorder log's lane mode (_Raw_LogFile.csv), then "
            f"LaneConfig/LvdsLaneConfig."
        )
    print(f"[load_raw_cube] Using numLanes={lanes} (source: {params.get('numLanes_source','unknown')} )")

    C = dims['chirps_per_frame']
    adc_fmt_str = str(params.get('dataFmt_adcFmt', '')).upper()
    if adc_fmt_str == 'REAL':
        datafmt = 'real'
        iq_order_lvds = "imag_real"
    elif 'COMPLEX2X' in adc_fmt_str:
        datafmt = 'complex2x'
        iq_order_lvds = "imag_real"  # Complex2X is also typically interleaved (I, then Q across all lanes)
    else:
        datafmt = 'complex'
        iq_order_lvds = "imag_real"

    # Hinty from log
    iq_pref = str(params.get('dataFmt_iqOrder') or 'IQ').upper()  # "IQ"/"QI"
    c2x_hint = str(params.get('dataFmt_iqOrder2x') or '').upper()  # "IQIQ"/"QIQI" for Complex2X
    F_target = dims['num_frames']
    adc_bits = int(params.get('dataFmt_adcBits') or 16)
    norm_factor = (2**(adc_bits - 1)) - 1.0

    print(f"[load_raw_cube] Decode settings: lanes={lanes}, adcFmt={adc_fmt_str}, tag='{datafmt}', iq_pref={iq_pref}")
    if datafmt == 'complex2x' and c2x_hint in ('IQIQ', 'QIQI'):
        print(f"[load_raw_cube] Complex2X order hint from LOG: {c2x_hint}")

    # Interleave hint: True→interleaved(0), False→non-interleaved(1)
    interleaved_hint = params.get('dataFmt_isInterleaved')
    if interleaved_hint is None:
        ch_try_list = [0, 1]  # prefer interleaved, then non-interleaved
        print("[load_raw_cube] No interleave hint in log; trying interleaved then non-interleaved.")
    else:
        desired_ch = 0 if bool(interleaved_hint) else 1
        ch_try_list = [desired_ch]
        print(f"[load_raw_cube] Using interleave from log: interleaved={bool(interleaved_hint)} -> ch_interleave={desired_ch}")

    # IQ/SWAP preference
    def _preferred_iq_swap():
        if datafmt == 'complex2x':
            if c2x_hint == 'IQIQ':
                return 0
            if c2x_hint == 'QIQI':
                return 1
        # fallback to IQ/QI
        return 0 if iq_pref == 'IQ' else 1

    iq_pref_swap = _preferred_iq_swap()
    iq_try_list = [iq_pref_swap, 1 - iq_pref_swap]

    # Sequential attempts (no scoring) – first successful wins
    best_cube = None
    used_ch = None
    used_swap = None

    for ch_interleave_try in ch_try_list:
        for iq_swap_try in iq_try_list:
            try:
                print(f"[decode] Trying: ch_interleave={ch_interleave_try}, iq_swap={iq_swap_try}, datafmt={datafmt}")
                cube_v = lvds_decode_concat(
                    bin_path, Ns=Ns, RX=RX, lanes=lanes, ch_interleave=ch_interleave_try,
                    iq_swap=iq_swap_try, C=C, datafmt=datafmt, iq_order=iq_order_lvds, F_target=F_target,
                    lane_order=lane_order
                )
                # normalizace
                nf = max(1.0, norm_factor)
                # lvds_decode_concat already returns complex64/float32, so astype is a
                # no-op cast here; copy=False + in-place divide avoids two full-size
                # temporaries of the whole raw cube.
                target_dtype = np.complex64 if np.iscomplexobj(cube_v) else np.float32
                cube_v = cube_v.astype(target_dtype, copy=False)
                if nf != 1.0:
                    cube_v /= nf
                best_cube = cube_v
                used_ch = ch_interleave_try
                used_swap = iq_swap_try
                print("[decode] Success with this combination.")
                break
            except Exception as e:
                print(f"[decode] Failed: ch_interleave={ch_interleave_try}, iq_swap={iq_swap_try}. Error: {e}")
        if best_cube is not None:
            break

    if best_cube is None:
        raise ValueError("LVDS decode failed for all tried combinations (log-driven).")

    # Persist chosen mode
    used_interleaved = not bool(used_ch)
    if datafmt == 'complex2x':
        used_iq_order = 'QIQI' if used_swap == 1 else 'IQIQ'
    else:
        used_iq_order = 'QI' if used_swap == 1 else 'IQ'
    dims['used_interleaved'] = used_interleaved
    dims['used_iq_order'] = used_iq_order
    dims['used_lanes'] = int(lanes)
    dims['num_frames_actual'] = best_cube.shape[0]
    dims['requested_interleaved'] = (bool(interleaved_hint) if interleaved_hint is not None else None)
    
    # Propagate physical stride to params for downstream processing
    params['unique_chirps_frame'] = dims.get('unique_chirps_frame')

    print(
        f"\n[load_raw_cube] Selected combination:\n"
        f"  lanes={lanes}  format={adc_fmt_str} (tag='{datafmt}')\n"
        f"  interleaved(used)={used_interleaved}, iq_order(used)={used_iq_order}\n"
        f"  interleaved(requested)={'auto' if interleaved_hint is None else bool(interleaved_hint)}\n"
        f"  frames={dims['num_frames_actual']} (log={dims['num_frames']}), chirps/frame={C}, rx={RX}, samples={Ns}"
    )

    return _apply_chirp_selection(best_cube, params, chirp_selection)


def _apply_chirp_selection(best_cube: np.ndarray, params: Dict, chirp_selection: Dict | None) -> np.ndarray:
    """Keep the chosen positions within each loop of unique chirps (chirp_selection:
    None or {'positions': [int...], 'pattern_len': eff_unique})."""
    try:
        if chirp_selection and isinstance(chirp_selection.get('positions'), list):
            sel_pos = list(chirp_selection['positions'])
            pattern_len = int(chirp_selection.get('pattern_len') or 0)
            eff_unique = int(params.get('uniqueChirps_effective') or pattern_len or 0)
            
            # Use physical stride for loops division
            unique_stride = params.get('unique_chirps_frame') or eff_unique
            unique_stride = max(1, int(unique_stride))

            if unique_stride > 0 and all(0 <= p < unique_stride for p in sel_pos):
                F, Ctot, RXn, NS = best_cube.shape
                loops = Ctot // unique_stride
                if loops * unique_stride != Ctot:
                    print(f"[load_raw_cube] WARNING: chirps/frame={Ctot} not divisible by stride={unique_stride}; skipping chirp selection.")
                    return best_cube
                resh = best_cube.reshape(F, loops, unique_stride, RXn, NS)
                sel = resh[:, :, sel_pos, :, :]  # (F, loops, Ksel, RX, Ns)
                Ksel = sel.shape[2]
                filtered = sel.reshape(F, loops * Ksel, RXn, NS)
                best_cube = filtered.astype(best_cube.dtype, copy=False)
                # Mutate params for downstream RD
                try:
                    params['cube_chirp_filtered'] = True
                    params['rd_selected_positions'] = sel_pos
                    params['rd_pattern_len_original'] = unique_stride  # IMPORTANT: distance mezi stejnými chirpy
                    print(f"[load_raw_cube] Chirp selection applied: positions={sel_pos} of stride={unique_stride} -> chirps/frame={loops*Ksel}")
                except Exception:
                    pass
            else:
                print("[load_raw_cube] Skipping chirp selection (invalid positions or pattern_len).")
    except Exception as e:
        print(f"[load_raw_cube] Chirp selection error: {e}")

    return best_cube


def _slope_hz_per_s(params: Dict) -> float:
    s_mhz_us = params.get('freqSlopeMHz_us')
    return float(s_mhz_us) * 1e12 if s_mhz_us is not None else 0.0

def _fs_hz(params: Dict) -> float:
    fs_mhz = params.get('digOutSampleRate_MHz')
    return float(fs_mhz) * 1e6 if fs_mhz is not None else 0.0

def _lambda_m(params: Dict) -> float:
    """
    Wavelength at the frequency the chirp actually has while the ADC is sampling.

    This used startFreqGHz, the frequency at the very start of the ramp, while the
    vital-signs and HR code use effectiveStartFreqGHz. On these captures the two differ
    by 0.15 to 0.9 %, so the same recording was measured on slightly different scales
    depending on which tab you opened.
    """
    f0_ghz = params.get('effectiveStartFreqGHz') or params.get('startFreqGHz')
    f0_hz = float(f0_ghz) * 1e9 if f0_ghz is not None else 0.0
    return C0 / f0_hz if f0_hz > 0 else 0.0

def wavelength_mid_m(params: Dict) -> float:
    """
    Wavelength at the middle of the ADC window (roadmap 1.9): capture_params.wavelength_m,
    the one function every view converts phase with. Falls back to _lambda_m when no
    frequency is known.
    """
    lam = capture_params.wavelength_m(params)
    return lam if lam else _lambda_m(params)


def _chirp_interval_s(params: Dict) -> float:
    idle_us = params.get('idleTime_us') or 0.0
    ramp_us = params.get('rampEndTime_us') or 0.0
    return (idle_us + ramp_us) * 1e-6

def range_bias_m(params: Dict | None) -> float:
    """
    Range offset to take off every range axis (roadmap 9.5): params['range_bias_m'],
    which the GUI sets from an active calibration (calibration.py, measured against a
    corner reflector at a known distance); 0 when there is none.
    """
    try:
        return float((params or {}).get('range_bias_m') or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _range_axis_m(n_fft_range: int, is_real: bool, params: Dict, use_full_spectrum: bool = False) -> np.ndarray:
    """
    Calculate range axis in meters.

    Args:
        n_fft_range: FFT size
        is_real: True for real input (rfft)
        params: Radar parameters
        use_full_spectrum: If True, return full spectrum range (all Nfft bins in natural FFT order)
    """
    fs = _fs_hz(params)
    slope = _slope_hz_per_s(params)
    if fs <= 0 or slope <= 0 or n_fft_range <= 0:
        return np.arange(max(1, n_fft_range), dtype=float)

    df = fs / float(n_fft_range)
    dR = C0 * df / (2.0 * slope)

    if use_full_spectrum:
        # Natural FFT order: [0..N/2-1, N/2..N-1]
        n_bins = n_fft_range
    else:
        # Positive half only (including Nyquist bin to match rfft)
        n_bins = n_fft_range // 2 + 1

    return np.arange(n_bins, dtype=float) * dR - range_bias_m(params)

@lru_cache(maxsize=32)
def _window_vector(n: int, window: str) -> np.ndarray:
    if window == 'hann':
        w = np.hanning(n)
    elif window == 'hamming':
        w = np.hamming(n)
    else:
        w = np.ones(n)
    w.flags.writeable = False
    return w

def apply_window(signal: np.ndarray, axis: int, window: str = 'hann') -> np.ndarray:
    n = signal.shape[axis]
    w = _window_vector(n, window)
    if signal.dtype in (np.float32, np.complex64):
        # a float64 window would turn a single-precision cube into a double one: twice the
        # memory (3 GB for a 3-min take) and a range FFT 2.8x slower
        w = w.astype(np.float32)
    shape = [1]*signal.ndim
    shape[axis] = n
    return signal * w.reshape(shape)


def fft(x: np.ndarray, n: int = None, axis: int = -1) -> np.ndarray:
    """np.fft.fft on every core (scipy.fft workers), in the precision of x."""
    return _sfft.fft(x, n=n, axis=axis, workers=-1)


def rfft(x: np.ndarray, n: int = None, axis: int = -1) -> np.ndarray:
    """np.fft.rfft on every core, in the precision of x."""
    return _sfft.rfft(x, n=n, axis=axis, workers=-1)

_fft_warned = set()


def range_fft_size(n_samples: int, requested: int | None = None, pad_pow2: bool = True) -> int:
    """
    Range FFT length for chirps of n_samples samples - never shorter than the chirp.

    requested=None: the next power of two (or n_samples itself without pad_pow2).
    A request shorter than the chirp used to be honoured, and np.fft.fft(x, n) then
    silently drops the samples past n: a 384- or 512-sample chirp analysed at the
    256 points this application used everywhere lost a third or half of its range
    resolution with no sign of it. Such a request is raised to the next power of two
    and reported once. Longer requests are zero padding and are kept.
    """
    n = max(1, int(n_samples))
    pow2 = int(1 << (n - 1).bit_length())
    if requested is None or int(requested) <= 0:
        return pow2 if pad_pow2 else n
    req = int(requested)
    if req < n:
        if (req, n) not in _fft_warned:
            _fft_warned.add((req, n))
            print(f"[range FFT] {req} points requested for {n} samples per chirp would "
                  f"drop {n - req} samples; using {pow2}.")
        return pow2
    return req


def compute_range_fft_cube(
    cube: np.ndarray,
    params: Dict,
    window: str = 'hann',
    pad_pow2: bool = True,
    remove_dc: bool = True,
    n_fft_range: int = None,
    use_full_spectrum: bool = True,
    return_complex: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    1D Range FFT over samples (last axis).
    - DC removal per (frame, chirp, RX) before windowing.
    - Hann window by default.
    - For complex input: use FFT and optionally show full spectrum (no fftshift).
    - For real input: use rFFT (positive half only).
    Returns magnitude or complex spectrum of the FFT.
    """
    is_complex = np.iscomplexobj(cube)
    # NEW: Complex2X / PseudoReal -> use half-spectrum (like Real), to avoid mirrored image
    try:
        fmt_str = str(params.get('dataFmt_adcFmt', '')).upper()
        is_c2x = ('COMPLEX2X' in fmt_str)
        is_pseudo = ('PSEUDOREAL' in fmt_str)
    except Exception:
        is_c2x = False
        is_pseudo = False

    if not is_complex:
        use_full_spectrum = False
    if is_complex and (is_c2x or is_pseudo):
        use_full_spectrum = False

    sig = cube.astype(np.complex64 if is_complex else np.float32, copy=False)

    if remove_dc:
        sig = sig - sig.mean(axis=-1, keepdims=True)

    sig = apply_window(sig, axis=-1, window=window)

    n_samples = sig.shape[-1]
    n_fft = range_fft_size(n_samples, n_fft_range, pad_pow2)

    if is_complex:
        spec = fft(sig, n=n_fft, axis=-1)
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
        # NEW: Use rfft for real data to get the correct one-sided spectrum
        spec = rfft(sig, n=n_fft, axis=-1)
        n_bins = spec.shape[-1]
        transform = 'rfft'
        is_real_input = True
        use_full_spectrum = False # rfft always returns half spectrum
    
    if return_complex:
        out = spec.astype(np.complex64, copy=False)
    else:
        out = np.abs(spec) + 1e-9
        if not use_full_spectrum and is_real_input:
            # Compensate for one-sided spectrum magnitude for REAL input
            # This makes peaks comparable to analytic Complex1X/2X signals
            out = out * 2.0
    
    range_m = _range_axis_m(n_fft, is_real_input, params, use_full_spectrum)
    if range_m.shape[-1] != n_bins:
        range_m = range_m[:n_bins]
    
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

def compute_rti(cube: np.ndarray, params: Dict, n_fft_range: int = 256) -> Tuple[np.ndarray, Dict]:
    """
    Computes Range-Time Intensity (RTI) map.
    Performs Range FFT and averages magnitude over RX channels.
    Uses full spectrum for complex input (I/Q), half for real input.
    Applies a notch filter to DC and edge bins.
    """
    # Detect input type
    is_complex_input = np.iscomplexobj(cube)
    # NEW: Complex2X / PseudoReal -> half-spectrum also for complex input
    try:
        fmt_str = str(params.get('dataFmt_adcFmt', '')).upper()
        is_c2x = ('COMPLEX2X' in fmt_str)
        is_pseudo = ('PSEUDOREAL' in fmt_str)
    except Exception:
        is_c2x = False
        is_pseudo = False

    range_fft_mag, meta_fft = compute_range_fft_cube(
        cube,
        params,
        n_fft_range=n_fft_range,
        remove_dc=True,
        window='hann',
        use_full_spectrum=(is_complex_input and not (is_c2x or is_pseudo))  # NEW
    ) # Shape (F, C, RX, R_bins)

    # 2. Reshape frames and chirps into a single time axis.
    num_frames, num_chirps_per_frame, num_rx, num_range_bins = range_fft_mag.shape
    total_chirps = num_frames * num_chirps_per_frame

    # Reshape to (total_chirps, num_rx, num_range_bins)
    rti_cube = range_fft_mag.reshape(total_chirps, num_rx, num_range_bins)

    # 3. Non-coherently average across RX channels.
    rti_map = rti_cube.mean(axis=1) # Shape (total_chirps, num_range_bins)

    # 4. Notch DC and edge bins to suppress artifacts
    if num_range_bins > 4:
        # Get a baseline noise/signal level from a clean part of the spectrum
        clean_region = rti_map[:, 10:num_range_bins//2]
        if clean_region.size > 0:
            min_val = np.percentile(clean_region, 1)
        else:
            min_val = np.percentile(rti_map, 1)

        # Suppress DC component (first 2 bins)
        rti_map[:, :2] = min_val
        # Suppress edge component (last 2 bins)
        rti_map[:, -2:] = min_val

    meta = {
        'range_m': meta_fft['range_m'],
        'chirp_index': np.arange(total_chirps),
        'total_chirps': total_chirps,
        'range_bins': num_range_bins,
        # NEW: expose FFT/meta flags so GUI can avoid “seam” at mid
        'use_full_spectrum': bool(meta_fft.get('use_full_spectrum')),
        'n_fft_range': int(meta_fft.get('n_fft_range') or n_fft_range),
        'is_complex_input': bool(is_complex_input),
    }
    return rti_map, meta

def doppler_slow_time(range_spec: np.ndarray, params: Dict) -> Tuple[np.ndarray, float, Dict]:
    """
    The chirps of one transmitter, as the slow-time signal a Doppler FFT runs along.

    range_spec: (frames, chirps, RX, range) after the range FFT.
    Returns (x, interval_s, info):
      x          (frames, RX, range, loops) - one sample per chirp pattern;
      interval_s time between two samples of x: the same TX transmits once per pattern;
      info       {'positions': chirp positions of the pattern that were taken (their
                 mean is where x sits in time within a loop), 'pattern_len': chirps per
                 pattern, 'loops': samples per frame}.

    The Range-Doppler map and the micro-Doppler spectrogram (limb_motion.py) both take
    their slow time from here, so their velocity axes cannot drift apart.
    """
    frame_cfg = params.get('frameConfig', {}) or {}
    num_frames, total_chirps, num_rx, num_range_bins = range_spec.shape

    eff_unique = _effective_unique_chirps(params, frame_cfg)
    eff_unique = max(1, int(eff_unique))
    
    # Use effective unique chirps for loop division (what is actually in ADC data)
    unique_stride = params.get('unique_chirps_frame') or eff_unique
    unique_stride = max(1, int(unique_stride))
    num_loops_total = total_chirps // unique_stride if unique_stride > 0 else 0

    # NEW: handle pre-filtered cube (chirp subset already selected)
    pre_filtered = bool(params.get('cube_chirp_filtered'))
    sel_positions_pf = params.get('rd_selected_positions') if pre_filtered else None
    pattern_len_orig = int(params.get('rd_pattern_len_original') or eff_unique)

    tx_order_masks = params.get('tx_order_masks')

    if pre_filtered and isinstance(sel_positions_pf, list) and pattern_len_orig >= 1:
        # Cube has Ksel chirps per loop; reshape by Ksel directly
        Ksel = len(sel_positions_pf)
        try:
            loops = total_chirps // Ksel
            reshaped = range_spec.reshape(num_frames, loops, Ksel, num_rx, num_range_bins)
        except ValueError:
            reshaped = None
        if reshaped is None:
            # fallback to legacy half-selection
            range_spec_tx1 = range_spec[:, 0::2, :, :]
            doppler_input = np.transpose(range_spec_tx1, (0, 2, 3, 1))
            effective_loops = doppler_input.shape[-1]
            chirp_interval_base = _chirp_interval_s(params)
            chirp_interval_tdm = chirp_interval_base * 2.0
            positions, pattern = [0], 2
        else:
            sel = reshaped  # already selected
            sel = sel.mean(axis=2) if Ksel > 1 else sel.squeeze(axis=2)
            doppler_input = np.transpose(sel, (0, 2, 3, 1))  # (F, RX, R, loops)
            effective_loops = doppler_input.shape[-1]
            # Doppler step uses ORIGINAL pattern length (distance between same TX in time)
            chirp_interval_base = _chirp_interval_s(params)
            chirp_interval_tdm = chirp_interval_base * max(1, pattern_len_orig)
            positions, pattern = list(sel_positions_pf), max(1, pattern_len_orig)
    else:
        # Original path: select TX positions from pattern
        if isinstance(tx_order_masks, list) and len(tx_order_masks) == eff_unique:
            pattern_len = len(tx_order_masks)
            active_masks = [m for m in tx_order_masks if int(m) != 0]
            if not active_masks:
                selected_positions = list(range(pattern_len))
            else:
                single_bit = [m for m in sorted(set(active_masks)) if m in (1, 2, 4)]
                selected_mask = single_bit[0] if single_bit else active_masks[0]
                selected_positions = [i for i, m in enumerate(tx_order_masks) if m == selected_mask]
        else:
            # No usable tx_order_masks: pick ONE TX (the first chirp of the pattern)
            # instead of assuming a 2-TX layout. The old `range(0, pattern_len, 2)`
            # took every second chirp, so a 3-TX config averaged TX0 and TX2 - two
            # different phase centres - into a single Doppler bin.
            pattern_len = eff_unique if eff_unique > 0 else 1
            selected_positions = [0]
            print(f"[compute_range_doppler_map] WARNING: tx_order_masks missing or "
                  f"inconsistent (eff_unique={eff_unique}); using only the first chirp "
                  f"of each pattern for Doppler.")


        try:
            reshaped = range_spec.reshape(num_frames, num_loops_total, unique_stride, num_rx, num_range_bins)
        except ValueError:
            reshaped = None

        if reshaped is not None and selected_positions:
            sel = reshaped[:, :, selected_positions, :, :]
            if sel.ndim == 5 and sel.shape[2] > 1:
                sel = sel.mean(axis=2)
            else:
                sel = sel.squeeze(axis=2)
            doppler_input = np.transpose(sel, (0, 2, 3, 1))
            effective_loops = doppler_input.shape[-1]
            chirp_interval_base = _chirp_interval_s(params)
            pattern = max(1, len(tx_order_masks) if isinstance(tx_order_masks, list) else pattern_len)
            chirp_interval_tdm = chirp_interval_base * pattern
            positions = list(selected_positions)
        else:
            # Reshape failed: fall back to one chirp per pattern, using the real
            # pattern length for the Doppler time step rather than assuming 2 TX.
            stride = max(1, int(unique_stride or pattern_len or 1))
            range_spec_tx1 = range_spec[:, 0::stride, :, :]
            doppler_input = np.transpose(range_spec_tx1, (0, 2, 3, 1))
            effective_loops = doppler_input.shape[-1]
            chirp_interval_base = _chirp_interval_s(params)
            chirp_interval_tdm = chirp_interval_base * float(stride)
            positions, pattern = [0], stride
            print(f"[compute_range_doppler_map] WARNING: could not reshape by chirp "
                  f"pattern; taking every {stride}. chirp for Doppler.")


    info = {'positions': [int(p) for p in positions], 'pattern_len': int(pattern),
            'loops': int(effective_loops)}
    return doppler_input, float(chirp_interval_tdm), info


FEW_DOPPLER_LOOPS = clutter.FEW_LOOPS   # fewer chirps per TX per frame: background over frames, no window
MIN_TARGET_RANGE_M = 0.3                # nearer than this is the TX-RX coupling, not a target


def motion_energy(range_cube: np.ndarray, max_frames: int = 400) -> np.ndarray:
    """
    Per range bin: the power of the chirp-averaged signal around its mean over frames -
    what changes from frame to frame (a person, breathing included), not the wall.
    range_cube: complex (frames, chirps, rx, range); up to max_frames evenly spread.
    """
    F = range_cube.shape[0]
    idx = np.unique(np.linspace(0, F - 1, min(F, max_frames)).round().astype(int))
    s = np.asarray(range_cube[idx]).mean(axis=1)
    return (np.abs(s - s.mean(axis=0, keepdims=True)) ** 2).mean(axis=(0, 1))


def strongest_range_bin(energy: np.ndarray, range_m, min_range_m: float = MIN_TARGET_RANGE_M) -> int:
    """
    The bin of the largest `energy` at min_range_m or beyond. Views that analyse one range
    started at bin 0 - the antenna coupling at 0 m, never a person.
    """
    energy = np.asarray(energy, dtype=float)
    r = np.asarray(range_m, dtype=float) if range_m is not None else None
    if r is None or r.shape != energy.shape or not np.any(r >= min_range_m):
        return int(np.argmax(energy))
    ok = np.flatnonzero(r >= min_range_m)
    return int(ok[np.argmax(energy[ok])])


def compute_range_doppler_map(
    cube: np.ndarray,
    params: Dict,
    n_fft_range: int = 256,
    n_fft_doppler: int = None,
    rx_mode: str = "MRC",
    doppler_window: str | None = "hann",  # NEW: allow selecting slow-time window
) -> Tuple[np.ndarray, Dict]:
    """
    Computes a 2D Range-Doppler map for each frame.
    Separates chirps by TX mask (TDM-MIMO); Doppler FFT is two-sided (fftshift).
    Returns (frames, doppler, range) with RX channels combined per rx_mode:
      - MRC: sqrt(sum(|X|^2) over RX)
      - SUM: |sum(X over RX)|
      - RXk: |X_k| for a single RX index (RX0, RX1)
    """
    is_complex_input = np.iscomplexobj(cube)
    try:
        fmt_str = str(params.get('dataFmt_adcFmt', '')).upper()
        is_c2x = ('COMPLEX2X' in fmt_str)
        is_pseudo = ('PSEUDOREAL' in fmt_str)
    except Exception:
        is_c2x = False
        is_pseudo = False

    # 1. Range FFT (fast-time)
    # No copy needed: the DC-removal below already allocates a new array rather than
    # modifying the input cube in place.
    sig = cube.astype(np.float32 if not is_complex_input else np.complex64, copy=False)
    sig = sig - sig.mean(axis=-1, keepdims=True)
    sig = apply_window(sig, axis=-1, window='hann')
    n_fft_range = range_fft_size(sig.shape[-1], n_fft_range)

    if is_complex_input:
        # Complex1X (full spectrum), Complex2X or PseudoReal (half spectrum)
        range_spec_full = fft(sig, n=n_fft_range, axis=-1)
        if not (is_c2x or is_pseudo):
            range_spec = range_spec_full
            use_full_range = True
        else:
            range_spec = range_spec_full[..., :n_fft_range // 2 + 1]
            use_full_range = False
    else:
        # Real data -> use rfft for one-sided spectrum
        range_spec = rfft(sig, n=n_fft_range, axis=-1)
        use_full_range = False

    # 2. Doppler FFT (slow-time) - TX separation using real ChirpConfig pattern
    doppler_input, chirp_interval_tdm, _ = doppler_slow_time(range_spec, params)
    effective_loops = doppler_input.shape[-1]

    # Static clutter across slow-time: the mean over the loops of each frame unless
    # params['clutter_method'] asks for another method (clutter.py). doppler_input is
    # (frames, RX, range, loops), so the background for "ema" is tracked over frames.
    #
    # A frame of a handful of chirps (vital-signs settings: 4) changes both: the mean of
    # the frame removes everything there is - a seated person's chest does not move
    # within 4 chirps - and a Hann window of 4 points is [0, .75, .75, 0], half the data.
    # The still scene is then the background over FRAMES (ema, from its first time
    # constant on, the time constant of the Analysis bar), and no window: the map shows
    # what moved in the last seconds at 0 m/s, as the RTI does.
    clutter_params = params
    few_note = None
    span_s = effective_loops * chirp_interval_tdm if chirp_interval_tdm > 0 else None
    if effective_loops >= FEW_DOPPLER_LOOPS and span_s is not None and span_s < clutter.SHORT_FRAME_S:
        # long enough for a window, too short for anything to move within it
        if (params.get("clutter_method") or "mean") == "mean":
            clutter_params = {**params, "clutter_method": "ema", "clutter_ema_warm": True}
            few_note = (f"the chirps of a TX span {1000 * span_s:.1f} ms a frame: background over frames "
                        f"(ema) instead of the frame mean")
    elif effective_loops < FEW_DOPPLER_LOOPS:
        if (params.get("clutter_method") or "mean") == "mean":
            clutter_params = {**params, "clutter_method": "ema", "clutter_ema_warm": True}
            few_note = (f"{effective_loops} chirps per TX per frame: background over frames "
                        f"(ema) instead of the frame mean, no window")
        else:
            few_note = f"{effective_loops} chirps per TX per frame: no window"
        doppler_window = None
    doppler_input = clutter.from_params(doppler_input, clutter_params, slow_axis=-1, frame_axis=0)
    # NEW: optional slow-time window (Hann by default)
    if isinstance(doppler_window, str) and doppler_window.lower() in ("hann", "hamming"):
        doppler_input = apply_window(doppler_input, axis=-1, window=doppler_window.lower())
    else:
        # keep as-is (no window)
        pass

    if n_fft_doppler is None:
        n_fft_doppler = doppler_input.shape[-1]

    # Full two-sided Doppler spectrum (shifted)
    doppler_spec = fft(doppler_input, n=n_fft_doppler, axis=-1)
    doppler_spec_shifted = np.fft.fftshift(doppler_spec, axes=-1)

    # 3. Magnitude and RX Combining → (F, D, R)
    # Convert from (F, RX, R, D) to (F, RX, D, R) for standard combination
    if doppler_spec_shifted.ndim == 4:
        rd_cplx = np.transpose(doppler_spec_shifted, (0, 1, 3, 2))  # (F, RX, D, R)
    else:
        # Fallback if somehow ndim is not 4 (e.g., RX or other dim squeezed)
        # We assume the last two are D, R or something similar
        rd_cplx = doppler_spec_shifted

    mode = (rx_mode or "MRC").upper()
    if mode == "SUM":
        rd_map_frames = np.abs(np.sum(rd_cplx, axis=1))
    elif mode.startswith("RX"):
        try:
            rx_idx = int(mode[2:])
        except Exception:
            rx_idx = 0
        rx_idx = max(0, min(rd_cplx.shape[1] - 1, rx_idx))
        rd_map_frames = np.abs(rd_cplx[:, rx_idx, :, :])
    else:
        rd_map_frames = np.sqrt(np.sum(np.abs(rd_cplx) ** 2, axis=1))

    # 4. Axes
    range_axis = _range_axis_m(
        n_fft_range, is_real=(not is_complex_input), params=params, use_full_spectrum=use_full_range
    )

    doppler_freqs = np.fft.fftshift(np.fft.fftfreq(n_fft_doppler, d=chirp_interval_tdm if chirp_interval_tdm > 0 else 1.0))
    wavelength = wavelength_mid_m(params)
    velocity_axis_full = doppler_freqs * wavelength / 2.0
    velocity_axis = velocity_axis_full

    meta = {
        'range_axis': range_axis,
        'velocity_axis': velocity_axis,
        'n_fft_range': n_fft_range,
        'n_fft_doppler': n_fft_doppler,
        'tx_mask_used': (params.get('tdm_active_masks', ['unknown'])[0] if params.get('tdm_active_masks') else 'unknown'),
        'effective_loops': effective_loops,
        'chirp_interval_tdm_s': chirp_interval_tdm,
        'doppler_view': 'full',
        'rx_mode': mode,
        # NEW: diagnostics for wrap-around mitigation
        'doppler_window': (doppler_window or 'none'),
        'fftshift_applied': True,
        'few_loops_note': few_note,
    }

    return rd_map_frames, meta


def _pick_map_for_single_lane_tag(datafmt: str, interleaved: bool, iq_order: str) -> str:
    """
    Decide how to map a single-lane LVDS stream to samples:
      - 'real'    : only I samples (Q=0)
      - 'iqiq'    : alternating I,Q,I,Q,...
      - 'i_then_q': first half I, then half Q
    """
    fmt = (datafmt or "").lower()           # 'real', 'complex', 'complex2x'
    iq = (iq_order or "").lower()
    if "real" in fmt:
        return "real"
    if "complex2x" in fmt or iq == "iqiq":
        return "iqiq"
    if "complex" in fmt:
        return "iqiq" if interleaved else "i_then_q"
    return "iqiq"


def get_all_bin_parts(first_bin_path: str) -> list[str]:
    """
    Checks if the given bin file is part of a split capture (e.g. _0.bin, _1.bin).
    Returns a sorted list of all parts.
    """
    if not first_bin_path or not os.path.isfile(first_bin_path):
        return [first_bin_path] if first_bin_path else []

    folder, name = os.path.split(first_bin_path)
    # Pattern to match: [prefix]_Raw_[N].bin or [prefix]_[N].bin
    # We want to extract the prefix that is common to all parts.
    match = re.search(r'^(.*?)(?:_Raw)?_(\d+)\.bin$', name)
    if not match:
        # If no index pattern, maybe it's just the only file
        return [first_bin_path]

    prefix_before_idx = match.group(1)
    # Determine if "_Raw" was part of the prefix before the index
    has_raw = "_Raw" in name[:match.start(2)]
    
    parts = []
    i = 0
    while True:
        if has_raw:
            p = os.path.join(folder, f"{prefix_before_idx}_Raw_{i}.bin")
            if not os.path.isfile(p):
                # The parts of one capture share one naming scheme. Falling back to
                # "<prefix>_<i>.bin" here used to glue on files from other recordings:
                # feb26/data_test2_Raw_0.bin (30 Jun, exactly its 1200 frames) picked up
                # data_test2_1.bin, a 150 MB capture from the day before, and decoded
                # as 10 371 frames of a 61 s recording.
                break
        else:
            p = os.path.join(folder, f"{prefix_before_idx}_{i}.bin")
            if not os.path.isfile(p):
                break
        
        parts.append(p)
        i += 1

    if not parts or first_bin_path not in parts:
        # Fallback if discovery failed to include the original file
        return [first_bin_path]

    return parts

def _drop_zero_padded_slots(raw, is_real_format):
    """
    Drop LVDS slots that carry nothing but zeros.

    The DCA1000 writes whatever its own lane mode reserves, even when the radar drives
    fewer lanes. An AWR1642 capture recorded with LaneConfig=1 (one active lane) into a
    DCA1000 running in 2-lane mode comes out as sample, 0, sample, 0, ... for the whole
    file: half of every group is a slot that was never filled.

    Read as-is, that is a zero-stuffed signal, which mirrors the spectrum and halves the
    range scale - on the capture this was found with, a target at 1.44 m was reported at
    0.72 m with a ghost at 5.28 m.

    Only applied to REAL captures, where a systematically empty slot cannot be anything
    but padding. Returns (raw, stride, kept_positions) with stride 1 when nothing was
    dropped.
    """
    if not is_real_format or raw.size < 4096:
        return raw, 1, None

    probe = raw[:min(raw.size, 1 << 21)]
    for stride in (2, 4, 8):
        if probe.size < stride * 64:
            break
        empty = [k for k in range(stride) if not probe[k::stride].any()]
        if empty and len(empty) < stride:
            keep = [k for k in range(stride) if k not in empty]
            # Confirm on the whole file before touching anything.
            if any(raw[k::stride].any() for k in empty):
                continue
            idx = np.zeros(raw.size, dtype=bool)
            for k in keep:
                idx[k::stride] = True
            print(f"[lvds_decode_concat] {len(empty)} of every {stride} int16 are zero "
                  f"through the whole file - unused LVDS slots. Keeping positions {keep} "
                  f"and dropping the padding; without this the range scale comes out 2x "
                  f"too small with a mirrored ghost.")
            return raw[idx], stride, keep
    return raw, 1, None


def _lag1(x: np.ndarray) -> float:
    """Correlation between a stream and itself shifted by one sample."""
    x = x - x.mean()
    denom = float(np.dot(x[:-1], x[:-1])) * float(np.dot(x[1:], x[1:]))
    return float(np.dot(x[:-1], x[1:]) / np.sqrt(denom)) if denom > 0 else 0.0


def _real_container_is_doubled(raw, lanes, expected_samples=None, probe=1 << 22):
    """
    Decide whether a REAL capture was written into a complex-shaped container.

    In REAL mode the recorder still lays down the width its own lane mode reserves, so a
    2-lane recorder writes groups of four int16 in which only the first two carry ADC
    samples. Where the surplus slots are zeros _drop_zero_padded_slots has already taken
    them out; where they carry anything else they survive, and reading the stream as one
    sample after another mirrors the spectrum about its midpoint and doubles the frame
    count. On the lane sweep this was found with, that is the difference between a range
    profile matching the same scene captured in Complex1X at 0.96 and at 0.08.

    Two independent things have to agree before the stream is re-cut, because a file
    longer than the log declares is otherwise just as likely to be a longer recording:

      - the file holds about twice the int16 the declared geometry needs, and
      - the first half of each group is a visibly smoother waveform than the stream read
        straight through. Consecutive ADC samples of a beat signal are strongly
        correlated; interleaving anything else between them breaks that, and decimating
        a genuinely sequential stream never makes it smoother.
    """
    w = 2 * lanes
    if raw.size < w * 1024:
        return False
    if expected_samples:
        ratio = raw.size / float(expected_samples)
        if not (1.6 < ratio < 2.4):
            return False

    n = min(raw.size, probe)
    n -= n % w
    blk = raw[:n].reshape(w, -1, order='F').astype(np.float64)
    straight = _lag1(blk.reshape(-1, order='F'))
    first_half = _lag1(blk[:lanes].reshape(-1, order='F'))

    doubled = first_half > 0.5 and (first_half - straight) > 0.15
    if doubled:
        print(f"[lvds_decode_concat] REAL capture written into a {w}-slot container: "
              f"reading it straight through gives lag-1 {straight:.3f}, the first "
              f"{lanes} slot(s) of each group give {first_half:.3f}. Keeping those and "
              f"dropping the rest; without this the spectrum mirrors and the frame "
              f"count doubles.")
    return doubled


def lvds_decode_concat(
    bin_path: str,
    Ns: int,
    RX: int,
    lanes: int,
    ch_interleave: int,
    iq_swap: int,
    C: int,
    datafmt: str,
    iq_order: str = "imag_real",
    F_target: int = None,
    lane_order: list[int] | None = None  # NEW: explicit lane mapping, e.g. [0, 2, 1, 3]
) -> np.ndarray:
    """
    Loads .bin (supporting split files), decodes LVDS (1/2/4 lanes) into complex or real data and reshapes to [F,C,RX,Ns].
    For REAL format returns float32 (purely real data).
    For Complex formats returns complex64.
    ch_interleave: 0=interleaved, 1=non-interleaved (opposite of DataFmtConfig)
    """
    bin_parts = get_all_bin_parts(bin_path)
    if len(bin_parts) > 1:
        print(f"[lvds_decode_concat] Loading {len(bin_parts)} parts: {[os.path.basename(p) for p in bin_parts]}")
        raw_list = []
        for p in bin_parts:
            raw_list.append(np.fromfile(p, dtype=np.int16))
        raw = np.concatenate(raw_list)
    else:
        raw = np.fromfile(bin_path, dtype=np.int16)

    if raw.size == 0:
        raise ValueError("BIN is empty.")

    lanes = int(lanes)
    if lanes not in (1, 2, 4):
        raise ValueError(f"Unsupported number of lanes: {lanes}. Must be 1, 2, or 4.")

    datafmt_l = str(datafmt).lower()
    is_real_format = (datafmt_l == "real")

    raw, _pad_stride, _pad_keep = _drop_zero_padded_slots(raw, is_real_format)

    # A REAL capture whose unused slots are not zeros survives the step above, so the
    # container width still has to be settled. Trimming it here rather than widening the
    # group below keeps the rest of the decode reading an ordinary sequential stream.
    if is_real_format and _pad_stride == 1:
        expected = (F_target * C * RX * Ns) if F_target else None
        if _real_container_is_doubled(raw, lanes, expected):
            w = 2 * lanes
            raw = raw[:(raw.size // w) * w].reshape(w, -1, order='F')[:lanes].reshape(-1, order='F')

    # Determine how many int16 values form a single LVDS group across lanes
    grp = lanes if is_real_format else (2 * lanes)

    # Align to whole LVDS groups
    valid_count = (raw.size // grp) * grp
    if valid_count != raw.size:
        print(f"[lvds_decode_concat] WARNING: raw size {raw.size} not aligned to LVDS group {grp}. "
              f"Truncating {raw.size - valid_count} int16.")
    raw = raw[:valid_count]
    blk = raw.reshape(grp, -1, order='F')

    # Apply lane reordering if requested (before demux)
    if lane_order is not None and len(lane_order) == lanes:
        print(f"[lvds_decode_concat] Applying lane reordering: {lane_order}")
        if is_real_format:
            # blk is [lanes, samples]
            blk = blk[lane_order, :]
        else:
            # blk is [2*lanes, samples], layout is typically [I0, I1, I2, I3, Q0, Q1, Q2, Q3]
            # or [I0, Q0, I1, Q1, ...] depending on iq_order
            # If imag_real (lane separation), it's [I0, I1, I2, I3, Q0, Q1, Q2, Q3]
            iq_order_l = str(iq_order).lower()
            if iq_order_l == "imag_real":
                # Reorder I lanes then Q lanes
                new_order = list(lane_order) + [x + lanes for x in lane_order]
                blk = blk[new_order, :]
            else:
                # Reorder pairs [I0, Q0, I1, Q1, ...]
                new_order = []
                for l in lane_order:
                    new_order.extend([2*l, 2*l+1])
                blk = blk[new_order, :]

    # Decode LVDS block into a flat array of samples (real or complex)
    if lanes == 4 and not is_real_format:
        # Complex streaming for 4 lanes (typical IWR6843/1843/AWR2243)
        iq_order_l = str(iq_order).lower()
        if iq_order_l == "imag_real":
            # Format: [I0, I1, I2, I3, Q0, Q1, Q2, Q3] across LVDS rows
            # blk has 8 rows (0..3 are I, 4..7 are Q)
            # IMPORTANT: For DCA1000 and 4 lanes Complex1x, I and Q samples are formed by interleaving bytes across lanes.
            # I sample = [lane0_byte1:lane0_byte0, lane1_byte1:lane1_byte0, lane2... lane3...] -> 4 samples
            # Then Q sample similarly.
            # blk[0:4] are thus I components for RX0..RX3, blk[4:8] are Q components for RX0..RX3.
            I = blk[0:4, :].reshape(-1, order='F')
            Q = blk[4:8, :].reshape(-1, order='F')
        else:
            # Format: [I0, Q0, I1, Q1, I2, Q2, I3, Q3]
            I = blk[0::2, :].reshape(-1, order='F')
            Q = blk[1::2, :].reshape(-1, order='F')
        
        if int(iq_swap) == 1:
            I, Q = Q, I
        flat = I.astype(np.float32) + 1j * Q.astype(np.float32)
    elif lanes == 2 and not is_real_format:
        # Complex streaming for 2 lanes
        iq_order_l = str(iq_order).lower()
        if iq_order_l == "imag_real":
            # [I0, I1, Q0, Q1]
            I = blk[0:2, :].reshape(-1, order='F')
            Q = blk[2:4, :].reshape(-1, order='F')
        else:
            # [I0, Q0, I1, Q1]
            I = blk[0::2, :].reshape(-1, order='F')
            Q = blk[1::2, :].reshape(-1, order='F')
        if int(iq_swap) == 1:
            I, Q = Q, I
        flat = I.astype(np.float32) + 1j * Q.astype(np.float32)
    elif is_real_format:
        # Real streaming
        if lanes == 4:
            # For 4 lanes, default interleaving usually works for most TI configs.
            # (Previously we tried [0, 2, 1, 3] but it caused artifacts in REAL mode).
            flat = blk.reshape(-1, order='F').astype(np.float32)
        else:
            # For 2 lanes or 1 lane, default interleaving usually works
            flat = blk.reshape(-1, order='F').astype(np.float32)
    else:
        # Complex streaming
        iq_order_l = str(iq_order).lower()
        # NEW FIX: For 1-lane complex data, always use iqiq-style demux regardless of iq_order hint.
        # The 'imag_real' style implies lane separation which isn't possible with 1 lane.
        if iq_order_l == "iqiq" or lanes == 1:
            # Complex2x or 1-lane Complex1x: I,Q,I,Q,...
            I = blk[0::2, :].reshape(-1, order='F')
            Q = blk[1::2, :].reshape(-1, order='F')
        else:
            # Complex1x (multi-lane): first half is I, second half is Q (per LVDS row)
            half_lanes = lanes // 2
            I = blk[0:half_lanes*2, :].reshape(-1, order='F')
            Q = blk[half_lanes*2:2*lanes, :].reshape(-1, order='F')
        if int(iq_swap) == 1:
            I, Q = Q, I
        flat = I.astype(np.float32) + 1j * Q.astype(np.float32)

    # Sanity check: how many “chirps” we really get
    base = Ns * RX  # both complex and real use Ns*RX samples per chirp (real without Q)
    if base <= 0:
        raise ValueError("Invalid Ns or RX for reshape.")
    total_chirps = flat.size // base
    leftover = flat.size - total_chirps * base
    if leftover != 0:
        print(f"[lvds_decode_concat] WARNING: samples not filling full chirps: leftover={leftover} scalars "
              f"(will be discarded).")
    if total_chirps % C != 0:
        print(f"[lvds_decode_concat] WARNING: total_chirps={total_chirps} not divisible by chirps_per_frame={C}. "
              f"Last frame will be truncated.")

    if total_chirps == 0:
        raise ValueError("After LVDS demux, not even one full chirp resulted. Check Ns/RX/lanes.")

    F_raw = max(1, total_chirps // C)
    # Prefer all available data in file (F_raw) over declared numFrames in log (F_target),
    # as users often record longer sessions than initially configured.
    F = F_raw
    need = F * C * base
    if flat.size != need:
        print(f"[lvds_decode_concat] INFO: using frames={F} (from raw={F_raw}, log={F_target}). "
              f"Discarding tail={flat.size - need} scalars.")
    flat = flat[:need]

    # Reshape logic (works for both real and complex `flat` array)
    temp = flat.reshape(Ns*RX, F*C, order='F').T  # [F*C, Ns*RX]

    dtype_out = np.float32 if is_real_format else np.complex64

    # Batch reshape over all chirps at once. Per chirp these are algebraically
    # identical to reshape(Ns,RX,'F').T resp. reshape(RX,Ns,'F'), but done in one
    # C-level operation instead of a Python loop over F*C chirps.
    if int(ch_interleave) == 1:
        # non-interleaved: temp contains blocks [Ns, RX]
        out = temp.reshape(F*C, RX, Ns)
    else:
        # interleaved: temp contains blocks [RX, Ns]
        out = temp.reshape(F*C, Ns, RX).transpose(0, 2, 1)

    result = np.ascontiguousarray(out, dtype=dtype_out).reshape(F, C, RX, Ns)
    print(f"[lvds_decode_concat] Output shape: {result.shape}, dtype: {result.dtype}")
    return result
