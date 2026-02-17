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
import numpy as np
from typing import Dict, Tuple
import json

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
    Prefer explicit ChirpConfig definitions (sum of each defined range).
    Fallback to FrameConfig uniqueChirps if no chirp list parsed.
    """
    chirps = params.get('chirps') or []
    if chirps:
        total = 0
        for ch in chirps:
            cs = ch.get('chirpStartIdx')
            ce = ch.get('chirpEndIdx')
            if cs is None or ce is None:
                continue
            if ce >= cs:
                total += (ce - cs + 1)
        if total > 0:
            return total
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
        num_rx = sum((rx_mask >> i) & 1 for i in range(4))
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

    print(f"[load_raw_cube] Using LVDS decode processing. lane_order={lane_order}")
    _pretty_print_dict(params, "Full Parameters for Processing")
    _pretty_print_dict(dims, "Full Dimensions for Processing")

    Ns = dims['num_samples']
    RX = dims['num_rx']
    lanes = params.get('numLanes')
    if lanes not in (1, 2, 4):
        raise ValueError(f"numLanes missing/invalid in log: {lanes}. Verify LaneConfig/LvdsLaneConfig.")
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
                if np.iscomplexobj(cube_v):
                    cube_v = cube_v.astype(np.complex64) / nf
                else:
                    cube_v = cube_v.astype(np.float32) / nf
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

    # --- NEW: apply chirp selection (filter within each loop of unique chirps) ---
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
    f0_ghz = params.get('startFreqGHz')
    f0_hz = float(f0_ghz) * 1e9 if f0_ghz is not None else 0.0
    return C0 / f0_hz if f0_hz > 0 else 0.0

def _chirp_interval_s(params: Dict) -> float:
    idle_us = params.get('idleTime_us') or 0.0
    ramp_us = params.get('rampEndTime_us') or 0.0
    return (idle_us + ramp_us) * 1e-6

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

    return np.arange(n_bins, dtype=float) * dR

def apply_window(signal: np.ndarray, axis: int, window: str = 'hann') -> np.ndarray:
    n = signal.shape[axis]
    if window == 'hann':
        w = np.hanning(n)
    elif window == 'hamming':
        w = np.hamming(n)
    else:
        w = np.ones(n)
    shape = [1]*signal.ndim
    shape[axis] = n
    return signal * w.reshape(shape)

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
        # NEW: Use rfft for real data to get the correct one-sided spectrum
        spec = np.fft.rfft(sig, n=n_fft, axis=-1)
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
    # Use a copy for real data to avoid modifying the input cube
    sig = cube.astype(np.float32, copy=True) if not is_complex_input else cube.astype(np.complex64, copy=False)
    sig = sig - sig.mean(axis=-1, keepdims=True)
    sig = apply_window(sig, axis=-1, window='hann')

    if is_complex_input:
        # Complex1X (full spectrum), Complex2X or PseudoReal (half spectrum)
        range_spec_full = np.fft.fft(sig, n=n_fft_range, axis=-1)
        if not (is_c2x or is_pseudo):
            range_spec = range_spec_full
            use_full_range = True
        else:
            range_spec = range_spec_full[..., :n_fft_range // 2 + 1]
            use_full_range = False
    else:
        # Real data -> use rfft for one-sided spectrum
        range_spec = np.fft.rfft(sig, n=n_fft_range, axis=-1)
        use_full_range = False

    # 2. Doppler FFT (slow-time) - TX separation using real ChirpConfig pattern
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
        else:
            sel = reshaped  # already selected
            sel = sel.mean(axis=2) if Ksel > 1 else sel.squeeze(axis=2)
            doppler_input = np.transpose(sel, (0, 2, 3, 1))  # (F, RX, R, loops)
            effective_loops = doppler_input.shape[-1]
            # Doppler step uses ORIGINAL pattern length (distance between same TX in time)
            chirp_interval_base = _chirp_interval_s(params)
            chirp_interval_tdm = chirp_interval_base * max(1, pattern_len_orig)
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
            pattern_len = eff_unique if eff_unique > 0 else 2
            selected_positions = list(range(0, pattern_len, 2))
        
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
            chirp_interval_tdm = chirp_interval_base * max(1, len(tx_order_masks) if isinstance(tx_order_masks, list) else pattern_len)
        else:
            range_spec_tx1 = range_spec[:, 0::2, :, :]
            doppler_input = np.transpose(range_spec_tx1, (0, 2, 3, 1))
            effective_loops = doppler_input.shape[-1]
            chirp_interval_base = _chirp_interval_s(params)
            chirp_interval_tdm = chirp_interval_base * 2.0

    # DC removal (static clutter) and windowing across slow-time
    doppler_input = doppler_input - doppler_input.mean(axis=-1, keepdims=True)
    # NEW: optional slow-time window (Hann by default)
    if isinstance(doppler_window, str) and doppler_window.lower() in ("hann", "hamming"):
        doppler_input = apply_window(doppler_input, axis=-1, window=doppler_window.lower())
    else:
        # keep as-is (no window)
        pass

    if n_fft_doppler is None:
        n_fft_doppler = doppler_input.shape[-1]

    # Full two-sided Doppler spectrum (shifted)
    doppler_spec = np.fft.fft(doppler_input, n=n_fft_doppler, axis=-1)
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
    wavelength = _lambda_m(params)
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
    Loads .bin, decodes LVDS (1/2/4 lanes) into complex or real data and reshapes to [F,C,RX,Ns].
    For REAL format returns float32 (purely real data).
    For Complex formats returns complex64.
    ch_interleave: 0=interleaved, 1=non-interleaved (opposite of DataFmtConfig)
    """
    raw = np.fromfile(bin_path, dtype=np.int16)
    if raw.size == 0:
        raise ValueError("BIN is empty.")

    lanes = int(lanes)
    if lanes not in (1, 2, 4):
        raise ValueError(f"Unsupported number of lanes: {lanes}. Must be 1, 2, or 4.")

    datafmt_l = str(datafmt).lower()
    is_real_format = (datafmt_l == "real")

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
    F = min(F_raw, int(F_target)) if F_target is not None else F_raw
    need = F * C * base
    if flat.size != need:
        print(f"[lvds_decode_concat] INFO: using frames={F} (from raw={F_raw}, log={F_target}). "
              f"Discarding tail={flat.size - need} scalars.")
    flat = flat[:need]

    # Reshape logic (works for both real and complex `flat` array)
    temp = flat.reshape(Ns*RX, F*C, order='F').T  # [F*C, Ns*RX]

    dtype_out = np.float32 if is_real_format else np.complex64
    out = np.empty((F*C, RX, Ns), dtype=dtype_out)

    if int(ch_interleave) == 1:
        # non-interleaved: temp contains blocks [Ns, RX]
        for i in range(F*C):
            out[i] = temp[i].reshape(Ns, RX, order='F').T
    else:
        # interleaved: temp contains blocks [RX, Ns]
        for i in range(F*C):
            out[i] = temp[i].reshape(RX, Ns, order='F')

    result = out.reshape(F, C, RX, Ns)
    print(f"[lvds_decode_concat] Output shape: {result.shape}, dtype: {result.dtype}")
    return result
