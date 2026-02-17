"""
AOA.py — Angle-of-Arrival (AoA) utilities

Bartlett beamforming on ULA via RX channels.
    Input cube: (Frames, Chirps, RX, RangeBins)
    """
from __future__ import annotations
import numpy as np

# Zkusíme importovat z projektu pro Range FFT a TX separaci
try:
    from radar_processing import compute_range_fft_cube, _chirp_interval_s, _range_axis_m
except ImportError:
    # Fallback pro případ, že by AOA.py byl spuštěn samostatně
    compute_range_fft_cube = None
    _chirp_interval_s = None
    _range_axis_m = None

C0 = 299_792_458.0  # m/s

def _center_frequency_hz(params: dict) -> float:
    """Center frequency from LOG parameters (fallback 77 GHz)."""
    try:
        f_start = params.get('effectiveStartFreqGHz') or params.get('startFreqGHz')
        f_stop = params.get('stopFreqGHz')
        if f_start and f_stop:
            return float((f_start + f_stop) / 2.0) * 1e9
        if f_start:
            return float(f_start) * 1e9
    except Exception:
        pass
    return 77.0e9

def _get_antenna_geometry(params: dict, num_virt_rx: int, num_phys_rx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns x, y, z coordinates of virtual antennas in meters (lambda units).
    
    Detects IWR6843AOP and returns its specific 2D geometry.
    Otherwise uses standard linear (ULA) construction.
    """
    # 1. Try to load explicit geometry from parameters
    try:
        g0 = params.get('antGeometry0') # x positions
        g1 = params.get('antGeometry1') # y positions
        
        if isinstance(g0, (list, np.ndarray)) and len(g0) >= num_virt_rx:
            x = np.array(g0[:num_virt_rx])
            if isinstance(g1, (list, np.ndarray)) and len(g1) >= num_virt_rx:
                y = np.array(g1[:num_virt_rx])
            else:
                y = np.zeros(num_virt_rx)
            z = np.zeros(num_virt_rx)
            return x, y, z
    except Exception:
        pass

    # 2. Detection of TI platforms and their geometries
    is_aop = False
    is_awr1843 = False
    is_awr2544 = False
    
    platform = str(params.get('platform', '')).upper()
    chip_hint = str(params.get('chip_hint', '')).upper()
    
    if "AOP" in platform or "AOP" in chip_hint or (num_phys_rx == 4 and num_virt_rx == 12):
        is_aop = True
    elif "1843" in platform or "1843" in chip_hint or (num_phys_rx == 4 and num_virt_rx == 8):
        is_awr1843 = True
    elif "2544" in platform or "2544" in chip_hint:
        is_awr2544 = True

    if is_aop:
        # IWR6843AOP: 4 RX (in 2x2 grid) and 3 TX
        # Virtual array is formed as superposition of TX positions and RX grid.
        # TX1 (0,0), TX3 (1,0), TX2 (0,1) in lambda units (approximately)
        # This matches TI SDK "Out-of-box" demo 2D AoA.
        print("[AOA] Using IWR6843AOP 2D geometry (4 RX, 3 TX).")
        # Standard RX order: RX1(0,0), RX2(0.5,0), RX3(0,0.5), RX4(0.5,0.5)
        # However, TI often uses different channel ordering.
        # Based on SDK:
        rx_x = np.array([0, 0.5, 0, 0.5])
        rx_y = np.array([0, 0, 0.5, 0.5])
        
        # TX offsets for TDM (TX1, TX2, TX3)
        # For AOP: TX1=(0,0), TX2=(0,1), TX3=(1,0)
        tx_offsets_x = [0.0, 0.0, 1.0] # TX1, TX2, TX3
        tx_offsets_y = [0.0, 1.0, 0.0]
        
        # NOTE: If the LOG contains TX1, TX3, TX2 order (which happens with AOP),
        # it would need to be swapped. We keep the standard order and add diagnostics.
        print(f"[AOA] AOP offsets: TX1={tx_offsets_x[0]},{tx_offsets_y[0]} | TX2={tx_offsets_x[1]},{tx_offsets_y[1]} | TX3={tx_offsets_x[2]},{tx_offsets_y[2]}")
        
        # If we have 2 TXs only (Azimuth extension for non-AOP), it's different.
        # But for AOP with 12 virt, it's 3 TX.
        
        x_list, y_list = [], []
        num_tx = num_virt_rx // num_phys_rx
        for i in range(num_tx):
            off_x = tx_offsets_x[i] if i < len(tx_offsets_x) else (i * 2.0)
            off_y = tx_offsets_y[i] if i < len(tx_offsets_y) else 0.0
            x_list.append(rx_x + off_x)
            y_list.append(rx_y + off_y)
            
        x = np.concatenate(x_list)[:num_virt_rx]
        y = np.concatenate(y_list)[:num_virt_rx]
        z = np.zeros(num_virt_rx)
        
        # Centering for beamforming
        x -= np.mean(x)
        y -= np.mean(y)
        return x, y, z

    if is_awr1843:
        # AWR1843/IWR1843: Often 4 RX and 2 TX (Azimuth extension) or 3 TX (Elevation)
        # For 2 TX: [TX1, TX2] -> 8 virt (ULA)
        # For 3 TX: [TX1, TX2, TX3] -> 12 virt (TX3 is usually elevation offset)
        print(f"[AOA] Using AWR1843 style geometry ({num_phys_rx} RX, {num_virt_rx//num_phys_rx} TX).")
        rx_pos = np.arange(num_phys_rx) * 0.5
        num_tx = num_virt_rx // num_phys_rx
        
        x_list, y_list = [], []
        for i in range(num_tx):
            # i=0 (TX1), i=1 (TX2), i=2 (TX3)
            if i == 2: # TX3 (Elevation) - TI SDK typically gives TX3 vertical offset
                x_list.append(rx_pos + 0.5) # Azimuthal offset of TX3 is usually 0.5 or 1.0 λ
                y_list.append(np.ones(num_phys_rx) * 0.5) # Elevation
            else: # TX1, TX2 (Azimuth extension)
                x_list.append(rx_pos + i * (num_phys_rx * 0.5))
                y_list.append(np.zeros(num_phys_rx))
        
        x = np.concatenate(x_list)[:num_virt_rx]
        y = np.concatenate(y_list)[:num_virt_rx]
        z = np.zeros(num_virt_rx)
        x -= np.mean(x)
        y -= np.mean(y)
        return x, y, z

    # 3. Fallback: Standard linear array (ULA)
    # Suitable for AWR2544LOPEVM (if it is ULA) and most planar EVMs.
    
    if num_phys_rx > 0:
        num_tx = num_virt_rx // num_phys_rx
    else:
        num_tx = 1
        num_phys_rx = num_virt_rx

    # RX: [0, 0.5, 1.0, 1.5]
    rx_pos = np.arange(num_phys_rx) * 0.5
    
    # TX offsets: [0, 2.0, 4.0, ...] (for azimuthal MIMO)
    tx_offsets = np.arange(num_tx) * (num_phys_rx * 0.5)
    
    x_list = []
    for t in range(num_tx):
        x_list.append(rx_pos + tx_offsets[t])
    
    x = np.concatenate(x_list)
    x = x[:num_virt_rx]
    
    # Centering
    x = x - np.mean(x)
    y = np.zeros(num_virt_rx)
    z = np.zeros(num_virt_rx)
        
    return x, y, z

def _apply_calibration(data: np.ndarray, params: dict, num_virt_rx: int) -> np.ndarray:
    """
    Applies phase calibration (antPhaseRot and compRangeBiasAndRxChanPhase).
    data shape: (..., num_virt_rx)
    """
    cal_phases = np.zeros(num_virt_rx, dtype=np.complex64)
    
    # 1. compRangeBiasAndRxChanPhase
    # Format: [range_bias, Q0, I0, Q1, I1, ...]
    try:
        comp = params.get('compRangeBiasAndRxChanPhase')
        if isinstance(comp, (list, np.ndarray)) and len(comp) >= (1 + 2 * num_virt_rx):
            for i in range(num_virt_rx):
                q = comp[1 + 2*i]
                i_val = comp[1 + 2*i + 1]
                # TI calibration is often stored as 16-bit fixed point Q, I
                # We want to multiply data by conj(cal) or similar.
                # Usually: x_cal = x * conj(I + jQ)
                cal_phases[i] = i_val + 1j * q
            
            # Normalization of calibration
            mag = np.abs(cal_phases)
            mag[mag == 0] = 1.0
            cal_phases /= mag
    except Exception as e:
        print(f"[AOA] Calibration error (compRangeBias): {e}")

    # 2. antPhaseRot
    # antPhaseRot is TI parameter that contains phase rotations for each channel (in radians)
    # Format: [rot0, rot1, ...]
    try:
        rot = params.get('antPhaseRot')
        if isinstance(rot, (list, np.ndarray)) and len(rot) >= num_virt_rx:
            # Add rotation to cal_phases
            # cal_phases might already have values from compRangeBias, so multiply them
            for i in range(num_virt_rx):
                phi = float(rot[i])
                rot_cplx = np.exp(1j * phi)
                if cal_phases[i] == 0:
                    cal_phases[i] = rot_cplx
                else:
                    cal_phases[i] *= rot_cplx
    except Exception as e:
        print(f"[AOA] Calibration error (antPhaseRot): {e}")
    
    if np.any(cal_phases != 0):
        # Apply calibration: data * conj(cal_phases)
        return data * np.conj(cal_phases)
    
    return data

def _is_safe_tdm(masks: list[int] | None) -> bool:
    """
    Check if the mask pattern is suitable for MIMO concatenation.
    Must hold:
    1. Masks exist.
    2. Each mask activates EXACTLY ONE TX (power of 2).
       Mask 5 (101) -> False. Mask 2 (010) -> True.
    """
    if not masks or len(masks) < 2:
        return False
    unique_masks = set()
    for m in masks:
        try:
            mv = int(m)
            if mv <= 0:
                return False
            # Check if exactly one bit is set (x & (x-1) == 0)
            if (mv & (mv - 1)) != 0:
                # Multi-TX mask (e.g. 5 = 101) -> not safe TDM for simple stacking
                return False
            unique_masks.add(mv)
        except (ValueError, TypeError):
            return False
    # Must have at least as many unique masks as chirps in pattern
    return len(unique_masks) == len(masks)

def compute_aoa_spectrum(
    cube: np.ndarray,
    params: dict,
    range_m: float | None = None,
    range_bin: int | None = None,
    angles_deg: np.ndarray | None = None,
    method: str = "bartlett",
    remove_static: bool = True,
    mimo: bool = True,
    invert_tx_phase: bool = False,
    mimo_mode: str = "MIMO", # "MIMO", "First TX", "All Chirps"
    rx_order_override: list[int] | str | None = None, # NEW: RX channel order or 'auto'
) -> tuple[np.ndarray, dict]:
    """
    AoA spectrum calculation across frames for the selected range bin.
    mimo_mode: 
        "MIMO": standard TDM-MIMO reconstruction of virtual array.
        "First TX": Uses only the first chirp from each unique set (e.g., only TX1).
        "All Chirps": Uses all chirps without TX distinction (suitable for SIMO).
    """
    if angles_deg is None:
        angles_deg = np.arange(-90.0, 91.0, 1.0)
    angles_deg = np.asarray(angles_deg).astype(float)

    if cube.ndim != 4:
        raise ValueError("cube must have shape (Frames, Chirps, RX, Samples/Bins)")
    
    F, C_total, RX, N_last = cube.shape

    # 1. Detection and optional Range FFT calculation
    is_freq = params.get('is_frequency_domain', False)
    if not is_freq:
        num_adc = params.get('numAdcSamples')
        if np.iscomplexobj(cube) and num_adc is not None and N_last != num_adc:
            is_freq = True
        elif np.iscomplexobj(cube) and N_last == 256 and num_adc is None:
            is_freq = True

    if not is_freq and compute_range_fft_cube is not None:
        n_fft = params.get('n_fft_range', 256)
        range_cube, rf_meta = compute_range_fft_cube(
            cube, params, window='hann', pad_pow2=False, remove_dc=True, n_fft_range=n_fft, return_complex=True
        )
    else:
        range_cube = cube
        range_m_axis = params.get('range_m')
        rf_meta = {'range_m': range_m_axis}

    F, C_total, RX, R = range_cube.shape

    # 2. MIMO / TX Selection - TX separation and virtual array construction
    eff_unique = int(params.get('uniqueChirps_effective') or 1)
    unique_stride = params.get('unique_chirps_frame') or eff_unique
    unique_stride = max(1, int(unique_stride))
    
    masks = params.get('tx_order_masks')
    current_masks = masks[:eff_unique] if masks else None
    
    is_subset = (C_total % unique_stride != 0) or (C_total < unique_stride)
    
    # MIMO decision
    do_mimo_expand = (
        mimo 
        and (mimo_mode == "MIMO")
        and (eff_unique > 1) 
        and not is_subset
        and _is_safe_tdm(current_masks)
    )
    
    if do_mimo_expand:
        num_loops = C_total // unique_stride
        try:
            # reshaped: (F, loops, unique_stride, RX, R)
            reshaped = range_cube.reshape(F, num_loops, unique_stride, RX, R)
            
            # Pick only the interesting (defined) chirps from the stride
            active_chirps = reshaped[:, :, :eff_unique, :, :].copy()
            
            # Invert phase of the second TX if requested
            if invert_tx_phase and eff_unique >= 2:
                active_chirps[:, :, 1::2, :, :] *= -1
                print("[AOA] Inverting phase for odd TX indices")

            # Virtual antenna array (for azimuth)
            virt_cube = active_chirps.reshape(F, num_loops, eff_unique * RX, R)
            num_virt_rx = eff_unique * RX
            if masks:
                print(f"[AOA] MIMO enabled. Virt RX count: {num_virt_rx}. Pattern masks: {current_masks}")
        except Exception as e:
            print(f"[AOA] MIMO reshape failed: {e}")
            virt_cube = range_cube
            num_virt_rx = RX
            num_loops = C_total
    elif mimo_mode == "First TX" and unique_stride > 0:
        num_loops = C_total // unique_stride
        reshaped = range_cube.reshape(F, num_loops, unique_stride, RX, R)
        virt_cube = reshaped[:, :, 0, :, :] # Only first chirp (TX1)
        num_virt_rx = RX
        print(f"[AOA] Using First TX only. RX count: {num_virt_rx}")
    else:
        virt_cube = range_cube
        num_virt_rx = RX
        num_loops = C_total
        if eff_unique > 1 and mimo:
            reason = "unsafe pattern" if not _is_safe_tdm(current_masks) else "subset/filtered data"
            print(f"[AOA] MIMO disabled ({reason}). Using {RX} antennas. Masks: {current_masks}")

    # 3. Range bin determination
    if range_bin is None:
        if range_m is not None:
            r_axis = rf_meta.get('range_m')
            if r_axis is None: r_axis = params.get('range_m')
            if isinstance(r_axis, list): r_axis = np.array(r_axis)
            if isinstance(r_axis, np.ndarray) and r_axis.ndim == 1 and r_axis.size == R:
                range_bin = int(np.argmin(np.abs(r_axis - float(range_m))))
            else:
                range_bin = R // 2
        else:
            range_bin = R // 2
    range_bin = int(max(0, min(R - 1, range_bin)))

    # 4. Data preparation
    x = virt_cube[:, :, :, range_bin]

    # 4a. Optional autodetection/swap of RX channels (safely applicable for num_virt_rx==RX)
    def _apply_perm(arr, perm):
        try:
            return arr[:, :, perm]
        except Exception:
            return arr

    if rx_order_override is not None:
        try:
            if isinstance(rx_order_override, str) and rx_order_override.lower() == 'auto':
                if num_virt_rx == RX and RX <= 6 and F > 0:
                    # Estimate RX order permutation by maximizing sharpness of angular spectrum
                    import itertools
                    test_angles = np.linspace(-90, 90, 181)
                    ang_rad_test = np.deg2rad(test_angles)
                    # Geometry for test (use current x_pos_lam computed below after geometry)
                    pass
                else:
                    print("[AOA] rx_order_override='auto' skipped (num_virt_rx != RX or RX>6)")
            elif isinstance(rx_order_override, (list, tuple)) and len(rx_order_override) == num_virt_rx:
                perm = [int(v) for v in rx_order_override]
                x = _apply_perm(x, perm)
                print(f"[AOA] Applied RX permutation override: {perm}")
        except Exception as e:
            print(f"[AOA] RX override failed: {e}")

    x = _apply_calibration(x, params, num_virt_rx)
    
    if remove_static:
        x = x - x.mean(axis=1, keepdims=True)

    # 5. Steering vector
    fc = _center_frequency_hz(params)
    # lam = C0 / max(fc, 1.0)
    
    # NEW: Pass physical RX count to geometry calculator
    x_pos_lam, y_pos_lam, z_pos_lam = _get_antenna_geometry(params, num_virt_rx, RX)
    if F > 0:
        print(f"[AOA] Platform: {params.get('platform','unknown')}, Chip: {params.get('chip_hint','unknown')}")
        print(f"[AOA] Geometry (lambda units): x={x_pos_lam[:min(8,len(x_pos_lam))]}..., y={y_pos_lam[:min(8,len(y_pos_lam))]}...")

    # If auto RX mapping requested and eligible, decide and apply now (needs geometry)
    if isinstance(rx_order_override, str) and rx_order_override and rx_order_override.lower() == 'auto':
        if num_virt_rx == RX and RX <= 6 and F > 0:
            try:
                import itertools
                test_angles = np.linspace(-90, 90, 181)
                sv_test = np.exp(1j * 2.0 * np.pi * np.outer(np.sin(np.deg2rad(test_angles)), x_pos_lam))
                x_f0 = x[0]  # (loops, RX)
                pwr = np.abs(x_f0)**2
                best_loop = int(np.argmax(pwr.sum(axis=1)))
                best_vec = x_f0[best_loop]
                
                base_metric = -1.0
                best_perm = None
                best_conj = False
                
                for do_conj in [False, True]:
                    vec_cand = np.conj(best_vec) if do_conj else best_vec
                    for perm in itertools.permutations(range(RX)):
                        v = vec_cand[list(perm)]
                        bf = np.abs(v @ sv_test.T.conj())**2
                        m = float(np.max(bf) - np.median(bf))
                        if m > base_metric:
                            base_metric = m
                            best_perm = list(perm)
                            best_conj = do_conj
                
                if best_perm is not None:
                    print(f"[AOA] AUTO RX selected: perm={best_perm}, conj={best_conj} (metric={base_metric:.3f})")
                    if best_conj:
                        x = np.conj(x)
                    if best_perm != list(range(RX)):
                        x = _apply_perm(x, best_perm)
                else:
                    print("[AOA] AUTO RX mapping kept identity.")
            except Exception as e:
                print(f"[AOA] AUTO RX mapping failed: {e}")

    ang_rad = np.deg2rad(angles_deg)
    # Steering vector for azimuth (assuming zero elevation)
    # Phase = 2*pi * (x*sin(theta) + y*cos(theta)*sin(phi)) -> phi=0 -> 2*pi*x*sin(theta)
    sv = np.exp(1j * 2.0 * np.pi * np.outer(np.sin(ang_rad), x_pos_lam))

    # 6. Bartlett Beamforming
    win = np.hamming(num_virt_rx)
    
    # --- DIAGNOSTICS: RX channel phases for the strongest loop ---
    try:
        if F > 0:
            # Find the strongest loop in frame 0 for the selected bin
            x_f0 = x[0] # (num_loops, num_virt_rx)
            pwr = np.abs(x_f0)**2
            best_loop = np.argmax(pwr.sum(axis=1))
            best_data = x_f0[best_loop]
            phases = np.angle(best_data)
            print(f"[AOA-DIAG] Frame 0, Loop {best_loop}, Bin {range_bin}")
            print(f"[AOA-DIAG] Magnitudes: {np.abs(best_data).round(3)}")
            print(f"[AOA-DIAG] Phases (rad): {phases.round(3)}")
            if num_virt_rx > 1:
                diffs = np.diff(phases)
                # Unwrap phase diffs to see trend
                diffs_unwrapped = (diffs + np.pi) % (2 * np.pi) - np.pi
                print(f"[AOA-DIAG] Phase Diffs: {diffs_unwrapped.round(3)}")
    except Exception as e:
        print(f"[AOA-DIAG] Error: {e}")
    # -------------------------------------------------------

    A = angles_deg.size
    spec = np.empty((F, A), dtype=float)
    for f in range(F):
        x_f = x[f] * win[None, :]
        bf_loops = np.abs(x_f @ sv.T.conj()) ** 2
        bf_avg = bf_loops.mean(axis=0)
        bf_db = 10.0 * np.log10(np.maximum(bf_avg, 1e-12))
        bf_db -= np.max(bf_db)
        spec[f, :] = bf_db

    meta = {
        'view_type': '1D Spectrum',
        'angles_deg': angles_deg,
        'range_bin': range_bin,
        'rx_count': RX,
        'virt_rx_count': num_virt_rx,
        'tx_count': eff_unique
    }
    
    r_axis = rf_meta.get('range_m')
    if r_axis is None and _range_axis_m is not None:
        try:
            is_complex_input = np.iscomplexobj(range_cube)
            n_fft_guess = 256
            if R == 129: n_fft_guess = 256
            elif R == 65: n_fft_guess = 128
            use_full = (is_complex_input and R == n_fft_guess)
            r_axis = _range_axis_m(n_fft_guess, not is_complex_input, params, use_full_spectrum=use_full)
            if r_axis.size != R:
                r_axis = np.arange(R) * (r_axis[1] if r_axis.size > 1 else 1.0)
        except Exception:
            r_axis = np.arange(R)

    if r_axis is not None:
        if r_axis.size > R: r_axis = r_axis[:R]
        elif r_axis.size < R: 
            step = r_axis[1]-r_axis[0] if r_axis.size > 1 else 1.0
            r_axis = np.concatenate([r_axis, r_axis[-1] + step * np.arange(1, R - r_axis.size + 1)])
        meta['range_axis'] = r_axis
        if range_bin < r_axis.size:
            meta['range_m'] = float(r_axis[range_bin])
    else:
        meta['range_axis'] = np.arange(R)

    return spec, meta

def compute_range_azimuth_map(
    cube: np.ndarray,
    params: dict,
    angles_deg: np.ndarray | None = None,
    remove_static: bool = True,
    mimo: bool = True,
    invert_tx_phase: bool = False,
    mimo_mode: str = "MIMO",
    rx_order_override: list[int] | str | None = None,
    method: str = "bartlett", # NEW: beamforming method
) -> tuple[np.ndarray, dict]:
    """
    2D Range-Azimuth map calculation (Bartlett beamforming for all range bins).
    """
    if angles_deg is None:
        angles_deg = np.arange(-90.0, 91.0, 1.0)
    angles_deg = np.asarray(angles_deg).astype(float)

    # 1. FFT a MIMO příprava
    F, C_total, RX, N_last = cube.shape
    is_freq = params.get('is_frequency_domain', False)
    if not is_freq:
        num_adc = params.get('numAdcSamples')
        if np.iscomplexobj(cube) and num_adc is not None and N_last != num_adc:
            is_freq = True
        elif np.iscomplexobj(cube) and N_last == 256 and num_adc is None:
            is_freq = True

    if not is_freq and compute_range_fft_cube is not None:
        n_fft = params.get('n_fft_range', 256)
        range_cube, rf_meta = compute_range_fft_cube(
            cube, params, window='hann', pad_pow2=False, remove_dc=True, n_fft_range=n_fft, return_complex=True
        )
    else:
        range_cube = cube
        range_m_axis = params.get('range_m')
        rf_meta = {'range_m': range_m_axis}

    F, C_total, RX, R = range_cube.shape
    
    eff_unique = int(params.get('uniqueChirps_effective') or 1)
    unique_stride = params.get('unique_chirps_frame') or eff_unique
    unique_stride = max(1, int(unique_stride))
    
    masks = params.get('tx_order_masks')
    current_masks = masks[:eff_unique] if masks else None
    
    # Check for subset/filtered data
    is_subset = (C_total % unique_stride != 0) or (C_total < unique_stride)
    
    do_mimo_expand = (
        mimo 
        and (mimo_mode == "MIMO")
        and (eff_unique > 1) 
        and not is_subset
        and _is_safe_tdm(current_masks)
    )

    if do_mimo_expand:
        num_loops = C_total // unique_stride
        try:
            reshaped = range_cube.reshape(F, num_loops, unique_stride, RX, R)
            
            # Pick active chirps
            active_chirps = reshaped[:, :, :eff_unique, :, :].copy()
            
            # Invert phase of the second TX if requested
            if invert_tx_phase and eff_unique >= 2:
                active_chirps[:, :, 1::2, :, :] *= -1
                print("[AOA 2D] Inverting phase for odd TX indices")
            
            virt_cube = active_chirps.reshape(F, num_loops, eff_unique * RX, R)
            num_virt_rx = eff_unique * RX
            if masks:
                print(f"[AOA 2D] MIMO enabled. Virt RX count: {num_virt_rx}. Pattern masks: {current_masks}")
        except Exception as e:
            print(f"[AOA 2D] MIMO reshape failed: {e}")
            virt_cube = range_cube
            num_virt_rx = RX
            num_loops = C_total
    elif mimo_mode == "First TX" and unique_stride > 0:
        num_loops = C_total // unique_stride
        reshaped = range_cube.reshape(F, num_loops, unique_stride, RX, R)
        virt_cube = reshaped[:, :, 0, :, :] # Pouze první chirp (TX1)
        num_virt_rx = RX
        print(f"[AOA 2D] Using First TX only. RX count: {num_virt_rx}")
    else:
        virt_cube = range_cube
        num_virt_rx = RX
        num_loops = C_total
        if eff_unique > 1 and mimo:
            reason = "unsafe pattern" if not _is_safe_tdm(current_masks) else "subset/filtered data"
            print(f"[AOA 2D] MIMO disabled ({reason}). Using {RX} antennas. Masks: {current_masks}")

    # Inform about rx_order_override received
    print(f"[AOA 2D] Received rx_order_override={rx_order_override}, mimo_mode={mimo_mode}")

    # Optional RX permutation/autodetect (safe when num_virt_rx == RX)
    def _apply_perm(arr, perm):
        try:
            return arr[:, :, perm, :]
        except Exception:
            return arr

    if rx_order_override is not None:
        try:
            if isinstance(rx_order_override, str) and rx_order_override.lower() == 'auto':
                if num_virt_rx == RX and RX <= 6 and F > 0:
                    # Determine a representative range bin (max energy in frame 0)
                    R_energy = np.sum(np.abs(virt_cube[0])**2, axis=(0,1))  # [R]
                    best_bin = int(np.argmax(R_energy))
                    print(f"[AOA 2D] AUTO RX: selected range_bin={best_bin} for permutation search")
                else:
                    print("[AOA 2D] rx_order_override='auto' skipped (num_virt_rx != RX or RX>6)")
                    best_bin = None
            else:
                best_bin = None
        except Exception as e:
            print(f"[AOA 2D] RX override pre-check failed: {e}")
            best_bin = None

    # 2. Clutter removal
    if remove_static:
        virt_cube = virt_cube - virt_cube.mean(axis=1, keepdims=True)

    # 3. Steering matrix
    fc = _center_frequency_hz(params)
    # lam = C0 / max(fc, 1.0)
    
    # NEW: Pass physical RX count to geometry calculator
    x_pos_lam, y_pos_lam, z_pos_lam = _get_antenna_geometry(params, num_virt_rx, RX)
    if F > 0:
        print(f"[AOA 2D] Platform: {params.get('platform','unknown')}, Chip: {params.get('chip_hint','unknown')}")
        print(f"[AOA 2D] Geometry (lambda units): x={x_pos_lam[:4]}..., y={y_pos_lam[:4]}...")

    # If auto RX mapping requested and eligible, decide and apply now (needs geometry)
    if isinstance(rx_order_override, str) and rx_order_override and rx_order_override.lower() == 'auto':
        if num_virt_rx == RX and RX <= 6 and F > 0 and best_bin is not None:
            try:
                import itertools
                test_angles = np.linspace(-90, 90, 181)
                ang_rad_test = np.deg2rad(test_angles)
                sv_test = np.exp(1j * 2.0 * np.pi * np.outer(np.sin(ang_rad_test), x_pos_lam))
                
                x_f0 = virt_cube[0, :, :, best_bin]  # (loops, RX)
                pwr = np.abs(x_f0)**2
                best_loop = int(np.argmax(pwr.sum(axis=1)))
                best_vec = x_f0[best_loop]
                
                base_metric = -1.0
                best_perm = None
                best_conj = False
                
                # Try permutations and complex conjugation (solves I/Q swap or opposite trend)
                for do_conj in [False, True]:
                    vec_cand = np.conj(best_vec) if do_conj else best_vec
                    for perm in itertools.permutations(range(RX)):
                        v = vec_cand[list(perm)]
                        bf = np.abs(v @ sv_test.T.conj())**2
                        # Change metric to SNR (Peak-to-Median) for 2D maps
                        m = float(np.max(bf) / (np.median(bf) + 1e-9))
                        if m > base_metric:
                            base_metric = m
                            best_perm = list(perm)
                            best_conj = do_conj
                
                if best_perm is not None:
                    print(f"[AOA 2D] AUTO RX selected: perm={best_perm}, conj={best_conj} (metric_snr={base_metric:.3f})")
                    if best_conj:
                        virt_cube = np.conj(virt_cube)
                    if best_perm != list(range(RX)):
                        virt_cube = _apply_perm(virt_cube, best_perm)
                    
                    # DIAG phases after fix
                    best_vec_after = virt_cube[0, best_loop, :, best_bin]
                    phases_after = np.angle(best_vec_after)
                    diffs_after = np.diff(phases_after)
                    diffs_after = (diffs_after + np.pi) % (2 * np.pi) - np.pi
                    print(f"[AOA 2D-DIAG] Phases after: {phases_after.round(3)}")
                    print(f"[AOA 2D-DIAG] Diffs after: {diffs_after.round(3)}")
                else:
                    print("[AOA 2D] AUTO RX mapping kept identity.")
            except Exception as e:
                print(f"[AOA 2D] AUTO RX mapping failed: {e}")
        else:
            print("[AOA 2D] rx_order_override='auto' skipped (num_virt_rx != RX or RX>6)")
    elif isinstance(rx_order_override, (list, tuple)) and len(rx_order_override) == num_virt_rx:
        try:
            perm = [int(v) for v in rx_order_override]
            virt_cube = _apply_perm(virt_cube, perm)
            print(f"[AOA 2D] Applied RX permutation override: {perm}")
        except Exception as e:
            print(f"[AOA 2D] RX override failed: {e}")

    ang_rad = np.deg2rad(angles_deg)
    # Steering matrix for azimuth (assuming zero elevation)
    sv = np.exp(1j * 2.0 * np.pi * np.outer(np.sin(ang_rad), x_pos_lam))

    # 4. Beamforming
    win = np.hamming(num_virt_rx)
    virt_cube = _apply_calibration(virt_cube, params, num_virt_rx)

    A = angles_deg.size
    ra_map = np.empty((F, R, A), dtype=float)
    meta = {
        'angles_deg': angles_deg,
        'virt_rx_count': num_virt_rx,
    }
    # Determine if it's 2D geometry (for heatmap in GUI)
    meta['is_2d'] = np.any(y_pos_lam != 0) or np.any(z_pos_lam != 0)

    # Use 'bartlett' as default if method_name not found
    method_name = str(method if 'method' in locals() else "bartlett").lower()
    
    # Check if 'Shift Spectrum' is enabled in params (passed from GUI)
    shift_spectrum = bool(params.get('aoa_shift_spectrum', True)) # DEFAULT ON now
    if shift_spectrum:
        print("[AOA 2D] Shift Spectrum ENABLED (swapping halves)")

    for f in range(F):
        x_f = virt_cube[f] # (num_loops, num_virt_rx, R)
        
        if method_name == "capon":
            # Capon (MVDR) for each range bin
            # Rxx = x * x.H. Regularization for stability.
            bf_avg = np.zeros((R, A))
            for r in range(R):
                x_fr = x_f[:, :, r] # (loops, rx)
                if x_fr.shape[0] < num_virt_rx:
                    # Lack of snapshots, fallback to Bartlett
                    bf_avg[r] = np.abs(x_fr @ sv.T.conj()).mean(axis=0)
                else:
                    Rxx = (x_fr.T.conj() @ x_fr) / x_fr.shape[0]
                    Rxx += np.eye(num_virt_rx) * (np.trace(Rxx) * 0.01 + 1e-9)
                    invR = np.linalg.inv(Rxx)
                    # P = 1 / (a.H * invR * a)
                    den = np.real(np.sum(sv.conj() * (sv @ invR), axis=1))
                    bf_avg[r] = 1.0 / (den + 1e-12)
        else:
            # Bartlett (default)
            x_f_swapped = np.transpose(x_f, (2, 0, 1)) # (R, loops, rx)
            x_f_win = x_f_swapped * win[None, None, :]
            bf_out = np.abs(x_f_win @ sv.T.conj()) ** 2
            bf_avg = bf_out.mean(axis=1) # (R, A)
        
        # Shift spectrum to put 0 degrees in the middle if it currently shows at edges
        if shift_spectrum:
            # ONLY AZIMUTH AXIS (last axis of bf_avg array)
            bf_avg = np.fft.fftshift(bf_avg, axes=-1)

        bf_db = 10.0 * np.log10(np.maximum(bf_avg, 1e-12))
        bf_db -= np.max(bf_db)

        ra_map[f] = bf_db

    r_axis = rf_meta.get('range_m')
    if r_axis is None and _range_axis_m is not None:
        try:
            is_complex_input = np.iscomplexobj(range_cube)
            n_fft_guess = 256
            if R == 129: n_fft_guess = 256
            elif R == 65: n_fft_guess = 128
            use_full = (is_complex_input and R == n_fft_guess)
            r_axis = _range_axis_m(n_fft_guess, not is_complex_input, params, use_full_spectrum=use_full)
            if r_axis.size != R:
                r_axis = np.arange(R) * (r_axis[1] if r_axis.size > 1 else 1.0)
        except Exception:
            r_axis = np.arange(R)

    if r_axis is not None:
        if r_axis.size > R: r_axis = r_axis[:R]
        elif r_axis.size < R:
            step = r_axis[1]-r_axis[0] if r_axis.size > 1 else 1.0
            r_axis = np.concatenate([r_axis, r_axis[-1] + step * np.arange(1, R - r_axis.size + 1)])
        meta['range_axis'] = r_axis
    else:
        meta['range_axis'] = np.arange(R)

    meta.update({
        'view_type': 'Range-Azimuth',
        'angles_deg': angles_deg,
        'virt_rx_count': num_virt_rx,
        'tx_count': eff_unique
    })

    return ra_map, meta

def compute_azimuth_elevation_map(
    cube: np.ndarray,
    params: dict,
    range_bin: int | None = None,
    angles_az_deg: np.ndarray | None = None,
    angles_el_deg: np.ndarray | None = None,
    remove_static: bool = True,
    mimo: bool = True,
    invert_tx_phase: bool = False,
    rx_order_override: list[int] | str | None = None,
    method: str = "bartlett",
) -> tuple[np.ndarray, dict]:
    """
    2D angular spectrum calculation (Azimuth-Elevation) for the selected range bin.
    Note: "method" parameter is accepted for consistency with other functions (Bartlett/Capon),
    currently Bartlett estimation is used (Capon is not yet implemented in this function).
    """
    if angles_az_deg is None:
        angles_az_deg = np.arange(-90.0, 91.0, 2.0)
    if angles_el_deg is None:
        angles_el_deg = np.arange(-90.0, 91.0, 2.0)
    
    F, C_total, RX, N_last = cube.shape
    # Assume cube is already in Range domain (frequency)
    range_cube = cube
    R = range_cube.shape[-1]
    
    if range_bin is None:
        range_bin = R // 4
    range_bin = int(max(0, min(R - 1, range_bin)))

    eff_unique = int(params.get('uniqueChirps_effective') or 1)
    if mimo and eff_unique > 1:
        num_loops = C_total // eff_unique
        try:
            active = range_cube[:, :num_loops*eff_unique, :, :].reshape(F, num_loops, eff_unique, RX, R).copy()
            if invert_tx_phase and eff_unique >= 2:
                active[:, :, 1::2, :, :] *= -1
            virt_cube = active.reshape(F, num_loops, eff_unique * RX, R)
            num_virt_rx = eff_unique * RX
        except Exception:
            virt_cube = range_cube
            num_virt_rx = RX
    else:
        virt_cube = range_cube
        num_virt_rx = RX

    if remove_static:
        virt_cube = virt_cube - virt_cube.mean(axis=1, keepdims=True)

    x_pos, y_pos, z_pos = _get_antenna_geometry(params, num_virt_rx, RX)
    az_rad = np.deg2rad(angles_az_deg)
    el_rad = np.deg2rad(angles_el_deg)
    AZ, EL = np.meshgrid(az_rad, el_rad)
    
    kx = np.sin(AZ) * np.cos(EL)
    ky = np.sin(EL)
    
    data = virt_cube[:, :, :, range_bin]
    data = _apply_calibration(data, params, num_virt_rx)
    
    F_count = data.shape[0]
    spec_2d = np.zeros((F_count, el_rad.size, az_rad.size), dtype=float)
    
    # Optional shift of azimuth axis (0° in the middle)
    shift_spectrum = bool(params.get('aoa_shift_spectrum', True))

    for f in range(F_count):
        x_f = data[f].mean(axis=0) # Average over loops
        for i_el in range(el_rad.size):
            for i_az in range(az_rad.size):
                phase = 2.0 * np.pi * (kx[i_el, i_az] * x_pos + ky[i_el, i_az] * y_pos)
                sv = np.exp(1j * phase)
                val = np.abs(np.vdot(sv, x_f))**2
                spec_2d[f, i_el, i_az] = val
        
        # Shift only across azimuth axis (last axis)
        if shift_spectrum:
            spec_2d[f] = np.fft.fftshift(spec_2d[f], axes=-1)
        
        m = np.max(spec_2d[f])
        if m > 0:
            spec_2d[f] = 10 * np.log10(spec_2d[f] / m + 1e-12)
        else:
            spec_2d[f] = -100.0

    meta = {
        'view_type': 'Azimuth-Elevation',
        'angles_az_deg': angles_az_deg,
        'angles_el_deg': angles_el_deg,
        'range_bin': range_bin,
        'is_2d_spectrum': True
    }
    return spec_2d, meta