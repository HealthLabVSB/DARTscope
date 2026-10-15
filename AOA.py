"""
AOA.py — Angle-of-Arrival (AoA) utilities

Bartlett beamforming on ULA via RX channels.
    Input cube: (Frames, Chirps, RX, RangeBins)
    """
from __future__ import annotations
import numpy as np

import clutter
import spectral

# Zkusíme importovat z projektu pro Range FFT a TX separaci
try:
    from radar_processing import compute_range_fft_cube, _chirp_interval_s, _range_axis_m
except ImportError:
    # Fallback pro případ, že by AOA.py byl spuštěn samostatně
    compute_range_fft_cube = None
    _chirp_interval_s = None
    _range_axis_m = None

try:
    import antenna_config
except ImportError:
    antenna_config = None

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

def _resolve_antenna_config(params: dict, num_virt_rx: int, num_phys_rx: int):
    """
    The antenna configuration to use: the one explicitly selected, or - when the
    capture looks like a 4 RX x 3 TX antenna-on-package board - the verified AOP
    preset. Returning the preset (rather than a private copy of the geometry) keeps
    the element positions and the per-element phase correction in one place, so the
    auto-detected case cannot drift from the explicitly selected one.
    """
    cfg = params.get('antenna_config')
    if cfg:
        return cfg
    if antenna_config is None:
        return None
    platform = str(params.get('platform', '')).upper()
    chip_hint = str(params.get('chip_hint', '')).upper()
    if "ULA" in platform or "LINEAR" in platform:
        return None

    if "AOP" in platform or "AOP" in chip_hint:
        return antenna_config.get_config(antenna_config.AOP_DEFAULT)

    # The chip name decides, not the virtual element count. The count used to be the
    # only route in, and it demanded 12 virtual elements - so a single-TX capture on an
    # antenna-on-package board fell through to the linear fallback and every angle was
    # computed on a 1.5 lambda row the board does not have. A board is what it is
    # whether it transmits on one TX or three.
    cfg = antenna_config.default_config_for_chip(platform or chip_hint)
    if cfg is None and chip_hint and platform:
        cfg = antenna_config.default_config_for_chip(chip_hint)
    if cfg is not None:
        return cfg

    if num_phys_rx == 4 and num_virt_rx == 12:
        return antenna_config.get_config(antenna_config.AOP_DEFAULT)
    return None


def _get_antenna_geometry(params: dict, num_virt_rx: int, num_phys_rx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns x, y, z coordinates of virtual antennas in meters (lambda units).
    
    Detects IWR6843AOP and returns its specific 2D geometry.
    Otherwise uses standard linear (ULA) construction.
    """
    # 0. An explicitly selected antenna configuration always wins over any guessing;
    #    an AOP-looking capture resolves to the verified preset.
    cfg = _resolve_antenna_config(params, num_virt_rx, num_phys_rx)
    if params.get('cascade') and not cfg:
        # no guess for the AWR2243 cascade: a wrong array gives wrong angles silently
        raise ValueError("The AWR2243 cascade (8 RX, 6 TX) needs its antenna configuration selected "
                         f"('{antenna_config.CASCADE_CONFIG}') and a corner-reflector calibration "
                         "(Estimate...) before angles mean anything.")
    if cfg and antenna_config is not None:
        ok, msg = antenna_config.validate(cfg)
        if ok:
            x, y = antenna_config.build_geometry(
                cfg, masks=params.get('tx_order_masks'),
                num_virt_rx=num_virt_rx, num_phys_rx=num_phys_rx)
            how = "selected" if params.get('antenna_config') else "auto-detected"
            print(f"[AOA] Antenna configuration ({how}): '{cfg.get('name')}' "
                  f"({num_virt_rx} virtual elements)")
            return x, y, np.zeros(num_virt_rx)
        print(f"[AOA] WARNING: selected antenna configuration is invalid ({msg}); "
              f"falling back to auto-detection.")

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

    # 2. Detection of TI platforms and their geometries.
    # Explicit platform/chip_hint strings always take priority. Only when neither is
    # known do we fall back to guessing from RX/virtual-channel counts - that guess is
    # ambiguous (a non-AOP 4RX x 3TX board also yields 12 virtual channels), so it is
    # logged loudly instead of being applied silently.
    is_aop = False
    is_awr1843 = False
    is_awr2544 = False

    platform = str(params.get('platform', '')).upper()
    chip_hint = str(params.get('chip_hint', '')).upper()
    is_forced_linear = "ULA" in platform or "LINEAR" in platform

    if is_forced_linear:
        print("[AOA] Antenna type forced to linear/ULA geometry via params['platform'].")
    elif "AOP" in platform or "AOP" in chip_hint:
        is_aop = True
    elif "1843" in platform or "1843" in chip_hint:
        is_awr1843 = True
    elif "2544" in platform or "2544" in chip_hint:
        is_awr2544 = True
    elif num_phys_rx == 4 and num_virt_rx == 12:
        is_aop = True
        print("[AOA] WARNING: platform/chip_hint unknown - guessing IWR6843AOP geometry "
              "from channel count alone (4 RX, 12 virtual channels). This is ambiguous "
              "with a non-AOP 3TX board; if AoA looks wrong, set params['platform'] "
              "explicitly (e.g. to 'ULA').")
    elif num_phys_rx == 4 and num_virt_rx == 8:
        is_awr1843 = True

    if is_aop:
        if num_virt_rx == 12:
            print("[AOA] Using IWR6843AOP 2D geometry (4 RX, 3 TX).")
            
            # Standard RX relative positions
            rx_x = np.array([0.5, 0.5, 0.0, 0.0])
            rx_y = np.array([0.0, 0.5, 0.0, 0.5])
            
            # Map of TX offsets based on mask
            tx_offsets = {
                1: (0.0, 0.0), # TX0
                2: (1.0, 0.0), # TX1
                4: (0.0, 1.0), # TX2
            }
            
            # Get actual masks for the virtual array
            masks = params.get('tx_order_masks')
            if masks and len(masks) >= 3:
                current_masks = masks[:3]
            else:
                current_masks = [4, 2, 1] # Fallback to original hardcoded order
            
            x_list, y_list = [], []
            for m in current_masks:
                off_x, off_y = tx_offsets.get(int(m), (0.0, 0.0))
                x_list.append(rx_x + off_x)
                y_list.append(rx_y + off_y)
                
            x_raw = np.concatenate(x_list)
            el_raw = np.concatenate(y_list)

            print("[AOA] --- VIRTUAL ANTENNA CHANNELS LOG ---")
            tx_names_map = {1: "TX0", 2: "TX1", 4: "TX2"}
            for idx in range(12):
                chirp_idx = idx // 4
                rx_idx = idx % 4 + 1
                m = current_masks[chirp_idx] if chirp_idx < len(current_masks) else 0
                tx_name = tx_names_map.get(int(m), f"TX{chirp_idx}")
                print(f"Virt Channel {idx:2d} ({tx_name}RX{rx_idx}): x_lambda={x_raw[idx]:.2f}, elevation_lambda={el_raw[idx]:.2f}")
            print("------------------------------------------")

            # Centering
            x = x_raw - np.mean(x_raw)
            y = el_raw - np.mean(el_raw)
            z = np.zeros(num_virt_rx)
            return x, y, z
        else:
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

def _apply_calibration(data: np.ndarray, params: dict, num_virt_rx: int,
                       axis: int = -1) -> np.ndarray:
    """
    Applies channel calibration: a measured one (params['channel_calibration'], from
    calibration.py) and TI's phase parameters (antPhaseRot, compRangeBiasAndRxChanPhase).
    data: the virtual channels on `axis` (the last one by default).
    """
    shape = [1] * np.ndim(data)
    shape[axis] = num_virt_rx
    measured = params.get('channel_calibration')
    if measured is not None:
        c = np.array([complex(*z) if isinstance(z, (list, tuple)) else complex(z) for z in measured])
        if c.size == num_virt_rx:
            data = data * c.astype(np.complex64).reshape(shape)
        else:
            print(f"[AOA] Channel calibration has {c.size} channels, the array {num_virt_rx}; "
                  f"not applied.")

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
        return data * np.conj(cal_phases).reshape(shape)
    
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

def _apply_aop_rx_phase_correction(virt_cube: np.ndarray, params: dict, num_virt_rx: int, RX: int) -> np.ndarray:
    """
    Per-element phase correction, taken from the selected antenna configuration.

    This used to apply a hardcoded alternating 180 degree flip to every AOP-looking
    capture. That was wrong: on the AOP layout the flipped elements share an x
    position with their unflipped neighbours, and the azimuth beamformer only uses x,
    so each pair cancelled. A synthetic target swept across the field of view came out
    with a mean error of 41 degrees with the correction, and 0.0 degrees without it.

    A correction is therefore applied only when the chosen antenna configuration
    explicitly defines one (element_phase_deg), which real boards with rotated
    antenna feeds can still do.

    virt_cube shape: (F, num_loops, num_virt_rx, R)
    """
    cfg = _resolve_antenna_config(params, num_virt_rx, RX)
    if not cfg or antenna_config is None:
        return virt_cube
    factors = antenna_config.element_phase_factors(cfg, num_virt_rx)
    if factors is None:
        return virt_cube
    print(f"[AOA] Applying per-element phase correction from '{cfg.get('name')}'")
    return virt_cube * factors[None, None, :, None]

def compensate_tdm_doppler(virt_cube: np.ndarray, num_tx: int, num_rx: int,
                           slots_per_loop: int | None = None, hypothesis=0):
    """
    Remove the phase a moving target adds between the TX slots of one TDM-MIMO loop.

    In TDM-MIMO the TX take turns: TX t transmits t chirp slots after TX 0, and a
    target moving at radial velocity v has meanwhile shifted its phase by
    t * 4 pi v Tc / lambda. The beamformer reads that as part of the angle, so a
    moving target is placed at the wrong angle, and the virtual array loses coherence
    between its TX blocks. The phase is estimated from the Doppler peak of each
    range bin and each frame (sub-bin, spectral.py) and taken out:

        x[t * num_rx + rx] *= exp(-1j * t * 2 pi (k + hypothesis * L) / (L * S))

    with k the signed Doppler bin over L loops and S slots per loop. A static target
    (k = 0) is left untouched, which is why a scene with only static clutter looks the
    same with or without it.

    hypothesis: velocity folding count (roadmap 1.4). The Doppler over loops is only
    known modulo the loop rate; each fold adds 2 pi hypothesis * t / S between TX
    slots. A scalar, or an array broadcastable to (frames, range).

    virt_cube: (frames, loops, num_tx * num_rx, range), virtual index t * num_rx + rx,
    TX t in chirp slot t of the loop.
    Returns (compensated cube, signed fractional Doppler bin per (frame, range)).
    """
    F, L, V, R = virt_cube.shape
    S = int(slots_per_loop or num_tx)
    if L < 2 or num_tx < 2 or V < num_tx * num_rx:
        return virt_cube, np.zeros((F, R))
    win = np.hanning(L + 2)[1:-1].astype(np.float32)          # no zero end points
    k = np.empty((F, R))
    step = max(1, int(2 ** 24 // max(1, L * V * R)))           # frames per ~128 MB chunk
    for f0 in range(0, F, step):
        spec = np.fft.fft(virt_cube[f0:f0 + step] * win[None, :, None, None], axis=1)
        power = np.sum(np.abs(spec) ** 2, axis=2)              # (f, L, R)
        k[f0:f0 + step] = spectral.circular_peak(power, axis=1)
    phi = 2.0 * np.pi * (k + np.asarray(hypothesis, dtype=float) * L) / (L * S)
    t = np.repeat(np.arange(num_tx), num_rx)                   # slot of each channel
    comp = np.ones((F, V, R), dtype=np.complex64)
    comp[:, :num_tx * num_rx, :] = np.exp(-1j * t[None, :, None] * phi[:, None, :])
    return virt_cube * comp[:, None, :, :], k


def compute_aoa_spectrum(
    cube: np.ndarray,
    params: dict,
    range_m: float | None = None,
    range_bin: int | None = None,
    angles_deg: np.ndarray | None = None,
    method: str = "bartlett",
    remove_static: bool = True,
    mimo: bool = True,
    invert_alternate_tx_phase: bool = False,
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
            if invert_alternate_tx_phase and eff_unique >= 2:
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

    virt_cube = _apply_aop_rx_phase_correction(virt_cube, params, num_virt_rx, RX)
    if params.get('tdm_doppler_compensation') and do_mimo_expand and num_virt_rx == eff_unique * RX:
        virt_cube, _ = compensate_tdm_doppler(virt_cube, eff_unique, RX, unique_stride)
        print("[AOA] TDM Doppler compensation applied")

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
        x = clutter.from_params(x, params, slow_axis=1, frame_axis=0)

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
        spec[f, :] = bf_db

    # Global normalization (fixed scale across all frames)
    if spec.size:
        spec -= np.max(spec)

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
    invert_alternate_tx_phase: bool = False,
    mimo_mode: str = "MIMO",
    rx_order_override: list[int] | str | None = None,
    method: str = "bartlett", # NEW: beamforming method
    progress_cb=None,  # optional callable(done_frames, total_frames) for GUI progress
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
            if invert_alternate_tx_phase and eff_unique >= 2:
                active_chirps[:, :, 1::2, :, :] *= -1
                print("[AOA 2D] Alternate TX phase inversion applied")
            
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

    # 1.5. AOP 180° RX phase rotation (shared with all other AoA views) and TDM Doppler compensation
    virt_cube = _apply_aop_rx_phase_correction(virt_cube, params, num_virt_rx, RX)

    if params.get('tdm_doppler_compensation') and num_virt_rx == eff_unique * RX > RX:
        virt_cube, _ = compensate_tdm_doppler(virt_cube, eff_unique, RX, unique_stride)
        print("[AOA 2D] TDM Doppler compensation applied")

    # 2. Clutter removal
    if remove_static:
        virt_cube = clutter.from_params(virt_cube, params, slow_axis=1, frame_axis=0)

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
    virt_cube = _apply_calibration(virt_cube, params, num_virt_rx, axis=2)   # (F, L, V, R)

    A = angles_deg.size
    # float32: the maps of a 3-min take at 100 fps are 18000 x 256 x 181 values - 6.7 GB in
    # float64, which the Movement view then copied twice more
    ra_map = np.empty((F, R, A), dtype=np.float32)
    meta = {
        'angles_deg': angles_deg,
        'virt_rx_count': num_virt_rx,
    }
    # Determine if it's 2D geometry (for heatmap in GUI)
    meta['is_2d'] = np.any(y_pos_lam != 0) or np.any(z_pos_lam != 0)

    # Use 'bartlett' as default if method_name not found
    method_name = str(method if 'method' in locals() else "bartlett").lower()
    
    shift_spectrum = False

    if method_name != "capon":
        # Bartlett on blocks of frames in single precision, from the RX covariance: the mean
        # over the loops of |x^T h|^2 is h^T C conj(h), C = mean x conj(x)^T - the same map
        # as frame by frame in complex128, which took 5 ms a frame (2 min for a 3-min take
        # at 100 fps); h is the steering vector with the RX window folded in
        svh = (win[:, None] * sv.T.conj()).astype(np.complex64)                 # (V, A)
        block = max(1, int(2e7 // max(1, R * num_virt_rx * A)))                  # ~160 MB at a time
        for f0 in range(0, F, block):
            xb = np.asarray(virt_cube[f0:f0 + block], dtype=np.complex64)        # (B, L, V, R)
            C = np.matmul(np.transpose(xb, (0, 3, 2, 1)),
                          np.conj(np.transpose(xb, (0, 3, 1, 2)))) / xb.shape[1] # (B, R, V, V)
            T = np.matmul(C, np.conj(svh))                                       # (B, R, V, A)
            bf_avg = np.einsum("brva,va->bra", T, svh).real                      # (B, R, A)
            if shift_spectrum:
                bf_avg = np.fft.fftshift(bf_avg, axes=-1)
            ra_map[f0:f0 + bf_avg.shape[0]] = 10.0 * np.log10(np.maximum(bf_avg, 1e-12))
            if progress_cb is not None:
                progress_cb(min(F, f0 + block), F)

    for f in range(F if method_name == "capon" else 0):
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

        ra_map[f] = bf_db

        if progress_cb is not None:
            progress_cb(f + 1, F)

    # Global normalization (fixed scale across all frames)
    if ra_map.size:
        ra_map -= np.max(ra_map)

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
    invert_alternate_tx_phase: bool = False,
    rx_order_override: list[int] | str | None = None,
    method: str = "bartlett",
) -> tuple[np.ndarray, dict]:
    """
    2D angular spectrum calculation (Azimuth-Elevation) for the selected range bin.
    Supports both "bartlett" and "capon" (MVDR) estimation via the "method" parameter.
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
            if invert_alternate_tx_phase and eff_unique >= 2:
                active[:, :, 1::2, :, :] *= -1
            virt_cube = active.reshape(F, num_loops, eff_unique * RX, R)
            num_virt_rx = eff_unique * RX
        except Exception:
            virt_cube = range_cube
            num_virt_rx = RX
    else:
        virt_cube = range_cube
        num_virt_rx = RX

    virt_cube = _apply_aop_rx_phase_correction(virt_cube, params, num_virt_rx, RX)
    if params.get('tdm_doppler_compensation') and num_virt_rx == eff_unique * RX > RX:
        virt_cube, _ = compensate_tdm_doppler(virt_cube, eff_unique, RX, eff_unique)
        print("[AOA Az-El] TDM Doppler compensation applied")

    if remove_static:
        virt_cube = clutter.from_params(virt_cube, params, slow_axis=1, frame_axis=0)

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
    
    shift_spectrum = False

    # Bartlett or Capon estimation
    method_name = str(method).lower()
    
    for f in range(F_count):
        x_f_frame = data[f] # (loops, rx)
        
        if method_name == "capon":
            # MVDR / Capon
            # Rxx = x * x.H. Regularization for stability.
            Rxx = (x_f_frame.T.conj() @ x_f_frame) / x_f_frame.shape[0]
            Rxx += np.eye(num_virt_rx) * (np.trace(Rxx) * 0.01 + 1e-9)
            invR = np.linalg.inv(Rxx)
            
            for i_el in range(el_rad.size):
                # Steering vectors for all azimuths at once for this elevation
                phase = 2.0 * np.pi * (kx[i_el, :] * x_pos[:, None] + ky[i_el, :] * y_pos[:, None])
                sv_mat = np.exp(1j * phase) # (rx, az)
                # P = 1 / (a.H * invR * a)
                # sv_mat.T.conj() has shape (az, rx)
                den = np.real(np.sum(sv_mat.conj() * (invR @ sv_mat), axis=0))
                spec_2d[f, i_el, :] = 1.0 / (den + 1e-12)
        else:
            # Bartlett. With clutter removal the power has to be averaged over the
            # snapshots: averaging the complex snapshots first would take the very loop
            # mean that clutter removal subtracted, leaving zero.
            for i_el in range(el_rad.size):
                phase = 2.0 * np.pi * (kx[i_el, :] * x_pos[:, None] + ky[i_el, :] * y_pos[:, None])
                sv_mat = np.exp(1j * phase) # (rx, az)
                # a^H x, conjugated as in the RAE view and in Capon above. Without the
                # conjugate this matched the mirror image: a target at (+20, -10) deg
                # was drawn at (-20, +10) while the RAE view put it where it is.
                if remove_static:
                    spec_2d[f, i_el, :] = (np.abs(x_f_frame @ sv_mat.conj()) ** 2).mean(axis=0)
                else:
                    spec_2d[f, i_el, :] = np.abs(x_f_frame.mean(axis=0) @ sv_mat.conj()) ** 2
        
        # Shift only across azimuth axis (last axis)
        if shift_spectrum:
            spec_2d[f] = np.fft.fftshift(spec_2d[f], axes=-1)
        
        spec_2d[f] = 10 * np.log10(np.maximum(spec_2d[f], 1e-12))

    # Global normalization (fixed scale across all frames)
    if spec_2d.size:
        spec_2d -= np.max(spec_2d)

    meta = {
        'view_type': 'Azimuth-Elevation',
        'angles_az_deg': angles_az_deg,
        'angles_el_deg': angles_el_deg,
        'range_bin': range_bin,
        'is_2d_spectrum': True
    }
    return spec_2d, meta

def compute_range_azimuth_elevation_map(
    cube: np.ndarray,
    params: dict,
    angles_az_deg: np.ndarray | None = None,
    angles_el_deg: np.ndarray | None = None,
    r_start: int = 0,
    r_end: int | None = None,
    remove_static: bool = True,
    mimo: bool = True,
    invert_alternate_tx_phase: bool = False,
    mimo_mode: str = "MIMO",
    rx_order_override: list[int] | str | None = None,
    method: str = "capon",
    diagonal_loading: float = 0.01,
    progress_cb=None,  # optional callable(done_frames, total_frames) for GUI progress
) -> tuple[np.ndarray, dict]:
    """
    3D Range-Azimuth-Elevation (RAE) map calculation using Capon (MVDR) or Bartlett
    beamforming for all range bins.

    Returns:
        rae_map: float32 array of shape (F, R, EL, AZ) in dB, normalized to the global
            maximum across all frames (fixed scale during frame playback).
        meta: dict with 'range_axis', 'angles_az_deg', 'angles_el_deg', 'view_type'='Range-Az-El'.

    Notes:
        - Capon uses diagonal loading (trace(Rxx) * diagonal_loading) for numerical stability.
        - If the number of snapshots (chirp loops) is lower than the number of virtual RX
          antennas, the method automatically falls back to Bartlett.
        - If antenna geometry has no vertical component (ULA), elevation is not resolvable
          and the elevation grid is reduced to [0.0] with a warning.
    """
    if angles_az_deg is None:
        angles_az_deg = np.arange(-90.0, 91.0, 2.0)
    if angles_el_deg is None:
        angles_el_deg = np.arange(-45.0, 46.0, 2.0)
    angles_az_deg = np.asarray(angles_az_deg).astype(float)
    angles_el_deg = np.asarray(angles_el_deg).astype(float)

    # 1. FFT and MIMO preparation (same pattern as compute_range_azimuth_map)
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

    fc = params.get('frameConfig') or {}
    expected_chirps = (fc.get('chirpEndIdx', 0) - fc.get('chirpStartIdx', 0) + 1) * fc.get('numLoops', 0) if ('chirpStartIdx' in fc and 'chirpEndIdx' in fc and 'numLoops' in fc) else None
    safe_tdm_check = _is_safe_tdm(current_masks)

    # 1. Diagnostika metadat a datového cube
    print("[AOA RAE] --- MIMO DIAGNOSTIC LOG ---")
    print(f"[AOA RAE] cube.shape = {cube.shape}")
    print(f"[AOA RAE] F = {F}, C_total = {C_total}, RX = {RX}, R = {R}")
    print(f"[AOA RAE] uniqueChirps_effective = {eff_unique}")
    print(f"[AOA RAE] unique_chirps_frame = {unique_stride}")
    print(f"[AOA RAE] tx_order_masks = {masks}")
    print(f"[AOA RAE] tx_order_names = {params.get('tx_order_names')}")
    print(f"[AOA RAE] frameConfig.chirpStartIdx = {fc.get('chirpStartIdx')}")
    print(f"[AOA RAE] frameConfig.chirpEndIdx = {fc.get('chirpEndIdx')}")
    print(f"[AOA RAE] frameConfig.numLoops = {fc.get('numLoops')}")
    print(f"[AOA RAE] expected_chirps = {expected_chirps}")
    print(f"[AOA RAE] safe_tdm_check = {safe_tdm_check}")
    print("[AOA RAE] ---------------------------")

    if mimo and mimo_mode == "MIMO":
        # Raise error if general TDM-MIMO prerequisites are not met
        if not masks:
            raise ValueError("MIMO Error: parameter 'tx_order_masks' is missing from the metadata.")
        if eff_unique <= 1:
            raise ValueError(f"MIMO Error: invalid 'uniqueChirps_effective'={eff_unique} (MIMO needs > 1).")
        if unique_stride <= 1:
            raise ValueError(f"MIMO Error: invalid 'unique_chirps_frame'={unique_stride} (MIMO needs > 1).")
        if C_total % unique_stride != 0:
            raise ValueError(f"MIMO Error: chirps per frame ({C_total}) is not divisible by the number of unique chirps ({unique_stride}).")
        if not safe_tdm_check:
            raise ValueError(f"MIMO Error: _is_safe_tdm() failed for masks {current_masks}. Each chirp must enable exactly one unique TX.")

    if do_mimo_expand:
        num_loops = C_total // unique_stride
        try:
            reshaped = range_cube.reshape(F, num_loops, unique_stride, RX, R)
            active_chirps = reshaped[:, :, :eff_unique, :, :].copy()
            if invert_alternate_tx_phase and eff_unique >= 2:
                active_chirps[:, :, 1::2, :, :] *= -1
                print("[AOA RAE] Alternate TX phase inversion applied")
            virt_cube = active_chirps.reshape(F, num_loops, eff_unique * RX, R)
            num_virt_rx = eff_unique * RX
            print("[AOA RAE] MIMO enabled")
            print(f"[AOA RAE] virt_rx_count={num_virt_rx}")
            print(f"[AOA RAE] tx_order_masks={current_masks}")
            print(f"[AOA RAE] num_loops={num_loops}")
        except Exception as e:
            print(f"[AOA RAE] MIMO reshape failed: {e}")
            virt_cube = range_cube
            num_virt_rx = RX
            num_loops = C_total
    elif mimo_mode == "First TX" and unique_stride > 0:
        num_loops = C_total // unique_stride
        reshaped = range_cube.reshape(F, num_loops, unique_stride, RX, R)
        virt_cube = reshaped[:, :, 0, :, :]  # Only first chirp (TX1)
        num_virt_rx = RX
        print(f"[AOA RAE] Using First TX only. RX count: {num_virt_rx}")
    else:
        virt_cube = range_cube
        num_virt_rx = RX
        num_loops = C_total
        if eff_unique > 1 and mimo:
            reason = "unsafe pattern" if not _is_safe_tdm(current_masks) else "subset/filtered data"
            print(f"[AOA RAE] MIMO disabled ({reason}). Using {RX} antennas. Masks: {current_masks}")

    # Optional RX permutation override (explicit list only; 'auto' is not supported here)
    if rx_order_override is not None:
        if isinstance(rx_order_override, str):
            print("[AOA RAE] rx_order_override='auto' not supported for RAE map (skipped)")
        elif isinstance(rx_order_override, (list, tuple)) and len(rx_order_override) == num_virt_rx:
            try:
                perm = [int(v) for v in rx_order_override]
                virt_cube = virt_cube[:, :, perm, :]
                print(f"[AOA RAE] Applied RX permutation override: {perm}")
            except Exception as e:
                print(f"[AOA RAE] RX override failed: {e}")

    # 1.5. AOP 180° RX phase rotation (shared with all other AoA views)
    virt_cube = _apply_aop_rx_phase_correction(virt_cube, params, num_virt_rx, RX)
    if params.get('tdm_doppler_compensation') and do_mimo_expand and num_virt_rx == eff_unique * RX:
        virt_cube, _ = compensate_tdm_doppler(virt_cube, eff_unique, RX, unique_stride)
        print("[AOA RAE] TDM Doppler compensation applied")

    # 2. Clutter removal
    if remove_static:
        virt_cube = clutter.from_params(virt_cube, params, slow_axis=1, frame_axis=0)

    # 3. Calibration (broadcast over RX axis)
    cal_test = np.ones(num_virt_rx, dtype=np.complex64)
    cal_applied = _apply_calibration(cal_test, params, num_virt_rx)
    if not np.allclose(cal_applied, cal_test):
        virt_cube = virt_cube * cal_applied[None, None, :, None]
        print("[AOA RAE] Applied phase calibration.")

    # 4. Antenna geometry and elevation capability check
    x_pos, y_pos, z_pos = _get_antenna_geometry(params, num_virt_rx, RX)
    if F > 0:
        print(f"[AOA RAE] Platform: {params.get('platform','unknown')}, Chip: {params.get('chip_hint','unknown')}")
        print(f"[AOA RAE] Geometry (lambda units): x={x_pos[:min(8,len(x_pos))]}..., y={y_pos[:min(8,len(y_pos))]}...")
    if not (np.any(y_pos != 0) or np.any(z_pos != 0)):
        print("[AOA RAE] WARNING: Antenna geometry has no vertical component (ULA). "
              "Elevation is not resolvable - reducing elevation grid to [0.0].")
        angles_el_deg = np.array([0.0])

    # 5. Range cropping
    if r_end is None:
        r_end = R
    r_start = int(max(0, min(R - 1, r_start)))
    r_end = int(max(r_start + 1, min(R, r_end)))
    virt_cube = virt_cube[:, :, :, r_start:r_end]
    R_sel = r_end - r_start

    # 6. Precompute 2D steering matrix SV (virt_rx x EL*AZ)
    az_rad = np.deg2rad(angles_az_deg)
    el_rad = np.deg2rad(angles_el_deg)
    AZ_grid, EL_grid = np.meshgrid(az_rad, el_rad)   # (EL, AZ)
    kx = (np.sin(AZ_grid) * np.cos(EL_grid)).ravel()  # (EL*AZ,)
    ky = np.sin(EL_grid).ravel()
    SV = np.exp(1j * 2.0 * np.pi * (x_pos[:, None] * kx[None, :] + y_pos[:, None] * ky[None, :]))  # (rx, K)

    EL_n = el_rad.size
    AZ_n = az_rad.size
    K = EL_n * AZ_n

    # 7. Method selection with snapshot-deficit fallback
    method_name = str(method).lower()
    if method_name == "capon" and num_loops < num_virt_rx:
        print(f"[AOA RAE] Not enough snapshots for Capon (loops={num_loops} < virt_rx={num_virt_rx}). "
              "Falling back to Bartlett.")
        method_name = "bartlett"

    shift_spectrum = False
    print(f"[AOA RAE] shift_spectrum={shift_spectrum}")

    rae_map = np.empty((F, R_sel, EL_n, AZ_n), dtype=np.float32)
    eye = np.eye(num_virt_rx, dtype=np.complex128)

    for f in range(F):
        x_f = virt_cube[f]  # (loops, rx, R_sel)

        if method_name == "capon":
            # Batch covariance matrices per range bin: (R_sel, rx, rx)
            X = np.transpose(x_f, (2, 0, 1))  # (R_sel, loops, rx)
            # Standard covariance: Rxx_ij = E[x_i * conj(x_j)]
            Rxx = np.einsum('rli,rlj->rij', X, X.conj()) / max(1, num_loops)
            tr = np.einsum('rii->r', Rxx).real
            Rxx = Rxx + eye[None, :, :] * (tr * diagonal_loading + 1e-9)[:, None, None]
            try:
                invR = np.linalg.inv(Rxx)
            except np.linalg.LinAlgError:
                invR = np.linalg.pinv(Rxx)
            P = np.empty((R_sel, K), dtype=np.float64)
            for r in range(R_sel):
                # P = 1 / (a.H * invR * a) for all steering vectors at once
                den = np.real(np.sum(SV.conj() * (invR[r] @ SV), axis=0))
                P[r] = 1.0 / (den + 1e-12)
        elif remove_static:
            # Bartlett on clutter-removed data: average the beamformer POWER over loops.
            # Averaging the complex snapshots first (below) would take exactly the loop
            # mean that clutter removal just subtracted, so the result would be zero and
            # the whole map would collapse into numerical noise - on a real recording
            # the entire cube spanned only 2.4 dB, which made every threshold useless.
            P = np.zeros((R_sel, K), dtype=np.float64)
            n_snap = x_f.shape[0]
            for snap in range(n_snap):
                P += np.abs(x_f[snap].T @ SV.conj()) ** 2
            P /= max(1, n_snap)
        else:
            # Without clutter removal the coherent mean over loops is a deliberate
            # static/DC beamformer, and a single matmul is much faster.
            xm = x_f.mean(axis=0)  # (rx, R_sel)
            P = np.abs(xm.T @ SV.conj()) ** 2  # (R_sel, K)

        P = P.reshape(R_sel, EL_n, AZ_n)

        # Shift spectrum only across the azimuth axis (last axis)
        if shift_spectrum:
            P = np.fft.fftshift(P, axes=-1)

        bf_db = 10.0 * np.log10(np.maximum(P, 1e-12))
        rae_map[f] = bf_db.astype(np.float32)

        if progress_cb is not None:
            progress_cb(f + 1, F)

    # Global normalization (fixed scale across all frames)
    if rae_map.size:
        rae_map -= np.max(rae_map)

    # 8. Range axis (same logic as compute_range_azimuth_map)
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
        r_axis = np.asarray(r_axis, dtype=float)
        if r_axis.size > R: r_axis = r_axis[:R]
        elif r_axis.size < R:
            step = r_axis[1]-r_axis[0] if r_axis.size > 1 else 1.0
            r_axis = np.concatenate([r_axis, r_axis[-1] + step * np.arange(1, R - r_axis.size + 1)])
    else:
        r_axis = np.arange(R, dtype=float)

    meta = {
        'view_type': 'Range-Az-El',
        'angles_az_deg': angles_az_deg,
        'angles_el_deg': angles_el_deg,
        'range_axis': r_axis[r_start:r_end],
        'r_start': r_start,
        'r_end': r_end,
        'virt_rx_count': num_virt_rx,
        'tx_count': eff_unique,
        'method_used': method_name,
    }

    return rae_map, meta