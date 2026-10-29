"""
test_capon_rae.py

Synthetic tests for the Capon (MVDR) Range-Azimuth-Elevation map (AOA.py)
and the RAE target detection pipeline (detection.py).

Generates a synthetic datacube for IWR6843AOP geometry (4 RX x 3 TX = 12 virt RX)
with targets at known range / azimuth / elevation and verifies that:
- the RAE map peak lands at the expected position (Capon and Bartlett),
- the map stays finite with a snapshot deficit (Bartlett fallback),
- static targets disappear with remove_static=True,
- the detector returns the expected point cloud.

Run: python test_capon_rae.py

Authors: Daniel Barvik, Dan Hruby, and AI
"""
import sys
import numpy as np

import antenna_config as ac
from AOA import (
    compute_range_azimuth_elevation_map,
    compute_range_azimuth_map,
    _get_antenna_geometry,
)
from detection import detect_targets_rae, cfar_range_1d_vectorized


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
RANGE_STEP_M = 0.1  # synthetic range bin size [m]


# Hardcoded reference geometries per Requirements
X_REF_RAW = np.array([
    0.5, 0.5, 0.0, 0.0,
    1.5, 1.5, 1.0, 1.0,
    0.5, 0.5, 0.0, 0.0
], dtype=float)
EL_REF_RAW = np.array([
    1.0, 1.5, 1.0, 1.5,
    0.0, 0.5, 0.0, 0.5,
    0.0, 0.5, 0.0, 0.5
], dtype=float)

AOP_X_REF = X_REF_RAW - np.mean(X_REF_RAW)
AOP_EL_REF = EL_REF_RAW - np.mean(EL_REF_RAW)


def make_params(R: int = 64) -> dict:
    """
    Parameters emulating an IWR6843AOP TDM-MIMO capture (3 TX, 4 RX).

    make_synthetic_cube() below bakes an alternating 180 degree phase into the
    generated wavefront ("rotated RX1/RX3 elements on the chip"), and the antenna
    configuration handed to the beamformer declares the same convention. That this
    matches real hardware was confirmed on an actual IWR6843AOP capture - see
    antenna_config.AOP_DEFAULT for the measurement.
    """
    return {
        'platform': 'AOP',
        'chip_hint': 'IWR6843AOP',
        'uniqueChirps_effective': 3,
        'unique_chirps_frame': 3,
        'tx_order_masks': [1, 2, 4],
        'is_frequency_domain': True,
        'aoa_shift_spectrum': False,   # keep steering-grid indexing intact
        'range_m': np.arange(R) * RANGE_STEP_M,
        'antenna_config': ac.get_config(ac.AOP_DEFAULT),
    }


def make_synthetic_cube(
    targets: list[dict],
    params: dict,
    F: int = 2,
    loops: int = 64,
    RX: int = 4,
    ntx: int = 3,
    R: int = 64,
    noise: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """
    Build a synthetic range-domain cube (F, loops*ntx, RX, R) with point targets.

    Each target dict: {'r_bin': int, 'az_deg': float, 'el_deg': float,
                       'amp': float, 'static': bool (optional)}
    Non-static targets get a random phase per snapshot (loop) so that the
    spatial covariance matrix has full rank for Capon.
    """
    rng = np.random.default_rng(seed)
    
    # Standard RX relative positions
    rx_x = np.array([0.5, 0.5, 0.0, 0.0])
    rx_y = np.array([0.0, 0.5, 0.0, 0.5])
    
    # Map of TX offsets based on mask
    tx_offsets = {
        1: (0.0, 0.0), # TX0
        2: (1.0, 0.0), # TX1
        4: (0.0, 1.0), # TX2
    }
    
    masks = params.get('tx_order_masks')
    if masks and len(masks) >= 3:
        current_masks = masks[:3]
    else:
        current_masks = [4, 2, 1]
    
    x_list, y_list = [], []
    for m in current_masks:
        off_x, off_y = tx_offsets.get(int(m), (0.0, 0.0))
        x_list.append(rx_x + off_x)
        y_list.append(rx_y + off_y)
        
    x_raw = np.concatenate(x_list)
    y_raw = np.concatenate(y_list)
    
    x_pos = x_raw - np.mean(x_raw)
    y_pos = y_raw - np.mean(y_raw)

    cube = noise * (rng.standard_normal((F, loops * ntx, RX, R))
                    + 1j * rng.standard_normal((F, loops * ntx, RX, R)))

    for tgt in targets:
        az = np.deg2rad(float(tgt['az_deg']))
        el = np.deg2rad(float(tgt['el_deg']))
        kx = np.sin(az) * np.cos(el)
        ky = np.sin(el)
        a = np.exp(1j * 2.0 * np.pi * (x_pos * kx + y_pos * ky))  # (virt,)
        amp = float(tgt.get('amp', 1.0))
        if tgt.get('static', False):
            s = amp * np.ones((F, loops))
        else:
            s = amp * np.exp(1j * 2.0 * np.pi * rng.random((F, loops)))
        # Phase sign representing the rotated RX1/RX3 elements on the chip
        # RX1 (idx 0) and RX3 (idx 2) are inverted (value -1)
        phase_sign = [
            -1, +1, -1, +1,
            -1, +1, -1, +1,
            -1, +1, -1, +1
        ]
        r_bin = int(tgt['r_bin'])
        for t in range(ntx):
            for rx in range(RX):
                v = t * RX + rx
                # chirp index = loop*ntx + t (TDM order TX1, TX2, TX3)
                cube[:, t::ntx, rx, r_bin] += s * a[v] * phase_sign[v]

    return cube.astype(np.complex64)


def _nearest_idx(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(axis) - value)))


# ---------------------------------------------------------------------------
# RAE map tests (Step 1)
# ---------------------------------------------------------------------------
def test_single_target_capon():
    """Peak of the Capon RAE map at known (r, az, el) within tolerance."""
    R = 64
    params = make_params(R)
    tgt = {'r_bin': 30, 'az_deg': 20.0, 'el_deg': -10.0, 'amp': 5.0}
    cube = make_synthetic_cube([tgt], params, R=R)

    rae, meta = compute_range_azimuth_elevation_map(
        cube, params, remove_static=False, method="capon"
    )
    F, Rn, ELn, AZn = rae.shape
    assert Rn == R and F == 2, f"unexpected shape {rae.shape}"
    assert np.all(np.isfinite(rae)), "map contains NaN/Inf"
    assert meta['method_used'] == 'capon', f"expected capon, got {meta['method_used']}"

    az_axis = meta['angles_az_deg']
    el_axis = meta['angles_el_deg']
    exp_az = _nearest_idx(az_axis, tgt['az_deg'])
    exp_el = _nearest_idx(el_axis, tgt['el_deg'])

    for f in range(F):
        r_i, el_i, az_i = np.unravel_index(np.argmax(rae[f]), rae[f].shape)
        assert abs(r_i - tgt['r_bin']) <= 1, f"frame {f}: range peak {r_i} != {tgt['r_bin']}"
        assert abs(az_i - exp_az) <= 2, f"frame {f}: az peak {az_axis[az_i]} != {tgt['az_deg']}"
        assert abs(el_i - exp_el) <= 2, f"frame {f}: el peak {el_axis[el_i]} != {tgt['el_deg']}"
    # Physical range axis check
    assert abs(meta['range_axis'][tgt['r_bin']] - tgt['r_bin'] * RANGE_STEP_M) < 1e-9


def test_bartlett_same_peak():
    """Bartlett returns a map of the same shape with the peak at the same position."""
    R = 64
    params = make_params(R)
    tgt = {'r_bin': 30, 'az_deg': 20.0, 'el_deg': -10.0, 'amp': 5.0}
    cube = make_synthetic_cube([tgt], params, R=R)

    rae_c, meta_c = compute_range_azimuth_elevation_map(
        cube, params, remove_static=False, method="capon"
    )
    rae_b, meta_b = compute_range_azimuth_elevation_map(
        cube, params, remove_static=False, method="bartlett"
    )
    assert rae_c.shape == rae_b.shape, "Capon/Bartlett shape mismatch"
    assert meta_b['method_used'] == 'bartlett'
    assert np.all(np.isfinite(rae_b)), "Bartlett map contains NaN/Inf"

    pc = np.unravel_index(np.argmax(rae_c[0]), rae_c[0].shape)
    pb = np.unravel_index(np.argmax(rae_b[0]), rae_b[0].shape)
    assert abs(pc[0] - pb[0]) <= 1, f"range peak mismatch: {pc[0]} vs {pb[0]}"
    assert abs(pc[1] - pb[1]) <= 2, f"el peak mismatch: {pc[1]} vs {pb[1]}"
    assert abs(pc[2] - pb[2]) <= 2, f"az peak mismatch: {pc[2]} vs {pb[2]}"


def test_frame_config_64_loops():
    """3TX x 4RX capture whose frameConfig declares 64 loops (192 chirps) is processed."""
    R = 64
    params = make_params(R)
    params['frameConfig'] = {'chirpStartIdx': 0, 'chirpEndIdx': 2, 'numLoops': 64}
    tgt = {'r_bin': 30, 'az_deg': 20.0, 'el_deg': -10.0, 'amp': 5.0}
    cube = make_synthetic_cube([tgt], params, F=1, loops=64, R=R)
    assert cube.shape[1] == 192

    rae, meta = compute_range_azimuth_elevation_map(
        cube, params, remove_static=False, method="capon"
    )
    assert np.all(np.isfinite(rae)), "map contains NaN/Inf"

    az_axis = meta['angles_az_deg']
    el_axis = meta['angles_el_deg']
    r_i, el_i, az_i = np.unravel_index(np.argmax(rae[0]), rae[0].shape)
    assert abs(r_i - tgt['r_bin']) <= 1, f"range peak {r_i} != {tgt['r_bin']}"
    assert abs(az_axis[az_i] - tgt['az_deg']) <= 4.0, f"az peak {az_axis[az_i]} != {tgt['az_deg']}"
    assert abs(el_axis[el_i] - tgt['el_deg']) <= 4.0, f"el peak {el_axis[el_i]} != {tgt['el_deg']}"


def test_snapshot_deficit_fallback():
    """loops < virt RX -> automatic Bartlett fallback, finite map values."""
    R = 32
    params = make_params(R)
    tgt = {'r_bin': 12, 'az_deg': -15.0, 'el_deg': 5.0, 'amp': 5.0}
    cube = make_synthetic_cube([tgt], params, F=1, loops=8, R=R)  # 8 < 12 virt RX

    rae, meta = compute_range_azimuth_elevation_map(
        cube, params, remove_static=False, method="capon"
    )
    assert meta['method_used'] == 'bartlett', "expected Bartlett fallback"
    assert np.all(np.isfinite(rae)), "fallback map contains NaN/Inf"
    r_i, el_i, az_i = np.unravel_index(np.argmax(rae[0]), rae[0].shape)
    assert abs(r_i - tgt['r_bin']) <= 1, f"range peak {r_i} != {tgt['r_bin']}"


def test_remove_static():
    """A static target stops standing out of the background when remove_static=True."""
    R = 32
    params = make_params(R)
    tgt = {'r_bin': 12, 'az_deg': 0.0, 'el_deg': 0.0, 'amp': 10.0, 'static': True}
    cube = make_synthetic_cube([tgt], params, F=1, R=R)

    rae_keep, meta = compute_range_azimuth_elevation_map(
        cube, params, remove_static=False, method="bartlett"
    )
    az_i = _nearest_idx(meta['angles_az_deg'], 0.0)
    el_i = _nearest_idx(meta['angles_el_deg'], 0.0)

    rae_rm, _ = compute_range_azimuth_elevation_map(
        cube, params, remove_static=True, method="bartlett"
    )
    assert np.all(np.isfinite(rae_rm)), "map contains NaN/Inf"

    # Compare the target against the background of its own map. An absolute dB
    # threshold is meaningless here because each map is renormalised to its own peak:
    # once the target is gone, the noise floor becomes the peak.
    keep_contrast = float(rae_keep[0, tgt['r_bin'], el_i, az_i] - np.median(rae_keep[0]))
    rm_contrast = float(rae_rm[0, tgt['r_bin'], el_i, az_i] - np.median(rae_rm[0]))

    assert keep_contrast > 20.0, \
        f"static target should dominate without removal (contrast {keep_contrast:.1f} dB)"
    assert rm_contrast < 3.0, \
        f"static target not removed (contrast {rm_contrast:.1f} dB above the background)"

def test_range_crop():
    """r_start/r_end crop the map and range axis consistently."""
    R = 64
    params = make_params(R)
    tgt = {'r_bin': 30, 'az_deg': 10.0, 'el_deg': 0.0, 'amp': 5.0}
    cube = make_synthetic_cube([tgt], params, F=1, R=R)

    rae, meta = compute_range_azimuth_elevation_map(
        cube, params, r_start=10, r_end=50, remove_static=False, method="capon"
    )
    assert rae.shape[1] == 40, f"cropped R size {rae.shape[1]} != 40"
    assert meta['range_axis'].size == 40
    r_i = np.unravel_index(np.argmax(rae[0]), rae[0].shape)[0]
    assert abs((r_i + 10) - tgt['r_bin']) <= 1, f"peak at global bin {r_i + 10} != {tgt['r_bin']}"


def test_ula_no_elevation():
    """ULA geometry (no vertical component) -> elevation grid reduced to [0]."""
    R = 32
    params = make_params(R)
    params['platform'] = 'GENERIC'
    params['chip_hint'] = 'ULA'
    params['uniqueChirps_effective'] = 2
    params['unique_chirps_frame'] = 2
    params['tx_order_masks'] = [1, 2]
    # This scenario is a linear 2TX x 4RX array, so declare that array explicitly
    # instead of inheriting the AOP configuration from make_params().
    params['antenna_config'] = ac.get_config("Azimuth ULA - 2TX x 4RX (8 virtual)")

    rng = np.random.default_rng(1)
    num_virt = 8
    # Hardcoded standard ULA geometry in lambda units
    x_pos = np.arange(num_virt) * 0.5
    x_pos = x_pos - np.mean(x_pos)
    az = np.deg2rad(25.0)
    a = np.exp(1j * 2.0 * np.pi * x_pos * np.sin(az))
    loops = 32
    cube = 0.05 * (rng.standard_normal((1, loops * 2, 4, R))
                   + 1j * rng.standard_normal((1, loops * 2, 4, R)))
    s = 5.0 * np.exp(1j * 2.0 * np.pi * rng.random((1, loops)))
    for t in range(2):
        for rx in range(4):
            cube[:, t::2, rx, 10] += s * a[t * 4 + rx]

    rae, meta = compute_range_azimuth_elevation_map(
        cube.astype(np.complex64), params, remove_static=False, method="capon"
    )
    assert meta['angles_el_deg'].size == 1 and meta['angles_el_deg'][0] == 0.0, \
        "elevation grid should collapse to [0.0] for ULA"
    r_i, el_i, az_i = np.unravel_index(np.argmax(rae[0]), rae[0].shape)
    assert r_i == 10, f"range peak {r_i} != 10"
    assert abs(meta['angles_az_deg'][az_i] - 25.0) <= 4.0, \
        f"az peak {meta['angles_az_deg'][az_i]} != 25"


# ---------------------------------------------------------------------------
# Detection tests (Step 2)
# ---------------------------------------------------------------------------
def test_detect_single_target():
    """One target -> exactly one detection with correct physical values."""
    R = 64
    params = make_params(R)
    tgt = {'r_bin': 30, 'az_deg': 20.0, 'el_deg': -10.0, 'amp': 5.0}
    cube = make_synthetic_cube([tgt], params, F=1, R=R)

    rae, meta = compute_range_azimuth_elevation_map(
        cube, params, remove_static=False, method="capon"
    )
    dets = detect_targets_rae(
        rae[0], meta['range_axis'], meta['angles_az_deg'], meta['angles_el_deg'],
        threshold_factor=6.0
    )
    assert len(dets) == 1, f"expected 1 detection, got {len(dets)}"
    d = dets[0]
    assert abs(d['range_m'] - tgt['r_bin'] * RANGE_STEP_M) <= 1.5 * RANGE_STEP_M, \
        f"range {d['range_m']:.2f} m != {tgt['r_bin'] * RANGE_STEP_M:.2f} m"
    assert abs(d['azimuth_deg'] - tgt['az_deg']) <= 4.0, f"az {d['azimuth_deg']} != {tgt['az_deg']}"
    assert abs(d['elevation_deg'] - tgt['el_deg']) <= 4.0, f"el {d['elevation_deg']} != {tgt['el_deg']}"
    assert d['snr_db'] > 0.0, "SNR should be positive"
    # Cartesian consistency: x^2 + y^2 + z^2 == r^2
    r_cart = np.sqrt(d['x_m']**2 + d['y_m']**2 + d['z_m']**2)
    assert abs(r_cart - d['range_m']) < 1e-9, "Cartesian coords inconsistent with range"
    assert d['z_m'] < 0.0, "negative elevation should give negative z"


def test_detect_two_targets_same_range():
    """Two targets at the same range, azimuths +-25 deg -> Capon separates them."""
    R = 64
    params = make_params(R)
    targets = [
        {'r_bin': 25, 'az_deg': -25.0, 'el_deg': 0.0, 'amp': 5.0},
        {'r_bin': 25, 'az_deg': 25.0, 'el_deg': 0.0, 'amp': 5.0},
    ]
    cube = make_synthetic_cube(targets, params, F=1, R=R)

    rae, meta = compute_range_azimuth_elevation_map(
        cube, params, remove_static=False, method="capon"
    )
    dets = detect_targets_rae(
        rae[0], meta['range_axis'], meta['angles_az_deg'], meta['angles_el_deg'],
        threshold_factor=6.0
    )
    assert len(dets) == 2, f"expected 2 detections, got {len(dets)}"
    az_found = sorted(d['azimuth_deg'] for d in dets)
    assert abs(az_found[0] - (-25.0)) <= 4.0, f"az1 {az_found[0]} != -25"
    assert abs(az_found[1] - 25.0) <= 4.0, f"az2 {az_found[1]} != 25"
    for d in dets:
        assert abs(d['range_m'] - 25 * RANGE_STEP_M) <= 1.5 * RANGE_STEP_M


def test_detect_noise_only():
    """Pure noise -> 0 detections with a reasonable threshold (false alarm check)."""
    R = 64
    params = make_params(R)
    cube = make_synthetic_cube([], params, F=1, R=R)

    rae, meta = compute_range_azimuth_elevation_map(
        cube, params, remove_static=False, method="capon"
    )
    dets = detect_targets_rae(
        rae[0], meta['range_axis'], meta['angles_az_deg'], meta['angles_el_deg'],
        threshold_factor=6.0
    )
    assert len(dets) == 0, f"expected 0 detections in noise, got {len(dets)}"


def test_detect_edge_of_grid():
    """Target near the azimuth grid edge (+88 deg) and range edge -> no crash."""
    R = 64
    params = make_params(R)
    targets = [
        {'r_bin': 2, 'az_deg': 88.0, 'el_deg': 0.0, 'amp': 5.0},
        {'r_bin': 62, 'az_deg': -88.0, 'el_deg': 0.0, 'amp': 5.0},
    ]
    cube = make_synthetic_cube(targets, params, F=1, R=R)

    rae, meta = compute_range_azimuth_elevation_map(
        cube, params, remove_static=False, method="capon"
    )
    dets = detect_targets_rae(
        rae[0], meta['range_axis'], meta['angles_az_deg'], meta['angles_el_deg'],
        threshold_factor=6.0
    )
    assert len(dets) >= 1, "expected at least one detection near grid edges"
    for d in dets:
        assert 0 <= d['range_idx'] < R
        assert 0 <= d['az_idx'] < meta['angles_az_deg'].size
        assert 0 <= d['el_idx'] < meta['angles_el_deg'].size


def test_consistency_with_range_azimuth_map():
    """Max-over-elevation projection of the RAE map matches the Capon R-A map peak."""
    R = 64
    params = make_params(R)
    tgt = {'r_bin': 30, 'az_deg': 20.0, 'el_deg': 0.0, 'amp': 5.0}
    cube = make_synthetic_cube([tgt], params, F=1, R=R)

    rae, meta = compute_range_azimuth_elevation_map(
        cube, params, remove_static=False, method="capon"
    )
    proj = rae[0].max(axis=1)  # (R, AZ) - max over elevation
    r1, a1 = np.unravel_index(np.argmax(proj), proj.shape)

    ra, meta2 = compute_range_azimuth_map(
        cube, params, remove_static=False, method="capon",
        angles_deg=np.arange(-90.0, 91.0, 2.0)
    )
    r2, a2 = np.unravel_index(np.argmax(ra[0]), ra[0].shape)

    assert abs(int(r1) - int(r2)) <= 1, f"range peak mismatch: {r1} vs {r2}"
    az1 = meta['angles_az_deg'][a1]
    az2 = meta2['angles_deg'][a2]
    assert abs(az1 - az2) <= 4.0, f"azimuth peak mismatch: {az1} vs {az2}"


def test_cfar_vectorized_basic():
    """cfar_range_1d_vectorized flags an isolated strong cell and nothing else."""
    rng = np.random.default_rng(3)
    block = 1.0 + 0.1 * rng.standard_normal((64, 5, 7))
    block[30, 2, 3] = 50.0
    mask = cfar_range_1d_vectorized(block, guard=2, train=8, factor=6.0)
    assert mask[30, 2, 3], "strong cell not detected"
    assert np.count_nonzero(mask) == 1, f"false alarms: {np.count_nonzero(mask) - 1}"


def test_angle_accuracies():
    """Verify that targets at specified angles (az 0, +-20, +-45 and el +-15) land within 2 to 4 deg tolerance."""
    R = 64
    params = make_params(R)
    
    # Test cases: (azimuth, elevation)
    cases = [
        (0.0, 0.0),
        (20.0, 0.0),
        (-20.0, 0.0),
        (45.0, 0.0),
        (-45.0, 0.0),
        (0.0, 15.0),
        (0.0, -15.0),
    ]
    
    for az_tgt, el_tgt in cases:
        tgt = {'r_bin': 30, 'az_deg': az_tgt, 'el_deg': el_tgt, 'amp': 8.0}
        cube = make_synthetic_cube([tgt], params, R=R)
        
        rae, meta = compute_range_azimuth_elevation_map(
            cube, params, remove_static=False, method="capon"
        )
        
        az_axis = meta['angles_az_deg']
        el_axis = meta['angles_el_deg']
        
        # Max index in the map
        r_i, el_i, az_i = np.unravel_index(np.argmax(rae[0]), rae[0].shape)
        
        az_peak = az_axis[az_i]
        el_peak = el_axis[el_i]
        
        az_err = abs(az_peak - az_tgt)
        el_err = abs(el_peak - el_tgt)
        
        # Verify tolerance of 4.0 degrees max (since angular step is 2 deg, peak search can be off by max 2-4 deg)
        assert az_err <= 4.0, f"Azimuth peak error too high for case (az={az_tgt}, el={el_tgt}): got {az_peak}, err {az_err}"
        assert el_err <= 4.0, f"Elevation peak error too high for case (az={az_tgt}, el={el_tgt}): got {el_peak}, err {el_err}"


def test_fftshift_disabled_or_ignored():
    """Verify that even if 'aoa_shift_spectrum' is set to True, the physical azimuth peak is still correct."""
    R = 64
    params = make_params(R)
    params['aoa_shift_spectrum'] = True # Try to enable it!
    
    tgt = {'r_bin': 30, 'az_deg': 20.0, 'el_deg': 0.0, 'amp': 5.0}
    cube = make_synthetic_cube([tgt], params, R=R)
    
    rae, meta = compute_range_azimuth_elevation_map(
        cube, params, remove_static=False, method="capon"
    )
    
    az_axis = meta['angles_az_deg']
    _, _, az_i = np.unravel_index(np.argmax(rae[0]), rae[0].shape)
    az_peak = az_axis[az_i]
    
    # Peak must still be at +20 deg (within 4 deg tolerance)
    assert abs(az_peak - 20.0) <= 4.0, f"fftshift altered physical axis! Peak at {az_peak} instead of 20"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main():
    tests = [
        test_single_target_capon,
        test_bartlett_same_peak,
        test_frame_config_64_loops,
        test_snapshot_deficit_fallback,
        test_remove_static,
        test_range_crop,
        test_ula_no_elevation,
        test_detect_single_target,
        test_detect_two_targets_same_range,
        test_detect_noise_only,
        test_detect_edge_of_grid,
        test_consistency_with_range_azimuth_map,
        test_cfar_vectorized_basic,
        test_angle_accuracies,
        test_fftshift_disabled_or_ignored,
    ]
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"[PASS] {name}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] {name}: {type(e).__name__}: {e}")
    print("-" * 60)
    print(f"{len(tests) - failed}/{len(tests)} tests passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
