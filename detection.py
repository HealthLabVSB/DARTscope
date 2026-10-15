"""
detection.py

Target detection algorithms for radar processing.
Implements CFAR (Constant False Alarm Rate) and other detection methods for Range-Doppler maps.

Includes:
- 2D CA-CFAR (Cell-Averaging CFAR) for Range-Doppler maps
- OS-CFAR (Ordered Statistic CFAR) variant
- Detection clustering and peak extraction

Authors: Daniel Barvik, Dan Hruby, and AI
"""
import numpy as np
from typing import Tuple, List, Dict
from scipy import ndimage

import spectral


def cfar_2d_ca(
    rd_map: np.ndarray,
    guard_cells: int = 2,
    training_cells: int = 8,
    threshold_factor: float = 3.0,
    method: str = 'average',
    guard_cells_range: int | None = None,
    training_cells_range: int | None = None,
) -> np.ndarray:
    """
    2D CA-CFAR (Cell-Averaging CFAR) detector for Range-Doppler map.

    Args:
        rd_map: 2D array (Doppler, Range) of magnitude values
        guard_cells: Number of guard cells around CUT (Cell Under Test)
        training_cells: Number of training cells for noise estimation
        threshold_factor: Multiplier for adaptive threshold (Pfa related)
        method: 'average' (CA-CFAR) or 'ordered' (OS-CFAR, uses median)
        guard_cells_range, training_cells_range: along range, when they differ from
            the Doppler ones - a zero-padded range FFT has more bins per resolution cell

    Returns:
        Boolean array of same shape as rd_map with detections marked as True

    A map with too few Doppler rows for the window (range_only: a few chirps per frame,
    4 with the vital-signs settings) is tested along range only, row by row - the 2D
    window never fitted there and nothing was ever detected.
    """
    rows, cols = rd_map.shape
    detections = np.zeros((rows, cols), dtype=bool)

    g_r = guard_cells if guard_cells_range is None else int(guard_cells_range)
    w_r = g_r + (training_cells if training_cells_range is None else int(training_cells_range))
    # Total window size (guard ring + training ring around the CUT), along range
    if cols <= 2 * w_r:
        return detections
    # along Doppler: the same window, or none at all on a map of a few rows
    window = guard_cells + training_cells
    w_d, g_d = (0, 0) if range_only(rows, guard_cells, training_cells) else (window, guard_cells)

    x = np.asarray(rd_map, dtype=np.float64)
    full_size = (2 * w_d + 1, 2 * w_r + 1)
    guard_size = (2 * g_d + 1, 2 * g_r + 1)

    if method == 'ordered':
        # OS-CFAR: median over the training annulus (square minus guard square).
        footprint = np.ones(full_size, dtype=bool)
        footprint[w_d - g_d:w_d + g_d + 1, w_r - g_r:w_r + g_r + 1] = False
        n_training = int(footprint.sum())
        if n_training % 2:
            noise_level = ndimage.rank_filter(x, n_training // 2, footprint=footprint, mode='nearest')
        else:
            # np.median averages the two central order statistics; rank_filter alone
            # would return only the lower one.
            lo = ndimage.rank_filter(x, n_training // 2 - 1, footprint=footprint, mode='nearest')
            hi = ndimage.rank_filter(x, n_training // 2, footprint=footprint, mode='nearest')
            noise_level = 0.5 * (lo + hi)
    else:
        # CA-CFAR: annulus mean = (sum over full window - sum over guard window) / cell count.
        n_full, n_guard = full_size[0] * full_size[1], guard_size[0] * guard_size[1]
        sum_full = ndimage.uniform_filter(x, size=full_size, mode='nearest') * n_full
        sum_guard = ndimage.uniform_filter(x, size=guard_size, mode='nearest') * n_guard
        n_training = n_full - n_guard
        noise_level = (sum_full - sum_guard) / n_training

    detected = x > noise_level * threshold_factor

    # Only cells whose full training window fits inside the map get a decision,
    # matching the border behaviour of the original implementation.
    detections[w_d:rows - w_d, w_r:cols - w_r] =         detected[w_d:rows - w_d, w_r:cols - w_r]
    return detections


def range_only(doppler_rows: int, guard_cells: int, training_cells: int) -> bool:
    """True when the 2D CFAR window does not fit the Doppler rows: cfar_2d_ca then tests
    along range only."""
    return doppler_rows <= 2 * (guard_cells + training_cells)


def cfar_1d(
    signal: np.ndarray,
    guard_cells: int = 2,
    training_cells: int = 8,
    threshold_factor: float = 3.0
) -> np.ndarray:
    """
    1D CA-CFAR detector for range or Doppler profiles.

    Args:
        signal: 1D array of magnitude values
        guard_cells: Number of guard cells on each side of CUT
        training_cells: Number of training cells on each side
        threshold_factor: Multiplier for adaptive threshold

    Returns:
        Boolean array of same shape as signal with detections marked as True
    """
    n = len(signal)
    detections = np.zeros(n, dtype=bool)
    window = guard_cells + training_cells
    if n <= 2 * window:
        return detections

    x = np.asarray(signal, dtype=np.float64)
    full_size = 2 * window + 1
    guard_size = 2 * guard_cells + 1

    sum_full = ndimage.uniform_filter1d(x, size=full_size, mode='nearest') * full_size
    sum_guard = ndimage.uniform_filter1d(x, size=guard_size, mode='nearest') * guard_size
    noise_level = (sum_full - sum_guard) / (full_size - guard_size)

    detected = x > noise_level * threshold_factor
    detections[window:n - window] = detected[window:n - window]
    return detections


def extract_peaks(
    detections: np.ndarray,
    rd_map: np.ndarray,
    min_distance: int = 3
) -> List[Dict]:
    """
    Extract peak locations and values from detection map.
    Performs clustering and selects local maxima.

    Args:
        detections: Boolean detection map
        rd_map: Original magnitude map
        min_distance: Minimum pixel distance between peaks

    Returns:
        List of dicts with keys: 'doppler_idx', 'range_idx', 'magnitude', and the
        interpolated fractional indices 'doppler_pos', 'range_pos' (spectral.py; equal
        to the integer indices when interpolation is off). rd_map must be linear
        magnitude or power, not dB.
    """
    # Label connected components
    labeled, num_features = ndimage.label(detections)

    peaks = []
    for label_idx in range(1, num_features + 1):
        # Get all pixels in this cluster
        cluster_mask = (labeled == label_idx)
        cluster_coords = np.argwhere(cluster_mask)

        # Find local maximum in cluster
        cluster_values = rd_map[cluster_mask]
        max_idx = np.argmax(cluster_values)
        peak_coord = cluster_coords[max_idx]
        d_pos, r_pos = spectral.refine_peak_nd(rd_map, peak_coord)

        peaks.append({
            'doppler_idx': int(peak_coord[0]),
            'range_idx': int(peak_coord[1]),
            'doppler_pos': d_pos,
            'range_pos': r_pos,
            'magnitude': float(rd_map[peak_coord[0], peak_coord[1]])
        })

    return peaks


def detections_to_range_velocity(
    peaks: List[Dict],
    range_axis: np.ndarray,
    velocity_axis: np.ndarray
) -> List[Dict]:
    """
    Convert detection indices to physical range and velocity values.

    Args:
        peaks: List of detections with 'doppler_idx', 'range_idx', 'magnitude'
        range_axis: Physical range axis in meters
        velocity_axis: Physical velocity axis in m/s

    Returns:
        List of dicts with added 'range_m' and 'velocity_ms' keys
    """
    for peak in peaks:
        r_idx = peak['range_idx']
        d_idx = peak['doppler_idx']

        if 0 <= r_idx < len(range_axis):
            peak['range_m'] = spectral.interp_axis(range_axis, peak.get('range_pos', r_idx))
        else:
            peak['range_m'] = None

        if 0 <= d_idx < len(velocity_axis):
            peak['velocity_ms'] = spectral.interp_axis(velocity_axis, peak.get('doppler_pos', d_idx))
        else:
            peak['velocity_ms'] = None

    return peaks


def compute_snr_db(magnitude: float, noise_level: float) -> float:
    """Calculate SNR in dB."""
    if noise_level > 0:
        return 20 * np.log10(magnitude / noise_level)
    return 0.0


def cfar_range_1d_vectorized(
    rae_lin: np.ndarray,
    guard: int = 2,
    train: int = 8,
    factor: float = 6.0,
    return_noise: bool = False,
):
    """
    Vectorized CA-CFAR along the range axis (axis 0) for a whole (R, EL, AZ) block.

    Uses scipy.ndimage.uniform_filter1d: the noise estimate is the mean of the
    training cells (full window minus guard window) around each cell under test.

    Args:
        rae_lin: array (R, ...) of LINEAR power values (not dB)
        guard: number of guard cells on each side of CUT
        train: number of training cells on each side
        factor: threshold multiplier (Pfa related)
        return_noise: if True, also return the noise estimate array

    Returns:
        Boolean detection mask of the same shape as rae_lin
        (and the noise estimate array if return_noise=True).
    """
    rae_lin = np.asarray(rae_lin, dtype=np.float64)
    R = rae_lin.shape[0]
    guard = max(0, int(guard))
    train = max(1, int(train))

    win_total = 2 * (guard + train) + 1
    win_guard = 2 * guard + 1
    if win_total >= 2 * R:
        # Degenerate case: too few range bins -> global mean as noise estimate
        noise = np.mean(rae_lin, axis=0, keepdims=True) * np.ones_like(rae_lin)
    else:
        total_sum = ndimage.uniform_filter1d(rae_lin, size=win_total, axis=0, mode='reflect') * win_total
        inner_sum = ndimage.uniform_filter1d(rae_lin, size=win_guard, axis=0, mode='reflect') * win_guard
        noise = (total_sum - inner_sum) / (2 * train)

    noise = np.maximum(noise, 1e-20)
    detections = rae_lin > noise * factor
    if return_noise:
        return detections, noise
    return detections


def detect_targets_rae(
    rae_db: np.ndarray,
    range_axis: np.ndarray,
    angles_az_deg: np.ndarray,
    angles_el_deg: np.ndarray,
    guard: int = 2,
    train: int = 8,
    threshold_factor: float = 6.0,
    min_separation: Tuple[int, int, int] = (2, 2, 2),
    angular_rel_threshold_db: float = 15.0,
) -> List[Dict]:
    """
    Target detection over a Range-Azimuth-Elevation map (single frame).

    Pipeline:
        1. dB -> linear power conversion,
        2. CA-CFAR mask along the range axis (vectorized),
        3. AND with local maxima in the Az-El plane (maximum_filter, size=(1,3,3)),
        4. clustering via ndimage.label + cluster peak selection,
        5. mapping to physical values incl. Cartesian coordinates.

    Args:
        rae_db: array (R, EL, AZ) in dB (one frame of the RAE map)
        range_axis: physical range axis [m], length R
        angles_az_deg: azimuth axis [deg], length AZ
        angles_el_deg: elevation axis [deg], length EL
        guard/train/threshold_factor: range-CFAR parameters
        min_separation: minimum index distance (range, el, az) between targets
        angular_rel_threshold_db: reject angular peaks more than this many dB below
            the strongest peak at the same range bin (sidelobe rejection)

    Returns:
        List of dicts: {'range_idx', 'az_idx', 'el_idx', 'range_m', 'azimuth_deg',
                        'elevation_deg', 'power_db', 'snr_db', 'x_m', 'y_m', 'z_m'}
    """
    rae_db = np.asarray(rae_db)
    if rae_db.ndim != 3:
        raise ValueError("rae_db must have shape (R, EL, AZ)")

    range_axis = np.asarray(range_axis, dtype=float)
    angles_az_deg = np.asarray(angles_az_deg, dtype=float)
    angles_el_deg = np.asarray(angles_el_deg, dtype=float)

    # 1. dB -> linear power (CFAR must operate on linear scale)
    rae_lin = 10.0 ** (rae_db / 10.0)

    # 2. Range CFAR mask + noise estimate
    cfar_mask, noise = cfar_range_1d_vectorized(
        rae_lin, guard=guard, train=train, factor=threshold_factor, return_noise=True
    )

    # 3. Local maxima in the Az-El plane
    local_max = rae_lin >= ndimage.maximum_filter(rae_lin, size=(1, 3, 3), mode='nearest')
    detections = cfar_mask & local_max

    if not np.any(detections):
        return []

    # 4. Cluster detections (26-connectivity) and pick the peak of each cluster
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, num_features = ndimage.label(detections, structure=structure)

    raw_peaks = []
    for label_idx in range(1, num_features + 1):
        cluster_mask = (labeled == label_idx)
        cluster_coords = np.argwhere(cluster_mask)
        cluster_values = rae_lin[cluster_mask]
        max_idx = int(np.argmax(cluster_values))
        r_i, el_i, az_i = (int(v) for v in cluster_coords[max_idx])
        raw_peaks.append((float(rae_lin[r_i, el_i, az_i]), r_i, el_i, az_i))

    # Sidelobe rejection: drop peaks far below the strongest peak of the same range bin
    if angular_rel_threshold_db is not None and angular_rel_threshold_db > 0:
        rel_lin = 10.0 ** (-float(angular_rel_threshold_db) / 10.0)
        raw_peaks = [
            p for p in raw_peaks
            if p[0] >= float(np.max(rae_lin[p[1]])) * rel_lin
        ]

    # Greedy suppression of peaks closer than min_separation (strongest first)
    raw_peaks.sort(key=lambda p: p[0], reverse=True)
    sep_r, sep_el, sep_az = (max(0, int(v)) for v in min_separation)
    kept = []
    for pwr, r_i, el_i, az_i in raw_peaks:
        too_close = False
        for _, kr, kel, kaz in kept:
            if (abs(r_i - kr) <= sep_r and abs(el_i - kel) <= sep_el
                    and abs(az_i - kaz) <= sep_az):
                too_close = True
                break
        if not too_close:
            kept.append((pwr, r_i, el_i, az_i))

    # 5. Map to physical values
    targets = []
    for pwr, r_i, el_i, az_i in kept:
        # Between the cells: the range, elevation and azimuth grids are coarse compared
        # with how well a strong target can be located (spectral.py).
        r_p, el_p, az_p = spectral.refine_peak_nd(rae_lin, (r_i, el_i, az_i))
        r_m = spectral.interp_axis(range_axis, r_p) if r_i < range_axis.size else float(r_i)
        az_deg = spectral.interp_axis(angles_az_deg, az_p) if az_i < angles_az_deg.size else 0.0
        el_deg = spectral.interp_axis(angles_el_deg, el_p) if el_i < angles_el_deg.size else 0.0
        az = np.deg2rad(az_deg)
        el = np.deg2rad(el_deg)

        noise_lvl = float(noise[r_i, el_i, az_i])
        snr_db = 10.0 * np.log10(pwr / noise_lvl) if noise_lvl > 0 else 0.0

        targets.append({
            'range_idx': r_i,
            'az_idx': az_i,
            'el_idx': el_i,
            'range_m': r_m,
            'azimuth_deg': az_deg,
            'elevation_deg': el_deg,
            'power_db': float(rae_db[r_i, el_i, az_i]),
            'snr_db': float(snr_db),
            'x_m': r_m * np.cos(el) * np.sin(az),
            'y_m': r_m * np.cos(el) * np.cos(az),
            'z_m': r_m * np.sin(el),
        })

    return targets


