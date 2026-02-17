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


def cfar_2d_ca(
    rd_map: np.ndarray,
    guard_cells: int = 2,
    training_cells: int = 8,
    threshold_factor: float = 3.0,
    method: str = 'average'
) -> np.ndarray:
    """
    2D CA-CFAR (Cell-Averaging CFAR) detector for Range-Doppler map.

    Args:
        rd_map: 2D array (Doppler, Range) of magnitude values
        guard_cells: Number of guard cells around CUT (Cell Under Test)
        training_cells: Number of training cells for noise estimation
        threshold_factor: Multiplier for adaptive threshold (Pfa related)
        method: 'average' (CA-CFAR) or 'ordered' (OS-CFAR, uses median)

    Returns:
        Boolean array of same shape as rd_map with detections marked as True
    """
    detections = np.zeros_like(rd_map, dtype=bool)
    rows, cols = rd_map.shape

    # Total window size
    window = guard_cells + training_cells

    for r in range(window, rows - window):
        for c in range(window, cols - window):
            # Extract training region (annulus around guard cells)
            training_region = []
            for i in range(-window, window + 1):
                for j in range(-window, window + 1):
                    # Include only training cells (outside guard region)
                    if abs(i) > guard_cells or abs(j) > guard_cells:
                        training_region.append(rd_map[r + i, c + j])

            # Estimate noise level
            if method == 'ordered':
                # OS-CFAR: use median or percentile
                noise_level = np.median(training_region)
            else:
                # CA-CFAR: use mean
                noise_level = np.mean(training_region)

            # Adaptive threshold
            threshold = noise_level * threshold_factor

            # Detection decision
            if rd_map[r, c] > threshold:
                detections[r, c] = True

    return detections


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
    detections = np.zeros_like(signal, dtype=bool)
    n = len(signal)
    window = guard_cells + training_cells

    for i in range(window, n - window):
        # Left training cells
        left_train = signal[i - window : i - guard_cells]
        # Right training cells
        right_train = signal[i + guard_cells + 1 : i + window + 1]

        # Noise estimate
        noise_level = np.mean(np.concatenate([left_train, right_train]))
        threshold = noise_level * threshold_factor

        if signal[i] > threshold:
            detections[i] = True

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
        List of dicts with keys: 'doppler_idx', 'range_idx', 'magnitude'
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

        peaks.append({
            'doppler_idx': int(peak_coord[0]),
            'range_idx': int(peak_coord[1]),
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
            peak['range_m'] = float(range_axis[r_idx])
        else:
            peak['range_m'] = None

        if 0 <= d_idx < len(velocity_axis):
            peak['velocity_ms'] = float(velocity_axis[d_idx])
        else:
            peak['velocity_ms'] = None

    return peaks


def compute_snr_db(magnitude: float, noise_level: float) -> float:
    """Calculate SNR in dB."""
    if noise_level > 0:
        return 20 * np.log10(magnitude / noise_level)
    return 0.0


