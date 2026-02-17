import numpy as np
from range_fft import compute_range_fft_cube

def compute_rti(cube: np.ndarray, params: dict, n_fft_range: int = 256):
    """
    Computes Range-Time Intensity (RTI) map.
    Performs Range FFT and averages magnitude over RX channels.
    Uses full spectrum for complex input (I/Q), half for real input.
    Applies a notch filter to DC and edge bins.
    """
    is_complex_input = np.iscomplexobj(cube)
    fmt_str = str(params.get('dataFmt_adcFmt', '')).upper()
    is_c2x = ('COMPLEX2X' in fmt_str)
    is_pseudo = ('PSEUDOREAL' in fmt_str)

    range_fft_mag, meta_fft = compute_range_fft_cube(
        cube,
        params,
        n_fft_range=n_fft_range,
        remove_dc=True,
        window='hann',
        use_full_spectrum=(is_complex_input and not (is_c2x or is_pseudo))
    )
    num_frames, num_chirps_per_frame, num_rx, num_range_bins = range_fft_mag.shape
    total_chirps = num_frames * num_chirps_per_frame
    rti_cube = range_fft_mag.reshape(total_chirps, num_rx, num_range_bins)
    rti_map = rti_cube.mean(axis=1)

    if num_range_bins > 4:
        clean_region = rti_map[:, 10:num_range_bins//2]
        if clean_region.size > 0:
            min_val = np.percentile(clean_region, 1)
        else:
            min_val = np.percentile(rti_map, 1)
        rti_map[:, :2] = min_val
        rti_map[:, -2:] = min_val

    meta = {
        'range_m': meta_fft['range_m'],
        'chirp_index': np.arange(total_chirps),
        'total_chirps': total_chirps,
        'range_bins': num_range_bins,
        'use_full_spectrum': bool(meta_fft.get('use_full_spectrum')),
        'n_fft_range': int(meta_fft.get('n_fft_range') or n_fft_range),
        'is_complex_input': bool(is_complex_input),
    }
    return rti_map, meta

