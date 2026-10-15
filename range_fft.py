"""
range_fft.py

Compatibility re-export. The canonical Range FFT implementation lives in
radar_processing.py; this module used to carry a second, near-identical copy
whose only real difference was defaulting `use_full_spectrum` to False.

That difference made the app inconsistent with itself: RTI and Range-Doppler
pass `use_full_spectrum` explicitly (full spectrum for Complex1X), while the
AoA and vital-signs call sites omit it and silently got only the positive half,
i.e. half the range span of the very same capture. Keeping one implementation
removes that class of bug for good.

Note: the range step dR is independent of `use_full_spectrum`, so range bin k
means the same distance either way - the full spectrum simply exposes the upper
half of the unambiguous range that complex (I/Q) sampling provides.
"""
from radar_processing import compute_range_fft_cube

__all__ = ["compute_range_fft_cube"]
