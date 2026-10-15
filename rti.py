"""
rti.py

Compatibility re-export. The canonical Range-Time-Intensity implementation
lives in radar_processing.py; this module used to carry a byte-for-byte
equivalent second copy, so a fix applied to one of them silently missed the
other (the GUI imports this one).
"""
from radar_processing import compute_rti

__all__ = ["compute_rti"]
