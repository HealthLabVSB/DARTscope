"""
Tests for capture_params.segment_frames - the time segment of a capture (roadmap 9.6).

Run with:  python -m pytest test_segment.py -q
"""
import pytest

from capture_params import segment_frames


def test_whole_capture_by_default():
    assert segment_frames(18000, 100.0) == (0, 18000)


def test_seconds_become_frames():
    assert segment_frames(18000, 100.0, 30, 90) == (3000, 9000)
    assert segment_frames(18000, 100.0, 30, None) == (3000, 18000)
    assert segment_frames(18000, 100.0, None, 12.5) == (0, 1250)
    assert segment_frames(1200, 20.0, 0.024, 0.126) == (0, 3)      # rounded to frames


def test_clamped_to_the_capture():
    assert segment_frames(6000, 100.0, 50, 500) == (5000, 6000)


@pytest.mark.parametrize("t_from, t_to", [(90, 30), (30, 30), (-1, 10), (70, 80), (59.999, None)])
def test_bad_segments_say_why(t_from, t_to):
    with pytest.raises(ValueError):
        segment_frames(6000, 100.0, t_from, t_to)


def test_no_frame_rate_is_an_error():
    with pytest.raises(ValueError):
        segment_frames(100, 0.0, 1, 2)
