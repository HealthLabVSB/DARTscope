"""
Tests for playback.next_play_frame: the global Play follows the wall clock and skips the
frames the views cannot draw in time.

Run with:  python -m pytest test_playback.py -q
"""
from playback import next_play_frame


def test_frames_follow_the_clock_and_slow_views_skip():
    clock, f = None, 100
    f, clock = next_play_frame(clock, f, 2999, 10.0, 20.0)       # starts: the next frame
    assert f == 101
    f, clock = next_play_frame(clock, f, 2999, 10.14, 20.0)      # a view took 140 ms: 7 frames on
    assert f == 107
    f, clock = next_play_frame(clock, f, 2999, 10.145, 20.0)     # never stands still
    assert f == 108


def test_a_moved_slider_and_the_end_start_the_clock_over():
    f, clock = next_play_frame(None, 10, 2999, 0.0, 20.0)
    f, clock = next_play_frame(clock, 500, 2999, 5.0, 20.0)      # the slider was dragged to 500
    assert f == 501
    f, clock = next_play_frame(clock, 2998, 2999, 6.0, 20.0)     # (moved again) near the end
    f, clock = next_play_frame(clock, f, 2999, 6.5, 20.0)
    assert f == 0 and clock[1] == 0                              # from the start again


def test_the_first_run_draws_every_shown_frame_the_next_ones_go_in_real_time():
    drawn = set()
    clock, f, t = None, 0, 0.0
    shown = []
    for _ in range(6):                                           # first run: nothing kept yet
        f, clock = next_play_frame(clock, f, 99, t, 20.0, step=2, cached=lambda k: k in drawn)
        drawn.add(f)
        shown.append(f)
        t += 0.1                                                 # a view takes 100 ms to draw
    assert shown == [2, 4, 6, 8, 10, 12]                         # every 2nd frame, none skipped
    clock, f, t = None, 0, 10.0
    f, clock = next_play_frame(clock, f, 99, t, 20.0, step=2, cached=lambda k: k in drawn)
    assert f == 2
    f, clock = next_play_frame(clock, f, 99, t + 0.2, 20.0, step=2, cached=lambda k: k in drawn)
    assert f == 10                                               # 200 ms later: frame 10, in time


def test_one_frame_is_shown_of_every_so_many():
    from playback import frame_step
    assert frame_step(20.0) == 2 and frame_step(50.0) == 1 and frame_step(10.0) == 4


def test_the_meter_tells_frames_shown_and_the_speed_against_real_time():
    from playback import Meter
    m = Meter()
    for k in range(11):                                          # 10 shown a second, every 2nd
        m.add(k * 0.1, 2 * k)                                    # frame of a 50 fps take
    fps, speed = m.rates(20.0)
    assert abs(fps - 10.0) < 1e-9 and abs(speed - 0.4) < 1e-9    # 0.4x real time
    m.add(1.1, 0)                                                # from the start again
    assert m.rates(20.0) is None
