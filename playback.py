"""
playback.py

The pace of the global Play: which frame comes next. Kept out of DCA_processing, which
builds its Tk root at import, so it can be tested.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

DISPLAY_FPS = 25.0             # frames shown a second at most: every 2nd of a 50 fps take


def frame_step(interval_ms: float) -> int:
    """Every how many frames of a recording with this frame period one is shown."""
    fps = 1000.0 / max(1.0, float(interval_ms))
    return max(1, int(round(fps / DISPLAY_FPS)))


def next_play_frame(clock, current: int, max_frame: int, now: float, interval_ms: float,
                    step: int = 1, cached=None):
    """
    The frame playback shows next, and its clock (t0, frame at t0, frame last shown).

    Without cached: the frames follow the wall clock - one every interval_ms - and those
    the view cannot draw in time are skipped (a cascade take has a frame every 20 ms, a
    view needs 70-140 ms to draw one: frame by frame the playback ran 4-7 times slower
    than the recording).

    With cached(frame) - whether the view has a picture of it (frame_cache): of the frames
    on the grid of step, the latest one due by the clock that has one; when none has, the
    next on the grid, drawn while the clock stands still - the first run goes at the pace
    of drawing and keeps every frame it shows, the next ones in real time.

    A clock that is None or whose last frame is not the current one (the slider was
    moved) starts over from here; after the last frame the first.
    """
    if clock is None or clock[2] != current:
        clock = (now, current, current)
    t0, f0, _ = clock
    due = f0 + int((now - t0) * 1000.0 / max(1.0, float(interval_ms)) + 1e-6)
    if cached is None:
        nxt = max(current + 1, due)
    else:
        step = max(1, int(step))
        ahead = (current // step + 1) * step
        nxt = None
        f = (min(due, max_frame) // step) * step
        while f >= ahead:
            if cached(f):
                nxt = f
                break
            f -= step
        if nxt is None:
            nxt = ahead
            if not (nxt <= max_frame and cached(nxt)):
                clock = (now, nxt, nxt)       # to be drawn: time stands still meanwhile
    if nxt > max_frame:                       # from the start again
        nxt, clock = 0, (now, 0, 0)
    return nxt, (clock[0], clock[1], nxt)


class Meter:
    """How fast playback really goes, over the last WINDOW_S: frames shown a second, and
    the recording's time a second of the clock (1.0 = real time)."""

    WINDOW_S = 2.0

    def __init__(self):
        self.rows = []                        # (clock s, frame)

    def reset(self):
        self.rows = []

    def add(self, now: float, frame: int):
        if self.rows and frame < self.rows[-1][1]:     # from the start again
            self.rows = []
        self.rows.append((float(now), int(frame)))
        while len(self.rows) > 2 and now - self.rows[1][0] >= self.WINDOW_S:
            self.rows.pop(0)

    def rates(self, interval_ms: float):
        """(frames shown a second, speed against real time), or None before two frames."""
        if len(self.rows) < 2:
            return None
        (t0, f0), (t1, f1) = self.rows[0], self.rows[-1]
        dt = t1 - t0
        if dt <= 0:
            return None
        return (len(self.rows) - 1) / dt, (f1 - f0) * float(interval_ms) / 1000.0 / dt
