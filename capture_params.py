"""
capture_params.py

Quantities derived from a parsed capture that more than one view needs to agree on.

These used to live inside DCA_processing.py, next to the GUI. That file builds its
Tk root at import time, so nothing outside the GUI could reach them: the command-line
tools, the tests and any other script had to re-derive the frame rate on their own,
which is exactly the kind of drift derive_fps was written to stop. Nothing here
touches Tk; the GUI passes a callback where it wants to show something.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

from typing import Callable

DEFAULT_FPS_FALLBACK = 15.0   # only used when the capture metadata has no frame period
C_LIGHT = 299_792_458.0       # m/s, exact


def wavelength_m(params: dict, default_ghz: float | None = None) -> float | None:
    """Wavelength that turns the phase of a range bin into range (roadmap 1.9).

    The frequency in the middle of the ADC window, f_eff + slope * (N - 1) / (2 fs): the
    phase of a range bin follows the beat frequency over the whole window, and referred to
    the chirp centre (spectral.centre_phase) it belongs to the middle of it. The frequency
    at the start of sampling (effectiveStartFreqGHz) gives displacements and velocities
    too large by half the sampled sweep over the frequency: 1.8 to 2.4 % on the lab's
    IWR6843 captures (256 samples, 90 MHz/us, 10 Msps: 1.90 %; 80 samples, 97 MHz/us,
    2.6 Msps: 2.43 %). Falls back to the start of sampling when the chirp is not fully
    known, and to default_ghz (or None) when no frequency is known at all.
    """
    f0 = params.get('effectiveStartFreqGHz') or params.get('startFreqGHz') \
        or params.get('startFreq_GHz') or default_ghz
    try:
        f0 = float(f0) * 1e9
    except (TypeError, ValueError):
        return None
    if not f0 > 0:
        return None
    try:
        slope = float(params.get('freqSlopeMHz_us')) * 1e12
        fs = float(params.get('digOutSampleRate_MHz')) * 1e6
        n = int(params.get('numAdcSamples'))
        if fs > 0 and n > 0:
            f_mid = f0 + slope * (n - 1) / (2.0 * fs)
            if f_mid > 0:
                return C_LIGHT / f_mid
    except (TypeError, ValueError):
        pass
    return C_LIGHT / f0


def derive_period_ms(params: dict) -> float | None:
    """Frame period in milliseconds, or None when the log carries none."""
    try:
        # Preferred: already selected value
        val = params.get('framePeriodicity_ms_selected')
        if isinstance(val, (int, float)) and val > 0:
            # A selected value of e.g. 2000 ms next to frameConfig ticks of 2 000 000
            # (= 10 ms at 5 ns per tick) is a unit slip, not a 2 s frame: prefer the
            # ticks when the two differ by more than an order of magnitude.
            fc = params.get('frameConfig') or {}
            ticks = fc.get('framePeriodicity_ticks')
            if isinstance(ticks, (int, float)) and ticks > 0:
                calc_ms = (ticks * 5.0) / 1000000.0
                if val > 100 and calc_ms < 100 and (val / calc_ms) > 10:
                    return float(calc_ms)
            return float(val)
        # Fallback: frameConfig.framePeriodicity_ms (if exposed)
        fc = params.get('frameConfig') or {}
        val = fc.get('framePeriodicity_ms')
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
        # Fallback: raw ticks in 5 ns units if present
        ticks = fc.get('framePeriodicity_ticks')
        if isinstance(ticks, (int, float)) and ticks > 0:
            return (ticks * 5.0) / 1000000.0
    except Exception:
        pass
    return None


def derive_fps(params: dict, context: str = "",
               on_fallback: Callable[[str], None] | None = None) -> float:
    """
    Frame rate of the capture, from a single place.

    Every band-pass filter for breathing and heart rate is normalised by fps/2, so a
    wrong value shifts all of them. The fallbacks used to be scattered and
    inconsistent (15 fps in the Vital Signs path, 20 fps in the AI-vital path), which
    made the two tabs disagree on the same recording for a reason unrelated to the
    signal. When the fallback is taken it is reported instead of being silent -
    printed, and handed to `on_fallback` so a GUI can show it too.
    """
    try:
        period_ms = derive_period_ms(params or {})
        if period_ms and float(period_ms) > 0:
            return 1000.0 / float(period_ms)
    except Exception as e:
        print(f"[fps] Could not derive frame period{' for ' + context if context else ''}: {e}")
    msg = (f"No frame period in capture metadata"
           f"{' (' + context + ')' if context else ''} - assuming "
           f"{DEFAULT_FPS_FALLBACK:g} fps; breathing/heart rates will be scaled wrong "
           f"if the real frame rate differs.")
    print(f"[fps] WARNING: {msg}")
    if on_fallback is not None:
        try:
            on_fallback(msg)
        except Exception:
            pass
    return DEFAULT_FPS_FALLBACK


def _tx_names(params: dict) -> list:
    """TX name per chirp of the unique set, from the log or from chirps[].txActive."""
    names = params.get("tx_order_names") or []
    if not names and isinstance(params.get("chirps"), list):
        names = [(ch.get("txActive") or "None") for ch in params["chirps"]]
    return list(names)


def tx_selection_options(params: dict):
    """
    Choices for picking one transmitter's chirps, in the order the log gives them.

    Returns (['All', 'TX:TX0', 'TX:TX1', ...], default_index).
    """
    try:
        seen, ordered_unique = set(), []
        for n in _tx_names(params):
            if n and n != "None" and n not in seen:
                seen.add(n)
                ordered_unique.append(n)
        if not ordered_unique:
            return (["All"], 0)
        return (["All"] + [f"TX:{n}" for n in ordered_unique], 0)
    except Exception:
        return (["All"], 0)


def selected_chirp_positions(params: dict, selection: str):
    """
    Chirp positions within one unique set that belong to the selected transmitter.

    Input: selection = "All" or "TX:TX0"/"TX:TX1"/...
    Returns: None (for All) or {'positions': [...], 'pattern_len': eff_unique}
    """
    try:
        if not selection or selection == "All" or not selection.startswith("TX:"):
            return None
        txname = selection.split("TX:", 1)[1].strip()
        names = _tx_names(params)
        if not names:
            return None
        positions = [i for i, n in enumerate(names) if n == txname]
        if not positions:
            return None
        return {"positions": positions, "pattern_len": len(names)}
    except Exception:
        return None


def chirp_period_s(params: dict) -> float:
    """Time from one chirp to the next: idle time + ramp end time (0 when unknown)."""
    try:
        idle = float(params.get('idleTime_us') or 0.0)
        ramp = float(params.get('rampEndTime_us') or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, (idle + ramp) * 1e-6)


def frame_period_s(params: dict, fallback_s: float | None = None) -> float | None:
    """Frame period in seconds from the log, or `fallback_s` when it has none."""
    ms = derive_period_ms(params or {})
    return float(ms) / 1000.0 if ms else fallback_s


def slow_time_axis(params: dict, n_frames: int, n_loops: int, interval_s: float,
                   offset_s: float = 0.0, t0_s: float = 0.0):
    """
    Time in seconds of every slow-time sample, shaped (frames, loops) (roadmap 1.11).

    Sample l of frame f was taken at t0 + f * T_frame + offset + l * interval: frames
    start one frame period apart, and inside a frame the samples of one transmitter
    follow each other every `interval_s` (radar_processing.doppler_slow_time gives it,
    with the chirp positions `offset_s` is taken from). t0_s is the time of frame 0 -
    the start of a segment, so times stay in recording time.

    Without a frame period in the log the frames are assumed back to back, and that
    is printed: a time axis that is silently wrong would scale every rate read off it.
    """
    import numpy as np

    n_frames, n_loops = int(n_frames), int(n_loops)
    period = frame_period_s(params)
    if not period or period <= 0:
        period = n_loops * float(interval_s)
        print(f"[time] No frame period in the log; assuming frames back to back "
              f"({period * 1e3:.2f} ms).")
    f = np.arange(n_frames, dtype=float)[:, None]
    l = np.arange(n_loops, dtype=float)[None, :]
    return float(t0_s) + f * period + float(offset_s) + l * float(interval_s)


def chirp_times(params: dict, n_frames: int, n_chirps: int, t0_s: float = 0.0):
    """
    Time in seconds of every chirp of the cube, shaped (frames, chirps) (roadmap 1.11).

    With a transmitter chosen on the Chirp selector the cube holds only that
    transmitter's chirps (params['cube_chirp_filtered']); their times are those of the
    positions they had in the full pattern.
    """
    import numpy as np

    tc = chirp_period_s(params)
    period = frame_period_s(params) or (int(n_chirps) * tc)
    c = np.arange(int(n_chirps))
    sel = params.get('rd_selected_positions') if params.get('cube_chirp_filtered') else None
    if isinstance(sel, list) and sel:
        pattern = int(params.get('rd_pattern_len_original') or len(sel))
        slot = (c // len(sel)) * pattern + np.asarray(sel)[c % len(sel)]
    else:
        slot = c
    f = np.arange(int(n_frames), dtype=float)[:, None]
    return float(t0_s) + f * period + slot[None, :] * tc


def segment_frames(n_frames: int, fps: float, t_from=None, t_to=None) -> tuple:
    """
    Frames [f0, f1) of the time segment t_from..t_to seconds (roadmap 9.6).

    None for either end means the start or the end of the capture. The bounds are
    clamped to the capture; a segment that is reversed, negative, or holds fewer than
    two frames raises ValueError saying why. Times are capture time: 0 is the first
    frame recorded.
    """
    n = int(n_frames)
    if n <= 0 or not fps or fps <= 0:
        raise ValueError("no frames or no frame rate to cut a segment from")
    for t in (t_from, t_to):
        if t is not None and float(t) < 0:
            raise ValueError(f"segment time {t} s is negative")
    if t_from is not None and t_to is not None and float(t_to) <= float(t_from):
        raise ValueError(f"segment {t_from}-{t_to} s ends before it starts")
    f0 = 0 if t_from is None else int(round(float(t_from) * fps))
    f1 = n if t_to is None else int(round(float(t_to) * fps))
    f0, f1 = max(0, min(f0, n)), max(0, min(f1, n))
    if f1 - f0 < 2:
        raise ValueError(f"segment {t_from}-{t_to} s holds {max(0, f1 - f0)} frame(s) of "
                         f"this {n / fps:.1f} s capture")
    return f0, f1


# Names the GUI has always used; kept so existing call sites read unchanged.
_derive_period_ms = derive_period_ms
_tx_selection_options = tx_selection_options
_selected_chirp_positions = selected_chirp_positions
