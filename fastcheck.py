"""
fastcheck.py

A quick check of a whole recording before it is analysed: is the data complete, does
the signal look sane from the first frame to the last, and is somebody breathing in
front of the radar - one pass over the data, for the Fastcheck tab.

The capture health report (capture_health.py) judges the logs, the files, the start of
the raw data and the first frames. This goes through every frame:

    integrity    frames the recorder filled with zeros because packets were lost: how
                 many, in how many separate dropouts, the longest dropout and the
                 longest clean stretch; ADC clipping; the power of every receiver over
                 time (a receiver that falls silent, a gain step); frames that do not
                 look like the frames around them (a burst of interference, a stream
                 that slipped out of step).
    overview     the range-time intensity (RTI) of the recording, one line per frame,
                 and the same with the static scene removed (motion only).
    vital signs  how much of the slow motion seen in every range bin lies in the
                 breathing band and in the heart band (vital_bands.py); the range with
                 the most breathing; there, the energy of both bands over time, the
                 breathing rate in the windows whose rhythm is clear, and the heart
                 rate as the phase-12 pipeline follows it (moving_hr.heart_spectra and
                 heart_track: a harmonic sum and a Viterbi path - a plain spectral peak
                 missed the somnograph's pulse by 20 bpm). On six seated recordings it
                 was within 4.9 bpm of the pulse on average.

Nothing here touches Tk, so the tab, the tests and a script get the same answer.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field

import numpy as np

import capture_health
import capture_params
import vital_bands
import vitalsigns
from capture_health import Finding
from radar_processing import compute_range_fft_cube, range_fft_size

# Limits. A frame is judged against the recording itself wherever possible, so the
# limits are ratios to what is typical for it, with a floor for very steady captures.
CLIP_LEVEL = 0.98              # of full scale (the decoder scales full scale to 1.0)
CLIP_FRAME = 1e-4              # fraction of a frame's samples at that level: clipped
RX_SILENT_DB = 20.0            # a receiver this far below its own median: silent
JUMP_MIN_DB = 3.0              # a frame's range profile changing this much ...
JUMP_TYPICAL_X = 6.0           # ... and this many times the typical change: a glitch
LEVEL_INFO_DB = 10.0           # signal level spread over the recording worth a note
DAMAGED_ERROR = 0.25           # this fraction of frames zero-filled: hardly usable
RTI_CHIRPS = 16                # chirps per frame averaged for the RTI
CHUNK_SAMPLES = 4_000_000      # complex samples decoded per block
MIN_RANGE_M = 0.2              # nearer than this is antenna coupling, not a person
BREATHING_X = 5.0              # breathing band this many times the noise: slow motion
BREATHING_Q = 6.0              # breathing peak / band average: a clear rhythm (as the
                               # Vitals view's Min peak/avg: clean ~12, noise ~4)
WINDOW_S = 30.0                # windows of the rate tracks, as the Vitals BR window ...
MIN_WINDOW_S = 12.0            # ... or a third of a shorter recording, at least this
STEP_S = 2.0
MIN_VITAL_S = 8.0              # shorter recordings are not judged for vital signs
STEADY_PER_MIN = 2.0           # a rate this close to its neighbours' is steady
IRREGULAR_CV = 0.25            # breath intervals spreading more than this: irregular breathing
RATE_DISAGREE = 0.25           # a clear spectral peak this far above the counted rate: its harmonic


@dataclass
class Fastcheck:
    """Everything the Fastcheck tab shows. Per-frame arrays have one entry per frame."""
    stem: str
    fps: float
    shape: tuple                             # (frames, chirps, RX, samples)
    t_s: np.ndarray                          # frame times [s]
    damaged: np.ndarray                      # frame holds zero-filled packets
    zero_frac: np.ndarray                    # fraction of samples exactly zero
    clip_frac: np.ndarray                    # fraction of samples at full scale
    rx_db: np.ndarray                        # (frames, RX) power per receiver [dBFS]
    jump_db: np.ndarray                      # change of the range profile from the frame before
    glitch: np.ndarray                       # frames that do not look like their neighbours
    range_m: np.ndarray
    rti_db: np.ndarray                       # (frames, range) mean power [dB]
    motion_db: np.ndarray                    # (frames, range) power with the static scene removed
    bands: vital_bands.Bands
    br_ratio: np.ndarray                     # (range,) breathing band power / noise
    hr_ratio: np.ndarray                     # (range,) heart band power / noise
    best_bin: int | None                     # the range with the most breathing
    win_t: np.ndarray                        # centres of the rate windows [s]
    br_db: np.ndarray                        # breathing band power over the noise per window [dB]
    hr_db: np.ndarray                        # heart band power over the noise per window [dB]
    br_rate: np.ndarray                      # breathing rate per window [/min]
    br_q: np.ndarray                         # breathing peak / band average per window
    hr_rate: np.ndarray                      # heart rate per window [bpm] (moving_hr tracker)
    hr_valid: np.ndarray                     # the heart peak stands out enough to count
    br_breaths: np.ndarray = field(default_factory=lambda: np.zeros(0))       # breaths counted per window
    br_breaths_ok: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # >= 2, above the noise
    findings: list = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    # --- summary ---------------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return int(self.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    @property
    def worst(self) -> str:
        sev = [f.severity for f in self.findings] or ["ok"]
        return max(sev, key=capture_health.SEVERITIES.index)

    def clear_rhythm(self) -> np.ndarray:
        """
        Windows with a clear breathing rhythm: a peak standing out of the band (see
        _clear_q) - or, at 60 % of that, a rate within 2 /min of the median of the
        windows around (two either side). Noise gives a different rate in every window,
        breathing the same one; on a recording with 6 % of its frames lost the peak
        alone passed 2 windows of 21 whose rates all read 14 /min like the Vitals view.
        Also every window with two or more breaths counted clearly above the noise
        (vitalsigns.count_breaths): irregular breathing has no peak to stand out - a seated
        minute of 9 breaths passed 2 windows of 21 by the peak, 21 of 21 by its breaths.
        """
        return clear_windows(self.br_rate, self.br_q, self.br_breaths_ok,
                             self.notes.get("clear_q", BREATHING_Q))

    def verdict(self) -> str:
        n_err = sum(f.severity == "error" for f in self.findings)
        n_warn = sum(f.severity == "warning" for f in self.findings)
        if n_err:
            return f"Problems: {n_err} error(s), {n_warn} warning(s) - see the list"
        if n_warn:
            return f"Usable with care: {n_warn} warning(s)"
        return "OK - nothing wrong found"

    def key_numbers(self) -> str:
        f, c, rx, s = self.shape
        missing = 100.0 * int(self.damaged.sum()) / max(f, 1)
        complete = f"{100.0 - missing:.{2 if 0 < missing < 1 else 1}f}"
        parts = [f"{self.duration_s:.1f} s", f"{f} frames at {self.fps:g} fps",
                 f"{c} chirps x {rx} RX x {s} samples", f"complete frames {complete} %"]
        spans = damaged_spans(self.damaged)
        if spans:
            longest = max(b - a for a, b in spans) / self.fps
            parts.append(f"dropouts {len(spans)} (longest {longest:.2f} s)")
        parts.append(f"longest clean stretch {longest_clean(self.damaged) / self.fps:.1f} s")
        return " | ".join(parts)

    def report(self) -> str:
        """The check as plain text (Export > Report)."""
        marks = {"ok": "  ok ", "info": " info", "warning": " WARN", "error": "ERROR"}
        lines = [f"Fastcheck of {self.stem}", self.verdict(), self.key_numbers(), ""]
        width = max((len(f.check) for f in self.findings), default=0)
        for f in sorted(self.findings, key=lambda x: -capture_health.SEVERITIES.index(x.severity)):
            lines.append(f"{marks[f.severity]}  {f.check:<{width}}  {f.message}")
        return "\n".join(lines)

    def frame_table(self):
        """(array, header) of the per-frame figures, for Export > Frames (CSV)."""
        cols = [self.t_s, self.damaged.astype(float), self.zero_frac, self.clip_frac,
                self.jump_db, self.glitch.astype(float)]
        names = ["time_s", "zero_filled", "zero_fraction", "clip_fraction", "jump_db", "glitch"]
        for r in range(self.rx_db.shape[1]):
            cols.append(self.rx_db[:, r])
            names.append(f"rx{r + 1}_dbfs")
        return np.column_stack(cols), ",".join(names)


# --- helpers ---------------------------------------------------------------------------

def _pct(p: float) -> str:
    """A percentage that does not round a small loss away to 0.0."""
    return f"{p:.1f}" if p == 0 or p >= 1 else f"{p:.2g}"


def damaged_spans(mask) -> list:
    """Contiguous [start, end) runs of True."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def longest_clean(mask) -> int:
    """Frames in the longest run without a zero-filled frame."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return 0
    edges = [-1] + np.flatnonzero(m).tolist() + [m.size]
    return int(max(b - a - 1 for a, b in zip(edges, edges[1:])))


def _fill(x: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """x (frames, ...) with the bad frames replaced by a line between their neighbours."""
    if not bad.any() or bad.all():
        return x
    good = np.flatnonzero(~bad)
    idx = np.arange(x.shape[0])
    flat = x.reshape(x.shape[0], -1)
    out = flat.copy()
    for j in range(flat.shape[1]):
        col = flat[good, j]
        if np.iscomplexobj(col):
            out[bad, j] = (np.interp(idx[bad], good, col.real)
                           + 1j * np.interp(idx[bad], good, col.imag))
        else:
            out[bad, j] = np.interp(idx[bad], good, col)
    return out.reshape(x.shape)


def _band_power(p: np.ndarray, freqs: np.ndarray, band) -> np.ndarray:
    sel = (freqs >= band[0]) & (freqs <= band[1])
    if not sel.any():
        return np.full(p.shape[1:], np.nan)
    return p[sel].mean(axis=0)


def _detrend(x: np.ndarray) -> np.ndarray:
    """x (time, ...) minus its straight-line trend: the static scene and a slow drift,
    which would otherwise leak into the bottom of the breathing band."""
    n = x.shape[0]
    t = (np.arange(n) - (n - 1) / 2.0).reshape((n,) + (1,) * (x.ndim - 1))
    slope = (x * t).sum(axis=0) / max(float((t * t).sum()), 1e-12)
    return x - x.mean(axis=0) - slope * t


def _folded_spectrum(x: np.ndarray, fps: float):
    """Power over |frequency| of complex slow-time signals x (time, ...), trend removed."""
    n = x.shape[0]
    w = np.hanning(n).reshape((n,) + (1,) * (x.ndim - 1))
    p = np.abs(np.fft.fft(_detrend(x) * w, axis=0)) ** 2
    half = n // 2
    folded = p[:half + 1].copy()
    folded[1:n - half] += p[:half:-1]                     # add the negative frequencies
    return np.arange(half + 1) * fps / n, folded


def _phase_power(x: np.ndarray, fps: float):
    """
    Power over frequency of the phase of x (time, RX, bins) - the chest displacement the
    Vitals view measures - summed over the receivers.

    The phase is taken around the window's own centre (as the Vitals view's IQ DC
    correction "mean"), unwrapped and detrended. The complex signal itself is no good
    for a rhythm: a breath moves the chest by millimetres, a turn or two of the phase
    at 60 GHz, and that spreads its power over many sidebands.
    """
    n = x.shape[0]
    ph = _detrend(np.unwrap(np.angle(x - x.mean(axis=0)), axis=0))
    w = np.hanning(n).reshape((n,) + (1,) * (ph.ndim - 1))
    p = np.abs(np.fft.rfft(ph * w, axis=0)) ** 2
    return np.fft.rfftfreq(n, 1.0 / fps), p.sum(axis=1)


def _chest_phase(x: np.ndarray, bad: np.ndarray, fps: float) -> np.ndarray:
    """
    Phase of x (time, RX, bins) around its centre, as a displacement in radians:
    unwrapped over the good frames only, the steps a dropout leaves (a turn lost across
    the gap) fitted away as in the phase-12 pipeline (moving_hr.stitch_steps), and the
    lost frames filled in between. Taking the phase of complex values interpolated
    across a dropout instead cut through the circle and scattered the breathing peak.
    """
    import moving_hr
    F = x.shape[0]
    good = ~np.asarray(bad, bool)
    if good.sum() < 3:
        return np.zeros(x.shape)
    z = x - x[good].mean(axis=0)
    flat = np.angle(z).reshape(F, -1)
    out = np.empty(flat.shape)
    idx = np.arange(F)
    no_slips = np.zeros(F, bool)
    for j in range(flat.shape[1]):
        col = np.empty(F)
        col[good] = np.unwrap(flat[good, j])
        col[~good] = np.interp(idx[~good], idx[good], col[good])
        col, _ = moving_hr.stitch_steps(col, fps, ~good, no_slips)
        col[~good] = np.interp(idx[~good], idx[good], col[good])
        out[:, j] = col
    return out.reshape(x.shape)


def _clear_q(n_bins: int) -> float:
    """
    Peak over band average above which a band holds a clear rhythm, for n_bins bins in
    the band: halfway (geometrically) between pure noise, whose largest of n
    exponential bins averages the harmonic number H_n, and a clean sinusoid under a
    Hann window, two thirds of the band's power in one bin. About 6 for the 30 s
    windows the Vitals view uses, as its Min peak/avg; less for shorter windows,
    which have fewer bins to be peaky in.
    """
    n = max(2, int(n_bins))
    noise = float(np.sum(1.0 / np.arange(1, n + 1)))
    return float(max(3.0, np.sqrt(noise * 0.67 * n)))


def _rx_power(slow: np.ndarray, fps: float):
    """Folded power of slow (time, RX, ...) summed over the receivers, one at a time."""
    total = freqs = None
    for r in range(slow.shape[1]):
        freqs, p = _folded_spectrum(slow[:, r], fps)
        total = p if total is None else total + p
    return freqs, total


# --- the check -----------------------------------------------------------------------------

def run(stem: str, params: dict, cube: np.ndarray, fps: float | None = None,
        bin_paths=(), n_fft_range: int | None = None, bands: vital_bands.Bands | None = None,
        progress_cb=None) -> Fastcheck:
    """
    Check a decoded capture, cube (frames, chirps, RX, samples) scaled to full scale 1.

    progress_cb(done, total) is called after every block of frames; it may raise to
    stop the check.
    """
    cube = np.asarray(cube)
    if cube.ndim != 4 or cube.shape[0] == 0:
        raise ValueError(f"expected a cube (frames, chirps, RX, samples), got {cube.shape}")
    F, C, RX, S = cube.shape
    if fps is None:
        with contextlib.redirect_stdout(io.StringIO()):
            fps = capture_params.derive_fps(params)
    fps = float(fps)
    bands = bands or vital_bands.current()
    nfft = range_fft_size(S, n_fft_range)

    damaged = np.zeros(F, bool)
    zero_frac = np.zeros(F)
    clip_frac = np.zeros(F)
    rx_pow = np.zeros((F, RX))
    rti = None
    slow = None
    range_m = None
    rti_chirps = np.unique(np.linspace(0, C - 1, min(C, RTI_CHIRPS)).round().astype(int))
    block = max(1, CHUNK_SAMPLES // max(1, C * RX * S))

    for f0 in range(0, F, block):
        x = cube[f0:f0 + block]
        n = x.shape[0]
        flat = x.reshape(n, -1)
        zero_frac[f0:f0 + n] = (flat == 0).mean(axis=1)
        # lost packets: long unbroken runs of zeros, as everywhere else in the application
        damaged[f0:f0 + n] = vitalsigns.detect_damaged_frames(x)
        if np.iscomplexobj(x):
            peak = np.maximum(np.abs(flat.real), np.abs(flat.imag))
        else:
            peak = np.abs(flat)
        clip_frac[f0:f0 + n] = (peak >= CLIP_LEVEL).mean(axis=1)
        ac = x - x.mean(axis=-1, keepdims=True)
        rx_pow[f0:f0 + n] = (np.abs(ac) ** 2).mean(axis=(1, 3))

        with contextlib.redirect_stdout(io.StringIO()):
            spec, meta = compute_range_fft_cube(x[:, rti_chirps], params, n_fft_range=nfft,
                                                use_full_spectrum=False, return_complex=True)
            mean_spec, _ = compute_range_fft_cube(x.mean(axis=1, keepdims=True), params,
                                                  n_fft_range=nfft, use_full_spectrum=False,
                                                  return_complex=True)
        if rti is None:
            range_m = np.asarray(meta["range_m"], float)
            rti = np.zeros((F, spec.shape[-1]), np.float32)
            slow = np.zeros((F, RX, spec.shape[-1]), np.complex64)
        rti[f0:f0 + n] = (np.abs(spec) ** 2).mean(axis=(1, 2))
        slow[f0:f0 + n] = mean_spec[:, 0]
        if progress_cb is not None:
            progress_cb(min(F, f0 + n), F)

    rti_db = 10 * np.log10(rti + 1e-20)
    rx_db = 10 * np.log10(rx_pow + 1e-20)

    # frames unlike their neighbours: the median change over the range bins ignores a
    # target moving in a few of them, and is the whole profile for a glitch
    R = rti_db.shape[1]
    b0 = min(max(2, R // 64), max(0, R - 3))
    jump = np.zeros(F)
    if F > 1:
        jump[1:] = np.median(np.abs(np.diff(rti_db[:, b0:R - 1], axis=0)), axis=1)
    typical = float(np.median(jump[1:][~damaged[1:]])) if F > 2 and (~damaged[1:]).any() else 0.0
    limit = max(JUMP_MIN_DB, JUMP_TYPICAL_X * typical)
    glitch = (jump > limit) & ~damaged
    if F > 1:
        glitch[1:] &= ~damaged[:-1]          # the frame after a dropout changes too

    # motion only: the static scene (mean over the recording) removed from the chirp mean
    good = ~damaged
    slow_f = _fill(slow, damaged)
    static = slow_f[good].mean(axis=0) if good.any() else slow_f.mean(axis=0)
    motion_db = 10 * np.log10((np.abs(slow_f - static) ** 2).sum(axis=1) + 1e-20).astype(np.float32)

    vit = _vital_energy(slow, fps, bands, range_m, damaged)
    win_s, clear_q = vit.pop("win_s", WINDOW_S), vit.pop("clear_q", BREATHING_Q)
    breaths_total = vit.pop("breaths_total", None)

    chk = Fastcheck(stem=stem, fps=fps, shape=(F, C, RX, S), t_s=np.arange(F) / fps,
                    damaged=damaged, zero_frac=zero_frac, clip_frac=clip_frac, rx_db=rx_db,
                    jump_db=jump, glitch=glitch, range_m=range_m, rti_db=rti_db,
                    motion_db=motion_db, bands=bands, **vit)
    chk.notes["jump_limit_db"] = limit
    chk.notes["vital_window_s"] = win_s
    if breaths_total is not None:
        chk.notes["breaths"] = breaths_total
    chk.notes["clear_q"] = clear_q
    chk.findings = _findings(chk, params, bin_paths, cube)
    return chk


def clear_windows(rate: np.ndarray, q: np.ndarray, breaths_ok: np.ndarray,
                  thr: float = BREATHING_Q) -> np.ndarray:
    """The windows Fastcheck.clear_rhythm calls clear, from the window arrays alone."""
    rate = np.asarray(rate, float)
    ok = np.isfinite(rate)
    steady = np.zeros(rate.size, bool)
    for i in range(rate.size):
        nb = rate[max(0, i - 2):i + 3]
        nb = nb[np.isfinite(nb)]
        if ok[i] and nb.size >= 3:
            steady[i] = abs(rate[i] - np.median(nb)) <= STEADY_PER_MIN
    counted = (np.asarray(breaths_ok, bool) if np.size(breaths_ok) == rate.size
               else np.zeros(rate.size, bool))
    q = np.asarray(q, float)
    return ok & ((q >= thr) | ((q >= 0.6 * thr) & steady) | counted)


def vital_windows(slow: np.ndarray, fps: float, bands=None, range_m=None,
                  damaged=None) -> dict:
    """
    Fastcheck's breathing and heart chain on any slow time (frames, channels, bins) -
    one person's cell in the Multi Vitals view (multi_vitals.py), beams as the channels:
    the bin among those given that breathes most, the breathing rate and the breaths
    counted in every window, the heart followed through the windows, and "clear" - the
    windows with a clear breathing rhythm (clear_windows). best_bin None: no bin breathes
    BREATHING_X times its noise.
    """
    slow = np.asarray(slow)
    bands = bands or vital_bands.current()
    range_m = np.arange(slow.shape[2], dtype=float) if range_m is None else np.asarray(range_m, float)
    out = _vital_energy(slow, float(fps), bands, range_m, damaged, min_range_m=-np.inf,
                        nyquist_last=False)
    out["clear"] = clear_windows(out["br_rate"], out["br_q"], out["br_breaths_ok"],
                                 out.get("clear_q", BREATHING_Q))
    return out


def breathing_bin(slow: np.ndarray, fps: float, bands, range_m, damaged=None,
                  min_range_m: float = MIN_RANGE_M, nyquist_last: bool = True):
    """
    (bin or None, br_ratio, hr_ratio): the range bin whose slow-time signal has the
    most breathing-band power above its noise (above the heart band if the frame rate
    leaves room), summed over the receivers - None when no bin breathes BREATHING_X
    times its noise. br_ratio and hr_ratio are those band powers over the noise, per
    bin. The Vitals view's Auto Bin starts from it as well; nyquist_last=False when slow
    holds only some of the bins (the last one is then not the Nyquist bin).
    """
    F, RX, R = slow.shape
    bad = np.zeros(F, bool) if damaged is None else np.asarray(damaged, bool)
    br, hr = bands.resp_hz, bands.heart_hz
    with_hr = fps > 2 * hr[1]
    filled = _fill(slow, bad)
    freqs, p = _rx_power(filled, fps)                # (freq, range)
    # noise: above the heart band if the frame rate leaves room, else outside both bands
    above = freqs > hr[1] + 0.5
    if above.sum() >= 4:
        noise = p[above].mean(axis=0)
    else:
        out = ((freqs > 0.02) & ~((freqs >= br[0]) & (freqs <= br[1]))
               & ~((freqs >= hr[0]) & (freqs <= hr[1])))
        noise = p[out].mean(axis=0) if out.any() else p[1:].mean(axis=0)
    noise = np.maximum(noise, 1e-30)
    br_power = _band_power(p, freqs, br)
    br_ratio = br_power / noise
    hr_ratio = _band_power(p, freqs, hr) / noise if with_hr else np.full(R, np.nan)

    allowed = np.asarray(range_m) >= min_range_m
    if nyquist_last:
        allowed[-1] = False                          # the Nyquist bin
    excess = np.where(allowed & (br_ratio >= BREATHING_X), br_power - noise, -np.inf)
    best = int(np.argmax(excess)) if np.isfinite(excess).any() and excess.max() > 0 else None
    return best, br_ratio, hr_ratio


def _vital_energy(slow: np.ndarray, fps: float, bands, range_m, damaged=None,
                  min_range_m: float = MIN_RANGE_M, nyquist_last: bool = True) -> dict:
    """Breathing and heart band power of every range bin, and over time at the best one."""
    F, RX, R = slow.shape
    bad = np.zeros(F, bool) if damaged is None else np.asarray(damaged, bool)
    empty = dict(br_ratio=np.full(R, np.nan), hr_ratio=np.full(R, np.nan), best_bin=None,
                 win_t=np.zeros(0), br_db=np.zeros(0), hr_db=np.zeros(0),
                 br_rate=np.zeros(0), br_q=np.zeros(0), hr_rate=np.zeros(0),
                 hr_valid=np.zeros(0, bool), br_breaths=np.zeros(0),
                 br_breaths_ok=np.zeros(0, bool))
    br, hr = bands.resp_hz, bands.heart_hz
    if F / fps < MIN_VITAL_S or fps <= 2 * br[1]:
        return empty
    with_hr = fps > 2 * hr[1]
    best, br_ratio, hr_ratio = breathing_bin(slow, fps, bands, range_m, bad,
                                             min_range_m=min_range_m, nyquist_last=nyquist_last)
    out = dict(empty, br_ratio=br_ratio, hr_ratio=hr_ratio, best_bin=best)
    if best is None:
        return out

    # at the best range and its neighbours (whichever breathes most in each window): the
    # chest displacement, stitched across the dropouts
    lo, hi = max(0, best - 1), min(R, best + 2)
    disp = _chest_phase(slow[:, :, lo:hi], bad, fps)                  # (F, RX, bins)
    # the slow drift of a whole recording (the chest settling, the static part changing)
    # would otherwise leak into the bottom of the breathing band of every window
    from scipy import signal as _sig
    if F > 30:
        sos = _sig.butter(2, 0.6 * br[0], btype="highpass", fs=fps, output="sos")
        disp = _sig.sosfiltfilt(sos, disp, axis=0)
    # the rate from the breathing band alone, as the Vitals view takes it: 74 dropouts
    # stitched over a minute leave a slow random walk that a high-pass alone passes
    # into the bottom bin of the band
    disp_br = disp
    if F > 60:
        sos = _sig.butter(4, [br[0], min(br[1], 0.45 * fps)], btype="band", fs=fps, output="sos")
        disp_br = _sig.sosfiltfilt(sos, disp, axis=0)
    win_s = min(WINDOW_S, max(MIN_WINDOW_S, F / fps / 3.0), F / fps)
    win = int(round(win_s * fps))
    step = max(1, int(round(STEP_S * fps)))
    nyq = 0.5 * fps
    noise_band = (hr[1] + 0.5, min(hr[1] + 5.0, 0.8 * nyq))
    t, brd, hrd, brr, q, nbr, nbr_ok = [], [], [], [], [], [], []
    # what a breath is, judged on the whole recording: a quiet window must not turn its
    # ripples into breaths (a 20 s window counted 4 where the minute had 3)
    med_br = np.median(disp_br, axis=1)                                   # (F, bins)
    swing_bin = np.percentile(med_br, 95, axis=0) - np.percentile(med_br, 5, axis=0)
    # and the breaths of the whole recording in each bin: a window's counted rate is the
    # breathing those intervals do over it (its own few peaks, without the pauses at its
    # edges, made 20 /min of a minute that breathed 7)
    whole = [vitalsigns.count_breaths(
                 med_br[:, k], fps, br,
                 vitalsigns.breathing_noise_rms(np.median(disp[:, :, k], axis=1), fps, br, hr[1]))
             for k in range(med_br.shape[1])]
    for s0 in range(0, F - win + 1, step):
        w = np.hanning(win).reshape((win, 1, 1))
        p_w = (np.abs(np.fft.rfft(_detrend(disp[s0:s0 + win]) * w, axis=0)) ** 2).sum(axis=1)
        p_b = (np.abs(np.fft.rfft(_detrend(disp_br[s0:s0 + win]) * w, axis=0)) ** 2).sum(axis=1)
        f_w = np.fft.rfftfreq(win, 1.0 / fps)
        pick = int(np.argmax(_band_power(p_w, f_w, br)))
        col, col_b = p_w[:, pick], p_b[:, pick]
        # the breaths themselves, one by one, in the chest displacement of this bin
        cnt = vitalsigns.count_breaths(
            med_br[s0:s0 + win, pick], fps, br,
            vitalsigns.breathing_noise_rms(np.median(disp[s0:s0 + win, :, pick], axis=1), fps, br, hr[1]),
            swing=swing_bin[pick])
        nbr.append(cnt["n"])
        nbr_ok.append(cnt["ok"])
        in_br = np.flatnonzero((f_w >= br[0]) & (f_w <= br[1]))
        in_noise = (f_w >= noise_band[0]) & (f_w <= noise_band[1])
        floor = float(col[in_noise].mean()) if in_noise.sum() >= 3 else float("nan")
        t.append((s0 + win / 2) / fps)
        hrd.append(10 * np.log10(_band_power(col[:, None], f_w, hr)[0] / floor)
                   if with_hr and np.isfinite(floor) else np.nan)
        if in_br.size < 2:
            brd.append(np.nan)
            brr.append(np.nan)
            q.append(0.0)
            continue
        brd.append(10 * np.log10(col[in_br].mean() / floor) if np.isfinite(floor) else np.nan)
        k = in_br[np.argmax(col_b[in_br])]
        if k == in_br[0]:
            # a peak on the lowest bin of the band is drift running into it, not a rhythm -
            # unless the breaths themselves stand out (long gaps put power there too)
            brr.append(_counted_rate(whole[pick], cnt, s0, win, fps))
            q.append(0.0)
            continue
        q.append(float(col_b[k] / col_b[in_br].mean()))
        # a clear peak gives the rate as it always did; without one (irregular breathing,
        # whose peak falls on a harmonic or the slowest gap) the counted breaths give it,
        # and they overrule a clear peak only well above them - the second harmonic, 16 /min
        # for 9 counted. A count above a clear peak is the count's mistake: a double hump
        # in a breath counted twice (15-18 /min where the peak and the minute said 12).
        spec = _peak_rate(col_b, f_w, br)
        counted = _counted_rate(whole[pick], cnt, s0, win, fps)
        if np.isfinite(counted) and (q[-1] < _clear_q(int(in_br.size))
                                     or not np.isfinite(spec)
                                     or spec > (1 + RATE_DISAGREE) * counted):
            brr.append(counted)
        else:
            brr.append(spec)
    n_bins = int(np.sum((np.fft.rfftfreq(win, 1.0 / fps) >= br[0])
                        & (np.fft.rfftfreq(win, 1.0 / fps) <= br[1])))
    # the heart, as the phase-12 pipeline follows it (moving_hr.heart_spectra and
    # heart_track): in the same windows, from the displacement of every receiver in the
    # bin that breathes most and in its neighbours (a neighbour where its arc stands above
    # its noise: vitalsigns.heart_channels) - the heart band and the envelope of its
    # 5-25 Hz content, breathing harmonics notched, a harmonic sum and a Viterbi path.
    # One bin's receivers were 3.5 bpm off the PPG beats on six seated recordings, these 2.5.
    hr_rate = np.full(len(t), np.nan)
    hr_valid = np.zeros(len(t), bool)
    if with_hr and t:
        import moving_hr
        j = int(np.argmax((disp_br ** 2).sum(axis=(0, 1))))
        d = np.mean([disp[:, i, b] for b in range(disp.shape[2]) for i in range(RX)
                     if b == j or vitalsigns.arc_snr(slow[:, i, lo + b]) >= vitalsigns.ARC_SNR_MIN],
                    axis=0)
        wins = [(s0, s0 + win) for s0 in range(0, F - win + 1, step)]
        grid, spectra = moving_hr.heart_spectra(d, fps, bands, wins, br_bpm=np.array(brr),
                                                method="auto")
        hr_rate, conf, _ = moving_hr.heart_track(spectra, grid, STEP_S)
        hr_valid = conf >= moving_hr.CONF_MIN
    # the breaths of the whole recording, in the bin that breathes most
    total = whole[int(np.argmax((disp_br ** 2).sum(axis=(0, 1))))]
    out.update(win_t=np.array(t), br_db=np.array(brd), hr_db=np.array(hrd),
               br_rate=np.array(brr), br_q=np.array(q), hr_rate=np.asarray(hr_rate, float),
               hr_valid=np.asarray(hr_valid, bool), br_breaths=np.array(nbr, float),
               br_breaths_ok=np.array(nbr_ok, bool), win_s=win_s, clear_q=_clear_q(n_bins),
               breaths_total=total)
    return out


def _counted_rate(whole: dict, window: dict, s0: int, win: int, fps: float) -> float:
    """The breathing rate the whole recording's breaths give over one window, if the
    window holds a breath of its own (a window inside a long pause has no rate)."""
    if not whole["ok"] or window["n"] < 1:
        return float("nan")
    return vitalsigns.breathing_rate_over(whole["times"], s0 / fps, (s0 + win) / fps)


def _peak_rate(p: np.ndarray, freqs: np.ndarray, band) -> float:
    """The strongest frequency in the band, per minute, between the bins (parabola)."""
    sel = np.flatnonzero((freqs >= band[0]) & (freqs <= band[1]))
    if sel.size == 0:
        return float("nan")
    k = int(sel[np.argmax(p[sel])])
    if p[k] <= 0:
        return float("nan")
    f = freqs[k]
    if 0 < k < len(p) - 1 and p[k - 1] > 0 and p[k + 1] > 0:
        a, b, c = np.log(p[k - 1]), np.log(p[k]), np.log(p[k + 1])
        d = a - 2 * b + c
        if d < 0:
            f += 0.5 * (a - c) / d * (freqs[1] - freqs[0])
    return float(np.clip(f, band[0], band[1]) * 60.0)


# --- findings --------------------------------------------------------------------------------

def _times(frames, fps, limit=4) -> str:
    frames = list(frames)
    txt = ", ".join(f"{f / fps:.1f} s" for f in frames[:limit])
    return txt + (f" and {len(frames) - limit} more" if len(frames) > limit else "")


def _findings(chk: Fastcheck, params: dict, bin_paths, cube) -> list:
    out: list[Finding] = []
    F, fps = chk.n_frames, chk.fps

    # what the logs, the files, the raw prefix and the first frames say
    try:
        rep = capture_health.assess(chk.stem, params, bin_paths=list(bin_paths or []),
                                    cube=cube, check_zero_fill=False)
        out.extend(rep.findings)
    except Exception as e:
        out.append(Finding("health report", "warning", f"could not run: {type(e).__name__}: {e}"))

    # lost packets, from the data itself
    n_bad = int(chk.damaged.sum())
    spans = damaged_spans(chk.damaged)
    clean = longest_clean(chk.damaged) / fps
    if n_bad == 0:
        out.append(Finding("zero-filled frames", "ok",
                           f"none in {F} frames - the data is complete"))
    else:
        longest = max(spans, key=lambda s: s[1] - s[0])
        pct = 100.0 * n_bad / F
        lost = float((chk.zero_frac * chk.damaged).sum() / F) * 100.0
        sev = "error" if n_bad / F > DAMAGED_ERROR else "warning"
        out.append(Finding("zero-filled frames", sev,
                           f"{n_bad} of {F} frames ({_pct(pct)} %) hold zero-filled packets "
                           f"(about {_pct(lost)} % of all samples), in {len(spans)} dropout(s); "
                           f"longest {(longest[1] - longest[0]) / fps:.2f} s at "
                           f"{longest[0] / fps:.1f} s; longest clean stretch {clean:.1f} s"))
    rec = params.get("dca_packets_zero_filled")
    if rec is not None:
        if int(rec) > 0 and n_bad == 0:
            out.append(Finding("recorder vs data", "info",
                               f"the recorder reports {int(rec)} zero-filled packet(s), but no "
                               f"decoded frame holds a run of zeros - they may lie outside "
                               f"the decoded frames"))
        elif int(rec) == 0 and n_bad:
            out.append(Finding("recorder vs data", "warning",
                               f"the recorder reports no lost packets, yet {n_bad} frame(s) "
                               f"hold long runs of zeros"))

    # ADC clipping in every frame
    clipped = np.flatnonzero(chk.clip_frac >= CLIP_FRAME)
    if clipped.size == 0:
        out.append(Finding("clipping", "ok", "no frame reaches full scale"))
    else:
        sev = "error" if clipped.size > 0.1 * F else "warning"
        out.append(Finding("clipping", sev,
                           f"{clipped.size} frame(s) reach full scale (worst "
                           f"{100 * chk.clip_frac.max():.2f} % of samples), first at "
                           f"{_times(clipped, fps, 3)} - the phase of clipped samples is lost"))

    # every receiver over time
    ok = ~chk.damaged
    silent = {}
    if ok.sum() >= 2:
        med = np.median(chk.rx_db[ok], axis=0)
        for r in range(chk.rx_db.shape[1]):
            fr = np.flatnonzero(ok & (chk.rx_db[:, r] < med[r] - RX_SILENT_DB))
            if fr.size:
                silent[r] = fr
    if silent:
        out.append(Finding("receivers over time", "warning", "; ".join(
            f"RX{r + 1} falls silent in {fr.size} frame(s) from {fr[0] / fps:.1f} s"
            for r, fr in silent.items())))
    elif ok.any():
        total = 10 * np.log10(np.mean(10 ** (chk.rx_db[ok] / 10), axis=1))
        spread = float(np.percentile(total, 99) - np.percentile(total, 1))
        if spread > LEVEL_INFO_DB:
            out.append(Finding("signal level", "info",
                               f"the received power changes by {spread:.0f} dB during the "
                               f"recording - something moved close to the radar, or the "
                               f"gain changed"))
        else:
            out.append(Finding("receivers over time", "ok",
                               f"all {chk.rx_db.shape[1]} steady (level within {spread:.1f} dB)"))

    # frames unlike their neighbours
    gl = np.flatnonzero(chk.glitch)
    if gl.size:
        out.append(Finding("frame glitches", "warning",
                           f"{gl.size} frame(s) differ from the frame before by more than "
                           f"{chk.notes.get('jump_limit_db', JUMP_MIN_DB):.1f} dB over the "
                           f"whole range, at {_times(gl, fps)} - interference, or the data "
                           f"stream out of step"))
    else:
        out.append(Finding("frame glitches", "ok", "every frame follows on from the one before"))

    # somebody breathing?
    def ratio_db(v):
        return 10 * np.log10(max(float(v), 1e-12))

    if chk.duration_s < MIN_VITAL_S or not np.isfinite(chk.br_ratio).any():
        out.append(Finding("vital signs", "info",
                           f"not judged: {chk.duration_s:.1f} s at {chk.fps:g} fps is too little "
                           f"for the breathing band (at least {MIN_VITAL_S:g} s needed)"))
    elif chk.best_bin is None:
        allowed = chk.range_m >= MIN_RANGE_M
        top = np.nanmax(chk.br_ratio[allowed]) if allowed.any() else np.nan
        out.append(Finding("vital signs", "info",
                           f"no slow motion anywhere (the breathing band at most "
                           f"{ratio_db(top):.0f} dB above the noise) - nobody in front of "
                           f"the radar"))
    else:
        r = chk.best_bin
        where = f"{chk.range_m[r]:.2f} m"
        good = chk.clear_rhythm()
        n, k = good.size, int(good.sum())
        hr_txt = ""
        if chk.hr_valid.any():
            hr_txt = (f"; heart about {np.median(chk.hr_rate[chk.hr_valid]):.0f} bpm "
                      f"({int(chk.hr_valid.sum())} of {n} windows)")
        # the breaths counted over the whole recording, when they stand above the noise
        cnt = chk.notes.get("breaths") or {}
        cnt_txt = ""
        if cnt.get("ok") and cnt.get("n", 0) >= 3:
            cnt_txt = (f" ({cnt['n']} breaths counted in {chk.duration_s:.0f} s"
                       + (", irregular" if cnt.get("cv", 0) > IRREGULAR_CV else "") + ")")
        # the headline rate is the windows' median (they take the counted breaths where the
        # spectrum has no clear peak or one on a harmonic); the count of the whole recording
        # stands beside it for the reader to check against their own
        rate = np.nanmedian(chk.br_rate[good]) if k else np.nan
        if n and k >= 0.5 * n:
            txt = (f"breathing at {where}: about {rate:.0f} /min"
                   f"{cnt_txt}, clear in {k} of {n} windows of "
                   f"{chk.notes.get('vital_window_s', WINDOW_S):.0f} s{hr_txt}"
                   f" - the Vitals, HR and AI views measure them in detail")
        elif k:
            txt = (f"breathing at {where} only in {k} of {n} windows (about "
                   f"{rate:.0f} /min){cnt_txt} - the person moved, or "
                   f"the signal is weak")
        else:
            txt = (f"the most slow motion is at {where} ({ratio_db(chk.br_ratio[r]):.0f} dB "
                   f"above the noise), but it has no clear breathing rhythm - nobody sitting "
                   f"still, or too much motion")
        out.append(Finding("vital signs", "info", txt))
    return out
