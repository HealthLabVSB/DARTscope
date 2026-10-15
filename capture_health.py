"""
capture_health.py

One report on whether a capture can be trusted, before anything is computed from it.

Every fault found in this code so far was silent: the lane width taken from the radar
instead of the recorder, REAL samples read out of a complex-shaped container, a start
time off by up to 28 minutes. None of them raised - the byte count always worked out -
and each only showed up as a disagreement between recordings that should have agreed.
The checks for several of those already existed, scattered through the decoder and the
log parser, printing to the console at load time or not at all. This adds them up.

Levels, cheapest first:
    assess_params  - what the logs say, no data read
    assess_files   - whether the raw files were written with this log
    assess_raw     - a prefix of the raw file: ADC saturation, container layout
    assess_cube    - the decoded cube: frame count, dead channels, DC, IQ balance,
                     noise spectrum

Nothing here touches Tk, so the GUI, the tools and the tests all get the same answer.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field

import numpy as np

import capture_params

SEVERITIES = ("ok", "info", "warning", "error")
C_LIGHT = 299_792_458.0

# Thresholds. The IQ and DC ones were set from the captures this was written against,
# where image rejection sat at 25-39 dB on every channel and DC at 17-40 dB below the
# signal - so the limits below flag a fault, not an ordinary recording.
SATURATION_WARN = 1e-4          # fraction of raw samples at >= 98 % of full scale
SATURATION_ERROR = 1e-2
UNSAMPLED_RAMP_INFO = 0.20      # fraction of the ramp outside the ADC window
DC_WARN_DB = -10.0              # DC relative to RMS, per channel
IMAGE_REJECTION_WARN_DB = 20.0
IMAGE_SWAP_DB = -15.0           # consistently this far negative: I and Q swapped
TARGET_SNR_DB = 20.0            # a target must clear the noise by this to count
IQ_MIN_TARGETS = 2              # targets that must agree before a channel is faulted
DEAD_RX_DB = -20.0              # channel power relative to the median channel
CUBE_FRAMES = 40                # frames used for the per-channel statistics
RAW_PREFIX_BYTES = 16 << 20
FILE_AGE_TOLERANCE_S = 3600.0   # raw data vs its recorder log, and part to part


@dataclass
class Finding:
    check: str
    severity: str
    message: str

    def __post_init__(self):
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}")


@dataclass
class HealthReport:
    stem: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def worst(self) -> str:
        if not self.findings:
            return "ok"
        return max((f.severity for f in self.findings), key=SEVERITIES.index)

    def of(self, severity: str) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]

    def format(self, only_problems: bool = False) -> str:
        marks = {"ok": "  ok ", "info": " info", "warning": " WARN", "error": "ERROR"}
        rows = sorted(self.findings, key=lambda f: -SEVERITIES.index(f.severity))
        if only_problems:
            rows = [f for f in rows if f.severity in ("warning", "error")]
        lines = [f"{self.stem}  [{self.worst}]"]
        width = max((len(f.check) for f in rows), default=0)
        for f in rows:
            lines.append(f"  {marks[f.severity]}  {f.check:<{width}}  {f.message}")
        return "\n".join(lines)

    def summary(self) -> str:
        n_err, n_warn = len(self.of("error")), len(self.of("warning"))
        if not (n_err or n_warn):
            return f"{self.stem}: no problems found"
        first = (self.of("error") or self.of("warning"))[0]
        return f"{self.stem}: {n_err} error(s), {n_warn} warning(s) - {first.message}"


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _num(params, key):
    v = params.get(key)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# --- what the logs say -------------------------------------------------------------

def assess_params(params: dict) -> list[Finding]:
    out: list[Finding] = []
    has_recorder = any(k in params for k in ("dca_lane_mode", "dca_packets_received"))

    if not has_recorder:
        out.append(Finding("recorder log", "warning",
                           "no _Raw_LogFile.csv beside the capture: the container's lane "
                           "width, packet loss and the capture's own start time are unknown"))

    src = params.get("radar_start_source")
    if src == "DCA1000 recorder log":
        drift = params.get("capture_start_vs_log_s")
        extra = f" (the API log's first entry is {drift:+.0f} s off)" if drift else ""
        out.append(Finding("capture start", "ok",
                           f"{params.get('radar_start_time')} from the recorder log{extra}"))
    elif src:
        out.append(Finding("capture start", "warning",
                           f"{params.get('radar_start_time')} taken from the {src}, which is "
                           f"when the session began, not this capture; syncing a reference "
                           f"against it can be minutes off"))
    else:
        out.append(Finding("capture start", "warning", "no start time in the logs"))

    lanes, lanes_src = params.get("numLanes"), params.get("numLanes_source")
    if lanes not in (1, 2, 4):
        out.append(Finding("lanes", "error", f"no usable LVDS lane count ({lanes})"))
    elif lanes_src == "DCA1000 recorder log":
        radar = params.get("numLanes_radar")
        if params.get("lane_mode_mismatch"):
            out.append(Finding("lanes", "info",
                               f"radar drives {radar}, recorder container is {lanes} wide; "
                               f"decoded with {lanes}"))
        else:
            out.append(Finding("lanes", "ok", f"{lanes}, confirmed by the recorder"))
    else:
        out.append(Finding("lanes", "info",
                           f"{lanes} from {lanes_src}, not confirmed by the recorder log"))

    if has_recorder:
        lost = int(params.get("dca_packets_zero_filled") or 0)
        oos = int(params.get("dca_out_of_sequence") or 0)
        if lost or oos:
            pct = params.get("dca_packet_loss_pct")
            pct_txt = f", {pct:.2f} % of the stream" if pct else ""
            out.append(Finding("packet loss", "warning",
                               f"{lost} zero-filled packet(s){pct_txt}, {oos} out of "
                               f"sequence - the zeros read as phase jumps"))
        else:
            out.append(Finding("packet loss", "ok", "none reported by the recorder"))

    out.extend(_adc_window(params))

    fallback = []
    fps = _quiet(capture_params.derive_fps, params, on_fallback=fallback.append)
    if fallback:
        out.append(Finding("frame rate", "warning", fallback[0]))
    else:
        out.append(Finding("frame rate", "ok", f"{fps:g} fps"))

    out.extend(_antenna(params))
    return out


def _adc_window(params) -> list[Finding]:
    ramp = _num(params, "rampEndTime_us")
    start = _num(params, "adcStartTime_us") or 0.0
    n = _num(params, "numAdcSamples")
    fs = _num(params, "digOutSampleRate_MHz")
    slope = _num(params, "freqSlopeMHz_us")
    if not (ramp and n and fs):
        return []
    t_adc = n / fs
    end = start + t_adc
    if end > ramp + 1e-6:
        over = end - ramp
        return [Finding("ADC window", "error",
                        f"sampling ends {over:.1f} us after the ramp; the last "
                        f"{int(round(over * fs))} samples of every chirp are off the chirp")]
    unused = (ramp - end) / ramp
    if unused > UNSAMPLED_RAMP_INFO and slope:
        dr_now = C_LIGHT / (2 * slope * t_adc * 1e6) * 100
        dr_full = C_LIGHT / (2 * slope * (ramp - start) * 1e6) * 100
        return [Finding("ADC window", "info",
                        f"{unused * 100:.0f} % of the ramp is not sampled: range "
                        f"resolution {dr_now:.1f} cm where sampling the whole ramp gives "
                        f"{dr_full:.1f} cm")]
    return [Finding("ADC window", "ok", f"samples {100 * (1 - unused):.0f} % of the ramp")]


def _antenna(params) -> list[Finding]:
    rx_mask = params.get("dataFmt_rxChanEnMask")
    if rx_mask is None:
        return []
    import AOA
    n_rx = bin(int(rx_mask)).count("1")
    n_tx = max(1, len(params.get("tx_order_masks") or []))
    cfg = _quiet(AOA._resolve_antenna_config, params, n_tx * n_rx, n_rx)
    if cfg:
        how = "selected" if params.get("antenna_config") else "from the chip name"
        return [Finding("antenna", "ok", f"'{cfg.get('name')}' ({how}), "
                                         f"{n_tx} TX x {n_rx} RX")]
    if params.get("cascade"):
        return [Finding("antenna", "warning",
                        f"no antenna configuration selected for the cascade ({n_tx} TX x {n_rx} RX); "
                        f"the angle views need '{_quiet(lambda: __import__('antenna_config').CASCADE_CONFIG)}' "
                        f"and a corner-reflector calibration")]
    return [Finding("antenna", "warning",
                    f"no configuration matches chip '{params.get('chip_hint')}'; angles "
                    f"will be computed on an assumed linear array - select the board's "
                    f"configuration or add one to antenna_configs.json")]


# --- whether the files belong together ---------------------------------------------

def assess_files(params: dict, bin_paths) -> list[Finding]:
    """
    Whether the raw files were written with this log, judged by when they were written.

    The recorder writes its CSV and the last raw part within seconds of each other at
    the end of a capture: across 314 recordings the two agreed to the minute on 312 and
    to the hour on 313. The odd one out paired a log and CSV from 19 June with a raw
    file from 17 April - a different recording under the same name, which decodes
    without complaint. Parts of one capture are also written in order, so a later part
    that is older than an earlier one does not belong to it either.
    """
    import os
    paths = [p for p in (bin_paths or []) if p and os.path.isfile(p)]
    if not paths:
        return []
    out: list[Finding] = []
    times = [os.path.getmtime(p) for p in paths]
    for a, b, ta, tb in zip(paths, paths[1:], times, times[1:]):
        if ta - tb > FILE_AGE_TOLERANCE_S:
            out.append(Finding("raw files", "error",
                               f"{os.path.basename(b)} is {_age(ta - tb)} older than "
                               f"{os.path.basename(a)} - it cannot be the next part of "
                               f"the same capture"))
    csv = params.get("dca_csv_path")
    if csv and os.path.isfile(csv):
        gap = max(times) - os.path.getmtime(csv)
        if abs(gap) > FILE_AGE_TOLERANCE_S:
            which = "older" if gap < 0 else "newer"
            out.append(Finding("raw files", "error",
                               f"the raw data is {_age(abs(gap))} {which} than its recorder "
                               f"log - it is probably a different recording under the "
                               f"same name"))
        elif not out:
            out.append(Finding("raw files", "ok", "written together with the recorder log"))
    return out


def _age(seconds: float) -> str:
    if seconds >= 86400:
        return f"{seconds / 86400:.0f} day(s)"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 60:.0f} min"


# --- a prefix of the raw file ------------------------------------------------------

def assess_raw(params: dict, bin_path: str, prefix_bytes: int = RAW_PREFIX_BYTES) -> list[Finding]:
    from radar_processing import _drop_zero_padded_slots, _real_container_is_doubled

    raw = np.fromfile(bin_path, dtype=np.int16, count=prefix_bytes // 2)
    if raw.size == 0:
        return [Finding("raw data", "error", "the binary is empty")]
    out: list[Finding] = []

    bits = int(_num(params, "dataFmt_adcBits") or 16)
    full = (1 << (bits - 1)) - 1
    a = np.abs(raw.astype(np.int32))
    peak = int(a.max())
    frac = float(np.mean(a >= 0.98 * full))
    peak_dbfs = 20 * np.log10(max(peak, 1) / full)
    if frac >= SATURATION_ERROR:
        out.append(Finding("ADC saturation", "error",
                           f"{frac * 100:.2f} % of samples at full scale - the phase of "
                           f"clipped samples is lost; lower the RX gain"))
    elif frac >= SATURATION_WARN:
        out.append(Finding("ADC saturation", "warning",
                           f"{frac * 100:.3f} % of samples at full scale (peak "
                           f"{peak_dbfs:+.1f} dBFS)"))
    else:
        out.append(Finding("ADC saturation", "ok", f"peak {peak_dbfs:+.1f} dBFS, no clipping"))

    if str(params.get("dataFmt_adcFmt", "")).upper() == "REAL":
        lanes = params.get("numLanes") or 1
        _, stride, keep = _quiet(_drop_zero_padded_slots, raw, True)
        if stride > 1:
            out.append(Finding("REAL container", "info",
                               f"{stride - len(keep)} of every {stride} int16 are empty "
                               f"slots; the decoder drops them"))
        elif lanes in (1, 2, 4) and _quiet(_real_container_is_doubled, raw, int(lanes), None):
            out.append(Finding("REAL container", "info",
                               f"samples sit in a {2 * lanes}-slot container with only the "
                               f"first {lanes} filled; the decoder keeps those"))
    return out


# --- the decoded cube --------------------------------------------------------------

def frame_count_finding(declared, decoded, size_ratio=None, fps=None,
                        recorded_s=None) -> Finding | None:
    """
    Whether the decode came out the length it should.

    The declared frame count alone cannot tell a decode fault from a recording that was
    stopped early or left running - the recorder's stop mode is usually "Infinite", and
    the loader deliberately keeps every frame the file holds. What it cannot do is hold
    more frames than fit in the time the recorder ran: REAL samples read out of their
    container straight through gave 499 frames at 25 fps, 20 s, from a 12 s recording,
    and that is a stream cut into the wrong shape. So the recorder's duration decides
    whether a mismatch is an error; without it, the mismatch is only reported.
    """
    if not decoded:
        return None
    ratio = f" (raw file is {size_ratio:.2f}x the declared geometry)" if size_ratio else ""
    if fps and recorded_s:
        decoded_s = decoded / float(fps)
        rec = float(recorded_s)
        if decoded_s > 1.25 * rec + 2.0:
            return Finding("frame count", "error",
                           f"decoded {decoded} frames = {decoded_s:.0f} s at {fps:g} fps "
                           f"from a {rec:.0f} s recording - more than was "
                           f"recorded, so the stream was cut into the wrong shape{ratio}")
        # The other direction is ambiguous, and the message must not pick a side: 92
        # captures once came out at half their recording's length because every
        # decoded frame held two real ones, which looked exactly like a short recording.
        # The 1 s allows for the recorder starting before the radar on short captures.
        if decoded_s < 0.6 * rec - 1.0:
            return Finding("frame count", "warning",
                           f"decoded {decoded} frames = {decoded_s:.0f} s at {fps:g} fps "
                           f"from a {rec:.0f} s recording - either the radar stopped early "
                           f"or each frame is decoded larger than what the radar sent "
                           f"(check chirps per frame against the log){ratio}")
    if not declared:
        return None
    drift = int(decoded) - int(declared)
    if drift > 1 or drift < -2:
        how = "longer" if drift > 0 else "shorter"
        span = (f" ({decoded / float(fps):.1f} s of frames in a {float(recorded_s):.0f} s "
                f"recording)" if fps and recorded_s else "")
        return Finding("frame count", "warning",
                       f"decoded {decoded} frames where the log configures {declared} - "
                       f"the recording ran {how} than configured{span}{ratio}")
    return Finding("frame count", "ok", f"{decoded} frames, as declared ({declared})")


def image_rejection_db(x: np.ndarray, n_targets: int = 5,
                       min_snr_db: float = TARGET_SNR_DB) -> float:
    """Median image rejection over the clear targets of one channel; NaN if none."""
    return _image_rejection(x, n_targets, min_snr_db)[0]


def _image_rejection(x: np.ndarray, n_targets: int = 5,
                     min_snr_db: float = TARGET_SNR_DB) -> tuple[float, int]:
    """
    How far the strongest targets stand above their mirror images, for one channel.

    x: complex samples shaped [..., samples]. An IQ imbalance puts a copy of every
    target at the negative beat frequency; the ratio is the standard measure of it and,
    unlike a correlation between I and Q, does not depend on what is in the room. A
    negative value means the targets landed in the negative half - I and Q swapped.

    Only targets standing at least `min_snr_db` above the noise floor are used, and NaN
    comes back when there are none. Without that, a scene with nothing strong in it
    compares noise with noise: on the rect and kleky recordings the five "strongest
    targets" sat 7-16 dB above the floor and their ratios scattered from -22 to +15 dB,
    which read as swapped I and Q on hardware that is fine. A target that clears the
    floor by that margin cannot produce a ratio below it unless its image is real.
    """
    ns = x.shape[-1]
    s = x - x.mean(axis=-1, keepdims=True)
    p = (np.abs(np.fft.fft(s * np.hanning(ns), axis=-1)) ** 2).reshape(-1, ns).mean(axis=0)
    half = ns // 2
    noise = float(np.median(p)) + 1e-30
    # Targets are looked for in both halves: with I and Q swapped every one of them is
    # in the negative half, and searching only the positive half would miss the swap.
    band = np.r_[4:half - 4, half + 4:ns - 4]            # skip DC and Nyquist
    pairs = []
    for k in band[np.argsort(p[band])[::-1]]:
        if 10 * np.log10(p[k] / noise) < min_snr_db:
            break                                        # sorted: the rest are weaker
        # A target is a peak. The near-field coupling sits tens of dB above the floor
        # right at DC, and the window spreads it evenly to both sides - on rect,
        # dataset_square and kleky the only bin clearing 20 dB was that slope at bin 4,
        # and its "image" on the other side of DC read as 5-8 dB of image rejection.
        if p[k] < p[k - 1] or p[k] < p[k + 1]:
            continue
        pos = int(k) if k < half else int(ns - k)        # the target's positive-side bin
        if all(abs(pos - q) > 3 for q in pairs):
            pairs.append(pos)
        if len(pairs) == n_targets:
            break
    if not pairs:
        return float("nan"), 0
    return (float(np.median([10 * np.log10((p[k] + 1e-30) / (p[ns - k] + 1e-30))
                             for k in pairs])), len(pairs))


def assess_cube(params: dict, cube: np.ndarray, size_ratio=None,
                check_zero_fill: bool = True) -> list[Finding]:
    out: list[Finding] = []
    declared = (params.get("frameConfig") or {}).get("numFrames")
    fallback = []
    fps = _quiet(capture_params.derive_fps, params, on_fallback=fallback.append)
    f = frame_count_finding(declared, cube.shape[0], size_ratio,
                            fps=None if fallback else fps,
                            recorded_s=params.get("dca_duration_s"))
    if f:
        out.append(f)

    x = cube[:CUBE_FRAMES]
    bad = int((~np.isfinite(x)).sum())
    if bad:
        out.append(Finding("finite", "error", f"{bad} non-finite samples in the first "
                                              f"{x.shape[0]} frames"))

    n_rx = x.shape[2]
    power = np.array([np.mean(np.abs(x[:, :, r, :]) ** 2) for r in range(n_rx)])
    ref = np.median(power)
    rel = 10 * np.log10(power / (ref + 1e-30) + 1e-30)
    dead = [r for r in range(n_rx) if rel[r] < DEAD_RX_DB]
    if dead:
        out.append(Finding("RX channels", "error",
                           "RX" + ", RX".join(str(r + 1) for r in dead) +
                           f" carr{'y' if len(dead) > 1 else 'ies'} nothing "
                           f"({min(rel[r] for r in dead):.0f} dB below the others)"))
    else:
        out.append(Finding("RX channels", "ok",
                           f"{n_rx} live, power spread {rel.max() - rel.min():.1f} dB"))

    dc = []
    for r in range(n_rx):
        s = x[:, :, r, :]
        rms = np.sqrt(np.mean(np.abs(s) ** 2)) + 1e-30
        dc.append(20 * np.log10(abs(np.mean(s)) / rms + 1e-12))
    worst = int(np.argmax(dc))
    if dc[worst] > DC_WARN_DB:
        out.append(Finding("DC offset", "warning",
                           f"RX{worst + 1}: DC only {-dc[worst]:.0f} dB below the signal"))
    else:
        out.append(Finding("DC offset", "ok", f"at least {-max(dc):.0f} dB below the signal"))

    live = [r for r in range(n_rx) if r not in dead]
    if np.iscomplexobj(x) and live:
        # One clean target is enough to show a channel is fine - a quiet mirror position
        # is what good image rejection means. Condemning a channel takes two: a single
        # peak on the wrong side can be a spur, or a reflector beyond the unambiguous
        # range folded into negative frequencies. On kleky rlk90s2_4 the only target
        # clearing the noise sat at bin -34, 20 dB up, on three channels at once - and
        # on its own it read as swapped I and Q.
        per = [_image_rejection(x[:, :, r, :]) for r in live]
        judged = [v for v, n in per if n >= IQ_MIN_TARGETS]
        good = [v for v, n in per if n >= 1 and v >= IMAGE_REJECTION_WARN_DB]
        swapped = [v for v in judged if v <= IMAGE_SWAP_DB]
        weak = [v for v in judged if v < IMAGE_REJECTION_WARN_DB]
        if swapped:
            out.append(Finding("IQ balance", "error",
                               f"targets land in the negative half of the spectrum "
                               f"({min(swapped):.0f} dB) - I and Q look swapped"))
        elif weak:
            out.append(Finding("IQ balance", "warning",
                               f"image rejection only {min(weak):.0f} dB - mirror targets "
                               f"appear at the negative range"))
        elif good:
            out.append(Finding("IQ balance", "ok",
                               f"image rejection {min(good):.0f}-{max(good):.0f} dB"))
        elif any(n for _, n in per) and "COMPLEX1X" in str(params.get("dataFmt_adcFmt") or "").upper():
            # complex 1x spans 0..fs: the far half of the band is range as well (the RTI
            # draws it), so a peak there is a reflector as likely as a mirror image
            out.append(Finding("IQ balance", "info",
                               "not measurable: only a single peak clears the noise, in the "
                               "far half of the band - with complex 1x sampling a reflector "
                               "there looks like a mirror image; not enough to judge the "
                               "channel by"))
        elif any(n for _, n in per):
            out.append(Finding("IQ balance", "info",
                               "not measurable: only a single peak clears the noise, and "
                               "it is on the wrong side - a spur or a reflection from "
                               "beyond the range, not enough to judge the channel by"))
        else:
            out.append(Finding("IQ balance", "info",
                               f"not measurable: no target stands {TARGET_SNR_DB:.0f} dB "
                               f"clear of the noise in the first {x.shape[0]} frames"))

    out += noise_findings(params, x, live)

    if not check_zero_fill:
        return out
    try:
        import vitalsigns
        damaged = vitalsigns.detect_damaged_frames(cube)
        n_bad = int(np.sum(damaged))
        if n_bad:
            out.append(Finding("zero-filled frames", "warning",
                               f"{n_bad} of {len(damaged)} frames hold runs of zeros "
                               f"from lost packets"))
    except Exception:
        pass
    return out


# --- noise spectrum ------------------------------------------------------------------

NOISE_RX_WARN_DB = 6.0          # one channel's floor this far above the others
NOISE_LOW_EXCESS_DB = 6.0       # floor near DC this far above the rest of the band
NOISE_LINE_DB = 10.0            # a narrow line this far above the local floor


def noise_spectrum(x: np.ndarray, stride: int = 1) -> dict:
    """
    Noise power spectrum of one RX channel, from x shaped [frames, chirps, samples].

    Chirps one loop apart (`stride` chirps: the same TX) are subtracted, which removes
    the static scene - a wall is the same in both. What is left is receiver noise,
    anything that moved, and interference not locked to the chirps. With fewer chirps
    per frame than that, consecutive frames are used.
    Returns 'psd_db': noise power per sample in each bin, dB relative to full scale
    (complex: all bins, fftshift order not applied; real: the positive half), and
    'floor_db', 'low_excess_db' (the lowest tenth of the band against the median) and
    'lines' [(bin, dB above the local floor)] of narrow peaks.
    """
    x = np.asarray(x)
    s = max(1, int(stride))
    d = x[:, s:] - x[:, :-s] if x.shape[1] > s else np.diff(x[:, :1], axis=0)
    d = d.reshape(-1, x.shape[-1])
    ns = d.shape[-1]
    w = np.hanning(ns)
    p = np.mean(np.abs(np.fft.fft((d - d.mean(axis=-1, keepdims=True)) * w, axis=-1)) ** 2,
                axis=0) / (np.sum(w * w) * 2.0)
    if not np.iscomplexobj(x):
        p = p[:ns // 2 + 1] * 2.0
    db = 10 * np.log10(p + 1e-30)
    n = db.size
    edge = max(2, n // 64)
    band = np.arange(edge, n - edge) if np.iscomplexobj(x) else np.arange(edge, n)
    if np.iscomplexobj(x):
        band = band[np.abs(band - n // 2) > edge]          # skip DC and Nyquist
    floor = float(np.median(db[band]))
    low = np.r_[1:max(2, n // 10)]
    low_excess = float(np.median(db[low]) - floor)
    from scipy.ndimage import median_filter
    local = median_filter(db, size=17, mode="wrap")
    lines = []
    for k in band:
        h = db[k] - local[k]
        peak = db[k] >= db[k - 1] and db[k] >= db[(k + 1) % n]
        narrow = db[k] - max(db[k - 3], db[(k + 3) % n]) >= NOISE_LINE_DB / 2
        if h >= NOISE_LINE_DB and peak and narrow:
            lines.append((int(k), float(h)))
    lines.sort(key=lambda t: -t[1])
    return {"psd_db": db, "floor_db": floor, "low_excess_db": low_excess, "lines": lines}


def _bin_range_m(params: dict, k: int, n: int, is_complex: bool) -> float | None:
    fs = (_num(params, "digOutSampleRate_MHz") or 0) * 1e6
    slope = (_num(params, "freqSlopeMHz_us") or 0) * 1e12
    if fs <= 0 or slope <= 0:
        return None
    kk = k - n if (is_complex and k > n // 2) else k
    return kk * fs / n * C_LIGHT / (2 * slope)


def noise_findings(params: dict, x: np.ndarray, live=None) -> list[Finding]:
    """Findings from the noise spectrum of every live RX of x [frames, chirps, rx, samples]."""
    live = list(range(x.shape[2])) if live is None else list(live)
    if not live or x.shape[0] * x.shape[1] < 3:
        return []
    stride = int(_num(params, "unique_chirps_frame") or _num(params, "uniqueChirps_effective") or 1)
    spec = {r: noise_spectrum(x[:, :, r, :], stride) for r in live}
    floors = np.array([spec[r]["floor_db"] for r in live])
    med = float(np.median(floors))
    out = []
    worst = live[int(np.argmax(floors))]
    if floors.max() - med > NOISE_RX_WARN_DB:
        out.append(Finding("noise floor", "warning",
                           f"RX{worst + 1} is {floors.max() - med:.1f} dB noisier than the "
                           f"others ({floors.max():.0f} vs {med:.0f} dBFS per sample)"))
    else:
        out.append(Finding("noise floor", "ok",
                           f"{med:.0f} dBFS per sample, channels within "
                           f"{floors.max() - floors.min():.1f} dB"))
    low = max(spec[r]["low_excess_db"] for r in live)
    if low > NOISE_LOW_EXCESS_DB:
        out.append(Finding("noise spectrum", "info",
                           f"the floor rises {low:.0f} dB towards short range - phase noise "
                           f"around strong near reflections, or something moving close"))
    lines = {}
    for r in live:
        for k, h in spec[r]["lines"][:5]:
            lines.setdefault(k, []).append(h)
    shared = sorted(((k, max(h)) for k, h in lines.items() if len(h) >= max(2, len(live) // 2)),
                    key=lambda t: -t[1])[:3]
    if shared:
        n = spec[live[0]]["psd_db"].size
        where = []
        for k, h in shared:
            r_m = _bin_range_m(params, k, x.shape[-1], np.iscomplexobj(x))
            where.append(f"{r_m:.2f} m" if r_m is not None else f"bin {k}")
            where[-1] += f" (+{h:.0f} dB)"
        out.append(Finding("noise spectrum", "info",
                           "narrow lines left after removing the static scene, on most "
                           "channels at once: " + ", ".join(where) + " - interference not "
                           "locked to the chirps, or a target that moved"))
    return out


# --- everything --------------------------------------------------------------------

def assess(stem: str, params: dict, bin_paths=(), cube: np.ndarray | None = None,
           size_ratio: float | None = None, check_zero_fill: bool = True) -> HealthReport:
    """
    The full report. Each level is skipped when its input is missing, and a check that
    fails is reported rather than allowed to stop the others.
    """
    rep = HealthReport(stem=stem)
    steps = [("logs", lambda: assess_params(params))]
    if bin_paths:
        steps.append(("raw files", lambda: assess_files(params, bin_paths)))
        steps.append(("raw data", lambda: assess_raw(params, bin_paths[0])))
    if cube is not None:
        steps.append(("decoded data",
                      lambda: assess_cube(params, cube, size_ratio, check_zero_fill)))
    for name, step in steps:
        try:
            rep.findings.extend(step())
        except Exception as e:
            rep.findings.append(Finding(name, "warning",
                                        f"check could not run: {type(e).__name__}: {e}"))
    return rep
