"""
acquisition/quality.py

A short report on a recording, right after it is made - so a person standing in front of
the radar sees at once whether the take is usable - and a score that ranks the
recordings of a series for the kind of movement being studied.

From the samples the unchanged loader decodes:

    data         frames complete, packets lost (from the recorder), ADC clipping
    static       the strongest fixed reflectors (range)
    motion       the scene with the static part removed: where the moving target is
                 (range of the most motion energy, and its spread over the recording),
                 how far above the noise it stands (motion SNR), in what part of the
                 frames something moved
    velocity     the range-Doppler content around the target, from the chirps of the
                 first transmitter of each loop (so TDM timing gives the right maximum
                 velocity): how fast it moved (95th percentile of |v|), how much of its
                 energy sits at the edge of the velocity axis - motion faster than the
                 maximum velocity folds there -, how many velocity bins it fills
    phase        the step of the target's phase from frame to frame; above about 90 deg
                 the frame rate is too low to follow the motion (vital signs)

These are heuristics for comparing takes of the same movement with different settings,
not a physiological analysis; DARTscope's views (Fastcheck, Range-Doppler, Micro-Doppler,
Vitals, AoA) remain the place to look at a recording properly.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .config import RadarConfig
from .params import MIN_RANGE_M, derived, virtual_antennas

C0 = 299_792_458.0
MAX_FRAMES = 400                 # frames analysed (evenly spread over a longer recording)
MAX_BYTES = 1.5e9                # larger recordings: open them in DARTscope instead
CHUNK_FRAMES = 40
MOTION_FRAME_DB = 10.0           # a frame "has motion" this far above the floor
NO_MOTION_DB = 8.0               # below this, nothing clearly moved
TRACK_HALF_BINS = 3              # range bins either side of the target for its Doppler
EDGE = 0.85                      # |v| above this fraction of v_max: the folding zone
EDGE_WARN = 0.03                 # this share of the motion's Doppler energy there: folding
OCCUPIED_DB = 20.0               # a velocity bin within this of the motion's peak is occupied
CLIP_LEVEL = 0.98
CLIP_WARN = 1e-4
PHASE_WARN_DEG = 90.0
MIN_DOPPLER_LOOPS = 8            # fewer chirps per TX per frame: no velocity figures

OBJECTIVES = {
    "general": ("General movement", "motion well above the noise, no folding, no data loss"),
    "walking": ("Walking / whole-body movement", "as general, folding weighs more, finer "
                "velocity detail helps"),
    "limbs": ("Fast limbs (micro-Doppler)", "no folding above all; many velocity bins filled"),
    "vitals": ("Vital signs (breathing, heart)", "a steady target, >= 20 fps, small phase "
               "steps between frames"),
    "angles": ("Angles (AoA, several people)", "more virtual antennas (TX x RX), motion SNR"),
}


@dataclass
class Quality:
    metrics: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)          # (level, what, message)

    @property
    def verdict(self) -> str:
        levels = [f[0] for f in self.findings]
        for lv in ("error", "warning", "info"):
            if lv in levels:
                return lv
        return "ok"

    def headline(self) -> str:
        m = self.metrics
        if m.get("skipped"):
            return f"not analysed: {m['skipped']}"
        parts = [f"{m.get('frames_recorded', '?')} frames"]
        if m.get("zero_filled_packets"):
            parts.append(f"{m['zero_filled_packets']} packets LOST")
        if m.get("motion_detected"):
            parts.append(f"motion at {m['target_range_m']:.2f} m "
                         f"({m['range_p05_m']:.2f}-{m['range_p95_m']:.2f} m), "
                         f"SNR {m['motion_snr_db']:.0f} dB")
            if m.get("v95_ms") is not None:
                parts.append(f"|v| up to {m['v95_ms']:.2f} of +-{m['max_vel_ms']:.2f} m/s")
        else:
            parts.append(f"no clear motion (SNR {m.get('motion_snr_db', 0):.0f} dB)")
        return ", ".join(parts)

    def lines(self) -> list[str]:
        out = [self.headline()]
        for level, what, msg in self.findings:
            out.append(f"{level.upper():7s} {what}: {msg}")
        return out

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "headline": self.headline(), "metrics": self.metrics,
                "findings": [list(f) for f in self.findings]}


# --- the analysis ---------------------------------------------------------------------------

def _hann(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32) if n > 1 else np.ones(1, np.float32)


def _fft(x, axis):
    try:
        import scipy.fft as sfft
        return sfft.fft(x, axis=axis, workers=-1)
    except ImportError:                                    # numpy returns complex128
        return np.fft.fft(x, axis=axis).astype(np.complex64)


def _peaks(profile: np.ndarray, ranges: np.ndarray, k0: int, n: int = 3) -> list:
    p = profile.copy()
    p[:k0] = 0
    out = []
    for _ in range(n):
        k = int(np.argmax(p))
        if p[k] <= 0:
            break
        out.append((round(float(ranges[k]), 2), round(float(10 * np.log10(p[k] + 1e-30)), 1)))
        p[max(0, k - 2):k + 3] = 0
    return out


def analyze_cube(cube: np.ndarray, cfg: RadarConfig, progress=None) -> Quality:
    """The metrics of a decoded cube (frames, chirps, RX, samples) of configuration cfg."""
    q = Quality()
    m = q.metrics
    d = derived(cfg)
    m.update({k: d[k] for k in ("range_res_cm", "max_range_m", "max_vel_ms", "vel_res_ms", "fps",
                                "rate_mb_s", "virtual_antennas")})
    F, C, R, N = cube.shape
    U = max(1, cfg.unique_chirps)
    L = max(1, C // U)
    ph = cfg.physical()
    slope = ph["slope_mhz_us"] * 1e12
    fs = cfg.profile.dig_out_sample_rate * 1e3
    # the middle of the sampled band, as derived() (chirp_design.performance) takes it: the
    # start frequency gave a folding limit 3 % above the max_vel_ms the report states
    f_mid = ph["start_ghz"] * 1e9 + slope * (ph["adc_start_us"] * 1e-6 + N / fs / 2)
    lam = C0 / f_mid
    complex_ = np.iscomplexobj(cube)
    K = int(0.9 * N) if complex_ else N // 2
    dr = C0 * fs / (2 * slope * N)
    ranges = np.arange(K) * dr
    k0 = min(K - 1, max(1, int(math.ceil(MIN_RANGE_M / dr))))
    tc_same = U * cfg.chirp_time_us * 1e-6
    v_max = lam / (4 * tc_same)
    m["frames_recorded"] = int(F)

    # ADC clipping on the whole recording
    step = max(1, F // 50)
    sub = cube[::step]
    clip = float(np.mean((np.abs(sub.real) >= CLIP_LEVEL) | (np.abs(sub.imag) >= CLIP_LEVEL))
                 if complex_ else np.mean(np.abs(sub) >= CLIP_LEVEL))
    m["clip_fraction"] = clip

    frames = np.arange(F) if F <= MAX_FRAMES else np.unique(
        np.linspace(0, F - 1, MAX_FRAMES).round().astype(int))
    m["frames_analysed"] = int(len(frames))
    win = _hann(N)[None, None, None, :]

    def slot0(fr):
        x = np.asarray(cube[fr][:, 0::U], dtype=np.complex64)           # (f, L, R, N)
        return x - x.mean(axis=-1, keepdims=True)

    # static scene: the mean chirp (the FFT is linear, so the mean first)
    acc = np.zeros((R, N), np.complex128)
    for i in range(0, len(frames), CHUNK_FRAMES):
        acc += slot0(frames[i:i + CHUNK_FRAMES]).sum(axis=(0, 1))
    mean_chirp = (acc / (len(frames) * L)).astype(np.complex64)
    S = _fft(mean_chirp[None, None] * win, axis=-1)[0, 0, :, :K]     # (R, K)
    m["static_reflectors"] = _peaks((np.abs(S) ** 2).sum(axis=0), ranges, k0)

    # motion energy per frame and range
    E = np.zeros((len(frames), K), np.float64)
    for i in range(0, len(frames), CHUNK_FRAMES):
        X = _fft(slot0(frames[i:i + CHUNK_FRAMES]) * win, axis=-1)[..., :K]
        M = X - S[None, None]
        E[i:i + len(X)] = (np.abs(M) ** 2).mean(axis=(1, 2))
        if progress:
            progress(0.5 * (i + len(X)) / len(frames))
    Emean = E.mean(axis=0)
    floor = float(np.percentile(Emean[k0:], 25)) if K > k0 + 4 else float(np.median(Emean) + 1e-30)
    floor = max(floor, 1e-30)
    kt = k0 + int(np.argmax(Emean[k0:]))
    snr = float(10 * np.log10(Emean[kt] / floor))
    frame_snr = 10 * np.log10(E[:, k0:].max(axis=1) / floor)
    moving = frame_snr >= MOTION_FRAME_DB
    track = k0 + np.argmax(E[:, k0:], axis=1)
    m["motion_snr_db"] = round(snr, 1)
    m["motion_detected"] = bool(snr >= NO_MOTION_DB and moving.any())
    m["moving_fraction"] = round(float(moving.mean()), 3)
    m["target_range_m"] = round(float(ranges[kt]), 3)
    tr = ranges[track[moving]] if moving.any() else ranges[[kt]]
    m["range_p05_m"] = round(float(np.percentile(tr, 5)), 3)
    m["range_p95_m"] = round(float(np.percentile(tr, 95)), 3)
    m["range_span_m"] = round(m["range_p95_m"] - m["range_p05_m"], 3)
    m["max_range_usable_m"] = round(float(K * dr), 2)
    m["target_near_max"] = bool(m["motion_detected"] and m["range_p95_m"] > 0.9 * K * dr)

    # speed from the range track - independent of the Doppler, so it sees folding the
    # velocity axis cannot: the centroid of the motion around the tracked bin, median of
    # three against jumps between body parts, 80th percentile of |dr/dt|
    m["v_range_p80_ms"] = None
    if m["motion_detected"] and moving.sum() >= 5:
        rc = np.full(len(frames), np.nan)
        for i in np.where(moving)[0]:
            lo, hi = max(k0, track[i] - 2), min(K, track[i] + 3)
            w = E[i, lo:hi]
            rc[i] = float((w * ranges[lo:hi]).sum() / max(w.sum(), 1e-30))
        med = rc.copy()
        for i in range(1, len(rc) - 1):
            trio = rc[i - 1:i + 2]
            if not np.isnan(trio).any():
                med[i] = float(np.median(trio))
        dt = np.diff(frames) * (cfg.frame.period_ms * 1e-3)
        speeds = np.abs(np.diff(med)) / np.where(dt > 0, dt, np.nan)
        speeds = speeds[np.isfinite(speeds)]
        if len(speeds) >= 4:
            m["v_range_p80_ms"] = round(float(np.percentile(speeds, 80)), 3)
    if not m["motion_detected"]:
        for k in ("target_range_m", "range_p05_m", "range_p95_m", "range_span_m"):
            m[k] = None

    # velocity: Doppler around the tracked target, frames with motion
    m["v95_ms"] = m["edge_fraction"] = m["occupied_bins"] = None
    # with fewer loops a velocity bin is metres per second wide: no velocity figures then
    if L >= MIN_DOPPLER_LOOPS and m["motion_detected"]:
        dwin = _hann(L)[None, :, None, None]
        noise_bins = np.where(Emean < floor * 2.0)[0]
        noise_bins = noise_bins[noise_bins >= k0]
        if len(noise_bins) < 4:
            noise_bins = np.argsort(Emean[k0:])[:max(4, (K - k0) // 4)] + k0
        V = np.zeros(L)
        Vn = np.zeros(L)
        idx = np.where(moving)[0]
        for i in range(0, len(idx), CHUNK_FRAMES):
            sel = idx[i:i + CHUNK_FRAMES]
            X = _fft(slot0(frames[sel]) * win, axis=-1)[..., :K] - S[None, None]
            D = np.fft.fftshift(_fft(X * dwin, axis=1), axes=1)
            P = (np.abs(D) ** 2).sum(axis=2)                              # (f, L, K)
            for j, f in enumerate(sel):
                lo, hi = max(k0, track[f] - TRACK_HALF_BINS), min(K, track[f] + TRACK_HALF_BINS + 1)
                V += P[j, :, lo:hi].sum(axis=1)
                Vn += (hi - lo) * np.median(P[j][:, noise_bins], axis=1)
            if progress:
                progress(0.5 + 0.5 * (i + len(sel)) / len(idx))
        v = (np.arange(L) - L // 2) * (2 * v_max / L)
        Vs = np.clip(V - Vn, 0, None)
        if Vs.sum() > 0:
            av = np.abs(v)
            order = np.argsort(av)
            cum = np.cumsum(Vs[order]) / Vs.sum()
            m["v95_ms"] = round(float(av[order][min(len(cum) - 1, int(np.searchsorted(cum, 0.95)))]), 3)
            m["edge_fraction"] = round(float(Vs[av >= EDGE * v_max].sum() / Vs.sum()), 3)
            # relative to the peak, not the noise: a strong target's window sidelobes
            # would otherwise fill every bin
            m["occupied_bins"] = int(np.sum(Vs > Vs.max() * 10 ** (-OCCUPIED_DB / 10)))

    # phase step from frame to frame at the target (consecutive frames)
    m["phase_step_p95_deg"] = None
    if m["motion_detected"] and F >= 8:
        n0 = max(0, F // 2 - 300)
        seg = np.arange(n0, min(F, n0 + 600))
        kern = (np.hanning(N) * np.exp(-2j * np.pi * kt * np.arange(N) / N)).astype(np.complex64)
        z = np.empty(len(seg), np.complex64)
        for i in range(0, len(seg), CHUNK_FRAMES):
            x = slot0(seg[i:i + CHUNK_FRAMES]).mean(axis=1)             # (f, R, N)
            z[i:i + len(x)] = (x @ kern).sum(axis=1)
        dphi = np.angle(z[1:] * np.conj(z[:-1]))
        m["phase_step_p95_deg"] = round(float(np.degrees(np.percentile(np.abs(dphi), 95))), 1)

    _findings(q, cfg)
    return q


def _findings(q: Quality, cfg: RadarConfig):
    m, f = q.metrics, q.findings
    if m.get("clip_fraction", 0) > CLIP_WARN:
        f.append(("warning", "ADC clipping", f"{100 * m['clip_fraction']:.3f} % of the samples at "
                                             f"full scale - lower the RX gain"))
    if not m.get("motion_detected"):
        f.append(("info", "motion", f"nothing clearly moved (motion {m.get('motion_snr_db', 0):.0f} dB "
                                    f"above the noise) - was somebody moving in front of the radar?"))
        return
    if m.get("moving_fraction", 1) < 0.3:
        f.append(("info", "motion", f"motion in {100 * m['moving_fraction']:.0f} % of the frames only"))
    if m.get("target_near_max"):
        f.append(("warning", "range", f"the target reaches {m['range_p95_m']:.2f} m of the "
                                      f"{m['max_range_usable_m']:.2f} m the configuration covers - "
                                      f"a lower slope or a higher sample rate"))
    if folded(m):
        why = []
        ef, vr = m.get("edge_fraction"), m.get("v_range_p80_ms")
        if ef is not None and ef > EDGE_WARN:
            why.append(f"{100 * ef:.0f} % of its Doppler energy at the edge of the velocity axis")
        if vr is not None and vr > m["max_vel_ms"]:
            why.append(f"its range changes at {vr:.2f} m/s")
        f.append(("warning", "velocity folding",
                  f"the movement is faster than the maximum velocity +-{m['max_vel_ms']:.2f} m/s "
                  f"({'; '.join(why)}) - fewer transmitters in the TDM pattern or shorter chirps "
                  f"(idle, ramp)"))
    ob = m.get("occupied_bins")
    if ob is not None and ob <= 2 and cfg.frame.num_loops >= 8:
        f.append(("info", "velocity detail", f"the motion fills {ob} velocity bin(s) of "
                                             f"{m['vel_res_ms']:.3f} m/s - more loops for finer detail"))
    ps = m.get("phase_step_p95_deg")
    if ps is not None and ps > PHASE_WARN_DEG:
        f.append(("info", "frame rate",
                  f"the target's phase steps {ps:.0f} deg from frame to frame (95 %): for "
                  f"vital-sign analysis of this movement more frames per second are needed"))


def folded(m: dict) -> bool:
    """Velocity folding: Doppler energy at the edge of the axis, or a range track faster
    than the maximum velocity."""
    ef, vr = m.get("edge_fraction"), m.get("v_range_p80_ms")
    return bool((ef is not None and ef > EDGE_WARN)
                or (vr is not None and m.get("max_vel_ms") and vr > m["max_vel_ms"]))


def score(metrics: dict, objective: str = "general") -> tuple[float, list]:
    """(score in dB-like points, [(term, points), ...]) - higher is better."""
    m = metrics
    terms = []
    if m.get("skipped"):
        return -999.0, [("not analysed", -999.0)]
    terms.append(("motion SNR", min(float(m.get("motion_snr_db") or 0.0), 45.0)))
    if m.get("zero_filled_packets") or m.get("incomplete"):
        terms.append(("data loss", -100.0))
    if (m.get("clip_fraction") or 0) > CLIP_WARN:
        terms.append(("ADC clipping", -10.0))
    w_fold = {"limbs": 80.0, "walking": 60.0, "general": 40.0, "vitals": 20.0, "angles": 30.0}
    ef = float(m.get("edge_fraction") or 0.0)
    vr, vmax = m.get("v_range_p80_ms"), m.get("max_vel_ms")
    if vr and vmax and vr > vmax:                   # folding the edge energy misses
        ef = max(ef, min(0.3, 0.1 * vr / vmax))
    if ef:
        terms.append(("velocity folding", -w_fold.get(objective, 40.0) * ef))
    if m.get("target_near_max"):
        terms.append(("target near maximum range", -10.0))
    ob = m.get("occupied_bins")
    if objective in ("limbs", "walking") and ob:
        w = 3.0 if objective == "limbs" else 1.5
        terms.append(("velocity detail", w * math.log2(max(1, min(int(ob), 64)))))
    if objective == "vitals":
        fps = float(m.get("fps") or 0)
        if fps < 20:
            terms.append(("frame rate below 20 fps", -(20 - fps)))
        ps = m.get("phase_step_p95_deg")
        if ps and ps > 45:
            terms.append(("phase steps between frames", -20.0 * min(1.0, (ps - 45) / 90)))
        span = float(m.get("range_span_m") or 0)
        if span > 0.15:
            terms.append(("target not steady in range", -min(10.0, 20 * (span - 0.15))))
    if objective == "angles":
        terms.append(("virtual antennas", 10 * math.log10(max(1, int(m.get("virtual_antennas") or 1)) / 4)))
    rate = float(m.get("rate_mb_s") or 0)
    if rate:
        terms.append(("data rate (lighter is better)", -0.05 * rate))
    return round(sum(t for _, t in terms), 2), [(n, round(t, 2)) for n, t in terms]


# --- a recording on disk --------------------------------------------------------------------------

def config_of(stem_path: str) -> RadarConfig:
    """The configuration of a recording: its sidecar, else its mmWave Studio log."""
    side = stem_path + "_acquisition.json"
    if os.path.isfile(side):
        with open(side, encoding="utf-8") as f:
            return RadarConfig.from_dict(json.load(f)["config"])
    from .studio_log import read_config
    return read_config(stem_path + "_LogFile.txt")


def analyze_recording(stem_path: str, cfg: RadarConfig | None = None, stats: dict | None = None,
                      verification: dict | None = None, progress=None) -> Quality:
    """Decode with the unchanged loader and analyse; stats/verification from the capture."""
    from dca1000_decode import parse_logfile_full
    from radar_processing import derive_capture_dimensions, get_all_bin_parts, load_raw_cube
    cfg = cfg or config_of(stem_path)
    first = stem_path + "_Raw_0.bin"
    parts = get_all_bin_parts(first) if os.path.isfile(first) else []
    size = sum(os.path.getsize(p) for p in parts)
    if size > MAX_BYTES:
        q = Quality(metrics={"skipped": f"{size / 1e9:.1f} GB - open it in DARTscope"})
        return q
    with contextlib.redirect_stdout(io.StringIO()):
        params, _ = parse_logfile_full(stem_path + "_LogFile.txt")
        dims = derive_capture_dimensions(params)
        cube = load_raw_cube(first, params, dims)
    q = analyze_cube(cube, cfg, progress=progress)
    del cube
    st = stats or {}
    q.metrics["zero_filled_packets"] = int(st.get("zero_filled_packets") or 0)
    q.metrics["packets"] = st.get("received")
    if verification is not None and not verification.get("ok", True):
        q.metrics["incomplete"] = True
    if q.metrics["zero_filled_packets"]:
        q.findings.insert(0, ("error", "data", f"{q.metrics['zero_filled_packets']} packets lost "
                                               f"(zero-filled) - repeat the take"))
    elif q.metrics.get("incomplete"):
        q.findings.insert(0, ("error", "data", "the loader check failed - see the recording's report"))
    return q


def save(q: Quality, stem_path: str) -> str:
    path = stem_path + "_quality.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(q.to_dict(), f, indent=2)
    return path
