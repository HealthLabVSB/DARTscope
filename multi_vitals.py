"""
multi_vitals.py

Vital signs of several people at once (the Multi Vitals view): find the people in range
and azimuth, follow them through the recording, cut each one's cell out of the radar
cube - the range bins around them, seen through beams steered at them - and measure
breathing and heart rate there with Fastcheck's chain (fastcheck.vital_windows), the
one Fastcheck, the scan and the validation take agree on.

Three ways to find the people (method):

  "motion"     every BLOCK_S a map of what moves in range and azimuth, from the chirps
               of a frame against their mean (a walking person) and from the frames of
               the block against theirs (a breathing chest, a fidget), plus the peaks of
               1.5 s of frames where nobody was found within STILL_SEP_M (motion_maps);
               the peaks are followed by a tracker (constant velocity Kalman filter,
               Hungarian assignment, a track confirmed after CONFIRM[0] of its first
               CONFIRM[1] blocks and dropped after LOST_S without a detection, cut where
               it jumps, joined to its continuation and merged with its duplicate).
  "breathing"  still people: the motion of the frames over BREATH_MAP_S (a chest that
               breathes moves its echo all the time, a wall or a chair never), followed
               by the same tracker; a track whose cell does not breathe (Fastcheck's
               test, fastcheck.vital_windows) is dropped. Slower to react than "motion",
               but a person sitting still for minutes stays found.
  "range"      no angles: people at different ranges, the same 10 s motion summed over
               the virtual channels, each confirmed by its breathing. For an array whose
               angles cannot be trusted (the cascade before its calibration) or that has
               none; two people at one range are one.

The maps measure motion as the power of the complex echo against its mean, not as
breathing-band power: a chest moving 4 mm swings the phase of a 5 mm wave by +-10 rad, so
its echo spreads far beyond the breathing band (to 3 Hz and more) - a "noise" band above
the heart band held the breathing person 30 dB above the real noise, and normalising by it
put people where there was nobody. Breathing is measured on the phase, in the cell.

An angle sidelobe of a strong person is not a second one: a peak no stronger than
SIDELOBE_DB above what the array pattern leaks there from the people already found in
the same range bins is dropped (as CLEAN does).

Two beams to look at a person (beam):

  "capon"      (default) MVDR: unit gain towards the person, the rest of the array's
               degrees of freedom spent on nulling what else moves. The maps are Capon
               spectra - a person 12 dB weaker 30 deg beside another at the same range is
               a peak of its own, under Bartlett's main lobe it is none - and the heart is
               measured through Capon beams, the breathing through Bartlett's unless
               someone else leaks into the person's cell (see analyse). Against the
               somnograph on 36 seated recordings: heart within 5 bpm in 73 % of the
               windows, breathing within 2 /min in 85 % (Bartlett 56 % and 78 %, Fastcheck
               on the same recordings 56 % and 81 %).
  "bartlett"   the steering vector: the most signal from that direction, and the
               array's fixed pattern for telling a sidelobe from a person.

Vital signs are measured only where a person stays: a track's blocks are split into
stays (slower than MOVING_M_S, at least MIN_STAY_S long), and every stay is measured
with its cell fixed at the stay's median position. Walking stretches are drawn on the
map but get no rates - breathing and heartbeat are millimetres and fractions of one,
walking moves the chest by centimetres every step.

A reflection of a person (a wall behind them, the floor) breathes exactly like them.
A target whose breathing correlates with a stronger one's (|r| >= ECHO_CORR) is kept but
marked as its echo.

The array: the chirps of every TX slot averaged per frame (one sample per frame and
virtual channel; the vital bands are far below the frame rate), the per-element phase
correction and the channel calibration of the AoA views (AOA.py), and the element
positions of the selected or auto-detected antenna configuration. A chirp transmitting
on several TX at once (mask 5: TX1 and TX3 together) is one element whose response is
the sum of theirs; AoA views drop such chirps, here they add azimuth positions.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy import signal as sps
from scipy.optimize import linear_sum_assignment

import AOA
import antenna_config
import capture_params
import fastcheck
import vital_bands
import vitalsigns
from radar_processing import compute_range_fft_cube, range_fft_size

METHODS = ("motion", "breathing", "range")
BEAMS = ("bartlett", "capon")

BLOCK_S = 0.5                  # map and tracker step
ANGLES_DEG = np.arange(-75.0, 75.5, 1.0)   # a peak on the first or last angle is not located
R_MIN_M = 0.3                  # the antenna coupling and the radar's own housing below
R_MAX_M = 6.0
FAST_FRAMES = 2                # frames per block whose chirps give the walking motion
NOISE_PCT = 20.0               # noise floor of a map: this percentile ...
NOISE_BINS = 16                # ... over all angles and this many range bins either side
MOTION_DB = 12.0               # a person in the motion map stands this far above its floor
REL_DB = 25.0                  # and within this of the strongest peak of the block (its power)
SIDELOBE_DB = 6.0              # and this far above the pattern leakage of stronger ones
POOR_FIT_SAME_RANGE_DB = 6.0   # echoes that fit the array poorly: a peak this much weaker than a stronger
                               # one within PATCH_M in range is its sidelobe (theirs reach -5 dB, the
                               # model's -14; one person 2 m away left them 12 cm behind at +-16..61 deg)
CAPON_SAME_RANGE_DB = 15.0     # Capon maps have no fixed pattern: in a stronger peak's range
                               # bins a peak this much weaker is its coherent echo's artefact
MIN_SEP_M = 0.5                # two people closer than this are one detection
STILL_SEP_M = 1.0              # a peak of the 1.5 s map counts this far from those of 0.5 s
EDGE_SUPPRESSES = True         # a peak on the edge of the angle grid suppresses its leakage
EDGE_DEG = 70.0                # a peak this far out or further is the edge's (the antennas hardly see there):
                               # a still spot at -74 deg was the strongest peak of a walk's map half the time
PROMINENCE_DB = 6.0            # a peak stands this far above the median of its range over all angles: Capon
                               # spreads a walker's motion evenly around their range ring 12 dB below them,
                               # and every ripple of that ring was a peak - four "people" a block, all 0.9 m
                               # from the radar, the tracker jumping between them
MAX_PEOPLE = 6
BREATH_MAP_S = 10.0            # the breathing map sums this much time around a block
GATE_M = 0.8                   # a detection further than this from a track is not its
MEAS_SIGMA_M = 0.15            # tracker: position noise of a detection
ACCEL_SIGMA = 1.0              # ... and acceleration noise [m/s^2]
CONFIRM = (3, 5)               # a track is real after 3 detections in its first 5 blocks
LOST_S = 2.0                   # and gone after this long without one
LINK_GAP_S = 1.5               # a track starting this soon after another ended ...
LINK_SPEED = 1.5               # ... no further than a person walks at this speed [m/s] + GATE_M
                               # continues it (a walker lost at a corner of the square)
LINK_SLACK_M_S = 0.5           # ... and than the track's own recent speed plus this
SAME_SPOT_GAP_S = 20.0         # one that starts where another ended (MIN_SEP_M) this much later
                               # too: a still person lost while breathing shallowly
SAME_TRACK = 0.8               # tracks within MIN_SEP_M of each other for this share of the
                               # time they overlap are one person (a person found twice)
JUMP_M = 0.45                  # a track that jumps further than this and JUMP_X times its
JUMP_X = 2.5                   # recent steps changed person: it is cut there
MOVING_M_S = 0.2               # faster than this: walking, no vital signs
MIN_STAY_S = fastcheck.MIN_VITAL_S
PATCH_M = 0.25                 # the cell: range bins within this of the person (a torso is ~0.3 m deep)
LOADING = 0.1                  # Capon diagonal loading of the maps, of the mean channel power
CELL_LOADING = 0.5             # ... and of a person's beams for the heart (see analyse)
CROWDED_ISR = 0.1              # another person leaking this much of one's power: Capon for all
ECHO_CORR = 0.8                # breathing correlation with a stronger target: its echo ...
MAYBE_ECHO_CORR = 0.5          # ... and from this on, possibly
MIN_TRACK_S = 3.0              # shorter tracks, or ones detected in less than MIN_DETECTED
MIN_DETECTED = 0.6             # of their blocks, are not reported
GHOST_DB = 10.0                # a target this much weaker than a person, behind them (at least
GHOST_GAP_M = 0.4              # this much further and within GHOST_DEG of their direction) for most
GHOST_DEG = 20.0               # of their common time is a reflection of them (the room's walls)
MIN_PERSON_S = 8.0             # a target without vital signs followed for less than this is no person:
                               # on the 36 single-person seated recordings 43 of the 69 extra "people"
                               # were such, 3-7 s, 2-4 m behind the person, 13-19 dB above the floor ...
WALK_M = 1.5                   # ... unless it walks: gets this far from where it began, along a path
WALK_STRAIGHT = 1.5            # at most this many times as long (noise jumps about, people walk on)
C0 = 299_792_458.0


# --- the array -------------------------------------------------------------------------

@dataclass
class Array:
    slow: np.ndarray               # (frames, virtual channels, range bins) complex64
    range_m: np.ndarray
    fps: float
    damaged: np.ndarray            # (frames,) frames the recorder zero-filled
    steer: np.ndarray | None       # (angles, channels) steering vectors, None: range only
    angles_deg: np.ndarray
    fast_maps: np.ndarray          # (blocks, range, angles) walking motion power
    block_frames: int
    wavelength_m: float
    note: str = ""
    steer_r: np.ndarray | None = None  # (range bins, angles, channels): the exact paths near a large array
    near: object = None                # its radar_imaging.Array (steering_at a person's range)


def _bits(mask: int) -> list[int]:
    return [1 << k for k in range(32) if (int(mask) >> k) & 1]


def _steering(params: dict, n_rx: int, masks: list[int], angles_deg) -> tuple:
    """
    (steer (A, V) or None, slots used, element phase factors (V,) or None, note).

    Slots are the TX patterns of a loop; V = len(slots) * n_rx. A slot whose mask holds
    several TX contributes the sum of their responses. None when the geometry is unknown.
    """
    n_slots = max(1, len(masks))
    masks = [int(m) for m in masks] or [1]
    multi = any(len(_bits(m)) > 1 for m in masks)
    V_all = n_slots * n_rx
    if params.get("cascade") and not params.get("antenna_config"):
        # never guessed: a wrong array gives wrong angles silently (as in AOA.py)
        return None, list(range(n_slots)), None, (
            "the cascade has no calibration (Measure -> Cascade radar -> Calibration) - range only")
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            cfg = AOA._resolve_antenna_config(params, V_all, n_rx)
        except Exception:
            cfg = None
    s = np.sin(np.deg2rad(np.asarray(angles_deg, float)))[:, None]
    if cfg and antenna_config.validate(cfg)[0]:
        rx = np.asarray(cfg["rx_positions"], float)
        offs = {int(m): np.asarray(o, float) for m, o in zip(cfg.get("tx_masks") or [], cfg["tx_offsets"])}
        if rx.shape[0] == n_rx and all(b in offs for m in masks for b in _bits(m)):
            xs = [[rx[:, 0] + offs[b][0] for b in _bits(m)] for m in masks]     # per slot, per TX
            centre = np.mean(np.concatenate([np.concatenate(x) for x in xs]))
            cols = []
            for per_tx in xs:
                cols.append(sum(np.exp(1j * 2 * np.pi * (x - centre)[None, :] * s) for x in per_tx))
            steer = np.concatenate(cols, axis=1).astype(np.complex64)          # (A, V)
            factors = antenna_config.element_phase_factors(cfg, V_all)
            how = "selected" if params.get("antenna_config") else "auto-detected"
            note = f"array '{cfg.get('name')}' ({how}), {V_all} channels"
            if multi:
                note += ", chirps on several TX at once used as their sum"
            if params.get("cascade") and params.get("channel_calibration") is None:
                note += (" - NOT CALIBRATED: the channels' own phases are in the angles "
                         "(Measure -> Cascade radar -> Calibration)")
            if params.get("cascade") and n_slots * n_rx <= 8:
                note += (" - one TX: the cascade's 8 RX are two groups 8 wavelengths apart, a "
                         "person shows at several angles 6 deg apart; angles need six TX")
            return steer, list(range(n_slots)), factors, note
    # no configuration: the AoA views' own guess, for single-TX slots only
    slots = list(range(n_slots)) if not multi else [0]
    V = len(slots) * n_rx
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            x, _, _ = AOA._get_antenna_geometry(params, V, n_rx)
    except Exception as e:
        return None, slots, None, f"no array geometry ({e}) - range only"
    x = np.asarray(x, float)
    steer = np.exp(1j * 2 * np.pi * (x - x.mean())[None, :] * s).astype(np.complex64)
    return steer, slots, None, f"array from the AoA views' geometry, {V} channels"


NEAR_FIELD_M = 0.5             # an array whose far field (2 D^2 / lambda) begins beyond this gets its
                               # steering for every range bin: the exact paths TX -> point -> RX


def _near_steering(params: dict, n_rx: int, V: int, range_m: np.ndarray, angles_deg: np.ndarray,
                   lam: float):
    """
    (steer (R, A, V), radar_imaging.Array) for the cascade, or (None, None).

    Plane waves are wrong near the cascade's 18.5 wavelengths (its far field begins at
    2.7 m): someone seated 1.1 m in front came out at four angles 35 deg apart (the TX
    2 wavelengths apart get different errors), 3 dB from each other, and their smear
    raised the noise floor until a second person at 2 m, 12 dB weaker, stood 15 dB
    above it instead of 40 and was lost.
    """
    import radar_imaging
    try:
        a = radar_imaging.array_for(params, n_rx)
    except Exception:
        return None, None
    if a is None or a.n_virtual != V or a.far_field_wl() * lam < NEAR_FIELD_M:
        return None, None
    az = np.asarray(angles_deg, float)
    S = np.stack([a.steering(az, np.zeros_like(az), max(r, 0.05) / lam).T for r in range_m])
    return S.astype(np.complex64), a


def _wavefront_fit(arr: Array, windows: int = 7, window_s: float = 1.0) -> float | None:
    """
    For an array with exact paths (the cascade): median, over a few 1 s windows, of how
    much of the strongest motion's main component one direction explains - see
    radar_imaging.wavefront_fit. None for other arrays.
    """
    if arr.steer_r is None:
        return None
    F = arr.slow.shape[0]
    W = max(4, int(round(window_s * arr.fps)))
    ok = np.flatnonzero(arr.range_m >= R_MIN_M)
    if F < W or not ok.size:
        return None
    fits = []
    for f0 in np.linspace(0, F - W, windows).astype(int):
        X = arr.slow[f0:f0 + W].astype(np.complex128)
        X = X - X.mean(axis=0, keepdims=True)
        pm = (np.abs(X) ** 2).sum(axis=(0, 1))
        b = ok[int(np.argmax(pm[ok]))]
        Y = X[:, :, b]
        w, U = np.linalg.eigh(Y.T @ Y.conj())
        if w[-1] < 0.6 * w.sum():
            continue                       # not one thing moving
        u = U[:, -1]
        S = arr.steer_r[b]                 # (A, V)
        fits.append(float(np.max(np.abs(S.conj() @ u) ** 2 / np.real(np.einsum("av,av->a", S.conj(), S)))))
    return float(np.median(fits)) if fits else None


def _noise_floor(p: np.ndarray) -> np.ndarray:
    """(R, 1): NOISE_PCT percentile of p over all angles and NOISE_BINS bins either side."""
    R = p.shape[0]
    out = np.empty((R, 1))
    for r in range(R):
        lo, hi = max(0, r - NOISE_BINS), min(R, r + NOISE_BINS + 1)
        out[r, 0] = np.percentile(p[lo:hi], NOISE_PCT)
    return np.maximum(out, 1e-30)


def _capon_power(S: np.ndarray, C: np.ndarray) -> np.ndarray:
    """(R, A) Capon spectrum V / (a^H C^-1 a) of covariances C (R, V, V), loaded by LOADING."""
    V = C.shape[-1]
    C = 0.5 * (C + np.conj(np.swapaxes(C, 1, 2)))                    # Hermitian
    tr = np.maximum(np.real(np.trace(C, axis1=1, axis2=2)), 0.0) / V
    floor = 1e-6 * float(np.median(tr)) + 1e-30                       # an empty bin
    load = LOADING * tr + floor
    Ci = np.linalg.inv(C + load[:, None, None] * np.eye(V)[None])
    if S.ndim == 3:                                                   # a steering per range bin
        den = np.real(np.einsum("rav,rvw,raw->ra", S.conj(), Ci, S))
    else:
        den = np.real(np.einsum("av,rvw,aw->ra", S.conj(), Ci, S))
    return V / np.maximum(den, 1e-30)


def build_array(cube: np.ndarray, params: dict, fps: float | None = None,
                r_max: float = R_MAX_M, n_fft_range: int | None = None,
                angles_deg=ANGLES_DEG, use_angles: bool = True, capon: bool = False,
                progress_cb=None) -> Array:
    """
    The virtual array's slow time and the walking-motion map of every block (Bartlett,
    or Capon with capon=True: two people at one range within the main lobe of a small
    array are one Bartlett peak, a 12 dB weaker one beside a stronger one none at all).
    """
    cube = np.asarray(cube)
    if cube.ndim != 4 or cube.shape[0] == 0:
        raise ValueError(f"expected a cube (frames, chirps, RX, samples), got {cube.shape}")
    F, C, RX, S = cube.shape
    if fps is None:
        with contextlib.redirect_stdout(io.StringIO()):
            fps = capture_params.derive_fps(params)
    fps = float(fps)
    eff = max(1, int(params.get("uniqueChirps_effective") or 1))
    stride = max(1, int(params.get("unique_chirps_frame") or eff))
    masks = [int(m) for m in (params.get("tx_order_masks") or [1])][:eff]
    if C % stride:
        stride, eff, masks = 1, 1, masks[:1]
    L = C // stride
    angles_deg = np.asarray(angles_deg, float)
    if use_angles:
        steer, slots, factors, note = _steering(params, RX, masks, angles_deg)
    else:
        steer, slots, factors, note = None, list(range(min(eff, len(masks) or 1))), None, "range only (chosen)"
    if steer is None:
        angles_deg = np.zeros(1)
    V = len(slots) * RX
    nfft = range_fft_size(S, n_fft_range)
    lam = capture_params.wavelength_m(params, default_ghz=60.0)   # middle of the ADC window (roadmap 1.9)

    nb = max(1, int(round(BLOCK_S * fps)))
    B = int(np.ceil(F / nb))
    slow = None
    damaged = np.zeros(F, bool)
    fast = None
    steer_r = near = None
    range_m = None
    keep = None

    def prep(spec):
        """(n, slots, RX, R) range spectra -> (n, V, R) corrected virtual channels."""
        x = spec.reshape(spec.shape[0], V, spec.shape[-1])
        if factors is not None and factors.size == V:
            x = x * factors[None, :, None]
        with contextlib.redirect_stdout(io.StringIO()):
            x = AOA._apply_calibration(x, params, V, axis=1)
        return x

    for b in range(B):
        f0, f1 = b * nb, min(F, (b + 1) * nb)
        x = cube[f0:f1]
        n = x.shape[0]
        damaged[f0:f1] = vitalsigns.detect_damaged_frames(x)
        xs = x.reshape(n, L, stride, RX, S)[:, :, slots]                 # (n, L, slots, RX, S)
        with contextlib.redirect_stdout(io.StringIO()):
            spec, meta = compute_range_fft_cube(xs.mean(axis=1), params, n_fft_range=nfft,
                                                use_full_spectrum=False, return_complex=True)
        if slow is None:
            range_m = np.asarray(meta["range_m"], float)
            keep = np.flatnonzero(range_m <= r_max)
            if keep.size == 0:
                keep = np.arange(range_m.size)
            range_m = range_m[keep]
            slow = np.zeros((F, V, keep.size), np.complex64)
            fast = np.zeros((B, keep.size, angles_deg.size), np.float32)
            steer_r, near = (_near_steering(params, RX, V, range_m, angles_deg, lam) if steer is not None
                             else (None, None))
            if steer_r is not None:
                note += "; exact paths near the radar (its far field begins at %.1f m)" % (
                    near.far_field_wl() * lam)
        slow[f0:f1] = prep(spec[..., keep])
        # walking: the chirps of a frame against their mean, in a few frames of the block
        good = np.flatnonzero(~damaged[f0:f1])
        if good.size and L > 1:
            pick = good[np.unique(np.linspace(0, good.size - 1, min(FAST_FRAMES, good.size)).round().astype(int))]
            acc = 0.0
            for k in pick:
                with contextlib.redirect_stdout(io.StringIO()):
                    sp, _ = compute_range_fft_cube(xs[k], params, n_fft_range=nfft,
                                                   use_full_spectrum=False, return_complex=True)
                X = prep(sp[..., keep])                                       # (L, V, R)
                X = X - X.mean(axis=0, keepdims=True)
                if steer is None:
                    acc = acc + (np.abs(X) ** 2).mean(axis=(0, 1))[:, None]
                elif capon:
                    acc = acc + np.einsum("lvr,lwr->rvw", X, X.conj()) / X.shape[0]
                elif steer_r is not None:
                    Y = np.einsum("rav,lvr->lra", steer_r.conj(), X) / np.sqrt(V)
                    acc = acc + (np.abs(Y) ** 2).mean(axis=0)
                else:
                    Y = np.einsum("av,lvr->lra", steer.conj(), X) / np.sqrt(V)
                    acc = acc + (np.abs(Y) ** 2).mean(axis=0)
            acc = acc / pick.size
            fast[b] = (_capon_power(steer_r if steer_r is not None else steer, acc)
                       if (capon and steer is not None) else acc)
        if progress_cb is not None:
            progress_cb(f1, F)
    return Array(slow=slow, range_m=range_m, fps=fps, damaged=damaged, steer=steer,
                 angles_deg=angles_deg, fast_maps=fast, block_frames=nb, wavelength_m=lam,
                 note=note, steer_r=steer_r, near=near)


# --- maps --------------------------------------------------------------------------------

def _beam_power(arr: Array, X: np.ndarray) -> np.ndarray:
    """(R, A) mean beam power of frames X (n, V, R); channels summed without angles."""
    if arr.steer is None:
        return (np.abs(X) ** 2).sum(axis=1).mean(axis=0)[:, None]
    if arr.steer_r is not None:
        Y = np.einsum("rav,fvr->fra", arr.steer_r.conj(), X) / np.sqrt(X.shape[1])
    else:
        Y = np.einsum("av,fvr->fra", arr.steer.conj(), X) / np.sqrt(X.shape[1])
    return (np.abs(Y) ** 2).mean(axis=0)


def _block_stats(arr: Array):
    """
    Per block: frames used, sum of x and sum of x x^H (or of |x|^2 without angles), of x
    less its mean over the recording. Without that the static echo dominates both
    E[x x^H] and m m^H, their float32 difference loses the motion to rounding and can turn
    negative - Capon then read 1e30 off the antenna coupling at 0 m.
    """
    B = arr.fast_maps.shape[0]
    nb = arr.block_frames
    F, V, R = arr.slow.shape
    good_all = ~arr.damaged
    centre = (arr.slow[good_all].mean(axis=0) if good_all.any() else arr.slow.mean(axis=0))
    n = np.zeros(B)
    s1 = np.zeros((B, R, V), np.complex128)
    s2 = (np.zeros((B, R, V, V), np.complex64) if arr.steer is not None
          else np.zeros((B, R, V), np.float64))
    for b in range(B):
        f0, f1 = b * nb, min(F, (b + 1) * nb)
        good = np.flatnonzero(~arr.damaged[f0:f1]) + f0
        if good.size == 0:
            continue
        X = arr.slow[good].astype(np.complex128) - centre[None]         # (n, V, R)
        n[b] = good.size
        s1[b] = X.sum(axis=0).T
        if arr.steer is not None:
            s2[b] = np.einsum("fvr,fwr->rvw", X, X.conj())
        else:
            s2[b] = (np.abs(X) ** 2).sum(axis=0).T
    return n, s1, s2


def slow_motion_maps(arr: Array, half_blocks: int, stats=None, capon: bool = False) -> np.ndarray:
    """
    (B, R, A) power of the frames against their mean over 2 half_blocks + 1 blocks: the
    beam power of the window's covariance, a^H (E[x x^H] - m m^H) a / V, from sums kept
    per block (a 10 s window at 100 fps beamformed frame by frame took 25 s a minute).
    """
    n, s1, s2 = stats if stats is not None else _block_stats(arr)
    B = n.size
    out = np.zeros_like(arr.fast_maps)
    S = arr.steer_r if arr.steer_r is not None else arr.steer
    V = arr.slow.shape[1]
    for b in range(B):
        lo, hi = max(0, b - half_blocks), min(B, b + half_blocks + 1)
        m = n[lo:hi].sum()
        if m < 2:
            continue
        # summed block by block in float64 (differences of running sums lose the motion
        # of a long recording to float32 rounding, as the static echo did)
        mu = s1[lo:hi].sum(axis=0) / m                                       # (R, V)
        sq = s2[lo:hi].sum(axis=0, dtype=np.float64 if S is None else np.complex128) / m
        if S is None:
            out[b, :, 0] = (sq - np.abs(mu) ** 2).sum(axis=1)
        else:
            Cv = sq - np.einsum("rv,rw->rvw", mu, mu.conj())
            if capon:
                out[b] = _capon_power(S, Cv)
            elif S.ndim == 3:
                out[b] = np.maximum(np.real(np.einsum("rav,rvw,raw->ra", S.conj(), Cv, S)), 0) / V
            else:
                out[b] = np.maximum(np.real(np.einsum("av,rvw,aw->ra", S.conj(), Cv, S)), 0) / V
    return out


def _snr_db(p: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.maximum(p / _noise_floor(p), 1e-12)) if p.any() else np.full(p.shape, -120.0)


LEVEL_RANGE_DB = 0.0           # REL_DB on the power plus this x log10(range): 40 (the power at 1 m)
                               # let far echoes in - 21 people too many in 35 single-person recordings


def _level_db(p: np.ndarray, range_m: np.ndarray) -> np.ndarray:
    """(R, A) power [dB] of the motion (+ LEVEL_RANGE_DB log10 r)."""
    return 10 * np.log10(np.maximum(p, 1e-30)) + LEVEL_RANGE_DB * np.log10(np.maximum(range_m, R_MIN_M))[:, None]


def levels(arr: Array, half_blocks: int, stats=None, capon: bool = False, fast: bool = True) -> np.ndarray:
    """(B, R, A) _level_db of the motion over 2 half_blocks + 1 blocks (and of the walking
    motion of the block with fast=True) - for REL_DB."""
    p = slow_motion_maps(arr, half_blocks, stats, capon)
    out = _level_db(p.reshape(-1, p.shape[-1]), np.tile(arr.range_m, p.shape[0])).reshape(p.shape)
    if fast:
        f = arr.fast_maps
        out = np.maximum(out, _level_db(f.reshape(-1, f.shape[-1]), np.tile(arr.range_m, f.shape[0])).reshape(f.shape))
    return out


def motion_maps(arr: Array, stats=None, capon: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    (B, R, A) motion SNR [dB] per block, twice: walking (the chirps of a frame) or a chest
    over the frames of the block (0.5 s); and the frames of 1.5 s around it. The short
    one places a walker where they are - over 1.5 s two people crossing at 1 m/s 0.6 m
    apart were smeared into three tracks jumping between them, 0.5 m behind; the long
    one keeps a seated person through the still second at the end of a breath (with the
    short one alone 20 % fewer heart windows of the seated recordings were measured).
    """
    short = slow_motion_maps(arr, 0, stats, capon)
    long_ = slow_motion_maps(arr, 1, stats, capon)
    a = np.zeros_like(arr.fast_maps)
    b_ = np.zeros_like(arr.fast_maps)
    for b in range(a.shape[0]):
        a[b] = np.maximum(_snr_db(short[b]), _snr_db(arr.fast_maps[b]))
        b_[b] = _snr_db(long_[b])
    return a, b_


def still_maps(arr: Array, stats=None, capon: bool = False) -> np.ndarray:
    """(B, R, A) SNR [dB] of the motion of the frames over BREATH_MAP_S around each block."""
    half = max(1, int(round(0.5 * BREATH_MAP_S / BLOCK_S)))
    slow = slow_motion_maps(arr, half, stats, capon)
    return np.stack([_snr_db(m) for m in slow]) if slow.size else slow


def pattern(arr: Array) -> np.ndarray | None:
    """(A, A) power the array leaks into angle i from a target at angle j (1 on the diagonal);
    (R, A, A), one for every range bin, with exact paths near the radar."""
    if arr.steer is None:
        return None
    if arr.steer_r is not None:
        S = arr.steer_r
        G = np.abs(np.einsum("rav,rbv->rab", S.conj(), S)) ** 2
        d = np.real(np.einsum("rav,rav->ra", S.conj(), S))
        return (G / (d[:, :, None] * d[:, None, :])).astype(np.float32)
    S = arr.steer
    G = np.abs(S.conj() @ S.T) ** 2
    d = np.real(np.einsum("av,av->a", S.conj(), S))
    return G / np.outer(d, d)


def detections(snr_db: np.ndarray, range_m: np.ndarray, angles_deg: np.ndarray,
               thresh_db: float, min_sep: float = MIN_SEP_M, max_n: int = MAX_PEOPLE,
               r_min: float = R_MIN_M, leak: np.ndarray | None = None,
               same_range_db: float | None = None, level_db: np.ndarray | None = None,
               same_range_m: float | None = None) -> list[tuple[float, float, float]]:
    """
    Peaks of one map: [(range m, angle deg, snr dB)], strongest first, min_sep apart and
    not a sidelobe of a stronger one (leak: pattern(arr)). REL_DB below the strongest:
    of level_db when given (the motion's power), else of the SNR - against the floor of its
    own range a person 2 m away behind the echoes of one 1.1 m in front stood 37 dB below
    them, in power 12 dB.
    """
    A = snr_db.shape[1]
    mx = ndimage.maximum_filter(snr_db, size=(3, 3 if A > 1 else 1), mode="nearest")
    ok = (snr_db >= mx) & (snr_db >= thresh_db) & (range_m[:, None] >= r_min)
    if A > 2:
        ok &= snr_db >= np.median(snr_db, axis=1, keepdims=True) + PROMINENCE_DB
    # the edge of the angle grid: the peak may lie beyond it, so it is not reported - but
    # it is someone, and what they leak elsewhere must not be reported either (a person
    # at 80 deg, outside the grid, came out at +16 deg)
    edge = np.zeros_like(ok)
    if A > 2:
        out_ = (np.abs(angles_deg) >= EDGE_DEG) | (np.arange(A) == 0) | (np.arange(A) == A - 1)
        edge[:, out_] = ok[:, out_]
        if not EDGE_SUPPRESSES:
            ok[:, out_] = False
    if not ok.any():
        return []
    if level_db is not None:
        ok &= level_db >= float(level_db[ok].max()) - REL_DB
    else:
        ok &= snr_db >= float(snr_db[ok].max()) - REL_DB
    r_i, a_i = np.nonzero(ok)
    order = np.argsort(-snr_db[r_i, a_i])
    out, pos, idx = [], [], []
    lin = 10 ** (snr_db / 10)
    for k in order:
        ri, ai = r_i[k], a_i[k]
        r, a = float(range_m[ri]), float(angles_deg[ai])
        xy = np.array([r * np.sin(np.deg2rad(a)), r * np.cos(np.deg2rad(a))])
        if any(np.hypot(*(xy - p)) < min_sep for p in pos):
            continue
        if leak is not None:
            near = [(rj, aj) for rj, aj in idx if abs(rj - ri) <= 1]
            lk = leak[ri] if leak.ndim == 3 else leak
            leaked = sum(lin[rj, aj] * lk[ai, aj] for rj, aj in near)
            if near and lin[ri, ai] < leaked * 10 ** (SIDELOBE_DB / 10):
                continue
        if same_range_db is not None and any(
                (abs(rj - ri) <= 1 if same_range_m is None else abs(range_m[rj] - range_m[ri]) <= same_range_m)
                and snr_db[rj, aj] - snr_db[ri, ai] > same_range_db
                for rj, aj in idx):
            continue
        pos.append(xy)
        idx.append((ri, ai))
        if edge[ri, ai]:
            continue
        out.append((r, a, float(snr_db[ri, ai])))
        if len(out) >= max_n:
            break
    return out


# --- tracking -------------------------------------------------------------------------------

@dataclass
class _Track:
    state: np.ndarray              # x, y, vx, vy
    P: np.ndarray
    first: int
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    id: int = 0
    rows: list = field(default_factory=list)     # (block, x, y, detected, snr)


def track(dets_per_block: list, dt: float = BLOCK_S) -> list[_Track]:
    """Follow the detections from block to block; the confirmed tracks."""
    Fm = np.eye(4)
    Fm[0, 2] = Fm[1, 3] = dt
    q = ACCEL_SIGMA ** 2
    Q = q * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0], [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                      [dt ** 3 / 2, 0, dt ** 2, 0], [0, dt ** 3 / 2, 0, dt ** 2]])
    H = np.zeros((2, 4))
    H[0, 0] = H[1, 1] = 1.0
    Rm = np.eye(2) * MEAS_SIGMA_M ** 2
    live: list[_Track] = []
    done: list[_Track] = []
    next_id = 1
    lost_blocks = max(1, int(round(LOST_S / dt)))
    for b, dets in enumerate(dets_per_block):
        Z = np.array([[r * np.sin(np.deg2rad(a)), r * np.cos(np.deg2rad(a))] for r, a, _ in dets]).reshape(-1, 2)
        for t in live:
            t.state = Fm @ t.state
            t.P = Fm @ t.P @ Fm.T + Q
        cost = np.full((len(live), len(Z)), 1e6)
        for i, t in enumerate(live):
            for j, z in enumerate(Z):
                d = float(np.hypot(*(z - t.state[:2])))
                if d <= GATE_M:
                    cost[i, j] = d
        matched_t, matched_d = set(), set()
        if len(live) and len(Z):
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] < 1e6:
                    t = live[i]
                    y = Z[j] - H @ t.state
                    Sm = H @ t.P @ H.T + Rm
                    K = t.P @ H.T @ np.linalg.inv(Sm)
                    t.state = t.state + K @ y
                    t.P = (np.eye(4) - K @ H) @ t.P
                    t.hits += 1
                    t.misses = 0
                    t.rows.append((b, t.state[0], t.state[1], True, dets[j][2]))
                    matched_t.add(i)
                    matched_d.add(j)
        for i, t in enumerate(live):
            if i in matched_t:
                continue
            t.misses += 1
            t.state[2:] *= 0.7               # a person who is not seen is most likely still
            t.rows.append((b, t.state[0], t.state[1], False, np.nan))
        for j, z in enumerate(Z):
            if j in matched_d:
                continue
            P0 = np.diag([MEAS_SIGMA_M ** 2, MEAS_SIGMA_M ** 2, 1.0, 1.0])
            live.append(_Track(state=np.array([z[0], z[1], 0.0, 0.0]), P=P0, first=b,
                               rows=[(b, z[0], z[1], True, dets[j][2])]))
        keep = []
        for t in live:
            if not t.confirmed and b - t.first < CONFIRM[1] and t.hits >= CONFIRM[0]:
                t.confirmed = True
                t.id = next_id
                next_id += 1
            young_dead = not t.confirmed and (b - t.first >= CONFIRM[1] - 1 and t.hits < CONFIRM[0]
                                              or t.misses > CONFIRM[1] - CONFIRM[0])
            if young_dead:
                continue
            if t.confirmed and t.misses > lost_blocks:
                done.append(t)
                continue
            keep.append(t)
        live = keep
    done += [t for t in live if t.confirmed]
    for t in done:                            # the misses at the end are not the person
        while t.rows and not t.rows[-1][3]:
            t.rows.pop()
    done = [piece for t in done if t.rows for piece in _split_jumps(t)]
    done = _merge_duplicates(done)
    done = _link(done, dt)
    done.sort(key=lambda t: t.rows[0][0])
    for i, t in enumerate(done):              # numbered in the order they appear
        t.id = i + 1
    return done


def _split_jumps(t: _Track) -> list:
    """
    A track cut where it jumps: a weak target 0.85 m behind a seated person jumped 0.5 m
    towards them in one block, its velocity carried its prediction onto the person, and
    it took over their detections - the person's own track died, their vital signs split
    in two. Cut, the piece at the person's place joins their track again (_link).
    """
    rows = t.rows
    det = [i for i, r in enumerate(rows) if r[3]]
    cuts = []
    for k in range(1, len(det)):
        i0, i1 = det[k - 1], det[k]
        step = float(np.hypot(rows[i1][1] - rows[i0][1], rows[i1][2] - rows[i0][2]))
        prev = [float(np.hypot(rows[det[j]][1] - rows[det[j - 1]][1], rows[det[j]][2] - rows[det[j - 1]][2]))
                / max(1, rows[det[j]][0] - rows[det[j - 1]][0]) for j in range(max(1, k - 4), k)]
        if len(prev) < 2:                  # a jump needs a pace to break: a walker's
            continue                       # first steps are 0.7 m a block at 1.4 m/s
        recent = float(np.mean(prev))
        if step > max(JUMP_M, JUMP_X * recent * max(1, rows[i1][0] - rows[i0][0])):
            cuts.append(i1)
    if not cuts:
        return [t]
    out, start = [], 0
    for c in cuts + [len(rows)]:
        piece = rows[start:c]
        while piece and not piece[-1][3]:
            piece = piece[:-1]
        if piece:
            out.append(_Track(state=t.state, P=t.P, first=int(piece[0][0]), hits=sum(r[3] for r in piece),
                              confirmed=True, id=t.id, rows=list(piece)))
        start = c
    return out


def _merge_duplicates(tracks: list) -> list:
    """
    One person followed by two tracks at once (a detection that jumped further than the
    gate, and the old track coasting until it was lost): keep the one detected more
    often, and fill in its blocks without a detection from the other.
    """
    tracks = sorted(tracks, key=lambda t: -sum(r[3] for r in t.rows))
    kept = []
    for t in tracks:
        rows_t = {r[0]: r for r in t.rows}
        for k in kept:
            rows_k = {r[0]: r for r in k.rows}
            both = sorted(set(rows_t) & set(rows_k))
            if not both:
                continue
            close = np.mean([np.hypot(rows_t[b][1] - rows_k[b][1], rows_t[b][2] - rows_k[b][2])
                             < MIN_SEP_M for b in both])
            if close >= SAME_TRACK:
                for b, r in rows_t.items():
                    if b not in rows_k or (r[3] and not rows_k[b][3]):
                        rows_k[b] = r
                b0, b1 = min(rows_k), max(rows_k)
                filled = []
                for b in range(b0, b1 + 1):            # a gap between them: as predicted
                    if b in rows_k:
                        filled.append(rows_k[b])
                    else:
                        prev = filled[-1]
                        filled.append((b, prev[1], prev[2], False, np.nan))
                k.rows = filled
                break
        else:
            kept.append(t)
    return kept


def _recent_speed(t: _Track, n: int = 4) -> float:
    """Mean speed [m/s] over the last n detected steps of a track."""
    det = [r for r in t.rows if r[3]][-(n + 1):]
    if len(det) < 2:
        return 0.0
    d = sum(float(np.hypot(b[1] - a[1], b[2] - a[2])) for a, b in zip(det, det[1:]))
    return d / max(1e-9, (det[-1][0] - det[0][0]) * BLOCK_S)


def _link(tracks: list, dt: float) -> list:
    """
    Join a track to one that starts within LINK_GAP_S (in the block it ends at the
    earliest) no further than it could have got at its own recent speed + LINK_SLACK_M_S,
    or to one that starts at the same spot within SAME_SPOT_GAP_S. A seated person must not
    be continued by a target a metre away that a walker could have reached.
    """
    tracks = sorted(tracks, key=lambda t: t.rows[0][0])
    used = set()
    for t in tracks:
        if id(t) in used:
            continue
        while True:
            end_b, ex, ey = t.rows[-1][0], t.rows[-1][1], t.rows[-1][2]
            v = _recent_speed(t)
            best, best_d = None, np.inf
            for u in tracks:
                if u is t or id(u) in used:
                    continue
                gap = u.rows[0][0] - end_b
                if gap < 0:
                    # a hand-over: u began elsewhere (a reflection), reached this person
                    # while they were still followed and went on as them after this
                    # track ended - it continues this track from there on. Only one that
                    # began after this one: a track that came near someone followed since
                    # long before took over their whole past (a person standing on four
                    # floor marks for two minutes vanished into a reflection's track)
                    at = {r[0]: r for r in u.rows}
                    if u.rows[0][0] <= t.rows[0][0] or u.rows[-1][0] <= end_b or end_b not in at:
                        continue
                    d = float(np.hypot(at[end_b][1] - ex, at[end_b][2] - ey))
                    if d < MIN_SEP_M and d < best_d:
                        best, best_d = u, d
                    continue
                if gap * dt > max(LINK_GAP_S, SAME_SPOT_GAP_S):
                    continue
                d = float(np.hypot(u.rows[0][1] - ex, u.rows[0][2] - ey))
                reach = min(LINK_SPEED, v + LINK_SLACK_M_S) * max(gap, 1) * dt + 0.5 * GATE_M
                walked = gap * dt <= LINK_GAP_S and d <= reach
                if (walked or d < MIN_SEP_M) and d < best_d:
                    best, best_d = u, d
            if best is None:
                break
            b0, x0, y0 = best.rows[0][0], best.rows[0][1], best.rows[0][2]
            for k in range(1, b0 - end_b):          # the gap, walked in a straight line
                w = k / (b0 - end_b)
                t.rows.append((end_b + k, ex + w * (x0 - ex), ey + w * (y0 - ey), False, np.nan))
            t.rows.extend(r for r in best.rows if r[0] > end_b)
            used.add(id(best))
    return [t for t in tracks if id(t) not in used]


# --- one person's cell -------------------------------------------------------------------------

def _beam_offsets(arr: Array) -> np.ndarray:
    """Three beams across a person: the centre and a quarter of the beamwidth either side."""
    if arr.steer is None:
        return np.zeros(1)
    a0 = arr.steer[np.argmin(np.abs(arr.angles_deg))]
    resp = np.abs(arr.steer.conj() @ a0) / np.vdot(a0, a0).real
    inside = arr.angles_deg[resp >= 0.5 ** 0.5]
    width = float(inside.max() - inside.min()) if inside.size else 20.0
    d = float(np.clip(width / 4.0, 2.0, 10.0))
    return np.array([-d, 0.0, d])


def steering_at(arr: Array, angle_deg: float, range_m: float | None = None) -> np.ndarray:
    """The steering vector towards angle_deg, interpolated on the map's grid (V,); near a
    large array the exact one at range_m."""
    if arr.near is not None and range_m is not None:
        return arr.near.steering([angle_deg], [0.0], max(range_m, 0.05) / arr.wavelength_m)[:, 0].astype(np.complex64)
    i = np.interp(angle_deg, arr.angles_deg, np.arange(arr.angles_deg.size))
    i0 = int(np.clip(np.floor(i), 0, arr.angles_deg.size - 1))
    i1 = min(i0 + 1, arr.angles_deg.size - 1)
    w = i - i0
    # the phase of each element grows linearly with sin(angle): interpolate the phase
    ph0, ph1 = np.angle(arr.steer[i0]), np.angle(arr.steer[i1])
    mag = (1 - w) * np.abs(arr.steer[i0]) + w * np.abs(arr.steer[i1])
    dph = np.angle(np.exp(1j * (ph1 - ph0)))
    return (mag * np.exp(1j * (ph0 + w * dph))).astype(np.complex64)


def cell(arr: Array, f0: int, f1: int, range_m: float, angle_deg: float,
         beam: str = "bartlett") -> tuple[np.ndarray, np.ndarray]:
    """
    One person's slow time: (frames, beams, bins) and the bins' ranges. Beams are the
    channels of fastcheck.vital_windows (all virtual channels when there are no angles).
    """
    dr = float(np.median(np.diff(arr.range_m))) if arr.range_m.size > 1 else PATCH_M
    half = max(PATCH_M, 1.5 * dr)
    bins = np.flatnonzero(np.abs(arr.range_m - range_m) <= half)
    if bins.size == 0:
        bins = np.array([int(np.argmin(np.abs(arr.range_m - range_m)))])
    X = arr.slow[f0:f1][:, :, bins]                                     # (n, V, nb)
    if arr.steer is None:
        return X, arr.range_m[bins]
    V = X.shape[1]
    A = np.stack([steering_at(arr, angle_deg + d, range_m) for d in _beam_offsets(arr)])   # (K, V)
    good = ~arr.damaged[f0:f1]
    out = np.zeros((X.shape[0], A.shape[0], bins.size), np.complex64)
    for j in range(bins.size):
        x = X[:, :, j]
        if beam == "capon" and good.sum() > 2 * V:
            g = x[good] - x[good].mean(axis=0, keepdims=True)
            Rc = (g.T @ g.conj()) / g.shape[0]                       # E[x x^H]
            Rc = Rc + np.eye(V) * CELL_LOADING * np.trace(Rc).real / V
            Ri = np.linalg.inv(Rc)
            W = (Ri @ A.T).T                                          # (K, V): R^-1 a
            W = W / np.einsum("kv,kv->k", A.conj(), W)[:, None]       # a^H w = 1
        else:
            W = A / np.einsum("kv,kv->k", A.conj(), A)[:, None].real  # a^H w = 1
        out[:, :, j] = x @ W.conj().T
    return out, arr.range_m[bins]


# --- the result ----------------------------------------------------------------------------------

@dataclass
class Person:
    id: int
    blocks: np.ndarray             # blocks where followed
    x_m: np.ndarray
    y_m: np.ndarray
    detected: np.ndarray
    moving: np.ndarray
    snr_db: float
    stays: list = field(default_factory=list)     # (f0, f1, range m, angle deg, best range m)
    win_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    win_half: np.ndarray = field(default_factory=lambda: np.zeros(0))   # half the window [s]
    br_rate: np.ndarray = field(default_factory=lambda: np.zeros(0))
    br_clear: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    hr_rate: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hr_valid: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    breaths: int = 0
    breaths_s: float = 0.0
    disp_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    disp_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    echo_of: int | None = None
    echo_corr: float = float("nan")    # breathing correlation with the stronger target below
    maybe_echo_of: int | None = None

    @property
    def range_m(self) -> np.ndarray:
        return np.hypot(self.x_m, self.y_m)

    @property
    def angle_deg(self) -> np.ndarray:
        return np.rad2deg(np.arctan2(self.x_m, self.y_m))

    def median_rates(self) -> tuple[float, float]:
        br = self.br_rate[self.br_clear]
        hr = self.hr_rate[self.hr_valid]
        return (float(np.median(br)) if br.size else float("nan"),
                float(np.median(hr)) if hr.size else float("nan"))

    def at(self, block: int, t_s: float, step_s: float) -> dict | None:
        """Where the person is in a block and the rates of the window around t_s."""
        k = np.flatnonzero(self.blocks == block)
        if k.size == 0:
            return None
        k = int(k[0])
        br = hr = float("nan")
        if self.win_t.size:
            # the window centred nearest t_s among those that hold it (the first window's
            # centre lies half a window into a stay: its start reads that window)
            d = np.abs(self.win_t - t_s)
            inside = d <= self.win_half + 1e-9
            if inside.any():
                i = int(np.flatnonzero(inside)[np.argmin(d[inside])])
                if self.br_clear[i]:
                    br = float(self.br_rate[i])
                if self.hr_valid[i]:
                    hr = float(self.hr_rate[i])
        return dict(x=float(self.x_m[k]), y=float(self.y_m[k]), moving=bool(self.moving[k]),
                    detected=bool(self.detected[k]), br=br, hr=hr)


@dataclass
class MultiVitals:
    stem: str
    fps: float
    frames: int
    method: str
    beam: str
    block_s: float
    block_t: np.ndarray            # centre of every block [s, from the segment start]
    range_m: np.ndarray
    angles_deg: np.ndarray
    map_db: np.ndarray             # (blocks, range, angles) the detection map
    map_thresh_db: float
    people: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    t0_s: float = 0.0

    def summary(self) -> str:
        real = [p for p in self.people if p.echo_of is None]
        if not real:
            return f"no person found ({self.method}, {self.beam})"
        parts = []
        for p in real:
            br, hr = p.median_rates()
            where = f"{np.median(p.range_m):.2f} m"
            if self.angles_deg.size > 1:
                where += f", {np.median(p.angle_deg):+.0f} deg"
            rates = []
            rates.append(f"breathing {br:.0f} /min" if np.isfinite(br) else "no clear breathing")
            if np.isfinite(hr):
                rates.append(f"heart {hr:.0f} bpm")
            if not p.stays:
                rates = (["moving, no vital signs"] if p.moving.mean() > 0.5 else
                         [f"still for less than {MIN_STAY_S:.0f} s, no vital signs"])
            if p.maybe_echo_of is not None:
                rates.append(f"possibly an echo of #{p.maybe_echo_of}")
            parts.append(f"#{p.id} at {where}: " + ", ".join(rates))
        echoes = [p for p in self.people if p.echo_of is not None]
        txt = f"{len(real)} person(s) - " + "; ".join(parts)
        if echoes:
            txt += "; " + ", ".join(f"#{p.id} is an echo of #{p.echo_of}" for p in echoes)
        return txt


def _stays(p_blocks, x, y, nb: int, F: int, dt: float):
    """(moving per block, [(f0, f1, block indices)]) for one track."""
    n = len(p_blocks)
    if n == 0:
        return np.zeros(0, bool), []
    k = max(1, int(round(1.0 / dt)))                      # speed over about two seconds
    # a median first: a detection of a seated person's head or a reflection for a block
    # is an outlier, not a step (averaged, it cut their stays)
    xs = ndimage.uniform_filter1d(ndimage.median_filter(np.asarray(x, float), size=2 * k + 1,
                                                        mode="nearest"), size=2 * k + 1, mode="nearest")
    ys = ndimage.uniform_filter1d(ndimage.median_filter(np.asarray(y, float), size=2 * k + 1,
                                                        mode="nearest"), size=2 * k + 1, mode="nearest")
    speed = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - k), min(n - 1, i + k)
        if b > a:
            speed[i] = np.hypot(xs[b] - xs[a], ys[b] - ys[a]) / ((p_blocks[b] - p_blocks[a]) * dt)
    moving = speed > MOVING_M_S
    # walking lasts: a moment of position noise of a seated person must not cut its stay
    runs = ndimage.label(moving)[0]
    for lab in range(1, runs.max() + 1):
        if (runs == lab).sum() * dt < 1.0:
            moving[runs == lab] = False
    out = []
    i = 0
    while i < n:
        if moving[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not moving[j + 1] and p_blocks[j + 1] == p_blocks[j] + 1:
            j += 1
        f0, f1 = p_blocks[i] * nb, min(F, (p_blocks[j] + 1) * nb)
        if f1 - f0 >= MIN_STAY_S / dt * nb:
            out.append((f0, f1, np.arange(i, j + 1)))
        i = j + 1
    return moving, out


def analyse(cube: np.ndarray, params: dict, *, fps: float | None = None, method: str = "motion",
            beam: str = "capon", r_max: float = R_MAX_M, max_people: int = MAX_PEOPLE,
            min_sep: float = MIN_SEP_M, bands=None, n_fft_range: int | None = None,
            stem: str = "", t0_s: float = 0.0, progress_cb=None) -> MultiVitals:
    """Find, follow and measure everyone in a capture (frames, chirps, RX, samples)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if beam not in BEAMS:
        raise ValueError(f"beam must be one of {BEAMS}")
    bands = bands or vital_bands.current()

    def prog(frac):
        if progress_cb is not None:
            progress_cb(int(round(100 * frac)), 100)

    F = np.asarray(cube).shape[0]
    capon = beam == "capon"
    arr = build_array(cube, params, fps=fps, r_max=r_max, n_fft_range=n_fft_range,
                      use_angles=(method != "range"), capon=capon,
                      progress_cb=lambda d, t: prog(0.55 * d / max(1, t)))
    notes = [arr.note]
    wf = _wavefront_fit(arr)
    poor_fit = False
    if wf is not None:
        import radar_imaging
        if wf < radar_imaging.WAVEFRONT_MIN:
            poor_fit = True
            notes.append(radar_imaging.WAVEFRONT_NOTE.format(wf))
            if capon:
                # Capon nulls an echo that does not fit its steering vector, in its own
                # direction too: one person seated in front of the cascade lit their
                # whole range ring evenly; Bartlett keeps the peak where they are
                capon, beam = False, "bartlett"
                arr = build_array(cube, params, fps=fps, r_max=r_max, n_fft_range=n_fft_range,
                                  use_angles=(method != "range"), capon=False,
                                  progress_cb=lambda d, t: prog(0.55 * d / max(1, t)))
                notes.append("Bartlett instead of Capon (the echoes do not fit the array's model)")
    if method != "range" and arr.steer is None:
        notes.append(f"'{method}' needs angles; found the people by range instead")
        method = "range"
    stats = _block_stats(arr)
    second = None
    lev2 = None
    if method == "motion":
        maps, second = motion_maps(arr, stats, capon)
        lev = levels(arr, 0, stats, capon, fast=True)
        lev2 = levels(arr, 1, stats, capon, fast=False)
    else:
        maps = still_maps(arr, stats, capon)
        lev = levels(arr, max(1, int(round(0.5 * BREATH_MAP_S / BLOCK_S))), stats, capon, fast=False)
    thresh = MOTION_DB
    prog(0.65)
    # Capon has no fixed pattern to leak through: its sidelobes are its own business
    leak = None if capon else pattern(arr)
    kw = dict(leak=leak, same_range_db=CAPON_SAME_RANGE_DB if capon else (POOR_FIT_SAME_RANGE_DB if poor_fit else None),
              same_range_m=PATCH_M if poor_fit else None)
    dets = []
    for b in range(maps.shape[0]):
        d = detections(maps[b], arr.range_m, arr.angles_deg, thresh, min_sep, max_people, level_db=lev[b], **kw)
        if second is not None and len(d) < max_people:
            # a still person in the pause of a breath, not the smear of a walker found above
            xy = [(r * np.sin(np.deg2rad(a)), r * np.cos(np.deg2rad(a))) for r, a, _ in d]
            for r, a, s in detections(second[b], arr.range_m, arr.angles_deg, thresh, min_sep,
                                      max_people, level_db=lev2[b], **kw):
                x, y = r * np.sin(np.deg2rad(a)), r * np.cos(np.deg2rad(a))
                if all(np.hypot(x - u, y - v) >= STILL_SEP_M for u, v in xy) and len(d) < max_people:
                    d.append((r, a, s))
                    xy.append((x, y))
        dets.append(d)
    tracks = track(dets, BLOCK_S)
    prog(0.7)
    nb, fps_ = arr.block_frames, arr.fps
    people = []
    # first where everyone is and where they stay, then their vital signs: a person's
    # beams depend on who else shares their range cell
    found = []
    for t in tracks:
        rows = np.array([(r[0], r[1], r[2], r[3], r[4]) for r in t.rows], dtype=float)
        if rows.size == 0:
            continue
        blocks = rows[:, 0].astype(int)
        if blocks.size * BLOCK_S < MIN_TRACK_S or rows[:, 3].mean() < MIN_DETECTED:
            continue
        x, y = rows[:, 1], rows[:, 2]
        if arr.steer is None:                     # range only: a person on the axis
            y, x = np.hypot(x, y), np.zeros_like(x)
        moving, stays = _stays(blocks, x, y, nb, F, BLOCK_S)
        snr = rows[:, 4]
        found.append((Person(id=t.id, blocks=blocks, x_m=x, y_m=y, detected=rows[:, 3].astype(bool),
                             moving=moving, snr_db=float(np.nanmedian(snr)) if np.isfinite(snr).any()
                             else float("nan")), stays))
    found = _merge_places(found, nb)
    leak_b = pattern(arr)
    dr = float(np.median(np.diff(arr.range_m))) if arr.range_m.size > 1 else PATCH_M

    def crowded(p, idx, r_c, a_c) -> bool:
        """
        Another person in this one's range cell leaking into its Bartlett beams at least a
        tenth of this person's own power (CROWDED_ISR): 12 dB stronger at -12.7 dB of
        leakage is as loud there as the person.
        """
        if leak_b is None:
            return False
        blocks = set(p.blocks[idx].tolist())
        ia = int(np.argmin(np.abs(arr.angles_deg - a_c)))
        for q, _ in found:
            if q is p:
                continue
            k = np.array([i for i, b in enumerate(q.blocks) if b in blocks], int)
            if k.size < 0.5 * len(blocks):
                continue
            r_o = float(np.median(q.range_m[k]))
            a_o = float(np.median(q.angle_deg[k]))
            if abs(r_o - r_c) > PATCH_M + dr:
                continue
            louder = 10 ** ((q.snr_db - p.snr_db) / 10) if np.isfinite(q.snr_db + p.snr_db) else 1.0
            lk = leak_b[int(np.argmin(np.abs(arr.range_m - r_c)))] if leak_b.ndim == 3 else leak_b
            if lk[ia, int(np.argmin(np.abs(arr.angles_deg - a_o)))] * louder >= CROWDED_ISR:
                return True
        return False

    short = 0
    for ti, (p, stays) in enumerate(found):
        x, y = p.x_m, p.y_m
        wt, wh, brr, brc, hrr, hrv, dt_, dm = [], [], [], [], [], [], [], []
        for f0, f1, idx in stays:
            r_c = float(np.median(np.hypot(x[idx], y[idx])))
            a_c = float(np.median(np.rad2deg(np.arctan2(x[idx], y[idx]))))
            # Capon: the heart from its beams, the breathing from Bartlett's. Against the
            # somnograph (36 seated recordings) Capon's beams put 73 % of the heart windows
            # within 5 bpm (Bartlett 63 %) but only 66 % of the breathing ones within
            # 2 /min (84 %): the breathing dominates the cell's covariance, and a beam
            # slightly off the person partly nulls and distorts it; the heart, 20 dB
            # weaker, gains more from what Capon suppresses around it. With someone else in
            # the cell and inside the main lobe Bartlett's breathing is theirs as much as
            # this person's: Capon gives both there.
            capon_all = beam == "capon" and crowded(p, idx, r_c, a_c)
            z, rr = cell(arr, f0, f1, r_c, a_c, "capon" if capon_all else "bartlett")
            with contextlib.redirect_stdout(io.StringIO()):
                v = fastcheck.vital_windows(z, fps_, bands, rr, arr.damaged[f0:f1])
                if (beam == "capon" and not capon_all and arr.steer is not None
                        and v.get("best_bin") is not None):
                    zc, _ = cell(arr, f0, f1, r_c, a_c, "capon")
                    vc = fastcheck.vital_windows(zc, fps_, bands, rr, arr.damaged[f0:f1])
                    if vc.get("best_bin") is not None and len(vc["win_t"]) == len(v["win_t"]):
                        v = dict(v, hr_rate=vc["hr_rate"], hr_valid=vc["hr_valid"])
            best = v.get("best_bin")
            p.stays.append((f0, f1, r_c, a_c, float(rr[best]) if best is not None else float("nan")))
            if best is None:
                continue
            wt.append(np.asarray(v["win_t"]) + f0 / fps_)
            wh.append(np.full(len(v["win_t"]), 0.5 * float(v.get("win_s", fastcheck.WINDOW_S))))
            brr.append(v["br_rate"])
            brc.append(v["clear"])
            hrr.append(v["hr_rate"])
            hrv.append(v["hr_valid"])
            b = v.get("breaths_total") or {}
            if b.get("ok"):
                p.breaths += int(b.get("n", 0))
                p.breaths_s += (f1 - f0) / fps_
            ph = fastcheck._chest_phase(z[:, :, best:best + 1], arr.damaged[f0:f1], fps_)[:, :, 0]
            # every beam's phase has its own offset: the median of the beams sample by sample
            # would jump between them
            ph = ph - np.median(ph, axis=0, keepdims=True)
            d = np.median(ph, axis=1) * arr.wavelength_m / (4 * np.pi) * 1e3
            if d.size > 3 * fps_:
                # the slow drift of the phase over a minute (20 mm on one recording) below
                # the breathing band, as Fastcheck removes it before its rates
                sos = sps.butter(2, 0.6 * bands.resp_hz[0], btype="highpass", fs=fps_, output="sos")
                d = sps.sosfiltfilt(sos, d)
            dt_.append(np.arange(f0, f1) / fps_)
            dm.append(d - np.median(d))
        if wt:
            p.win_t = np.concatenate(wt)
            p.win_half = np.concatenate(wh)
            p.br_rate = np.concatenate(brr)
            p.br_clear = np.concatenate(brc).astype(bool)
            p.hr_rate = np.concatenate(hrr)
            p.hr_valid = np.concatenate(hrv).astype(bool)
            p.disp_t = np.concatenate(dt_)
            p.disp_mm = np.concatenate(dm)
        # still people are confirmed by their breathing: a track that never breathes is
        # something else that moves (a fan, a curtain, the sidelobe of a walker)
        if method != "motion" and not wt:
            notes.append(f"a target at {np.median(p.range_m):.2f} m moved but did not breathe - dropped")
            continue
        if not wt and (p.blocks[-1] - p.blocks[0] + 1) * BLOCK_S < MIN_PERSON_S:
            net = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
            path = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
            if not (net >= WALK_M and path <= WALK_STRAIGHT * net):
                short += 1                         # a reflection, a hand, a moment of noise
                continue
        people.append(p)
        prog(0.7 + 0.28 * (ti + 1) / max(1, len(found)))
    if short:
        notes.append(f"{short} short target(s) without vital signs not shown (under {MIN_PERSON_S:g} s)")
    for i, p in enumerate(people):                # 1, 2, 3 ... after the dropped ones
        p.id = i + 1
    _mark_echoes(people, bands, fps_)
    _mark_ghosts(people)
    B = maps.shape[0]
    res = MultiVitals(stem=stem, fps=fps_, frames=F, method=method, beam=beam, block_s=BLOCK_S,
                      block_t=(np.arange(B) * nb + 0.5 * nb) / fps_, range_m=arr.range_m,
                      angles_deg=arr.angles_deg, map_db=maps.astype(np.float16),
                      map_thresh_db=thresh, people=people, notes=notes, t0_s=t0_s)
    prog(1.0)
    return res


def _merge_places(found: list, nb: int) -> list:
    """
    One still person is one place: two tracks that stay at the same spot (MIN_SEP_M) at
    different times - overlapping by at most a quarter of the shorter stay - are the same
    person. A reflection 1.4 m behind a seated person drifted onto them over 15 blocks
    and followed them from there, the person's own track ending: no jump to cut, and at
    the hand-over the two were 0.5 m apart. Merged here, the stays of both are measured
    as one person's; the path keeps the track detected more often where both exist.
    """
    def centre(p, idx):
        return float(np.median(p.x_m[idx])), float(np.median(p.y_m[idx]))

    def same_place(p, sp, q, sq) -> bool:
        for f0, f1, ia in sp:
            ca = centre(p, ia)
            for g0, g1, ib in sq:
                cb = centre(q, ib)
                over = max(0, min(f1, g1) - max(f0, g0))
                if np.hypot(ca[0] - cb[0], ca[1] - cb[1]) < MIN_SEP_M and \
                        over <= 0.25 * min(f1 - f0, g1 - g0):
                    return True
        return False

    changed = True
    while changed:
        changed = False
        for i in range(len(found)):
            for j in range(len(found)):
                p, sp = found[i]
                q, sq = found[j]
                if i == j or not sp or not sq or p.detected.sum() < q.detected.sum():
                    continue
                if not same_place(p, sp, q, sq):
                    continue
                rows = {int(b): (p.x_m[k], p.y_m[k], p.detected[k], p.moving[k]) for k, b in enumerate(p.blocks)}
                for k, b in enumerate(q.blocks):
                    rows.setdefault(int(b), (q.x_m[k], q.y_m[k], q.detected[k], q.moving[k]))
                blocks = np.array(sorted(rows))
                arr_ = np.array([rows[b] for b in blocks], dtype=float)
                m = Person(id=p.id, blocks=blocks, x_m=arr_[:, 0], y_m=arr_[:, 1],
                           detected=arr_[:, 2].astype(bool), moving=arr_[:, 3].astype(bool),
                           snr_db=max(p.snr_db, q.snr_db))
                spans = sorted([(f0, f1) for f0, f1, _ in sp])
                for g0, g1, _ in sq:                   # q's stays, less what p already has
                    for f0, f1 in spans:
                        if f0 <= g0 < f1:
                            g0 = f1
                        if f0 < g1 <= f1:
                            g1 = f0
                    if g1 - g0 >= MIN_STAY_S * nb / BLOCK_S:
                        spans.append((g0, g1))
                stays = []
                for f0, f1 in sorted(spans):
                    idx = np.flatnonzero((blocks >= f0 // nb) & (blocks * nb < f1))
                    if idx.size:
                        stays.append((f0, f1, idx))
                found[i] = (m, stays)
                del found[j]
                changed = True
                break
            if changed:
                break
    return found


def _mark_ghosts(people: list):
    """
    A target GHOST_DB or more weaker than a person and behind them - GHOST_GAP_M further,
    within GHOST_DEG of their direction - for at least half of the time both are followed
    is an echo of them: on the 36 single-person seated recordings 18 of the 26 extra
    "people" left after MIN_PERSON_S lay so, 14-36 dB below the person, whether their
    breathing correlated with the person's (up to 0.68) or not (from 0.04); 10 were left
    (with the walkers of MIN_PERSON_S), the heart and breathing of the person unchanged.
    """
    order = sorted(people, key=lambda p: -(p.snr_db if np.isfinite(p.snr_db) else -1e9))
    for i, p in enumerate(order):
        if p.echo_of is not None or not np.isfinite(p.snr_db):
            continue
        for q in order[:i]:
            if q.echo_of is not None or not q.snr_db - p.snr_db >= GHOST_DB:
                continue
            common, ip, iq = np.intersect1d(p.blocks, q.blocks, return_indices=True)
            if common.size < 0.5 * p.blocks.size:
                continue
            rp, rq = np.hypot(p.x_m[ip], p.y_m[ip]), np.hypot(q.x_m[iq], q.y_m[iq])
            da = np.rad2deg(np.arctan2(p.x_m[ip], p.y_m[ip]) - np.arctan2(q.x_m[iq], q.y_m[iq]))
            if np.mean((rp >= rq + GHOST_GAP_M) & (np.abs(da) <= GHOST_DEG)) >= 0.5:
                p.echo_of = q.id
                break


def _mark_echoes(people: list, bands, fps: float):
    """
    A target breathing in step with a stronger one is its echo (|r| >= ECHO_CORR) or
    possibly so (|r| >= MAYBE_ECHO_CORR). On the seated recordings the reflections of the
    subject correlated 0.85-0.99 with them, an unrelated target -0.2 to 0.05.
    """
    br = bands.resp_hz
    if fps <= 2 * br[1]:
        return
    sos = sps.butter(2, [br[0], min(br[1], 0.45 * fps)], btype="band", fs=fps, output="sos")
    order = sorted(people, key=lambda p: -(p.snr_db if np.isfinite(p.snr_db) else -1e9))
    for i, p in enumerate(order):
        if p.disp_t.size < int(MIN_STAY_S * fps):
            continue
        best, best_q = 0.0, None
        for q in order[:i]:
            if q.echo_of is not None or q.disp_t.size == 0:
                continue
            common, ip, iq = np.intersect1d(np.round(p.disp_t * fps).astype(int),
                                            np.round(q.disp_t * fps).astype(int),
                                            return_indices=True)
            if common.size < max(30, int(MIN_STAY_S * fps)):
                continue
            a = sps.sosfiltfilt(sos, p.disp_mm[ip])
            b = sps.sosfiltfilt(sos, q.disp_mm[iq])
            if a.std() == 0 or b.std() == 0:
                continue
            r = abs(float(np.corrcoef(a, b)[0, 1]))
            if r > best:
                best, best_q = r, q
        if best_q is not None:
            p.echo_corr = best
            if best >= ECHO_CORR:
                p.echo_of = best_q.id
            elif best >= MAYBE_ECHO_CORR:
                p.maybe_echo_of = best_q.id
