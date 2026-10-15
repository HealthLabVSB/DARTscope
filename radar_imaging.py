"""
radar_imaging.py

4D radar imaging (the Imaging view): every frame a point cloud - where each reflection is
in three dimensions and how fast it moves towards or away from the radar - the objects it
forms, followed through the recording, and images of the scene: from above (bird's-eye),
from the radar's point of view (front, "radar vision") and in range and Doppler.

What a point is depends on the mode (MODES): what moves within a frame (Doppler), what
moves from frame to frame (micro-motion: a breathing chest), the still scene (the room);
people = the first two, all = everything. Every point records what found it (kind).

The chain, per frame:

  1. Range FFT of every chirp (DC removed, Hann; for complex samples the whole spectrum,
     twice the range of its positive half, less the bins just below DC). The chirps of
     every TX slot of a loop as slow time; their mean - the still scene - apart from
     what moves, and a Doppler FFT (Hann) of the rest.
  2. Moving: the range-Doppler power of all channels without the zero-Doppler row; cells
     standing threshold_db above an ordered statistic of their surroundings (_cfar), less
     the window sidelobes of stronger cells and the echo of every mover the still
     scene's removal leaves beside zero velocity. Every cell's snapshot is corrected for
     the time between the TX slots, its velocity unfolded beyond one slot's limit where
     two directions explain it clearly better (_unfold).
     Micro-motion: the change of the still scene over micro_window_s of frames around the
     frame against the thermal noise; the main direction of a range bin's change
     (_micro_points). Echoes - a wall's reflection of a breathing person - move in step
     with a nearer, stronger point (_mark_echoes).
     Still: range bins above the thermal noise (_still_noise).
  3. Angles (_angles): Bartlett over azimuth and elevation with the exact element
     positions of the antenna configuration (a chirp sent on several TX at once is the
     sum of their responses) and the channel calibration as in the AoA views; up to
     per_cell sources a cell, a further one only where it explains second_gain of what
     the others leave; near echoes with the exact paths TX -> point -> RX; sources of a
     cell refined against each other (RELAX) and between grid points on a quadratic
     surface; peaks on the edge of the field of view dropped.
  4. A source is kept in the cell where its own echo peaks in range and Doppler, its
     range and velocity refined there (_cell_sources).
  5. Points: range, velocity, azimuth, elevation, power, SNR, coherence, kind; x (across,
     angle sign as in the AoA views), y (away), z (up), with the radar's height and
     downward tilt applied.
  6. Objects: DBSCAN over the points of cluster_frames frames, a Kalman tracker; tracks
     whose velocity swings in step with a nearer, stronger one's are echoes (_echo_tracks).

Images of one frame (frame_images) are computed when shown.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import ndimage

import AOA
import antenna_config
import capture_params
import vitalsigns

C0 = 299_792_458.0
MODES = ("people", "moving", "micro", "still", "all")
KIND_MOVING, KIND_MICRO, KIND_STILL, KIND_ECHO = 0, 1, 2, 3   # what found a point (an echo: micro-motion
                                                              # in step with a nearer, stronger one)
POINT_FIELDS = ("frame", "range_m", "velocity_m_s", "azimuth_deg", "elevation_deg", "x_m", "y_m",
                "z_m", "power_db", "snr_db", "coherence", "kind", "cluster")


def _uses(mode: str, kind: int) -> bool:
    """Whether a mode looks for points of a kind: people = moving + micro, all = everything."""
    return {"people": (KIND_MOVING, KIND_MICRO), "moving": (KIND_MOVING,), "micro": (KIND_MICRO,),
            "still": (KIND_STILL,), "all": (KIND_MOVING, KIND_MICRO, KIND_STILL)}[mode].__contains__(kind)


@dataclass
class Settings:
    mode: str = "people"           # people (moving + micro), moving (Doppler in a frame), micro (motion
                                   # across frames: breathing), still (the room), all
    threshold_db: float = 13.0     # CFAR: a cell this far above its surroundings
    cfar: str = "os"               # "os" (ordered statistics: neighbouring targets do not hide each other) or "ca"
    guard: int = 2
    train: int = 8
    per_cell: int = 2              # points one range-Doppler cell may give
    still_per_cell: int = 4        # points one range bin of the still scene may give
    second_db: float = 10.0        # a second peak within this of the first ...
    second_gain: float = 0.5       # ... explaining this part of what the first leaves
    az_limit_deg: float = 70.0
    az_step_deg: float = 1.0
    el_limit_deg: float = 40.0
    el_step_deg: float = 2.0
    r_min_m: float = 0.3
    r_max_m: float | None = None
    micro_window_s: float = 1.0    # micro: the motion of this many seconds of frames ...
    echo_window_s: float = 10.0    # ... an echo: moves in step with a nearer one over this long
    unfold: bool = True            # velocities beyond the TDM-MIMO limit ...
    v_limit_m_s: float = 10.0      # ... up to this (people, cars in a car park)
    height_m: float = 0.0          # the radar above the floor
    tilt_deg: float = 0.0          # pointed down by
    cluster_eps_m: float = 0.5     # DBSCAN ...
    cluster_min: int = 3           # ... points in ...
    cluster_frames: int = 3        # ... the last this many frames
    max_points_frame: int = 500
    max_cells_frame: int = 160     # the strongest cells of a frame looked at for angles
    n_fft_range: int | None = None # range FFT size (the Analysis bar's), next power of two by default
    min_coherence: float = 0.0     # a cell's echo must be this much plane waves (0-1)
    confirm_people: bool = True    # a person's track must lie at someone Multi Vitals finds (its motion
                                   # maps reject the room's reflections far better than single frames)


# --- the array ---------------------------------------------------------------------------

@dataclass
class Array:
    slots: list                    # slot index in the loop -> list of TX element (x, y) offsets
    rx: np.ndarray                 # (RX, 2)
    used_slots: list               # which slots of the loop form the array
    factors: np.ndarray | None     # per-element phase correction of the configuration
    elevation: bool                # the array has a vertical extent
    note: str

    @property
    def n_virtual(self) -> int:
        return len(self.used_slots) * self.rx.shape[0]

    def _elements(self):
        """(tx (B, V, 2), rx (V, 2), on (B, V)): every channel's TX positions (B: the most
        TX one chirp sends on) and RX position, in wavelengths."""
        if getattr(self, "_el_cache", None) is None:
            nb = max(len(self.slots[s]) for s in self.used_slots)
            RX = self.rx.shape[0]
            V = len(self.used_slots) * RX
            tx = np.zeros((nb, V, 2))
            on = np.zeros((nb, V))
            for i, s in enumerate(self.used_slots):
                for b, off in enumerate(self.slots[s]):
                    tx[b, i * RX:(i + 1) * RX] = np.asarray(off, float)[None]
                    on[b, i * RX:(i + 1) * RX] = 1.0
            self._el_cache = (tx, np.tile(self.rx, (len(self.used_slots), 1)), on)
        return self._el_cache

    def steering(self, az_deg, el_deg, r_wl: float | None = None) -> np.ndarray:
        """
        (V, K) responses to directions (az_deg[k], el_deg[k]), AoA-view convention: plane
        waves, or with r_wl (the range in wavelengths) the exact paths TX -> point -> RX
        (as cascade/calibration.steering; the same plane waves far away). A chirp sent on
        several TX is the sum of their responses.
        """
        az = np.deg2rad(np.atleast_1d(np.asarray(az_deg, float)))
        el = np.deg2rad(np.atleast_1d(np.asarray(el_deg, float)))
        kx = np.sin(az) * np.cos(el)
        ky = np.sin(el)
        tx, rx, on = self._elements()
        if r_wl is None:
            x = tx[..., 0] + rx[None, :, 0]
            y = tx[..., 1] + rx[None, :, 1]
            ph = x[..., None] * kx + y[..., None] * ky                                   # (B, V, K)
        else:
            R = float(r_wl)
            P = R * np.stack([kx, ky, np.cos(az) * np.cos(el)], axis=1)                   # (K, 3)
            d_r = np.sqrt((P[None, :, 0] - rx[:, :1]) ** 2 + (P[None, :, 1] - rx[:, 1:]) ** 2
                          + P[None, :, 2] ** 2)                                           # (V, K)
            d_t = np.sqrt((P[None, None, :, 0] - tx[..., :1]) ** 2 + (P[None, None, :, 1] - tx[..., 1:]) ** 2
                          + P[None, None, :, 2] ** 2)                                     # (B, V, K)
            ph = 2 * R - d_t - d_r[None]
        A = np.exp(1j * 2 * np.pi * ph)
        return (A * on[..., None]).sum(axis=0) if A.shape[0] > 1 else A[0]

    def far_field_wl(self) -> float:
        """2 D^2 / lambda in wavelengths: nearer, plane waves are off by more than 22 degrees."""
        pos = np.concatenate([self.rx + np.asarray(o)[None] for s in self.used_slots for o in self.slots[s]])
        pos = np.vstack([pos, np.zeros((1, 2))])
        d = max(np.ptp(pos[:, 0]), np.ptp(pos[:, 1]), np.hypot(np.ptp(pos[:, 0]), np.ptp(pos[:, 1])))
        return 2.0 * d * d


def _bits(mask: int) -> list[int]:
    return [1 << k for k in range(32) if (int(mask) >> k) & 1]


def array_for(params: dict, n_rx: int) -> Array | None:
    """The virtual array of a capture, or None when its geometry is unknown (no angles)."""
    eff = max(1, int(params.get("uniqueChirps_effective") or 1))
    masks = [int(m) for m in (params.get("tx_order_masks") or [1])][:eff] or [1]
    V_all = len(masks) * n_rx
    if params.get("cascade") and not params.get("antenna_config"):
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            cfg = AOA._resolve_antenna_config(params, V_all, n_rx)
        except Exception:
            cfg = None
    if cfg and antenna_config.validate(cfg)[0]:
        rx = np.asarray(cfg["rx_positions"], float)
        offs = {int(m): np.asarray(o, float) for m, o in zip(cfg.get("tx_masks") or [], cfg["tx_offsets"])}
        if rx.shape[0] == n_rx and all(b in offs for m in masks for b in _bits(m)):
            slots = [[offs[b] for b in _bits(m)] for m in masks]
            ys = np.concatenate([rx[:, 1] + o[1] for s in slots for o in s])
            how = "selected" if params.get("antenna_config") else "auto-detected"
            return Array(slots=slots, rx=rx, used_slots=list(range(len(masks))),
                         factors=antenna_config.element_phase_factors(cfg, V_all),
                         elevation=bool(np.ptp(ys) > 0.2),
                         note=f"array '{cfg.get('name')}' ({how}), {V_all} channels")
    if params.get("cascade"):
        return None
    multi = any(len(_bits(m)) > 1 for m in masks)
    used = list(range(len(masks))) if not multi else [0]
    V = len(used) * n_rx
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            x, y, _ = AOA._get_antenna_geometry(params, V, n_rx)
    except Exception:
        return None
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.column_stack([x[:n_rx] - x[0], y[:n_rx] - y[0]])
    slots = [[np.array([x[i * n_rx] - x[0], y[i * n_rx] - y[0]])] for i in range(len(used))]
    slots += [[np.zeros(2)]] * (len(masks) - len(slots))
    return Array(slots=slots, rx=rx, used_slots=used, factors=None,
                 elevation=bool(np.ptp(y) > 0.2), note=f"array from the AoA views' geometry, {V} channels")


# --- one frame ---------------------------------------------------------------------------

@dataclass
class _Layout:
    L: int                         # loops
    stride: int                    # chirps per loop
    used: list                     # slot indices
    tc_s: float                    # time from one chirp to the next
    lam: float
    nfft_r: int
    keep: np.ndarray               # range bins kept
    range_m: np.ndarray
    nfft_d: int
    velocity: np.ndarray           # fftshifted velocity axis of one slot's Doppler
    f_d: np.ndarray                # its Doppler frequencies


def _layout(cube_shape, params: dict, arr: Array | None, s: Settings, complex_input: bool = True) -> _Layout:
    from radar_processing import _range_axis_m, range_fft_size
    F, C, RX, S = cube_shape
    eff = max(1, int(params.get("uniqueChirps_effective") or 1))
    stride = max(1, int(params.get("unique_chirps_frame") or eff))
    if C % stride:
        stride, eff = 1, 1
    L = C // stride
    used = arr.used_slots if arr is not None else list(range(eff))
    tc = (float(params.get("idleTime_us") or 0) + float(params.get("rampEndTime_us") or 0)) * 1e-6
    # middle of the ADC window, the wavelength of every view (capture_params.wavelength_m,
    # roadmap 1.9): the Doppler axis and the range of the near-field steering use it
    lam = capture_params.wavelength_m(params, default_ghz=60.0)
    nfft_r = range_fft_size(S, s.n_fft_range)
    fmt = str(params.get("dataFmt_adcFmt", "")).upper()
    # complex samples: the whole spectrum, as the other views (real ones mirror it)
    full = complex_input and "COMPLEX2X" not in fmt and "PSEUDOREAL" not in fmt
    r_axis = _range_axis_m(nfft_r, False, params, use_full_spectrum=full)
    r_max = s.r_max_m if s.r_max_m else float(r_axis[-1])
    ok = (r_axis >= 0) & (r_axis <= r_max)
    if full:
        ok[-3:] = False            # just below DC: the DC and its window, not the farthest range
    keep = np.flatnonzero(ok)
    nfft_d = max(L, 8)
    t_slot = stride * tc if tc > 0 else 1.0                    # one slot to its next sample
    f_d = np.fft.fftshift(np.fft.fftfreq(nfft_d, d=t_slot))
    return _Layout(L=L, stride=stride, used=used, tc_s=tc, lam=lam, nfft_r=nfft_r, keep=keep,
                   range_m=r_axis[keep], nfft_d=nfft_d, velocity=f_d * lam / 2.0, f_d=f_d)


def _spectra(frame: np.ndarray, params: dict, lay: _Layout, arr: Array | None):
    """
    (X (D, V, R) moving part per Doppler bin, S (V, R) still part) of one frame
    (C, RX, S), with the configuration's element phases and the calibration applied.
    """
    C, RX, Ns = frame.shape
    x = frame.astype(np.complex64) if np.iscomplexobj(frame) else frame.astype(np.float32)
    x = x - x.mean(axis=-1, keepdims=True)
    x = x * np.hanning(Ns)[None, None, :]
    spec = np.fft.fft(x, n=lay.nfft_r, axis=-1)[..., lay.keep]                # (C, RX, R)
    spec = spec.reshape(lay.L, lay.stride, RX, -1)[:, lay.used]                # (L, U, RX, R)
    spec = spec.reshape(lay.L, len(lay.used) * RX, -1)                          # (L, V, R)
    V = spec.shape[1]
    if arr is not None and arr.factors is not None and arr.factors.size >= V:
        spec = spec * arr.factors[:V][None, :, None]
    with contextlib.redirect_stdout(io.StringIO()):
        spec = AOA._apply_calibration(spec, params, V, axis=1)
    still = spec.mean(axis=0)
    mov = spec - still[None]
    if lay.L > 1:
        mov = mov * np.hanning(lay.L + 2)[1:-1][:, None, None]
    X = np.fft.fftshift(np.fft.fft(mov, n=lay.nfft_d, axis=0), axes=0)
    return X, still


OS_PERCENTILE = 25.0           # the noise of a cell: this percentile of its training cells
SIDELOBE_SPAN = 8              # a peak this close in range or Doppler to a stronger one ...
SIDELOBE_DB = 25.0             # ... and this much weaker is its window sidelobe (Hann: -31 dB)
CAND_DB = 12.0                 # a cell within this of the strongest near it may hold an echo of its own


OS_STEP = 3                    # the ordered statistic every this many cells, interpolated between


def _os_noise(padded: np.ndarray, fp: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    OS_PERCENTILE of the footprint's cells around every cell of the (rows, cols) map
    inside `padded`, computed every OS_STEP cells and interpolated (the noise changes
    slowly; a rank filter over every cell took 150 ms of a 12-channel frame).
    """
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(padded, fp.shape)                                   # (rows, cols, h, w)
    rr = np.unique(np.r_[np.arange(0, rows, OS_STEP), rows - 1])
    cc = np.unique(np.r_[np.arange(0, cols, OS_STEP), cols - 1])
    sub_ = win[rr][:, cc].reshape(rr.size, cc.size, -1)[..., fp.ravel()]
    coarse = np.percentile(sub_, OS_PERCENTILE, axis=-1)
    if rr.size == 1:
        return np.interp(np.arange(cols), cc, coarse[0])[None].repeat(rows, axis=0)
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((rr, cc), np.log(np.maximum(coarse, 1e-30)))
    R, Cg = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return np.exp(f(np.column_stack([R.ravel(), Cg.ravel()]))).reshape(rows, cols)


def _cfar(power: np.ndarray, s: Settings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (peaks, noise, cells) of a (D, R) power map. peaks: local maxima standing threshold_db
    above the OS_PERCENTILE of their training cells (2D, or along range only for a few
    Doppler bins), less the window sidelobes of stronger peaks on the same range or
    Doppler line. cells: every cell above the threshold within CAND_DB of the strongest
    around it and not such a sidelobe - an echo next to a stronger one is no local
    maximum of the summed power, but is one in its own direction (frame_points).

    Cell averaging, and even the median, lost a room: six reflectors 4 range cells apart
    (walls, furniture) raised each other's noise estimate until none stood out, 50-70 dB
    above the real noise. With "ca" the training cells are averaged (as the CFAR view).
    """
    rows, cols = power.shape
    g, tr = s.guard, s.train
    w = g + tr
    if rows > 2 * w:
        fp = np.ones((2 * w + 1, 2 * w + 1), bool)
        fp[tr:tr + 2 * g + 1, tr:tr + 2 * g + 1] = False
    else:
        fp = np.ones((1, 2 * w + 1), bool)
        fp[0, tr:tr + 2 * g + 1] = False
    # Doppler is circular (the fastest bins touch), range is not
    ph, pw = fp.shape[0] // 2, fp.shape[1] // 2
    padded = np.pad(power, ((ph, ph), (0, 0)), mode="wrap") if ph else power
    padded = np.pad(padded, ((0, 0), (pw, pw)), mode="edge")
    if s.cfar == "ca":
        k = fp.astype(float) / fp.sum()
        noise = ndimage.convolve(padded, k, mode="nearest")[ph:ph + rows, pw:pw + cols]
    else:
        noise = _os_noise(padded, fp, rows, cols)
    noise = np.maximum(noise, 1e-30)
    thr = 10 ** (s.threshold_db / 10)
    peaks = power >= ndimage.maximum_filter(power, size=(3 if rows >= 3 else 1, 3),
                                            mode=("wrap", "nearest"))
    cand = np.argwhere(peaks & (power > noise * thr))
    cand = cand[np.argsort(-power[cand[:, 0], cand[:, 1]])]
    mask = np.zeros_like(power, bool)
    kept = []
    lim = 10 ** (-SIDELOBE_DB / 10)
    for d, r in cand:
        p = power[d, r]
        if any(((dd == d and abs(rr - r) <= SIDELOBE_SPAN)
                or (rr == r and min(abs(dd - d), rows - abs(dd - d)) <= SIDELOBE_SPAN))
               and p < power[dd, rr] * lim for dd, rr in kept):
            continue
        kept.append((d, r))
        mask[d, r] = True
    near = ndimage.maximum_filter(power, size=(5 if rows >= 5 else 1, 5), mode=("wrap", "nearest"))
    cells = (power > noise * thr) & (power >= near * 10 ** (-CAND_DB / 10))
    for d, r in kept:                     # the window sidelobes of the peaks along their lines
        pk = power[d, r] * lim
        lo, hi = max(0, r - SIDELOBE_SPAN), min(cols, r + SIDELOBE_SPAN + 1)
        line = cells[d, lo:hi]
        line &= ~(power[d, lo:hi] < pk)
        dd = np.arange(d - SIDELOBE_SPAN, d + SIDELOBE_SPAN + 1) % rows
        cells[dd, r] &= ~(power[dd, r] < pk)
    cells |= mask
    return mask, noise, cells


def _parabolic(y0, y1, y2) -> float:
    """Offset of a peak between bins, from the power of three: a parabola through their logs
    (a Hann main lobe is close to a Gaussian - a parabola through the powers was off by
    up to a tenth of a bin)."""
    if min(y0, y1, y2) <= 0:
        return 0.0
    y0, y1, y2 = np.log(y0), np.log(y1), np.log(y2)
    den = y0 - 2 * y1 + y2
    return float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if den < 0 else 0.0


@dataclass
class _Grid:
    az: np.ndarray
    el: np.ndarray
    A: np.ndarray                  # (V, EL*AZ) steering, columns el-major
    norm: np.ndarray               # (EL*AZ,) |a|^2 of every column
    az_k: np.ndarray               # (EL*AZ,) direction of every column
    el_k: np.ndarray


def _grid(arr: Array, s: Settings) -> _Grid:
    az = np.arange(-s.az_limit_deg, s.az_limit_deg + 1e-9, s.az_step_deg)
    el = (np.arange(-s.el_limit_deg, s.el_limit_deg + 1e-9, s.el_step_deg) if arr.elevation
          else np.zeros(1))
    AZ, EL = np.meshgrid(az, el)
    A = arr.steering(AZ.ravel(), EL.ravel()).astype(np.complex64)
    return _Grid(az=az, el=el, A=A, norm=np.real(np.einsum("vk,vk->k", A.conj(), A)),
                 az_k=AZ.ravel(), el_k=EL.ravel())


NEAR_FACTOR = 20.0             # exact paths nearer than this many times 2 D^2 / lambda
WINDOW = 3                     # grid steps around a far-field peak searched with exact paths
SECOND_MIN = 0.02              # no further source once the found ones leave less of the echo


def _vec(cols, az: float, el: float) -> np.ndarray:
    return cols.arr.steering([az], [el], cols.r)[:, 0].astype(np.complex64)


class _Columns:
    """Steering columns of grid directions for one snapshot: the grid's plane waves, or
    for a near echo the exact paths at its range (computed for the few directions asked)."""

    def __init__(self, g: _Grid, arr: Array, r_wl: float | None):
        self.g, self.arr, self.r = g, arr, r_wl
        self.cache = {}

    def get(self, idx) -> np.ndarray:
        idx = np.atleast_1d(np.asarray(idx, int))
        if self.r is None:
            return self.g.A[:, idx]
        miss = [int(k) for k in idx if int(k) not in self.cache]
        if miss:
            A = self.arr.steering(self.g.az_k[miss], self.g.el_k[miss], self.r).astype(np.complex64)
            for j, k in enumerate(miss):
                self.cache[k] = A[:, j]
        return np.stack([self.cache[int(k)] for k in idx], axis=1)

    def beam(self, y: np.ndarray, idx) -> np.ndarray:
        A = self.get(idx)
        return np.abs(y.conj() @ A) ** 2 / np.real(np.einsum("vk,vk->k", A.conj(), A))


def _window(g: _Grid, k: int, w: int = WINDOW):
    nA, nE = g.az.size, g.el.size
    e0, a0 = divmod(int(k), nA)
    ee = np.arange(max(0, e0 - w), min(nE, e0 + w + 1))
    aa = np.arange(max(0, a0 - w), min(nA, a0 + w + 1))
    return ee, aa, (ee[:, None] * nA + aa[None, :]).ravel()


def _residual(y: np.ndarray, M: np.ndarray) -> np.ndarray:
    return y - M @ np.linalg.lstsq(M, y, rcond=None)[0]


_yy, _xx = (q.ravel().astype(float) for q in np.mgrid[-1:2, -1:2])
_QUAD_PINV = np.linalg.pinv(np.column_stack([np.ones(9), _yy, _xx, _yy * _yy, _yy * _xx, _xx * _xx]))


def _quad2d(img: np.ndarray) -> tuple[float, float]:
    """
    Peak offset (d_row, d_col) of a 3x3 power patch around its centre: a quadratic
    surface with a cross term through the logs. The cascade's elevation comes from one
    row 0.8 wavelengths up at one side: its beam is a long diagonal ridge in azimuth and
    elevation, and refining azimuth and elevation each on its own put an echo between
    two azimuth grid points 2.5 degrees too high.
    """
    z = np.log(np.maximum(img, 1e-30)).ravel()
    c = _QUAD_PINV @ z
    H = np.array([[2 * c[3], c[4]], [c[4], 2 * c[5]]])
    if np.linalg.det(H) <= 0 or H[0, 0] >= 0:
        return 0.0, 0.0
    d = -np.linalg.solve(H, c[1:3])
    return float(np.clip(d[0], -1, 1)), float(np.clip(d[1], -1, 1))


def _refine(y: np.ndarray, cols: _Columns, e: int, a: int) -> tuple[float, float, float]:
    """(az, el, power) of the peak at grid point (e, a), between grid points: a quadratic
    surface through the 3x3 around it, then once more through 3x3 half steps around that."""
    g = cols.g
    nA, nE = g.az.size, g.el.size
    sa = g.az[1] - g.az[0] if nA > 1 else 0.0
    se = g.el[1] - g.el[0] if nE > 1 else 0.0
    if nE == 1:
        if 0 < a < nA - 1:
            b = cols.beam(y, [a - 1, a, a + 1])
            az = g.az[a] + _parabolic(*b) * sa
        else:
            az = g.az[a]
        for _ in range(1):
            az_k = az + np.array([-0.5, 0.0, 0.5]) * sa
            b = _beam_at(y, cols, az_k, np.zeros(3))
            az = az + _parabolic(*b) * 0.5 * sa
        return float(az), 0.0, float(_beam_at(y, cols, [az], [0.0])[0])
    if not (0 < a < nA - 1 and 0 < e < nE - 1):
        return float(g.az[a]), float(g.el[e]), float(cols.beam(y, [e * nA + a])[0])
    idx = ((np.arange(e - 1, e + 2)[:, None]) * nA + np.arange(a - 1, a + 2)[None, :]).ravel()
    de, da = _quad2d(cols.beam(y, idx).reshape(3, 3))
    az, el = g.az[a] + da * sa, g.el[e] + de * se
    ee, aa = np.mgrid[-1:2, -1:2]
    b = _beam_at(y, cols, (az + aa.ravel() * 0.5 * sa), (el + ee.ravel() * 0.5 * se)).reshape(3, 3)
    de, da = _quad2d(b)
    az, el = az + da * 0.5 * sa, el + de * 0.5 * se
    return float(az), float(el), float(_beam_at(y, cols, [az], [el])[0])


def _beam_at(y: np.ndarray, cols: _Columns, az, el) -> np.ndarray:
    """Bartlett power of y towards directions between grid points."""
    A = cols.arr.steering(np.asarray(az, float), np.asarray(el, float), cols.r)
    return np.abs(y.conj() @ A) ** 2 / np.real(np.einsum("vk,vk->k", A.conj(), A))


def _peak_near(y: np.ndarray, cols: _Columns, k: int) -> tuple[int, float, float, float]:
    """
    The strongest direction around grid column k: (column, az, el, power), refined. The
    window follows the peak when it lies on its edge: a plane-wave peak of a near echo can
    be further off than the window (the cascade at 3.5 m: 8 degrees in elevation).
    """
    g = cols.g
    nA = g.az.size
    for _ in range(8):
        ee, aa, idx = _window(g, k)
        img = cols.beam(y, idx).reshape(ee.size, aa.size)
        ie, ia = np.unravel_index(np.argmax(img), img.shape)
        k_new = int(ee[ie] * nA + aa[ia])
        inner = (0 < ia < aa.size - 1 or aa[ia] in (0, nA - 1)) and \
                (0 < ie < ee.size - 1 or ee[ie] in (0, g.el.size - 1))
        if k_new == k or inner:
            k = k_new
            break
        k = k_new
    az, el, p = _refine(y, cols, k // nA, k % nA)
    return k, az, el, p


def _angles(Y: np.ndarray, arr: Array, g: _Grid, s: Settings, r_wl=None) -> tuple[list, np.ndarray]:
    """
    For snapshots Y (N, V) at ranges r_wl (N,) in wavelengths: up to per_cell (az, el,
    power) each, and how much of each snapshot's energy they explain together (1: plane
    waves from them alone; a mixture of many directions - multipath, noise - explains
    little).

    The strongest peak of the angle spectrum is a source; a further peak (within
    second_db of it) one when it explains second_gain of what the others leave - a
    sidelobe of a source, or the error of the steering model around it, explains little.
    The leak-based test (CLEAN) let a reflector 1.5 m from the cascade through as four
    sources at 30-57 degrees: the wavefront there is not plane (the far field of its
    18.5 wavelengths begins at 2.7 m) and a quarter of its energy fits no plane wave.
    Echoes nearer than NEAR_FACTOR times that are searched with the exact paths around
    the plane-wave peaks.

    A peak on the edge of the field of view is a source outside it: it takes its share
    out of the others but is no point (real captures put a tenth of their points on the
    edges of the grid).
    """
    nA, nE = g.az.size, g.el.size
    N = Y.shape[0]
    B = (np.abs(Y.conj() @ g.A) ** 2 / g.norm[None]).reshape(N, nE, nA)            # (N, EL, AZ)
    is_peak = B >= ndimage.maximum_filter(B, size=(1, 3, 3), mode="nearest")
    near_wl = NEAR_FACTOR * arr.far_field_wl()
    out, fit = [], np.zeros(N)
    second = 10 ** (-s.second_db / 10)
    sa = g.az[1] - g.az[0] if nA > 1 else 1.0
    se = g.el[1] - g.el[0] if nE > 1 else 1.0
    edge = lambda k: (k % nA) in (0, nA - 1) or (nE > 1 and (k // nA) in (0, nE - 1))
    for n in range(N):
        y = Y[n]
        r = None if r_wl is None or not (0 < r_wl[n] < near_wl) else float(r_wl[n])
        cols = _Columns(g, arr, r)
        energy = float(np.sum(np.abs(y) ** 2))
        img = B[n]
        cand = np.argwhere(is_peak[n])
        cand = cand[np.argsort(-img[cand[:, 0], cand[:, 1]])][: 4 * s.per_cell]
        found = []                                            # (column, az, el, power, steering)
        before = energy
        for e, a in cand:
            if found and img[e, a] < img[cand[0][0], cand[0][1]] * second:
                break
            if found:
                if before < SECOND_MIN * energy:
                    break                 # the sources so far are the whole echo
                if any(abs(g.az[a] - q[1]) <= 1.5 * sa and abs(g.el[e] - q[2]) <= 1.5 * se for q in found):
                    continue
                # first on the grid (cheap), half the gain: refining only moves it closer
                M = np.column_stack([q[4] for q in found])
                after = np.sum(np.abs(_residual(y, np.column_stack([M, cols.get([e * nA + a])[:, 0]]))) ** 2)
                if after > (1 - 0.5 * s.second_gain) * before:
                    continue
            k, az, el, p = _peak_near(y, cols, e * nA + a)
            v = _vec(cols, az, el)
            if found:
                if any(abs(az - q[1]) <= 1.5 * sa and abs(el - q[2]) <= 1.5 * se for q in found):
                    continue
                after = np.sum(np.abs(_residual(y, np.column_stack([M, v]))) ** 2)
                if after > (1 - s.second_gain) * before:
                    continue
            found.append((k, az, el, p, v))
            before = np.sum(np.abs(_residual(y, np.column_stack([q[4] for q in found]))) ** 2)
            if len(found) >= s.per_cell:
                break
        if len(found) > 1:
            found = _relax(y, found, cols)
        if found:
            res = _residual(y, np.column_stack([q[4] for q in found]))
            fit[n] = 1.0 - float(np.sum(np.abs(res) ** 2) / max(np.sum(np.abs(y) ** 2), 1e-30))
        out.append([(az, el, p) for k, az, el, p, _ in found if not edge(k)])
    return out, fit


RELAX_ITER = 6


def _relax(y: np.ndarray, found: list, cols: _Columns) -> list:
    """
    Sources of one snapshot refined against each other (RELAX): each one's peak looked
    for again, near where it was, in the snapshot without the others' fitted echoes.
    Bartlett peaks of two close sources pull towards each other (3 degrees for two
    people 30 degrees apart seen by 12 channels).
    """
    found = list(found)

    def rest(i):
        M = np.column_stack([q[4] for q in found])
        amp = np.linalg.lstsq(M, y, rcond=None)[0]
        return y - np.delete(M, i, axis=1) @ np.delete(amp, i)

    for _ in range(RELAX_ITER):
        moved = 0.0
        for i in range(len(found)):
            k, az, el, p = _peak_near(rest(i), cols, found[i][0])
            moved = max(moved, abs(az - found[i][1]), abs(el - found[i][2]))
            found[i] = (k, az, el, p, _vec(cols, az, el))
        if moved < 0.01:
            break
    return found


UNFOLD_RESIDUAL = 0.35         # a velocity beyond one slot's limit must leave at most this part
UNFOLD_FIT = 0.9               # of what the slot's own velocity leaves unexplained, and explain this much


def _explained(Y: np.ndarray, A: np.ndarray, norm: np.ndarray) -> np.ndarray:
    """
    Fraction of the energy of every snapshot Y (N, V) that two directions of the grid
    explain (greedy: the strongest, then the strongest of what is left orthogonal to it).
    """
    tot = np.maximum(np.sum(np.abs(Y) ** 2, axis=1), 1e-30)
    c = Y.conj() @ A                                                                # (N, K)
    b = np.abs(c) ** 2 / norm[None]
    k1 = np.argmax(b, axis=1)
    a1 = A[:, k1].T                                                                 # (N, V)
    n1 = norm[k1]
    g = a1.conj() @ A                                                               # a1^H A
    # Y projected off a1 against every column projected off a1
    c2 = c - (np.einsum("nv,nv->n", Y.conj(), a1) / n1)[:, None] * g
    n2 = norm[None] - np.abs(g) ** 2 / n1[:, None]
    ok = n2 > 0.05 * norm[None]
    b2 = np.where(ok, np.abs(c2) ** 2 / np.maximum(n2, 1e-30), 0.0)
    return (b[np.arange(Y.shape[0]), k1] + b2.max(axis=1)) / tot


def _unfold(Y: np.ndarray, f_d: np.ndarray, arr: Array, lay: _Layout, g: _Grid, s: Settings):
    """
    Snapshots corrected for the time between TX slots, and the velocity hypothesis m
    (f_d + m / (U T_c)) whose correction the array explains best with two directions:
    (Y corrected, f_d true). A wrong hypothesis bends the snapshot off every plane wave.

    The sharpest single beam - the first criterion - picked a wrong velocity for two
    people at one range and speed: their two beams are each weaker than the one blurred
    beam of a wrong correction (0.64 against 0.66 of the energy), two directions explain
    0.99 against 0.78. A small margin unfolded a third of a real capture's points of a
    person sitting still to 17 m/s: real echoes are no clean plane waves - two
    directions explain 0.8 of a weak one whatever the velocity - so a velocity beyond
    the limit needs clear evidence (UNFOLD_RESIDUAL, UNFOLD_FIT), and is not tried
    beyond v_limit_m_s.
    """
    U = len(lay.used)
    if U <= 1 or lay.tc_s <= 0:
        return Y, f_d
    RX = arr.rx.shape[0]
    k = np.repeat(np.asarray(lay.used, float), RX)                              # slot index per channel
    span = 1.0 / (lay.stride * lay.tc_s)                                         # f_d ambiguity
    # the correction of hypothesis m repeats every `stride`: U distinct ones around zero,
    # m = 0 first: another one has to explain the echo clearly better
    hyps = [0] if not s.unfold else sorted(range(-(lay.stride // 2), lay.stride - lay.stride // 2),
                                           key=lambda m: (abs(m), m))
    best_Y = best_f = base = best_val = None
    f_lim = 2.0 * s.v_limit_m_s / lay.lam
    hyps = [m for m in hyps if m == 0 or np.any(np.abs(f_d + m * span) <= f_lim)]
    for i, m in enumerate(hyps):
        fd = f_d + m * span
        corr = np.exp(-1j * 2 * np.pi * np.outer(fd, k * lay.tc_s))             # (N, V)
        Yc = Y * corr
        if len(hyps) == 1:
            return Yc, fd
        val = _explained(Yc, g.A, g.norm)
        if i == 0:
            base, best_val, best_Y, best_f = val, val.copy(), Yc.copy(), fd.copy()
            continue
        better = (((1 - val) <= UNFOLD_RESIDUAL * (1 - base)) & (val >= UNFOLD_FIT) & (val > best_val)
                  & (np.abs(fd) <= f_lim))
        best_val = np.where(better, val, best_val)
        best_Y[better] = Yc[better]
        best_f[better] = fd[better]
    return best_Y, best_f


def frame_points(frame: np.ndarray, params: dict, arr: Array | None, lay: _Layout,
                 g: _Grid | None, s: Settings, history=None) -> tuple[np.ndarray, dict]:
    """
    The points of one frame (rows range, velocity, az, el, power, SNR, coherence, kind)
    and its maps. history: a _History (maxlen: the frames of the micro window) the frame's
    still scene is appended to, for micro-motion.
    """
    X, still = _spectra(frame, params, lay, arr)
    rows = []
    dr = float(np.median(np.diff(lay.range_m))) if lay.range_m.size > 1 else 1.0
    dfd = float(np.median(np.diff(lay.f_d))) if lay.f_d.size > 1 else 0.0
    rmin = np.searchsorted(lay.range_m, s.r_min_m)
    images = {}
    if _uses(s.mode, KIND_MOVING) and lay.L >= MIN_LOOPS_DOPPLER:
        # without the zero-Doppler row (the still scene): what drifts by less than a bin in
        # the frame (someone shifting slowly) is in the rows either side of it - as one
        # echo, not two with opposite velocities
        zero = lay.nfft_d // 2
        Xm = np.delete(X, zero, axis=0)
        fd_axis = np.delete(lay.f_d, zero)
        P = (np.abs(Xm) ** 2).sum(axis=1)                                        # (D-1, R)
        peaks, noise, cells = _cfar(P, s)
        # taking the still scene out takes a mover's mean over the frame out too: an echo
        # SIDELOBE_DB and more weaker than the strongest at its range, beside zero Doppler
        if P.shape[0] >= 3:
            near0 = [zero - 1, zero]
            rest = np.delete(P, near0, axis=0).max(axis=0)
            weak = P[near0] < rest[None] * 10 ** (-SIDELOBE_DB / 10)
            for m_ in (peaks, cells):
                m_[near0] &= ~weak
        for m_ in (peaks, cells):
            m_[:, :rmin] = False
        images["rd"] = P
        images["rd_mask"] = peaks
        mov = _points(Xm, P, noise, peaks if arr is None or g is None else cells, arr, lay, g, s,
                      dr, dfd, moving=True, fd_axis=fd_axis)
        kinds = [KIND_MOVING] * len(mov)
        if history is not None:
            history.env.append(P.sum(axis=0))
            kinds = _mark_moving_echoes(mov, history.env.array(), lay, history.fps)
        rows += [q + (k,) for q, k in zip(mov, kinds)]
    noise_s = None
    if _uses(s.mode, KIND_STILL):
        Ps = (np.abs(still) ** 2).sum(axis=0)[None]                              # (1, R)
        noise_s = _still_noise(X, lay)[None]
        thr = 10 ** (s.threshold_db / 10)
        peaks = (Ps > noise_s * thr) & (Ps >= ndimage.maximum_filter(Ps, size=(1, 3), mode="nearest"))
        peaks[:, :rmin] = False
        images["still_profile"] = Ps[0]
        images["still_noise"] = noise_s[0]
        ss = s if arr is None else _with(s, per_cell=max(s.per_cell, s.still_per_cell))
        rows += [q + (KIND_STILL,) for q in _points(still[None], Ps, noise_s, peaks, arr, lay, g, ss, dr, 0.0,
                                                    moving=False)]
    if _uses(s.mode, KIND_MICRO) and history is not None:
        history.append(still)
        w_det = getattr(history, "w_det", history.maxlen)
        if len(history) >= max(4, w_det // 2):
            noise = (noise_s[0] if noise_s is not None else _still_noise(X, lay))
            rows += _micro_points(history.array(), w_det, noise, arr, lay, g, s, dr,
                                  getattr(history, "fps", 20.0))
    return np.array(rows, float).reshape(-1, 8), images


MICRO_SECOND = 0.25            # a second mover in a range bin: its motion this part of the first's
MICRO_EXTRA_DB = 6.0           # micro-motion: the threshold this much higher - against the thermal noise
                               # everything in a room moves a little; on seated recordings 13 dB kept
                               # two weak points away from the person for every three at them, 19 dB one
                               # for nine
MIN_LOOPS_DOPPLER = 8          # fewer chirps of a TX a frame: no Doppler in the frame (micro instead)
ECHO_CORR = 0.7                # |correlation| of the motion with a nearer, stronger mover: its echo
ECHO_MIN_S = 4.0               # of at least this long
ECHO_GAP_M = 0.4               # nearer by at least this


class _Ring:
    """
    The last maxlen arrays appended, readable as one array without a copy. A deque of them
    turned into an array at every frame copied the whole window each time: at 100 fps the
    10 s echo window is 1000 still scenes, and a 3-min single-chip take spent ~90 % of its
    Imaging time (9-10 min) copying them. Each item is written twice, at i and i + maxlen,
    so the last n are always the contiguous slice ending at i + maxlen.
    """

    def __init__(self, maxlen: int):
        self.maxlen = max(1, int(maxlen))
        self._buf = None
        self._i = self._n = 0

    def append(self, x):
        x = np.asarray(x)
        if self._buf is None or self._buf.shape[1:] != x.shape or self._buf.dtype != x.dtype:
            self._buf = np.empty((2 * self.maxlen,) + x.shape, x.dtype)
            self._i = self._n = 0
        self._buf[self._i] = x
        self._buf[self._i + self.maxlen] = x
        self._i = (self._i + 1) % self.maxlen
        self._n = min(self._n + 1, self.maxlen)

    def __len__(self) -> int:
        return self._n

    def clear(self):
        self._i = self._n = 0

    def array(self) -> np.ndarray:
        """(n, ...) the items, oldest first - a view, valid until the next append."""
        if self._buf is None:
            return np.zeros((0,))
        end = self._i + self.maxlen
        return self._buf[end - self._n:end]


class _History(_Ring):
    """The still scenes of the last frames (micro-motion), with the detection window, and
    the moving power of every range bin (echoes of movers)."""

    def __init__(self, maxlen: int, w_det: int, fps: float):
        super().__init__(maxlen)
        self.w_det, self.fps = w_det, fps
        self.env = _Ring(maxlen)

    def clear(self):
        super().clear()
        self.env.clear()


def _mark_moving_echoes(rows: list, env: np.ndarray, lay: _Layout, fps: float) -> list:
    """
    Kind of every moving row: an echo when the moving power of its range swells and
    fades with that of a nearer, stronger point's (env: (T, R) the moving power of the
    last frames). Someone stepping in place 1.7 m from the radar was a smear of moving
    points to 4.5 m, the room's reflections of them, eight objects a frame; the energy
    of each reflection follows the person's steps.
    """
    if not rows or env.shape[0] < ECHO_MIN_S * fps:
        return [KIND_MOVING] * len(rows)
    dr = float(np.median(np.diff(lay.range_m))) if lay.range_m.size > 1 else 1.0
    R = env.shape[1]
    k = max(3, int(round(fps)))
    kern = np.ones(k) / k

    def series(rng):
        r = int(np.clip(round((rng - lay.range_m[0]) / dr), 1, R - 2))
        x = np.log(env[:, r - 1:r + 2].sum(axis=1) + 1e-30)
        return x - np.convolve(x, kern, mode="same")             # its swings, not its level

    ser = [series(q[0]) for q in rows]
    order = sorted(range(len(rows)), key=lambda i: -rows[i][4])
    kinds = [KIND_MOVING] * len(rows)
    for pos, i in enumerate(order):
        for j in order[:pos]:
            if kinds[j] != KIND_MOVING or rows[j][0] > rows[i][0] - ECHO_GAP_M:
                continue
            a, b = ser[i][k:-k], ser[j][k:-k]
            if a.size < 4 or a.std() == 0 or b.std() == 0:
                continue
            if float(np.corrcoef(a, b)[0, 1]) >= ECHO_CORR:
                kinds[i] = KIND_ECHO
                break
    return kinds


def _motion_series(hist: np.ndarray, r: int, a: np.ndarray) -> np.ndarray:
    """The displacement-like motion of the echo of direction a in range bin r over the
    frames of hist: the phase of its change against the mean, unwrapped, detrended."""
    z = (hist[:, :, r] - hist[:, :, r].mean(axis=0)) @ a.conj()
    ph = np.unwrap(np.angle(z))
    k = np.arange(ph.size)
    return ph - np.polyval(np.polyfit(k, ph, 1), k)


def _mark_echoes(rows: list, series: list, fps: float) -> list:
    """Kind of every micro row: an echo when it moves in step with a nearer, stronger one.
    A breathing chest reflected by a wall or the floor breathes with it - on seated
    recordings three of four micro points lay behind the person."""
    from scipy import signal as sps
    if not rows or series[0] is None or series[0].size < ECHO_MIN_S * fps:
        return [KIND_MICRO] * len(rows)
    hi = min(0.6, 0.45 * fps)
    sos = sps.butter(2, [0.1, hi], btype="band", fs=fps, output="sos") if hi > 0.15 else None
    filt = [sps.sosfiltfilt(sos, x) if sos is not None else x for x in series]
    order = sorted(range(len(rows)), key=lambda i: -rows[i][4])
    kinds = [KIND_MICRO] * len(rows)
    for pos, i in enumerate(order):
        for j in order[:pos]:
            if kinds[j] != KIND_MICRO or rows[j][0] > rows[i][0] - ECHO_GAP_M:
                continue
            a, b = filt[i], filt[j]
            if a.std() == 0 or b.std() == 0:
                continue
            if abs(float(np.corrcoef(a, b)[0, 1])) >= ECHO_CORR:
                kinds[i] = KIND_ECHO
                break
    return kinds


def _micro_points(hist_all: np.ndarray, w_det: int, noise: np.ndarray, arr, lay: _Layout, g, s: Settings,
                  dr: float, fps: float) -> list:
    """
    Points of what moves from frame to frame - a breathing chest, a hand, someone shifting
    on a chair - from the still scenes of the micro window hist (W, V, R): their change
    against the window's mean. A seated person is no Doppler peak in a frame of a few
    chirps and no peak of the still scene among the walls (2 % of the still points of
    seated people lay at the person); their motion is.

    The motion power of every range bin against the thermal noise of the frames (noise:
    a frame's still noise, (R,)); a bin is a peak of it. Its direction: the main motion
    of the bin (the strongest eigenvector of the motion's covariance over the window, a
    second one when it has MICRO_SECOND of the first's power), kept where the beam's
    motion peaks in range. Coherence: the part of the motion in the first eigenvector.
    """
    hist = hist_all[-w_det:]
    W = hist.shape[0]
    M = hist - hist.mean(axis=0, keepdims=True)                                  # (W, V, R)
    Pm = (np.abs(M) ** 2).sum(axis=(0, 1))                                       # (R,)
    floor = np.maximum(noise, 1e-30) * (W - 1)
    thr = 10 ** ((s.threshold_db + MICRO_EXTRA_DB) / 10)
    R = Pm.size
    rmin = np.searchsorted(lay.range_m, s.r_min_m)
    peaks = (Pm > floor * thr) & (Pm >= ndimage.maximum_filter(Pm, size=3, mode="nearest"))
    peaks[:rmin] = False
    r_i = np.flatnonzero(peaks)
    r_i = r_i[np.argsort(-Pm[r_i])][: s.max_cells_frame]
    rows = []
    if not r_i.size:
        return rows
    snr = 10 * np.log10(Pm[r_i] / floor[r_i])
    if arr is None or g is None:
        for n, r in enumerate(r_i):
            rng = lay.range_m[r] + (_parabolic(*Pm[r - 1:r + 2]) * dr if 0 < r < R - 1 else 0.0)
            rows.append((rng, 0.0, np.nan, np.nan, float(Pm[r]), snr[n], np.nan, KIND_MICRO))
        return rows
    snaps, owner, coh = [], [], []
    covs = {}
    for n, r in enumerate(r_i):
        for rr in (r - 1, r, r + 1):
            if 0 <= rr < R and rr not in covs:
                Y = M[:, :, rr]
                covs[rr] = (Y.T @ Y.conj()) / W                                  # (V, V): sum of y y^H
        w, U = np.linalg.eigh(covs[r])
        for k in (1, 2):
            if k == 1 or w[-k] >= MICRO_SECOND * w[-1]:
                snaps.append(np.sqrt(max(w[-k], 0.0)) * U[:, -k])
                owner.append(n)
    snaps = np.asarray(snaps)
    one = _with(s, per_cell=1)
    angs, fit = _angles(snaps, arr, g, one, r_wl=lay.range_m[r_i[np.asarray(owner)]] / lay.lam)
    # coherence: how much of the bin's main motion one direction explains
    coh = {}
    for j, n in enumerate(owner):
        coh.setdefault(n, float(fit[j]))
    found = {}
    for j, n in enumerate(owner):
        for az, el, p in angs[j]:
            if any(abs(az - a2) < 3 * (g.az[1] - g.az[0] if g.az.size > 1 else 1) and
                   abs(el - e2) < 3 * (g.el[1] - g.el[0] if g.el.size > 1 else 1) for a2, e2, _ in found.get(n, [])):
                continue
            found.setdefault(n, []).append((az, el, p))
    series = []
    long = hist_all.shape[0] >= ECHO_MIN_S * fps
    for n, r in enumerate(r_i):
        for az, el, p in found.get(n, []):
            r_wl = lay.range_m[r] / lay.lam
            a = arr.steering([az], [el], r_wl if 0 < r_wl < NEAR_FACTOR * arr.far_field_wl() else None)[:, 0]
            beam = [float(np.real(a.conj() @ covs[rr] @ a)) if rr in covs else 0.0 for rr in (r - 1, r, r + 1)]
            if 0 < r < R - 1 and (beam[0] >= beam[1] or beam[2] > beam[1]):
                continue                  # this mover's motion peaks in the next bin
            rng = lay.range_m[r] + (_parabolic(*beam) * dr if 0 < r < R - 1 else 0.0)
            rows.append((rng, 0.0, az, el, p, snr[n], coh[n]))
            series.append(_motion_series(hist_all, r, a) if long else None)
    kinds = _mark_echoes(rows, series, fps)
    return [q + (k,) for q, k in zip(rows, kinds)]


def _still_noise(X: np.ndarray, lay: _Layout) -> np.ndarray:
    """
    (R,) noise power of the still scene's summed channels: the moving part's power at
    most Doppler bins is noise; the still scene, the mean of L loops, has 1/L of a chirp's
    noise, a Doppler bin sum(hann^2) of it. A room reflects at every range - a CFAR on the
    still profile measured it against itself and kept a few bins of the whole room.
    """
    L = lay.L
    w = np.hanning(L + 2)[1:-1] if L > 1 else np.ones(1)
    P = (np.abs(X) ** 2).sum(axis=1)                                             # (D, R)
    per_chirp = np.median(P, axis=0) / np.log(2) / max(np.sum(w ** 2), 1e-12)   # median of exponential
    floor = np.median(per_chirp)
    return np.maximum(np.minimum(per_chirp, 4 * floor), floor / 4) / max(L, 1)


def _contiguous(fd_axis: np.ndarray, d: int, dfd: float) -> bool:
    """Whether the rows either side of row d are one Doppler bin from it (not across the
    removed zero row, nor across the wrap)."""
    D = fd_axis.size
    if D < 3 or dfd <= 0:
        return False
    lo, hi = fd_axis[(d - 1) % D], fd_axis[(d + 1) % D]
    return abs(fd_axis[d] - lo - dfd) < 0.1 * dfd and abs(hi - fd_axis[d] - dfd) < 0.1 * dfd


def _with(s: Settings, **kw) -> Settings:
    d = asdict(s)
    d.update(kw)
    return Settings(**d)


def _points(X, P, noise, cells, arr, lay, g, s, dr, dfd, moving, fd_axis=None) -> list:
    """Rows (range, velocity, az, el, power, SNR) of the echoes in the cells of X (D, V, R),
    whose rows are at the Doppler frequencies fd_axis (D,)."""
    d_i, r_i = np.nonzero(cells)
    order = np.argsort(-P[d_i, r_i])[: s.max_cells_frame]
    d_i, r_i = d_i[order], r_i[order]
    if not d_i.size:
        return []
    D, R = P.shape
    snr = 10 * np.log10(P[d_i, r_i] / noise[d_i, r_i])
    fd_axis = lay.f_d if fd_axis is None else fd_axis
    fd = np.array([(fd_axis[d] + (_parabolic(P[d - 1, r], P[d, r], P[(d + 1) % D, r]) * dfd
                                  if _contiguous(fd_axis, d, dfd) else 0.0))
                   if moving else 0.0 for d, r in zip(d_i, r_i)])
    rows = []
    if arr is None or g is None:
        for n, (d, r) in enumerate(zip(d_i, r_i)):
            rng = lay.range_m[r] + (_parabolic(*P[d, r - 1:r + 2]) * dr if 0 < r < R - 1 else 0.0)
            rows.append((rng, fd[n] * lay.lam / 2.0, np.nan, np.nan, float(P[d, r]), snr[n], np.nan))
        return rows[: s.max_points_frame]
    Y = X[d_i, :, r_i]                                                           # (N, V)
    fd_bin = np.array([fd_axis[d] if moving else 0.0 for d in d_i])
    if moving:
        Y, fd_true = _unfold(Y, fd, arr, lay, g, s)
        fd_bin, fd = fd_bin + (fd_true - fd), fd_true
    angs, fit = _angles(Y, arr, g, s, r_wl=lay.range_m[r_i] / lay.lam)
    for n, (d, r) in enumerate(zip(d_i, r_i)):
        if fit[n] < s.min_coherence or not angs[n]:
            continue
        for rng, f, az, el, p in _cell_sources(X, d, r, fd[n], fd_bin[n], angs[n], arr, lay, dr,
                                               dfd if (not moving or _contiguous(fd_axis, d, dfd)) else 0.0):
            rows.append((rng, f * lay.lam / 2.0, az, el, p, snr[n], fit[n]))
    rows.sort(key=lambda q: -q[4])
    return rows[: s.max_points_frame]


def _cell_sources(X: np.ndarray, d: int, r: int, fd: float, fd_bin: float, srcs: list, arr: Array,
                  lay: _Layout, dr: float, dfd: float) -> list:
    """
    The sources (az, el, power) found in cell (d, r) that are echoes of their own, with
    their own range and Doppler frequency: [(range, f_d, az, el, power)].

    The cell's neighbourhood is split into the sources' echoes (least squares on their
    steering vectors, corrected for the TX slots at fd, the cell's unfolded Doppler
    frequency; fd_bin: that of the bin's centre). A source is kept where its echo peaks - a cell next to a stronger one
    repeats that one's echo through the window, and is the only place a weaker echo in
    another direction peaks (the summed power of the channels does not). Its range and
    Doppler are refined around its own peak: two people 14 cm apart share a 20 cm cell,
    and the cell's peak put both halfway.
    """
    D, V, R = X.shape
    r_wl = lay.range_m[r] / lay.lam
    A = arr.steering([q[0] for q in srcs], [q[1] for q in srcs],
                     r_wl if 0 < r_wl < NEAR_FACTOR * arr.far_field_wl() else None)   # (V, S)
    k = np.repeat(np.asarray(lay.used, float), arr.rx.shape[0])
    corr = np.exp(-1j * 2 * np.pi * fd * k * lay.tc_s)
    rr = np.clip(np.arange(r - 1, r + 2), 0, R - 1)
    dd = np.arange(d - 1, d + 2) % D if D >= 3 else np.array([d])
    Z = X[dd][:, :, rr] * corr[None, :, None]                                       # (Dn, V, 3)
    amp, *_ = np.linalg.lstsq(A, Z.transpose(1, 0, 2).reshape(V, -1), rcond=None)
    pw = (np.abs(amp) ** 2).reshape(len(srcs), dd.size, 3)
    c = dd.size // 2
    out = []
    for i, (az, el, p) in enumerate(srcs):
        m = pw[i]
        here = m[c, 1]
        flat = m.ravel()
        centre = c * 3 + 1
        # a peak, and of two equal cells (an echo halfway between) the later one only
        if 0 < r < R - 1 and (np.any(flat[:centre] >= here) or np.any(flat[centre + 1:] > here)):
            continue                      # its echo peaks in a neighbouring cell
        rng = lay.range_m[r] + (_parabolic(m[c, 0], here, m[c, 2]) if 0 < r < R - 1 else 0.0) * dr
        f = fd_bin + (_parabolic(m[0, 1], here, m[2, 1]) if dd.size == 3 else 0.0) * dfd
        out.append((rng, f, az, el, p))
    return out


def _to_xyz(r, az, el, s: Settings):
    az, el = np.deg2rad(np.nan_to_num(az)), np.deg2rad(np.nan_to_num(el))
    x = r * np.cos(el) * np.sin(az)
    y = r * np.cos(el) * np.cos(az)
    z = r * np.sin(el)
    t = np.deg2rad(s.tilt_deg)                       # pointed down by t
    y_w = y * np.cos(t) + z * np.sin(t)
    z_w = -y * np.sin(t) + z * np.cos(t) + s.height_m
    return x, y_w, z_w


# --- objects -----------------------------------------------------------------------------

CLUSTER_Z_WEIGHT = 0.3         # people are tall: a metre of height counts as 0.3 m
CLUSTER_V_WEIGHT = 0.15        # limbs swing at +-1 m/s around the body: 1 m/s counts as 0.15 m
TRACK_GATE_M = 0.5             # an object this near where a track is expected is its object ...
TRACK_GATE_CHI2 = 9.21         # ... and within the 99 % of the track's uncertainty
TRACK_KEEP_S = 1.5             # a track waits this long for its object to come back ...
TRACK_KEEP_STILL_S = 6.0       # ... and this long when it stood still (slower than TRACK_STILL_M_S):
TRACK_STILL_M_S = 0.3          # a seated person's breathing pauses; 14 tracks of one in a minute
TRACK_CONFIRM = 3              # frames with an object before a track is shown as one
TRACK_ACCEL = 2.0              # m/s^2, how fast people change their velocity
TRACK_MEAS_M = 0.25            # m, how far an object's centroid is off its person


def clusters(points: np.ndarray, s: Settings) -> np.ndarray:
    """DBSCAN labels (-1: noise) of points, in x, y, z and velocity."""
    if points.shape[0] < s.cluster_min:
        return np.full(points.shape[0], -1, int)
    from sklearn.cluster import DBSCAN
    feats = np.column_stack([points["x_m"], points["y_m"], points["z_m"] * CLUSTER_Z_WEIGHT,
                             points["velocity_m_s"] * CLUSTER_V_WEIGHT])
    return DBSCAN(eps=s.cluster_eps_m, min_samples=s.cluster_min).fit_predict(feats)


def _frame_objects(window: np.ndarray, current: np.ndarray, s: Settings) -> tuple[np.ndarray, list]:
    """
    Objects of a frame from the points of the last cluster_frames frames (a person at
    3 m gives the 12 channels of the IWR6843AOP two points a frame - too few for any
    cluster - and six in three frames), and the object index of every point of the
    frame itself (-1: none).
    """
    labels = clusters(window, s)
    own = np.full(int(current.sum()), -1, int)
    objs = []
    for c in sorted(set(labels.tolist()) - {-1}):
        m = labels == c
        q = window[m]
        mine = q[current[m]]
        use = mine if mine.size else q
        w = 10 ** (use["snr_db"] / 20)
        objs.append({"n": int(q.size), "echo_share": float(np.mean(q["kind"] == KIND_ECHO)),
                     "x_m": float(np.average(use["x_m"], weights=w)),
                     "y_m": float(np.average(use["y_m"], weights=w)),
                     "z_m": float(np.average(use["z_m"], weights=w)),
                     "velocity_m_s": float(np.median(q["velocity_m_s"])),
                     "size_m": (float(np.ptp(q["x_m"])), float(np.ptp(q["y_m"])), float(np.ptp(q["z_m"]))),
                     "snr_db": float(q["snr_db"].max())})
        own[labels[current] == c] = len(objs) - 1
    return own, objs


class _Tracker:
    """
    Objects followed from frame to frame: a constant-velocity Kalman filter per track,
    assignment by the Hungarian method within a statistical gate. An alpha-beta filter
    came first: with two or three points a person the centroid jumps by 0.3 m from frame
    to frame, its velocity estimate by 1 m/s, and tracks lost their people after a gap.
    """

    def __init__(self, dt: float):
        self.dt = dt
        self.tracks = []
        self.next_id = 1
        self.trails = {}

    @staticmethod
    def _predict(tr, n: int, dt: float):
        T = n * dt
        F = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        G = np.array([[T * T / 2, 0], [0, T * T / 2], [T, 0], [0, T]])
        Q = G @ G.T * TRACK_ACCEL ** 2
        return F @ tr["s"], F @ tr["P"] @ F.T + Q

    def step(self, f: int, objs: list) -> list:
        """Track id of every object of frame f (-1 while a track is tentative)."""
        from scipy.optimize import linear_sum_assignment
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        Rm = np.eye(2) * TRACK_MEAS_M ** 2
        preds = [self._predict(tr, max(1, f - tr["f"]), self.dt) for tr in self.tracks]
        ids = [-1] * len(objs)
        used = set()
        if self.tracks and objs:
            Z = np.array([[o["x_m"], o["y_m"]] for o in objs])
            D = np.full((len(self.tracks), len(objs)), 1e6)
            for i, (sp, Pp) in enumerate(preds):
                S = H @ Pp @ H.T + Rm
                d = Z - (H @ sp)[None]
                m2 = np.einsum("ni,ij,nj->n", d, np.linalg.inv(S), d)
                ok = (m2 <= TRACK_GATE_CHI2) | (np.hypot(d[:, 0], d[:, 1]) <= TRACK_GATE_M)
                D[i, ok] = m2[ok]
            rows, cols = linear_sum_assignment(D)
            for i, j in zip(rows, cols):
                if D[i, j] >= 1e6:
                    continue
                tr, o = self.tracks[i], objs[j]
                sp, Pp = preds[i]
                S = H @ Pp @ H.T + Rm
                K = Pp @ H.T @ np.linalg.inv(S)
                tr["s"] = sp + K @ (Z[j] - H @ sp)
                tr["P"] = (np.eye(4) - K @ H) @ Pp
                tr["z"] = o["z_m"]
                tr["f"], tr["hits"] = f, tr["hits"] + 1
                used.add(j)
                if tr["hits"] >= TRACK_CONFIRM:
                    ids[j] = tr["id"]
                    self.trails.setdefault(tr["id"], []).append((f, tr["s"][0], tr["s"][1], tr["z"]))
        for j, o in enumerate(objs):
            if j not in used:
                P0 = np.diag([TRACK_MEAS_M ** 2, TRACK_MEAS_M ** 2, 1.5 ** 2, 1.5 ** 2])
                self.tracks.append({"id": self.next_id, "s": np.array([o["x_m"], o["y_m"], 0.0, 0.0]),
                                    "P": P0, "z": o["z_m"], "f": f, "hits": 1})
                self.next_id += 1
        keep = max(1, int(round(TRACK_KEEP_S / self.dt)))
        keep_still = max(keep, int(round(TRACK_KEEP_STILL_S / self.dt)))

        def wait(tr):
            if tr["hits"] < TRACK_CONFIRM:
                return 2
            return keep_still if np.hypot(tr["s"][2], tr["s"][3]) < TRACK_STILL_M_S else keep

        self.tracks = [tr for tr in self.tracks if f - tr["f"] <= wait(tr)]
        return ids


def _objects_and_tracks(points: np.ndarray, starts: np.ndarray, frames: int, fps: float,
                        s: Settings) -> tuple[list, dict]:
    """Objects of every frame with their track ids; points' "cluster" set to their track."""
    tracker = _Tracker(1.0 / fps if fps > 0 else 0.05)
    by_frame = []
    w = max(1, int(s.cluster_frames))
    for f in range(frames):
        lo, hi = starts[max(0, f - w + 1)], starts[f + 1]
        win = points[lo:hi]
        if s.mode == "still":
            sel = np.arange(win.size)
        else:                                     # people (and their echoes), not the room
            sel = np.flatnonzero(win["kind"] != KIND_STILL)
        cur = (np.arange(lo, hi)[sel] >= starts[f])
        own, objs = _frame_objects(win[sel], cur, s) if sel.size else (np.zeros(0, int), [])
        ids = tracker.step(f, objs)
        for o, i in zip(objs, ids):
            o["track"] = int(i)
        by_frame.append(objs)
        lab = np.full(hi - starts[f], -1, np.float32)
        cur_idx = sel[cur] - (starts[f] - lo)
        lab[cur_idx] = [ids[k] if k >= 0 else -1 for k in own]
        points["cluster"][starts[f]:hi] = lab
    trails = {k: np.array(v, float) for k, v in tracker.trails.items()}
    echo_of = _echo_tracks(by_frame, fps)
    # a track made mostly of echo points is an echo, all of it: the motion of a point is
    # marked frame by frame and a fifth to a third of a reflection's points were missed
    share = {}
    for objs in by_frame:
        for o in objs:
            if o.get("track", -1) >= 0:
                share.setdefault(o["track"], []).append(o.get("echo_share", 0.0))
    for tid, v in share.items():
        if tid not in echo_of and float(np.mean(v)) >= TRACK_ECHO_SHARE:
            echo_of[tid] = 0                      # an echo, of whom is not known
    for objs in by_frame:
        for o in objs:
            o["echo_of"] = echo_of.get(o.get("track", -1), -1)
    if echo_of:
        hit = (np.isin(points["cluster"], np.array(list(echo_of), np.float32))
               & ((points["kind"] == KIND_MOVING) | (points["kind"] == KIND_MICRO)))
        points["kind"][hit] = KIND_ECHO
    return by_frame, trails


PERSON_MIN_S = 3.0             # a track followed this long is a person; a shorter one is noise or a passing reflection
GHOST_DB = 6.0                 # a track this much weaker than a person, behind them (GHOST_DEG) ...
GHOST_DEG = 20.0               # ... for most of their common time is a reflection of them
PERSON_NEAR_M = 0.6            # a track or a point this near a person is part of them
VITALS_NEAR_M = 0.8            # a person's track this near someone Multi Vitals finds ...
VITALS_SHARE = 0.5             # ... for this share of the time Multi Vitals finds anyone is confirmed
WALK_M_S = 0.3                 # a person's track moving this fast (over a second) walks: Multi Vitals
                               # does not follow walkers on the cascade - it is a person in the frames it
WALK_REL_DB = 3.0              # is within this of the strongest person there (the room's reflections
                               # of a walker were 6 dB or more weaker than the strongest in 98 % of frames;
                               # 3 dB: a second person in 10 % of a walk's frames, 6 dB: 18 %)


def people(points: np.ndarray, by_frame: list, trails: dict, fps: float) -> set:
    """
    The tracks that are people - the rest of the points (noise, short-lived and still
    reflections, echoes) is the separate layer under them. Followed for PERSON_MIN_S at
    least, no echo track (TRACK_ECHO_SHARE), not a weaker reflection behind a person, and
    one track a person: a second track on someone (an arm, the scatter of a body over two clusters)
    is merged into theirs - the points and objects get its number. Sets every object's
    "person" (True / False) and returns the people's track ids.
    """
    echo_track = {o["track"] for objs in by_frame for o in objs if o.get("echo_of", -1) >= 0}
    snr = {}
    for objs in by_frame:
        for o in objs:
            if o.get("track", -1) >= 0:
                snr.setdefault(o["track"], []).append(o["snr_db"])
    cand = []
    for tid, tr in trails.items():
        if tr.shape[0] == 0 or tid in echo_track:
            continue
        span = (tr[-1, 0] - tr[0, 0] + 1) / max(fps, 1e-9)
        if span < PERSON_MIN_S:
            continue
        cand.append((float(np.median(snr.get(tid, [-np.inf]))), tid))
    def common(a, b):
        """a's frames within b's span, a's positions there and b's (interpolated over b's gaps)"""
        ta, tb = trails[a], trails[b]
        m = (ta[:, 0] >= tb[0, 0]) & (ta[:, 0] <= tb[-1, 0])
        f = ta[m, 0]
        return f, ta[m, 1:3], np.column_stack([np.interp(f, tb[:, 0], tb[:, 1]), np.interp(f, tb[:, 0], tb[:, 2])])

    keep, merged = [], {}
    for s_db, tid in sorted(cand, reverse=True):
        mine = trails[tid].shape[0]
        verdict = None
        for s_p, pid in keep:
            fs, a, b = common(tid, pid)
            if len(fs) < 0.5 * mine:
                continue
            if np.mean(np.hypot(*(a - b).T) < PERSON_NEAR_M) >= 0.7:
                verdict = ("part", pid)
                break
            ra, rb = np.hypot(*a.T), np.hypot(*b.T)
            da = np.abs(np.rad2deg(np.arctan2(a[:, 0], a[:, 1]) - np.arctan2(b[:, 0], b[:, 1])))
            if s_db <= s_p - GHOST_DB and np.mean((ra >= rb + ECHO_GAP_M) & (da <= GHOST_DEG)) >= 0.5:
                verdict = ("ghost", pid)
                break
        if verdict is None:
            keep.append((s_db, tid))
        elif verdict[0] == "part":
            merged[tid] = verdict[1]
    persons = {tid for _, tid in keep}
    if merged and points.size:
        c = points["cluster"]
        for a, b in merged.items():
            c[c == a] = b
    for objs in by_frame:
        for o in objs:
            t = merged.get(o.get("track", -1), o.get("track", -1))
            o["track"] = t
            o["person"] = t in persons
    one_a_person(by_frame)
    for a, b in merged.items():
        tr = np.concatenate([trails[b], trails.pop(a)])      # where both are: the person's own
        tr = tr[np.argsort(tr[:, 0], kind="stable")]
        _, first = np.unique(tr[:, 0], return_index=True)
        trails[b] = tr[first]
    return persons


def one_a_person(by_frame: list):
    """One object a person and frame: of the people's objects of a frame within
    PERSON_NEAR_M of each other (one track, or two on one body) the one with the most
    points stays the person, the others become part of them."""
    for objs in by_frame:
        ppl = sorted((o for o in objs if o.get("person")), key=lambda o: -o["n"])
        kept = []
        for o in ppl:
            host = next((k for k in kept if np.hypot(k["x_m"] - o["x_m"], k["y_m"] - o["y_m"]) < PERSON_NEAR_M
                         or k["track"] == o["track"]), None)
            if host is None:
                kept.append(o)
            else:
                o["person"] = False
                o["part_of"] = host["track"]


def vital_people(cube: np.ndarray, params: dict, fps: float, progress_cb=None) -> list:
    """Where Multi Vitals finds people (no echoes): [(t_s, x_m, y_m)] per person, or [] when it
    cannot tell (no angles)."""
    import multi_vitals
    with contextlib.redirect_stdout(io.StringIO()):
        res = multi_vitals.analyse(cube, params, fps=fps, method="motion", beam="capon",
                                   progress_cb=progress_cb)
    if res.angles_deg.size < 2:                  # range only: no places to compare
        return []
    return [(p.blocks * res.block_s + 0.5 * res.block_s, p.x_m, p.y_m) for p in res.people if p.echo_of is None]


def _speed(tr: np.ndarray, fps: float) -> float:
    """Median speed of a trail over 1 s steps (frame to frame its jitter alone is 0.5 m/s)."""
    if tr.shape[0] < 2:
        return 0.0
    f = np.arange(tr[0, 0], tr[-1, 0] + 1)
    x, y = np.interp(f, tr[:, 0], tr[:, 1]), np.interp(f, tr[:, 0], tr[:, 2])
    k = max(1, int(round(fps)))
    if f.size <= k:
        return float(np.hypot(x[-1] - x[0], y[-1] - y[0]) / max(f.size / fps, 1e-9))
    return float(np.median(np.hypot(x[k:] - x[:-k], y[k:] - y[:-k])) * fps / k)


def confirm_people(persons: set, by_frame: list, trails: dict, fps: float, found: list) -> set:
    """
    Which person tracks stay people. One that walks (WALK_M_S) is one in the frames where
    it is within WALK_REL_DB of the strongest person there (with a walker in a room full of
    reflections the tracker swaps them: it is decided frame by frame). One that stays must lie within VITALS_NEAR_M of
    someone Multi Vitals found (found: vital_people) for VITALS_SHARE of the time it found
    anyone (a track while it finds nobody stays), and is a person in a frame only there -
    unless it is within WALK_REL_DB of the strongest person most of the time and in that
    frame (someone standing where Multi Vitals lost them, at a reflection of theirs). On the cascade the room reflects a
    person to several places at once, each followed frame by frame as long as they;
    Multi Vitals' maps over half a second with Capon's beams keep only the person (one
    person 2 m away: 20 % of the frames showed a second person, 1 % with this).
    """
    strongest = {f: max((o["snr_db"] for o in objs if o.get("person")), default=None)
                 for f, objs in enumerate(by_frame)}
    walks = {tid for tid in persons if _speed(trails[tid], fps) >= WALK_M_S}
    dominant = lambda f, o: o["snr_db"] >= strongest[f] - WALK_REL_DB
    keep = set()
    for tid in persons:
        tr = trails[tid]
        if tid in walks:                         # frame by frame: the tracker swaps walkers and reflections
            keep.add(tid)
            continue
        if not found:
            keep.add(tid)
            continue
        t = tr[:, 0] / fps
        near, seen = np.zeros(t.size, bool), np.zeros(t.size, bool)
        for ts, xs, ys in found:
            inside = (t >= ts[0] - 1.0) & (t <= ts[-1] + 1.0)
            if not inside.any():
                continue
            x, y = np.interp(t[inside], ts, xs), np.interp(t[inside], ts, ys)
            seen |= inside
            k = np.flatnonzero(inside)
            near[k[np.hypot(tr[k, 1] - x, tr[k, 2] - y) < VITALS_NEAR_M]] = True
        fs = tr[:, 0].astype(int)
        strong = [dominant(f, o) for f in fs for o in by_frame[f] if o.get("track") == tid and o.get("person")]
        if not seen.any() or near.sum() >= VITALS_SHARE * seen.sum() or \
                (strong and np.mean(strong) >= VITALS_SHARE):
            keep.add(tid)                         # (Multi Vitals finding nobody then says nothing)
    for f, objs in enumerate(by_frame):
        t = f / fps
        here = [(np.interp(t, ts, xs), np.interp(t, ts, ys)) for ts, xs, ys in found if ts[0] - 1.0 <= t <= ts[-1] + 1.0]
        for o in objs:
            if not o.get("person"):
                continue
            if o["track"] not in keep:
                o["person"] = False
            elif o["track"] in walks:
                o["person"] = dominant(f, o)
            elif here and min(np.hypot(o["x_m"] - x, o["y_m"] - y) for x, y in here) >= VITALS_NEAR_M \
                    and not dominant(f, o):
                o["person"] = False               # the track on a reflection for a moment
    one_a_person(by_frame)
    return keep


TRACK_ECHO_CORR = 0.6          # velocity swings in step with a nearer, stronger track's: its echo
TRACK_ECHO_MIN_S = 3.0         # over at least this much common time
TRACK_ECHO_SHARE = 0.5         # a track whose points are mostly echoes is one


def _echo_tracks(by_frame: list, fps: float) -> dict:
    """
    {track: the track it is an echo of}: a track whose objects' velocity swings in step
    with those of a nearer, stronger track - the room reflecting a walking person; every
    step of theirs is a step of the reflection. The velocities less their 1 s mean (two
    people walking towards the radar share the trend, not the steps).
    """
    series = {}
    for f, objs in enumerate(by_frame):
        for o in objs:
            tid = o.get("track", -1)
            if tid < 0:
                continue
            d = series.setdefault(tid, {"f": [], "v": [], "r": [], "snr": []})
            d["f"].append(f)
            d["v"].append(o["velocity_m_s"])
            d["r"].append(float(np.hypot(o["x_m"], o["y_m"])))
            d["snr"].append(o["snr_db"])
    k = max(3, int(round(fps)))
    kern = np.ones(k) / k
    info = {}
    for tid, d in series.items():
        f = np.asarray(d["f"])
        if f.size < TRACK_ECHO_MIN_S * fps:
            continue
        v = np.full(f[-1] - f[0] + 1, np.nan)
        v[f - f[0]] = d["v"]
        idx = np.arange(v.size)
        ok = np.isfinite(v)
        v = np.interp(idx, idx[ok], v[ok])
        info[tid] = (f[0], v - np.convolve(v, kern, mode="same"), float(np.median(d["r"])),
                     float(np.median(d["snr"])))
    echo_of = {}
    order = sorted(info, key=lambda t: -info[t][3])
    for pos, a in enumerate(order):
        fa, va, ra, _ = info[a]
        for b in order[:pos]:
            if b in echo_of:
                continue
            fb, vb, rb, _ = info[b]
            if rb > ra - ECHO_GAP_M:
                continue
            lo, hi = max(fa, fb), min(fa + va.size, fb + vb.size)
            if hi - lo < TRACK_ECHO_MIN_S * fps:
                continue
            x, y = va[lo - fa:hi - fa][k:-k], vb[lo - fb:hi - fb][k:-k]
            if x.size < 4 or x.std() == 0 or y.std() == 0:
                continue
            if float(np.corrcoef(x, y)[0, 1]) >= TRACK_ECHO_CORR:
                echo_of[a] = b
                break
    return echo_of


# --- the whole recording -------------------------------------------------------------------

@dataclass
class ImagingResult:
    points: np.ndarray                         # structured, POINT_FIELDS
    frames: int
    fps: float
    settings: Settings
    range_m: np.ndarray
    velocity_m_s: np.ndarray                   # one slot's unambiguous axis
    v_max_unfolded: float
    az_deg: np.ndarray
    el_deg: np.ndarray
    has_angles: bool
    has_elevation: bool
    notes: list = field(default_factory=list)
    stem: str = ""
    t0_s: float = 0.0
    starts: np.ndarray = field(default_factory=lambda: np.zeros(1, int))   # first point of every frame
    objects_by_frame: list = field(default_factory=list)
    trails: dict = field(default_factory=dict)  # track id -> (n, 4) frame, x, y, z
    person_tracks: set = field(default_factory=set)   # the tracks that are people (see people())

    def frame(self, f: int, persistence: int = 0) -> np.ndarray:
        f0 = max(0, int(f) - int(persistence))
        f1 = min(self.frames, int(f) + 1)
        return self.points[self.starts[f0]:self.starts[f1]]

    def objects(self, f: int, confirmed: bool = True, echoes: bool = False,
                people_only: bool = False) -> list[dict]:
        """Objects of frame f (x, y, z, velocity, size, n, snr, track, echo_of, person),
        tracked ones only unless confirmed=False, echoes of others only with echoes=True,
        only the people with people_only=True."""
        objs = self.objects_by_frame[int(f)] if 0 <= int(f) < len(self.objects_by_frame) else []
        return [o for o in objs if (o.get("track", -1) >= 0 or not confirmed)
                and (echoes or o.get("echo_of", -1) < 0) and (o.get("person", False) or not people_only)]

    def person_mask(self, points: np.ndarray) -> np.ndarray:
        """Which of these points (of any frames) are people's: of a person's object in their
        frame, or within PERSON_NEAR_M of a person there (a body's scattered points - also
        those taken for an echo of someone else: a second person 2 m from the cascade
        breathed enough like the first that half of their points were)."""
        if not points.size or not getattr(self, "person_tracks", set()):
            return np.zeros(points.size, bool)
        tracks, near = self._person_frames()
        mine = np.zeros(points.size, bool)
        frames = points["frame"].astype(int)
        for f in np.unique(frames):
            i = np.flatnonzero(frames == f)
            tr = tracks.get(int(f))
            if tr:
                mine[i[np.isin(points["cluster"][i].astype(int), np.fromiter(tr, int))]] = True
            xy = near.get(int(f))
            if xy is None:
                continue
            j = i[~mine[i]]
            if j.size:
                d = np.hypot(points["x_m"][j][:, None] - xy[None, :, 0], points["y_m"][j][:, None] - xy[None, :, 1])
                mine[j[d.min(axis=1) < PERSON_NEAR_M]] = True
        return mine

    def _person_frames(self) -> tuple:
        """frame -> the tracks of the people's objects (and of their parts) there, and frame ->
        (k, 2) where the people are (across gaps of a track up to a second)."""
        cache = self.__dict__.get("_person_cache")
        if cache is None:
            tracks, pts = {}, {}
            for f, objs in enumerate(self.objects_by_frame):
                for o in objs:
                    if o.get("person") or "part_of" in o:
                        tracks.setdefault(f, set()).add(int(o["track"]))
                    if o.get("person"):
                        pts.setdefault(int(o["track"]), []).append((f, o["x_m"], o["y_m"]))
            near = {}
            gap = max(1, int(round(self.fps)))
            for rows in pts.values():
                r = np.array(rows, float)
                for (f0, x0, y0), (f1, x1, y1) in zip(r[:-1], r[1:]):
                    n = int(f1 - f0)
                    fs = np.arange(int(f0), int(f1)) if n <= gap else np.array([int(f0)])
                    w = (fs - f0) / max(n, 1)
                    for f, a, b in zip(fs, x0 + w * (x1 - x0), y0 + w * (y1 - y0)):
                        near.setdefault(int(f), []).append((a, b))
                near.setdefault(int(r[-1, 0]), []).append((r[-1, 1], r[-1, 2]))
            cache = (tracks, {f: np.array(v) for f, v in near.items()})
            self.__dict__["_person_cache"] = cache
        return cache

    def trail(self, track: int, f: int, length_s: float = 3.0) -> np.ndarray:
        """(n, 3) x, y, z of a track over the last length_s up to frame f."""
        tr = self.trails.get(int(track))
        if tr is None:
            return np.zeros((0, 3))
        m = (tr[:, 0] <= f) & (tr[:, 0] > f - length_s * self.fps)
        return tr[m, 1:]

    def summary(self) -> str:
        n = self.points.size
        per = n / max(1, self.frames)
        sample = range(0, self.frames, max(1, self.frames // 200))
        objs = [len(self.objects(f)) for f in sample]
        moving = self.points[np.abs(self.points["velocity_m_s"]) > 0]
        txt = (f"{n} points in {self.frames} frames ({per:.0f} a frame), "
               f"{np.median(objs) if objs else 0:.0f} objects a frame (median)")
        persons = getattr(self, "person_tracks", set())
        if persons:
            ppl = [len(self.objects(f, people_only=True)) for f in sample]
            txt += (f"; people: {np.median(ppl):.0f} a frame (median), {len(persons)} tracks, "
                    f"{self.person_mask(self.points).mean():.0%} of the points")
        if moving.size:
            txt += (f"; moving at up to {np.percentile(np.abs(moving['velocity_m_s']), 99):.1f} m/s "
                    f"(unambiguous +-{self.v_max_unfolded:.1f} m/s)")
        return txt


WAVEFRONT_MIN = 0.4            # the strongest echoes of a large array fit point waves at least this well
WAVEFRONT_NOTE = ("the echoes fit the array's model poorly (median {:.2f} of their energy; people in "
                  "front of the cascade 0.65-0.91) - the antenna configuration or the calibration does "
                  "not fit this radar, or something near it distorts the wavefront (a surface, an object "
                  "or a cover close to the antennas): angles and the number of people are unreliable")


def wavefront_fit(points: np.ndarray, arr) -> float | None:
    """
    Median coherence of the strongest tenth of the moving and micro-motion points of an
    array of more than 16 channels, or None. On the cascade with its TX in the user
    guide's order - wrong, see antenna_config - every take with six TX fitted 0.30-0.49
    and one person showed at five angles; with the measured order people fit 0.65-0.91,
    someone seated 0.9 m away (wider than the beam) 0.46 in Multi Vitals' 1 s windows.
    """
    if arr is None or arr.n_virtual <= 16 or not points.size:
        return None
    q = points[((points["kind"] == KIND_MOVING) | (points["kind"] == KIND_MICRO)) & np.isfinite(points["coherence"])]
    if q.size < 20:
        return None
    top = q[q["snr_db"] >= np.percentile(q["snr_db"], 90)]
    return float(np.median(top["coherence"]))


def analyse(cube: np.ndarray, params: dict, fps: float | None = None, settings: Settings | None = None,
            stem: str = "", t0_s: float = 0.0, progress_cb=None) -> ImagingResult:
    """The 4D point cloud of every frame of a capture (frames, chirps, RX, samples)."""
    s = settings or Settings()
    if s.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    cube = np.asarray(cube)
    if cube.ndim != 4 or cube.shape[0] == 0:
        raise ValueError(f"expected a cube (frames, chirps, RX, samples), got {cube.shape}")
    F, C, RX, S = cube.shape
    if fps is None:
        with contextlib.redirect_stdout(io.StringIO()):
            fps = capture_params.derive_fps(params)
    arr = array_for(params, RX)
    lay = _layout(cube.shape, params, arr, s, np.iscomplexobj(cube))
    g = _grid(arr, s) if arr is not None else None
    notes = [arr.note if arr is not None else
             ("the cascade has no calibration - range and velocity only (Measure -> Cascade radar -> "
              "Calibration)" if params.get("cascade") else "no array geometry - range and velocity only")]
    if arr is not None and not arr.elevation:
        notes.append("the array has no vertical extent: no elevation (z from the radar's height)")
    U = len(lay.used)
    v1 = float(np.abs(lay.velocity).max()) if lay.velocity.size else 0.0
    v_unf = max(v1, min(v1 * lay.stride, s.v_limit_m_s)) if (s.unfold and U > 1) else v1
    damaged = vitalsigns.detect_damaged_frames(cube)
    chunks, starts = [], [0]
    w_det = max(4, int(round(s.micro_window_s * float(fps))))
    history = (_History(max(w_det, int(round(s.echo_window_s * float(fps)))), w_det, float(fps))
               if (_uses(s.mode, KIND_MICRO) or _uses(s.mode, KIND_MOVING)) else None)
    if _uses(s.mode, KIND_MOVING) and lay.L < MIN_LOOPS_DOPPLER:
        notes.append(f"{lay.L} chirps of a TX a frame: no Doppler in a frame - "
                     + ("moving people show as micro-motion" if _uses(s.mode, KIND_MICRO) else "no moving points"))
    # micro-motion over the frames around a frame, not before it: a walker's motion of
    # the last second lagged half a metre behind them and made a second track
    half = w_det // 2
    rows_of = [[] for _ in range(F)]
    for f in range(F):
        if damaged[f]:
            if history is not None:
                history.clear()
        else:
            pts, _ = frame_points(cube[f], params, arr, lay, g, s, history)
            # micro rows (echoes among them) have no velocity; a moving row always has one
            micro = (pts[:, 7] == KIND_MICRO) | ((pts[:, 7] == KIND_ECHO) & (pts[:, 1] == 0)) if pts.size else \
                np.zeros(0, bool)
            rows_of[f].append(pts[~micro])
            if micro.any() and f - half >= 0:
                rows_of[f - half].append(pts[micro])
            if f == F - 1 and micro.any():     # the last half window: the last window's points,
                for k in range(max(0, F - half), F):            # but not behind a mover there
                    own = np.concatenate(rows_of[k]) if rows_of[k] else np.zeros((0, 8))
                    mv = own[own[:, 7] == KIND_MOVING] if own.size else own
                    keep = pts[micro]
                    if mv.size:
                        xm, ym, _ = _to_xyz(mv[:, 0], mv[:, 2], mv[:, 3], s)
                        xk, yk, _ = _to_xyz(keep[:, 0], keep[:, 2], keep[:, 3], s)
                        far = np.min(np.hypot(xk[:, None] - xm[None], yk[:, None] - ym[None]), axis=1) > 1.0
                        keep = keep[far]
                    rows_of[k].append(keep)
        if progress_cb is not None and (f % 10 == 0 or f == F - 1):
            progress_cb(f + 1, F)
    for f in range(F):
        pts = np.concatenate(rows_of[f]) if rows_of[f] else np.zeros((0, 8))
        rows_of[f] = None
        rec = np.zeros(pts.shape[0], dtype=[(k, "f4") for k in POINT_FIELDS])
        if pts.size:
            rec["frame"] = f
            rec["range_m"], rec["velocity_m_s"] = pts[:, 0], pts[:, 1]
            rec["azimuth_deg"], rec["elevation_deg"] = pts[:, 2], pts[:, 3]
            x, y, z = _to_xyz(pts[:, 0], pts[:, 2], pts[:, 3], s)
            if arr is None:                                   # range only: on the axis
                x, y, z = np.zeros_like(x), pts[:, 0], np.full_like(x, s.height_m)
            rec["x_m"], rec["y_m"], rec["z_m"] = x, y, z
            rec["power_db"] = 10 * np.log10(np.maximum(pts[:, 4], 1e-30))
            rec["snr_db"] = pts[:, 5]
            rec["coherence"] = pts[:, 6]
            rec["kind"] = pts[:, 7]
            rec["cluster"] = -1
        chunks.append(rec)
        starts.append(starts[-1] + rec.size)
    points = np.concatenate(chunks) if chunks else np.zeros(0, dtype=[(k, "f4") for k in POINT_FIELDS])
    starts = np.asarray(starts, int)
    by_frame, trails = (_objects_and_tracks(points, starts, F, float(fps), s) if arr is not None
                        else ([[] for _ in range(F)], {}))
    persons = people(points, by_frame, trails, float(fps)) if s.mode != "still" else set()
    if persons:
        found = []
        if s.confirm_people:
            try:
                found = vital_people(cube, params, float(fps))
            except Exception as e:                # the layer then rests on the tracks alone
                notes.append(f"people not confirmed by Multi Vitals ({e})")
        persons = confirm_people(persons, by_frame, trails, float(fps), found)
    wf = wavefront_fit(points, arr)
    if wf is not None and wf < WAVEFRONT_MIN:
        notes.append(WAVEFRONT_NOTE.format(wf))
    if int(damaged.sum()):
        notes.append(f"{int(damaged.sum())} frame(s) lost packets - no points there")
    return ImagingResult(points=points, frames=F, fps=float(fps), settings=s, range_m=lay.range_m,
                         velocity_m_s=lay.velocity, v_max_unfolded=v_unf,
                         az_deg=g.az if g is not None else np.zeros(1),
                         el_deg=g.el if g is not None else np.zeros(1), has_angles=arr is not None,
                         has_elevation=bool(arr is not None and arr.elevation), notes=notes, stem=stem,
                         t0_s=t0_s, starts=starts, objects_by_frame=by_frame, trails=trails,
                         person_tracks=persons)


# --- images of one frame ---------------------------------------------------------------------

_RANGE_STEERING = {}


def _range_steering(arr: Array, az: np.ndarray, range_m: np.ndarray, lam: float) -> np.ndarray:
    """(R, V, A) exact-path steering towards azimuths az (elevation 0) for every range bin,
    kept for the next frame of the same array and axis."""
    key = (tuple(np.round(arr.rx.ravel(), 4)), tuple(np.round(np.concatenate([np.ravel(o) for s in arr.used_slots
                                                                               for o in arr.slots[s]]), 4)),
           tuple(arr.used_slots), float(lam), float(az[0]), float(az[-1]), az.size,
           float(range_m[0]), float(range_m[-1]), range_m.size)
    if key not in _RANGE_STEERING:
        _RANGE_STEERING.clear()
        _RANGE_STEERING[key] = np.stack([arr.steering(az, np.zeros_like(az), max(r, 0.05) / lam)
                                         for r in range_m]).astype(np.complex64)
    return _RANGE_STEERING[key]


FRONT_NEAR_M = 0.15            # the front image: bins this near the frame's points
MICRO_IMAGE_FRAMES = 12        # frames of the micro window an image uses at most


def frame_images(cube: np.ndarray, params: dict, f: int, settings: Settings | None = None,
                 front_window_m: tuple | None = None, fps: float = 20.0, at_ranges_m=None) -> dict:
    """
    Images of frame f for the view, of what its mode looks at - what moves in the frame,
    what moves from frame to frame (micro; up to MICRO_IMAGE_FRAMES frames of the micro
    window), the still scene: range-Doppler power (all channels), bird's-eye (range x azimuth) and front -
    what the radar sees, elevation x azimuth, over the ranges of front_window_m (all by
    default). Plane waves, slot timing corrected for the velocity of every Doppler bin
    within one slot's limit. Bartlett power; the cascade's front image at a coarser grid.
    at_ranges_m: the front image of the bins within FRONT_NEAR_M of these ranges only (the
    frame's points) - over all ranges, the motion of a whole room outweighed a seated
    person's, and the front image of a real take pointed 70 degrees away from them.
    """
    s = settings or Settings()
    cube = np.asarray(cube)
    F, C, RX, S = cube.shape
    arr = array_for(params, RX)
    lay = _layout(cube.shape, params, arr, s, np.iscomplexobj(cube))
    X, still = _spectra(cube[int(f)], params, lay, arr)
    P = (np.abs(X) ** 2).sum(axis=1)
    out = {"range_m": lay.range_m, "velocity_m_s": lay.velocity, "rd": P,
           "still_profile": (np.abs(still) ** 2).sum(axis=0)}
    if arr is None:
        return out
    zero = lay.nfft_d // 2
    parts = []
    if _uses(s.mode, KIND_MOVING) and lay.L >= MIN_LOOPS_DOPPLER:
        k = np.repeat(np.asarray(lay.used, float), arr.rx.shape[0])
        corr = np.exp(-1j * 2 * np.pi * np.outer(lay.f_d, k * lay.tc_s))                 # (D, V)
        Xc = np.delete(X * corr[:, :, None], zero, axis=0)
        parts.append(Xc)
    if _uses(s.mode, KIND_STILL):
        parts.append(still[None])
    if _uses(s.mode, KIND_MICRO):
        W = max(4, int(round(s.micro_window_s * fps)))
        lo = max(0, int(f) - W // 2)
        hi = min(F - 1, lo + W - 1)
        fr = np.unique(np.linspace(lo, hi, min(W, MICRO_IMAGE_FRAMES)).astype(int))
        if fr.size > 2:
            H = np.stack([_spectra(cube[i], params, lay, arr)[1] for i in fr])             # (W', V, R)
            parts.append((H - H.mean(axis=0, keepdims=True)) * np.sqrt(W / fr.size))
    if not parts:
        return out
    Z = np.concatenate(parts, axis=0)                                                      # (D', V, R)
    az = np.arange(-s.az_limit_deg, s.az_limit_deg + 1e-9, s.az_step_deg)
    near = NEAR_FACTOR * arr.far_field_wl() * lay.lam
    if lay.range_m.size and lay.range_m[0] < near and arr.far_field_wl() * lay.lam > 0.3:
        # near a large array the exact paths of every range bin: with plane waves a person
        # 2 m in front of the cascade (far field from 2.7 m) was an arc a metre to each side
        A_r = _range_steering(arr, az, lay.range_m, lay.lam)                               # (R, V, nA)
        nrm = np.real(np.einsum("rvk,rvk->rk", A_r.conj(), A_r))
        B = np.abs(np.einsum("dvr,rvk->rdk", Z, A_r.conj(), optimize=True)) ** 2
        out["bev"] = B.sum(axis=1) / nrm
    else:
        A = arr.steering(az, np.zeros_like(az))                                            # (V, nA)
        nrm = np.real(np.einsum("vk,vk->k", A.conj(), A))
        B = np.abs(np.einsum("dvr,vk->rdk", Z, A.conj(), optimize=True)) ** 2
        out["bev"] = B.sum(axis=1) / nrm[None]                                             # (R, nA)
    out["az_deg"] = az
    if arr.elevation:
        big = arr.n_virtual > 16
        az_f = np.arange(-s.az_limit_deg, s.az_limit_deg + 1e-9, s.az_step_deg * (2 if big else 1))
        el_f = np.arange(-s.el_limit_deg, s.el_limit_deg + 1e-9, s.el_step_deg * (2 if big else 1))
        AZ, EL = np.meshgrid(az_f, el_f)
        Af = arr.steering(AZ.ravel(), EL.ravel())
        r0, r1 = front_window_m or (s.r_min_m, lay.range_m[-1])
        sel = (lay.range_m >= r0) & (lay.range_m <= r1)
        if at_ranges_m is not None and len(at_ranges_m):
            near = np.min(np.abs(lay.range_m[:, None] - np.asarray(at_ranges_m, float)[None]), axis=1)
            if np.any(sel & (near <= FRONT_NEAR_M)):
                sel &= near <= FRONT_NEAR_M
        Zs = Z[:, :, sel]
        Cv = np.einsum("dvr,dwr->vw", Zs, Zs.conj())
        Pf = np.real(np.einsum("vk,vw,wk->k", Af.conj(), Cv, Af)) / np.real(np.einsum("vk,vk->k", Af.conj(), Af))
        out["front"] = np.maximum(Pf, 1e-30).reshape(el_f.size, az_f.size)
        out["front_az_deg"], out["front_el_deg"] = az_f, el_f
    return out
