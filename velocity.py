"""
velocity.py

Velocities beyond the unambiguous limit (roadmap 1.4).

The Doppler FFT runs over loops, one sample per loop period T, so it measures the
velocity only modulo 2 v_max, v_max = lambda / (4 T). A target at 1.3 v_max reads as
-0.7 v_max. Two ways to recover the fold count are offered:

  Multiple PRFs. The same target measured with two (or more) loop periods folds
  differently in each; the velocity that is consistent with all of them is the
  answer. With v_max in the ratio p:q (coprime) the unambiguous span grows to
  q * 2 v_max1 = p * 2 v_max2. Needs a capture with alternating chirp periods or
  sub-frames with different periods - none of the captures here has that, so this is
  offered as a function, tested on synthetic numbers.

  TDM hypotheses. In TDM-MIMO the TX take turns, S slots per loop, so each loop
  also samples the target S times, one slot apart. The fold count m adds
  2 pi m t / S to TX slot t - the loops cannot see it, the virtual array can: only the
  right m makes the array coherent again. Testing the S hypotheses against the
  beamformer peak extends the span to +-S v_max. It needs a real TDM-MIMO capture
  (one TX per chirp) and an array whose TX blocks are not trivially redundant;
  the margin between the best and the second-best hypothesis says how sure it is.
  Scoring by the beamformer peak alone is weak: a wrong hypothesis is a phase step
  between TX blocks, and the beamformer absorbs much of it by moving to another angle
  (on the IWR6843AOP array the wrong ones still fit 62 % of the energy, a 2.1 dB
  margin with no noise at all). The score is therefore taken at the angle the RX
  alone point to - each TX block beamformed on its own and the powers added, which
  the TX phases cannot affect. Noise-free worst-case margins over a grid of angles:

                              peak only   RX-anchored
      IWR6843AOP 3TX x 4RX      2.1 dB       3.6 dB
      ULA 2TX x 4RX             2.7 dB       4.3 dB
      1843/ISK 3TX x 4RX        1.0 dB       1.6 dB
      AWR2544LOP 4TX x 4RX      1.0 dB       1.9 dB

  With 10 dB SNR per channel the RX-anchored test still chose the right fold in every
  one of the 45-60 cases per array. A target that is really several scatterers in one
  cell breaks the plane-wave assumption either way. On the one real TDM capture here
  (AOA/capon_TDM_MIMO_60s, v_max = 2.94 m/s, people walking, so nothing actually
  folds) 2.6 % of the CFAR detections were still given a fold of +-1 with a margin of
  3 dB or more - the default threshold - and none of the strongest detections of a
  frame was. Treat a lone unfolded detection with suspicion; one that keeps its fold
  over consecutive frames is the real thing.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import numpy as np


def wrap(v, v_max):
    """Fold a velocity into [-v_max, v_max), as the Doppler FFT does."""
    v = np.asarray(v, dtype=float)
    return (v + v_max) % (2.0 * v_max) - v_max


# --- multiple PRFs ------------------------------------------------------------------

def unfold_multi_prf(v_measured, v_max, v_limit: float):
    """
    The velocity within +-v_limit that agrees best with every measurement.

    v_measured[i] was measured with unambiguous velocity v_max[i] (each in
    [-v_max[i], v_max[i])). Every fold of every measurement within the limit is a
    candidate; the one with the smallest RMS disagreement (each measurement compared
    modulo its own 2 v_max) wins.
    Returns (velocity, rms_residual). A residual comparable to the velocity
    resolution means the measurements agree; much larger means they are from
    different targets or v_limit is beyond what these PRFs can resolve.
    """
    vm = np.asarray(v_measured, dtype=float).ravel()
    vx = np.asarray(v_max, dtype=float).ravel()
    if vm.size != vx.size or vm.size < 2:
        raise ValueError("need one v_max per measurement, and at least two measurements")
    cands = []
    for v, m in zip(vm, vx):
        n = int(np.ceil(v_limit / (2.0 * m))) + 1
        c = v + 2.0 * m * np.arange(-n, n + 1)
        cands.append(c[np.abs(c) <= v_limit])
    cands = np.concatenate(cands)
    if cands.size == 0:
        return float("nan"), float("inf")
    diffs = wrap(cands[:, None] - vm[None, :], vx[None, :])
    rms = np.sqrt(np.mean(diffs ** 2, axis=1))
    best = int(np.argmin(rms))
    # the candidate itself is one fold of one measurement; refine it to the mean of
    # all measurements unfolded next to it
    v0 = cands[best]
    v = float(np.mean(v0 - diffs[best]))
    return v, float(rms[best])


# --- TDM hypotheses -----------------------------------------------------------------

def fold_candidates(v_measured: float, v_max: float, num_slots: int):
    """
    The velocities the fold hypotheses stand for, one per residue m mod S.

    Hypotheses m and m + S shift every TX slot by a whole turn, so they look the same;
    of each pair the one within [-S v_max, S v_max) is kept.
    Returns {residue: (m, velocity)}.
    """
    S = int(num_slots)
    out = {}
    for m in range(-(S - 1), S):
        v = v_measured + 2.0 * m * v_max
        if -S * v_max <= v < S * v_max:
            r = m % S
            if r not in out or abs(v) < abs(out[r][1]):
                out[r] = (m, float(v))
    return out


def hypothesis_gains(snapshot, k_loops: float, n_loops: int, num_tx: int, num_rx: int,
                     steering, slots_per_loop: int | None = None):
    """
    Beamformer coherence of a virtual-array snapshot under each fold hypothesis,
    taken where the RX alone point (see the module docstring).

    snapshot: (V,) at the target's Doppler peak, virtual index t * num_rx + rx, element
    phases and calibration already applied, NOT yet Doppler compensated.
    k_loops: measured Doppler in loop bins, signed and fractional.
    steering: (V, K) steering vectors over the angle grid.
    Returns an array of S gains in [0, 1] (residue m mod S); 1 = a plane wave where
    the RX point.
    """
    S = int(slots_per_loop or num_tx)
    x = np.asarray(snapshot, dtype=complex).ravel()
    n = num_tx * num_rx
    t = np.repeat(np.arange(num_tx), num_rx)
    x = x[:n] * np.exp(-1j * t * 2.0 * np.pi * k_loops / (n_loops * S))
    a = np.asarray(steering)[:n]
    p_rx = np.zeros(a.shape[1])
    for b in range(num_tx):
        blk = slice(b * num_rx, (b + 1) * num_rx)
        p_rx += np.abs(a[blk].conj().T @ x[blk]) ** 2
    weight = p_rx / (p_rx.max() + 1e-30)
    norm = n * np.sum(np.abs(x) ** 2) + 1e-30
    gains = np.empty(S)
    for r in range(S):
        xr = x * np.exp(-1j * t * 2.0 * np.pi * r / S)
        gains[r] = float(np.max(np.abs(a.conj().T @ xr) ** 2 * weight) / norm)
    return gains


def _steering(params: dict, num_virt: int, num_rx: int):
    import AOA

    x, y, _ = AOA._get_antenna_geometry(params, num_virt, num_rx)
    az = np.deg2rad(np.arange(-80.0, 80.25, 0.5))
    el = np.deg2rad(np.arange(-50.0, 50.5, 1.0)) if np.any(np.asarray(y) != 0) else np.zeros(1)
    AZ, EL = np.meshgrid(az, el)
    kx, ky = (np.sin(AZ) * np.cos(EL)).ravel(), np.sin(EL).ravel()
    return np.exp(1j * 2.0 * np.pi * (np.outer(x, kx) + np.outer(y, ky)))


def tdm_layout(params: dict, num_chirps: int):
    """(num_tx, slots_per_loop, loops) when the capture is plain TDM-MIMO, else None."""
    import AOA

    eff = int(params.get("uniqueChirps_effective") or 1)
    stride = max(1, int(params.get("unique_chirps_frame") or eff))
    masks = params.get("tx_order_masks")
    if (eff < 2 or params.get("cube_chirp_filtered") or num_chirps % stride
            or not AOA._is_safe_tdm(masks[:eff] if masks else None)):
        return None
    return eff, stride, num_chirps // stride


def extend_detections(peaks: list, frame: np.ndarray, params: dict, rd_meta: dict,
                      min_margin_db: float = 3.0, v_limit: float | None = None) -> list:
    """
    Add unfolded velocities to Range-Doppler detections of one frame (TDM hypotheses).

    frame: the frame's chirps as the Range-Doppler map got them, (chirps, RX, samples)
    of ADC data - or range bins when params['is_frequency_domain'].
    Adds to each peak: 'velocity_unfolded_ms', 'fold' (m), 'fold_decided',
    'fold_margin_db', 'fold_gains', and the best hypothesis whether accepted or not
    ('fold_best', 'velocity_best_ms'). Leaves the list unchanged when the capture is
    not TDM-MIMO. The fold stays 0 ('fold_decided' False) when the best hypothesis
    beats the next by less than min_margin_db, or when it stands for a speed above
    v_limit (m/s) - e.g. faster than anything in the scene can move.
    """
    import AOA
    from radar_processing import _window_vector, apply_window

    C, RX = frame.shape[0], frame.shape[1]
    layout = tdm_layout(params, C)
    v_axis = np.asarray(rd_meta.get("velocity_axis", []), dtype=float)
    if layout is None or v_axis.size < 2 or not peaks:
        return peaks
    S, stride, L = layout
    V = S * RX
    nd = int(rd_meta.get("n_fft_doppler") or v_axis.size)
    v_max = abs(float(v_axis[0]))
    SV = _steering(params, V, RX)
    if params.get("is_frequency_domain"):
        rng = frame
    else:
        x = frame - frame.mean(axis=-1, keepdims=True)
        rng = np.fft.fft(apply_window(x, axis=-1, window="hann"),
                         n=int(rd_meta.get("n_fft_range") or 256), axis=-1)
    virt = rng.reshape(L, stride, RX, -1)[:, :S].reshape(L, V, -1)   # (loops, V, range)
    w = _window_vector(L, "hann")
    for p in peaks:
        r = int(p["range_idx"])
        if r >= virt.shape[-1] or p.get("velocity_ms") is None:
            continue
        k = (float(p.get("doppler_pos", p["doppler_idx"])) - nd // 2) * L / nd
        snap = (w[:, None] * virt[:, :, r]
                * np.exp(-2j * np.pi * k * np.arange(L) / L)[:, None]).sum(axis=0)
        snap = AOA._apply_aop_rx_phase_correction(snap[None, None, :, None], params, V, RX)[0, 0, :, 0]
        snap = AOA._apply_calibration(snap, params, V)
        gains = hypothesis_gains(snap, k, L, S, RX, SV, stride)
        order = np.argsort(gains)[::-1]
        margin = 10.0 * np.log10(gains[order[0]] / max(gains[order[1]], 1e-12))
        cands = fold_candidates(float(p["velocity_ms"]), v_max, stride)
        m, v = cands.get(int(order[0]), (0, float(p["velocity_ms"])))
        p.update(fold_best=int(m), velocity_best_ms=float(v))
        decided = margin >= min_margin_db and (v_limit is None or abs(v) <= v_limit)
        if not decided:
            m, v = 0, float(p["velocity_ms"])
        p.update(velocity_unfolded_ms=v, fold=int(m), fold_decided=bool(decided),
                 fold_margin_db=float(margin), fold_gains=[float(g) for g in gains])
    return peaks
