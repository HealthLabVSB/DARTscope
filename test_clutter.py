"""
Tests for clutter - static clutter suppression along slow time.

Run with:  python -m pytest test_clutter.py -q
"""
import numpy as np
import pytest

import clutter

LOOPS, RANGE = 64, 32


def _scene(frames=40, breathing=False, seed=0):
    """
    (frames, loops, range): a wall at bin 5, a target moving at Doppler bin 8 at
    range bin 20 and, optionally, a seated person at bin 12 whose phase follows
    breathing from frame to frame but is constant within a frame.
    """
    rng = np.random.default_rng(seed)
    n = np.arange(LOOPS)
    x = np.zeros((frames, LOOPS, RANGE), complex)
    x[:, :, 5] += 10.0 * np.exp(0.7j)
    x[:, :, 20] += np.exp(2j * np.pi * 8 * n / LOOPS)[None, :]
    if breathing:
        # 4 mm of chest travel at 5 mm wavelength: 4 pi * 4/5 rad peak to peak
        phase = 0.5 * 4 * np.pi * 0.8 * np.sin(2 * np.pi * 0.25 * np.arange(frames) * 0.1)
        x[:, :, 12] += 2.0 * np.exp(1j * phase)[:, None]
    x += 1e-3 * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    return x


def _power(x, r):
    return float(np.mean(np.abs(x[..., r]) ** 2))


_S8 = np.sin(np.pi * 8 / LOOPS)


@pytest.mark.parametrize("method, mover_gain", [
    ("mean", 1.0), ("svd", 1.0),
    ("mti2", (2 * _S8) ** 2),        # a slow target loses power in an MTI canceller,
    ("mti3", (4 * _S8 ** 2) ** 2),   # the three-pulse one more than the two-pulse one
])
def test_every_method_removes_a_wall_and_keeps_a_mover(method, mover_gain):
    x = _scene()
    y = clutter.suppress(x, method, slow_axis=1, frame_axis=0)
    assert y.shape == x.shape
    assert _power(y, 5) < 1e-6 * _power(x, 5)
    assert _power(y, 20) == pytest.approx(mover_gain * _power(x, 20), rel=0.05)


def test_ema_removes_a_wall_after_it_has_seen_it():
    x = _scene()
    y = clutter.suppress(x, "ema", slow_axis=1, frame_axis=0, alpha=0.2)
    assert _power(y[1:], 5) < 1e-6 * _power(x, 5)
    assert _power(y, 20) > 0.9 * _power(x, 20)


def test_ema_keeps_a_seated_breathing_person_that_mean_removes():
    """The reason to offer it: per-frame mean takes out everything at zero Doppler."""
    x = _scene(frames=200, breathing=True)
    alpha = clutter.alpha_from_time_constant(10.0, 0.1)
    y_ema = clutter.suppress(x, "ema", slow_axis=1, frame_axis=0, alpha=alpha)
    y_mean = clutter.suppress(x, "mean", slow_axis=1, frame_axis=0)
    assert _power(y_mean, 12) < 1e-6 * _power(x, 12)
    assert _power(y_ema[50:], 12) > 0.3 * _power(x, 12)
    assert _power(y_ema[50:], 5) < 1e-4 * _power(x, 5)


@pytest.mark.parametrize("k", [1, 3, 8, 20, 32])
def test_mti_frequency_responses(k):
    """Circular differencing is an exact filter on the DFT grid."""
    n = np.arange(LOOPS)
    x = np.exp(2j * np.pi * k * n / LOOPS)
    g2 = np.abs(clutter.suppress(x, "mti2", slow_axis=0, frame_axis=None)).mean()
    g3 = np.abs(clutter.suppress(x, "mti3", slow_axis=0, frame_axis=None)).mean()
    s = np.sin(np.pi * k / LOOPS)
    assert g2 == pytest.approx(2 * abs(s), rel=1e-9)
    assert g3 == pytest.approx(4 * s * s, rel=1e-9)


def test_svd_follows_clutter_whose_amplitude_drifts():
    """A wall whose return fades by 20 % across the frame: rank-1, but not constant."""
    x = _scene(frames=4)
    drift = np.linspace(0.9, 1.1, LOOPS)[None, :]
    x[:, :, 5] = 10.0 * np.exp(0.7j) * drift
    x[:, :, 7] = 4.0 * np.exp(-0.2j) * drift
    y_mean = clutter.suppress(x, "mean", slow_axis=1, frame_axis=0)
    y_svd = clutter.suppress(x, "svd", slow_axis=1, frame_axis=0, rank=1)
    assert _power(y_svd, 5) < 1e-3 * _power(y_mean, 5)
    assert _power(y_svd, 20) > 0.8 * _power(x, 20)


def test_svd_without_frame_axis_on_a_slow_time_series():
    t = np.arange(500)
    x = np.zeros((500, RANGE), complex)
    x[:, 3] = 5.0
    x[:, 9] = np.exp(2j * np.pi * 0.05 * t)
    y = clutter.suppress(x, "svd", slow_axis=0, frame_axis=None, rank=1)
    assert _power(y, 3) < 1e-9 * 25
    assert _power(y, 9) > 0.9


def test_from_params_defaults_to_mean_and_rejects_unknown():
    x = _scene(frames=3)
    assert np.allclose(clutter.from_params(x, {}), clutter.suppress(x, "mean"))
    assert np.allclose(clutter.from_params(x, None), clutter.suppress(x, "mean"))
    assert clutter.from_params(x, {"clutter_method": "none"}) is not None
    with pytest.raises(ValueError):
        clutter.suppress(x, "wavelet")


def test_alpha_from_time_constant():
    assert clutter.alpha_from_time_constant(1.0, 1.0) == pytest.approx(1 - np.exp(-1))
    assert clutter.alpha_from_time_constant(0.0, 0.1) == 1.0


def test_range_doppler_map_uses_the_chosen_method():
    """compute_range_doppler_map reads params['clutter_method']; 'mean' is unchanged."""
    from radar_processing import compute_range_doppler_map
    from test_capon_rae import make_params

    params = make_params()
    F, L, RX, N = 6, 16, 4, int(params.get("numAdcSamples", 64))
    t = np.arange(N)
    tone = np.exp(2j * np.pi * 0.2 * t)
    cube = np.broadcast_to(tone, (F, L * max(1, int(params.get("unique_chirps_frame", 1))), RX, N))
    cube = np.ascontiguousarray(cube).astype(np.complex64)
    rd_mean, _ = compute_range_doppler_map(cube, params)
    rd_none, _ = compute_range_doppler_map(cube, {**params, "clutter_method": "none"})
    # A static tone: gone with the default, dominant without suppression.
    assert rd_mean.max() < 1e-3 * rd_none.max()


def test_ema_warm_start_keeps_the_first_frames():
    """Cold, the first frame is its own background (blank); warm, it is compared with the
    mean of the first time constant, so a breathing person shows from frame 0 on."""
    x = _scene(frames=60, breathing=True)
    cold = clutter.suppress(x, "ema", slow_axis=1, frame_axis=0, alpha=0.05)
    warm = clutter.suppress(x, "ema", slow_axis=1, frame_axis=0, alpha=0.05, warm=True)
    assert _power(cold[:1], 12) < 1e-5
    assert _power(warm[:1], 12) > 0.1
    assert _power(warm[:1], 5) < 1e-3 * _power(x[:1], 5)        # the wall still goes
    assert np.array_equal(cold[1:].shape, warm[1:].shape)


def test_the_gui_choice_keeps_a_seated_person_with_a_few_chirps():
    """
    Movement Map, Range-Angle and AoA take the Analysis bar's clutter choice through
    from_params. With the vital-signs settings (4 chirps a frame) the default "mean" left
    nothing of a breathing person; the background is then taken over frames. 16 chirps
    keep the frame mean.
    """
    x = _scene(frames=60, breathing=True)[:, :4]
    out = clutter.from_params(x, {"clutter_method": "mean", "clutter_alpha": 0.05})
    assert _power(clutter.suppress(x, "mean"), 12) < 1e-5     # what "mean" did
    assert _power(out, 12) > 0.1                                # the person stays
    assert _power(out, 5) < 1e-3 * _power(x, 5)                 # the wall goes
    assert clutter.few_loops_method("mean", 4) == ("ema", True)
    assert clutter.few_loops_method("mti2", 4) == ("mti2", False)
    assert clutter.few_loops_method("mean", None) == ("mean", False)   # no frame axis
    many = _scene(frames=10, breathing=True)[:, :16]
    assert np.allclose(clutter.from_params(many, {"clutter_method": "mean"}),
                       clutter.suppress(many, "mean"))


def test_range_doppler_map_with_a_few_chirps_keeps_a_seated_person():
    """
    Vital-signs settings: 4 chirps per frame. The frame mean would remove the whole scene
    (and a 4-point Hann window half the chirps); the map instead takes the background over
    frames and no window, so a breathing person stays at 0 m/s and a wall goes.
    """
    from radar_processing import compute_range_doppler_map
    from test_capon_rae import make_params

    params = {**make_params(), "uniqueChirps_effective": 1, "unique_chirps_frame": 1,
              "tx_order_masks": [1], "clutter_alpha": 0.01}
    params.pop("is_frequency_domain")
    F, L, RX, N = 120, 4, 4, 64
    t = np.arange(N)
    breathing = 2.0 * np.sin(2 * np.pi * 0.25 * np.arange(F) * 0.05)        # rad, frame to frame
    wall = 10.0 * np.exp(2j * np.pi * 0.10 * t)
    person = np.exp(1j * breathing)[:, None] * np.exp(2j * np.pi * 0.30 * t)[None, :]
    cube = np.broadcast_to((wall[None, :] + person)[:, None, None, :], (F, L, RX, N)).astype(np.complex64)
    rd, meta = compute_range_doppler_map(cube, params)                  # the default: 'mean', Hann
    assert meta["few_loops_note"] and meta["doppler_window"] == "none"
    zero = L // 2                                   # 0 m/s after fftshift (no timing here)
    n = rd.shape[-1]
    k_wall, k_person = int(round(0.10 * n)), int(round(0.30 * n))
    for f in (0, 30, 90):
        col = rd[f, zero]
        assert col[k_person] > 10 * col[k_wall], f
    many = {**params}
    cube16 = np.broadcast_to(cube[:, :1], (F, 16, RX, N)).astype(np.complex64)
    _, meta16 = compute_range_doppler_map(cube16, many)
    assert meta16["few_loops_note"] is None and meta16["doppler_window"] == "hann"


@pytest.mark.parametrize("loops", [4, 5, 64])
def test_range_doppler_pixels_sit_on_their_velocities(loops):
    """
    The map is drawn with every pixel centred on its velocity. An extent from the first
    to the last bin put 4 pixels into 3 bins: the zero-velocity one at -v_res/8.
    """
    from plot_interactivity import bin_extent

    v = np.fft.fftshift(np.fft.fftfreq(loops, d=47.8e-6)) * 0.005 / 2
    r = np.arange(80) * 0.05
    x0, x1, y0, y1 = bin_extent(v, r)
    centres = x0 + (np.arange(loops) + 0.5) * (x1 - x0) / loops
    assert np.allclose(centres, v)
    assert centres[loops // 2] == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(y0 + (np.arange(r.size) + 0.5) * (y1 - y0) / r.size, r)


def test_a_handful_of_velocity_bins_is_drawn_as_blocks():
    """Smoothed, 4 velocity bins 12.9 m/s wide drew a blob suggesting finer velocity."""
    from plot_interactivity import few_bins_interpolation
    assert few_bins_interpolation(4) == few_bins_interpolation(5) == "nearest"
    assert few_bins_interpolation(64) is None



def test_a_short_frame_takes_its_background_over_frames():
    """8 chirps of a TX in 0.3 ms (the cascade's multi-person preset: 1.9 ms): the frame mean
    took a breathing person out of the Range-Doppler map; the background over frames keeps
    them at 0 m/s, and the window stays (8 chirps are enough for one)."""
    from radar_processing import compute_range_doppler_map
    from test_capon_rae import make_params

    params = {**make_params(), "uniqueChirps_effective": 1, "unique_chirps_frame": 1,
              "tx_order_masks": [1], "clutter_alpha": 0.01, "idleTime_us": 7.0, "rampEndTime_us": 33.0}
    params.pop("is_frequency_domain")
    F, L, RX, N = 120, 8, 4, 64
    t = np.arange(N)
    breathing = 2.0 * np.sin(2 * np.pi * 0.25 * np.arange(F) * 0.05)
    wall = 10.0 * np.exp(2j * np.pi * 0.10 * t)
    person = np.exp(1j * breathing)[:, None] * np.exp(2j * np.pi * 0.30 * t)[None, :]
    cube = np.broadcast_to((wall[None, :] + person)[:, None, None, :], (F, L, RX, N)).astype(np.complex64)
    rd, meta = compute_range_doppler_map(cube, params)
    assert meta["few_loops_note"] and "span" in meta["few_loops_note"] and meta["doppler_window"] == "hann"
    zero = L // 2
    n = rd.shape[-1]
    k_wall, k_person = int(round(0.10 * n)), int(round(0.30 * n))
    for f in (30, 90):
        col = rd[f, zero]
        assert col[k_person] > 10 * col[k_wall], f
