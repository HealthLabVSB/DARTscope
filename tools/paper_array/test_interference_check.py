"""Synthetic checks of tools/paper_array/interference_check.py (python -m pytest tools/paper_array)."""
import numpy as np

from tools.paper_array import interference_check as ic


def _scene(F=24, C=10, RX=4, S=128, seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    n = np.arange(S)
    x = np.zeros((F, C, RX, S), np.complex64)
    for k, a in ((9.3, 300.0), (21.7, 80.0), (40.2, 30.0)):            # static reflectors
        ph = rng.uniform(0, 2 * np.pi, RX)
        x += (a * np.exp(1j * (2 * np.pi * k * n / S)[None, None, None, :] + 1j * ph[None, None, :, None]))
    drift = np.exp(1j * 0.3 * np.sin(2 * np.pi * np.arange(F) / F))       # slow chest-like motion
    x *= drift[:, None, None, None]
    x += noise * (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)) / np.sqrt(2)
    return x.astype(np.complex64), rng


def _sweep(S, start, length, amp, rate=0.02):
    """A foreign chirp crossing the IF band: a linear-FM burst of `length` samples."""
    n = np.arange(S, dtype=float)
    z = np.zeros(S, complex)
    m = (n >= start) & (n < start + length)
    t = n[m] - start
    z[m] = amp * np.exp(1j * np.pi * rate * t ** 2) * np.hanning(length)
    return z


def test_clean_scene_has_no_hits():
    x, _ = _scene()
    d = ic.detect(x, pattern=2)
    assert d["hit"].sum() == 0


def test_injected_sweep_on_all_rx_fires_once():
    x, _ = _scene()
    x[7, 3, :, :] += _sweep(x.shape[-1], 30, 40, 20.0)[None, :]
    d = ic.detect(x, pattern=2)
    assert d["hit"][7, 3]
    assert d["hit"].sum() == 1
    assert d["frame_hit"].sum() == 1 and d["frame_hit"][7]


def test_sweep_on_two_rx_is_not_counted():
    x, _ = _scene()
    x[12, 4, :2, :] += _sweep(x.shape[-1], 30, 40, 20.0)[None, :]
    d = ic.detect(x, pattern=2)
    assert not d["hit"][12, 4]
    assert d["hit"].sum() == 0


def test_zero_filled_frame_is_excluded():
    x, _ = _scene()
    x[5] = 0
    damaged = np.zeros(x.shape[0], bool)
    damaged[5] = True
    x[9, 1, :, :] += _sweep(x.shape[-1], 10, 60, 20.0)[None, :]
    d = ic.detect(x, pattern=2, damaged=damaged)
    assert not d["hit"][5].any()
    assert d["hit"][9, 1] and d["hit"].sum() == 1


def _moving(x, frame, k=15.4, amp=300.0, turns=2.0):
    """A strong target whose phase turns `turns` cycles across the chirps of one frame
    (a walking person over a long TDM frame)."""
    F, C, RX, S = x.shape
    n = np.arange(S)
    ph = 2 * np.pi * turns * np.arange(C) / C
    x[frame] += (amp * np.exp(1j * ph)[:, None, None] * np.exp(1j * 2 * np.pi * k * n / S)[None, None, :]
                 ).astype(np.complex64)
    return x


def test_moving_target_fires_mean_rule_not_specmed():
    x, _ = _scene()
    x = _moving(x, 10)
    assert ic.detect(x, pattern=2)["frame_hit"][10]                  # the rule as specified
    assert ic.detect(x, pattern=2, mode="specmed")["hit"].sum() == 0  # narrowband: not a burst


def test_specmed_still_fires_on_sweep():
    x, _ = _scene()
    x[7, 3, :, :] += _sweep(x.shape[-1], 30, 40, 20.0)[None, :]
    d = ic.detect(x, pattern=2, mode="specmed")
    assert d["hit"][7, 3] and d["hit"].sum() == 1


def test_collision_rate_matches_monte_carlo():
    rng = np.random.default_rng(3)
    t_s, s_v, b_if = 25.6e-6, 90e12, 10e6            # victim: config A
    s_i, rho = 30e12, 5e4                            # interferer: 30 MHz/us, 50 k chirps/s
    lam = ic.collision_rate(t_s, s_v, b_if, rho, s_i, overlap=1.0)
    tau = b_if / abs(s_v - s_i)
    span = 2e-3                                      # interferer start times around the window
    counts = []
    for _ in range(4000):
        n = rng.poisson(rho * span)
        t0 = rng.uniform(-span / 2, span / 2, n)
        f0 = -1e9                                    # every interferer chirp starts 1 GHz below
        # crossing time of f_v(t) = s_v t with f_i(t) = f0 + s_i (t - t0)
        tc = (f0 - s_i * t0) / (s_v - s_i)
        # a 5 GHz interferer sweep covers the victim's whole sampled band (kappa = 1)
        dur_ok = (tc - t0 >= 0) & (tc - t0 <= 5e9 / s_i)
        counts.append(np.sum(dur_ok & (tc >= -tau / 2) & (tc <= t_s + tau / 2)))
    assert abs(np.mean(counts) / lam - 1) < 0.1


def test_adc_duty_of_config_a():
    p = {"frameConfig": {"chirpsPerFrame": 5, "framePeriodicity_ms": 10.0}, "numAdcSamples": 256,
         "digOutSampleRate_MHz": 10.0, "idleTime_us": 10, "rampEndTime_us": 30}
    d = ic.adc_duty(p)
    assert abs(d["adc_duty"] - 5 * 25.6e-6 / 10e-3) < 1e-12
