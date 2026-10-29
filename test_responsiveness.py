"""
Tests for what keeps the application responsive with long recordings: a wide image drawn
averaged to the screen and finer when zoomed (plot_interactivity.ColumnPooledImage), its
display limits from a subsample, and the range FFT kept in single precision.

Run with:  python -m pytest test_responsiveness.py -q
"""
import gc
import time

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from plot_interactivity import ColumnPooledImage, image_percentiles


def _figure():
    fig = Figure(figsize=(8, 3), dpi=100)
    FigureCanvasAgg(fig)
    return fig, fig.add_subplot()


def test_a_wide_image_is_averaged_and_refined_when_zoomed():
    fig, ax = _figure()
    rng = np.random.default_rng(0)
    data = rng.standard_normal((8, 10_000)).astype(np.float32)
    pooled = ColumnPooledImage(ax, data, [0, 10_000, 0, 8], max_cols=100, aspect="auto", origin="lower")
    im = pooled.im
    assert im.get_array().shape == (8, 100)
    assert np.allclose(im.get_array()[:, 3], data[:, 300:400].mean(axis=1))
    assert ax.get_xlim() == (0.0, 10_000.0)
    del pooled
    gc.collect()                                     # kept alive by the image it draws
    ax.set_xlim(5000, 5050)                          # zoomed in: the columns themselves
    assert np.array_equal(im.get_array(), data[:, 5000:5050])
    assert im.get_extent() == [5000.0, 5050.0, 0.0, 8.0]
    assert ax.get_xlim() == (5000.0, 5050.0) and ax.get_ylim() == (0.0, 8.0)
    ax.set_xlim(4000, 6000)                          # 2000 columns: groups of 20
    assert im.get_array().shape == (8, 100)
    assert np.allclose(im.get_array()[:, 0], data[:, 4000:4020].mean(axis=1))
    ax.set_xlim(-500, 12_000)                        # beyond the data: all of it again
    assert im.get_array().shape == (8, 100) and im.get_extent()[:2] == [0.0, 10_000.0]
    fig.canvas.draw()


def test_a_narrow_image_is_drawn_as_it_is():
    fig, ax = _figure()
    data = np.arange(24, dtype=float).reshape(4, 6)
    im = ColumnPooledImage(ax, data, [0, 6, 0, 4]).im
    assert np.array_equal(im.get_array(), data)


def test_an_rti_sized_image_redraws_fast():
    """180 000 chirps x 256 range bins (3 min at 100 fps, 10 chirps a frame): a redraw
    took 3.4 s drawn whole."""
    fig, ax = _figure()
    data = np.random.default_rng(1).standard_normal((256, 180_000)).astype(np.float32)
    pooled = ColumnPooledImage(ax, data, [0, 180_000, 0, 9], aspect="auto", origin="lower")
    fig.colorbar(pooled.im, ax=ax)
    fig.canvas.draw()
    t = time.perf_counter()
    fig.set_size_inches(7.5, 2.8)
    fig.canvas.draw()
    assert time.perf_counter() - t < 1.0


def test_display_limits_from_a_subsample():
    img = np.random.default_rng(2).standard_normal((256, 50_000)).astype(np.float32)
    assert image_percentiles(img, [5, 99]) == pytest.approx(np.percentile(img, [5, 99]), abs=0.02)
    small = np.arange(100.0)
    assert np.array_equal(image_percentiles(small, [0, 100]), [0.0, 99.0])
    # every 8th of 256 range bins strong: a stride of 24 saw only those and put the median
    # of a Range-Doppler buffer at the targets' level
    rd = np.random.default_rng(5).standard_normal((2000, 10, 256)).astype(np.float32)
    rd[..., ::8] += 30.0
    assert image_percentiles(rd, [50, 99.5], max_values=200_000) == pytest.approx(np.percentile(rd, [50, 99.5]), abs=0.1)


def test_the_range_fft_stays_in_single_precision():
    """A float64 window turned the complex64 cube into complex128: twice the memory and
    a range FFT 2.8x slower. The spectrum is the same."""
    from radar_processing import apply_window, compute_range_fft_cube
    rng = np.random.default_rng(3)
    cube = (rng.standard_normal((20, 4, 4, 256)) + 1j * rng.standard_normal((20, 4, 4, 256))).astype(np.complex64)
    assert apply_window(cube, axis=-1).dtype == np.complex64
    assert apply_window(cube.real, axis=-1).dtype == np.float32
    assert apply_window(cube.astype(np.complex128), axis=-1).dtype == np.complex128
    spec, _ = compute_range_fft_cube(cube, {}, return_complex=True)
    ref = np.fft.fft((cube - cube.mean(axis=-1, keepdims=True)).astype(np.complex128) * np.hanning(256), axis=-1)
    assert spec.dtype == np.complex64
    assert np.max(np.abs(spec - ref)) < 1e-5 * np.max(np.abs(ref))


def test_a_window_drag_redraws_a_plot_once_it_settles(tk_root, tk_pump):
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import plot_interactivity
    plot_interactivity.install_settled_resize()      # what DCA_processing does at start

    def pump(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            tk_pump(top, 0.01)
            time.sleep(0.005)

    top = tk.Toplevel(tk_root)
    try:
        fig = Figure(figsize=(4, 3), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        fig.add_subplot().plot([0, 1], [0, 1])
        draws = []
        canvas.mpl_connect("draw_event", lambda e: draws.append(fig.get_size_inches().copy()))
        top.geometry("400x300")
        pump(0.6)
        draws.clear()
        for i in range(8):                           # a drag: a new size every 30 ms
            top.geometry(f"{410 + 10 * i}x{305 + 5 * i}")
            pump(0.03)
        assert len(draws) <= 1                       # not redrawn at every step (a window
        pump(0.8)                                    # manager may add a late Configure)
        assert 1 <= len(draws) <= 2
        w = canvas.get_tk_widget()                   # the last draw: the size it ended with
        assert np.allclose(draws[-1] * fig.dpi, [w.winfo_width(), w.winfo_height()], atol=2)
    finally:
        top.destroy()


def test_rows_of_subplots():
    from plot_interactivity import figure_rows
    fig = Figure()
    gs = fig.add_gridspec(6, 2)
    ax = fig.add_subplot(gs[0, :])
    for r in range(1, 6):
        fig.add_subplot(gs[r, 0])
        fig.add_subplot(gs[r, 1])
    fig.colorbar(ax.imshow(np.zeros((2, 2))), ax=ax)
    assert figure_rows(fig) == 6
    assert figure_rows(Figure()) == 1
    one = Figure()
    one.add_subplot()
    assert figure_rows(one) == 1


def test_a_plot_of_many_rows_keeps_its_minimum(tk_root):
    """Six rows of subplots squeezed to the 160 px of a one-row plot drew six flat lines."""
    import tkinter as tk
    from flow_layout import cap_plot_heights
    top = tk.Toplevel(tk_root)
    try:
        one = tk.Canvas(top, height=1200, highlightthickness=0)
        many = tk.Canvas(top, height=1200, highlightthickness=0)
        small_many = tk.Canvas(top, height=300, highlightthickness=0)
        for c in (one, many, small_many):
            c.pack()
        mins = {str(many): 600, str(small_many): 600}
        assert cap_plot_heights(top, 160, min_for=lambda c: mins.get(str(c)))
        assert [c.winfo_reqheight() for c in (one, many, small_many)] == [160, 600, 600]
        assert not cap_plot_heights(top, 160, min_for=lambda c: mins.get(str(c)))   # settled
    finally:
        top.destroy()


def test_a_window_too_big_for_the_screen_opens_within_it(tk_root, monkeypatch):
    import tkinter as tk
    import theme
    top = tk.Toplevel(tk_root)
    try:
        monkeypatch.setattr(theme, "work_area", lambda r: (0, 0, 1440, 852))     # a laptop at 200 %
        geoms = []
        monkeypatch.setattr(top, "geometry", lambda g=None: geoms.append(g))
        monkeypatch.setattr(top, "state", lambda s=None: geoms.append(s))
        theme.initial_geometry(top, 1400, 900)
        assert geoms[0] == "1400x812+0+0"
        assert geoms[1:] == (["zoomed"] if theme.IS_WIN else [])
        geoms.clear()
        monkeypatch.setattr(theme, "work_area", lambda r: (0, 0, 3840, 2100))    # 4K at 125 %
        theme.initial_geometry(top, 1400, 900, 1.25)
        assert geoms == ["1750x1125"]
    finally:
        top.destroy()


def test_the_imaging_history_ring_reads_like_the_deque_it_replaced():
    from collections import deque
    from radar_imaging import _History
    rng = np.random.default_rng(4)
    h, d = _History(7, 3, 100.0), deque(maxlen=7)
    for step in range(40):
        if step == 23:
            h.clear()
            d.clear()
        x = (rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))).astype(np.complex64)
        h.append(x)
        d.append(x)
        h.env.append(np.abs(x).sum(axis=0))
        assert len(h) == len(d)
        assert np.array_equal(h.array(), np.asarray(d))
        assert h.env.array().shape == (min(len(h.env), 7), 5)
