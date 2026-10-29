"""
Tests for frame_cache: a frame drawn once is shown again from its picture, and a canvas
drawn any other way forgets its pictures.

Run with:  python -m pytest test_frame_cache.py -q
"""
import numpy as np
import pytest

import frame_cache


@pytest.fixture
def canvas(tk_root):
    import tkinter as tk
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    frame_cache.install()
    top = tk.Toplevel(tk_root)
    fig = Figure(figsize=(4, 3), dpi=50)
    c = FigureCanvasTkAgg(fig, master=top)
    c.get_tk_widget().pack()
    c.draw()
    yield c, fig.add_subplot(111)
    frame_cache.forget()
    top.destroy()


def test_a_frame_comes_back_from_its_picture(canvas):
    c, ax = canvas
    line, = ax.plot([0, 1], [0, 1])
    pixels = {}
    for f in (3, 7):
        def update(f=f):
            line.set_ydata([0, f])
            ax.set_ylim(0, 8)
            c.draw_idle()
        frame_cache.render(c, f, update)
        pixels[f] = np.asarray(c.get_renderer().buffer_rgba()).copy()
    assert frame_cache.has(c, 3) and frame_cache.has(c, 7) and not frame_cache.has(c, 5)
    assert getattr(c, "_idle_draw_id", None) is None             # drawn now, not again later
    assert frame_cache.show(c, 3)
    assert np.array_equal(np.asarray(c.get_renderer().buffer_rgba()), pixels[3])
    assert not frame_cache.show(c, 5)


def test_any_other_drawing_forgets_the_pictures(canvas):
    c, ax = canvas
    frame_cache.render(c, 1, lambda: None)
    assert frame_cache.has(c, 1)
    ax.set_xlim(0, 2)                                            # a zoom, another setting ...
    c.draw()
    assert not frame_cache.has(c, 1) and frame_cache.size_bytes() == 0


def test_the_oldest_pictures_go_first(canvas, monkeypatch):
    c, ax = canvas
    frame_cache.render(c, 0, lambda: None)
    one = frame_cache.size_bytes()
    monkeypatch.setattr(frame_cache, "BUDGET_BYTES", int(2.5 * one))
    for f in (1, 2, 3):
        frame_cache.render(c, f, lambda: None)
    assert not frame_cache.has(c, 0) and not frame_cache.has(c, 1) and frame_cache.has(c, 3)
