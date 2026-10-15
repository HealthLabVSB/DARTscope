"""
frame_cache.py

The frames the views have drawn, kept as pictures. A view draws a frame of a recording
in 70-140 ms (matplotlib renders the whole figure, ~65-70 ms, and some views compute
images or rebuild their axes on top), a cascade take has one every 20 ms: the global
Play could not keep up. A frame drawn once is kept - the canvas's pixels, zlib-compressed
(5 MB -> ~0.4 MB in 11 ms) - and shown again in ~11 ms (6 ms to unpack, 5 ms to hand
it to Tk) without asking the view at all.

What a picture shows must still be what the view would draw. Every canvas drawn outside
a frame drawn through render() - a new Run, another setting, a zoom, a resized window -
forgets its pictures (install() hooks FigureCanvasTkAgg.draw), so no view has to say
what its drawing depends on. A picture shown leaves the view's figure a frame behind:
the host draws the view itself once the frames stop coming (see DCA_processing).

The pictures of all canvases together stay under BUDGET_BYTES; the oldest go first.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import zlib
from collections import OrderedDict

import numpy as np

BUDGET_BYTES = 2_000_000_000   # all pictures together (~5000 frames of a full-window view)
LEVEL = 1                      # zlib level: 3 saved 14 % and cost 20 % more time

_pictures: OrderedDict = OrderedDict()   # (id(canvas), frame) -> (bytes, shape), oldest first
_size = 0
_busy = 0                      # > 0 while render() draws: those draws are the frames' own
_installed = False


def install():
    """Make every TkAgg canvas forget its pictures when it is drawn outside render()."""
    global _installed
    if _installed:
        return
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    original = FigureCanvasTkAgg.draw

    def draw(self):
        if not _busy:
            forget(self)
        return original(self)

    FigureCanvasTkAgg.draw = draw
    _installed = True


def forget(canvas=None):
    """Drop the pictures of a canvas (of every canvas with None)."""
    global _size
    cid = None if canvas is None else id(canvas)
    for key in [k for k in _pictures if cid is None or k[0] == cid]:
        _size -= len(_pictures.pop(key)[0])


def has(canvas, frame: int) -> bool:
    return (id(canvas), int(frame)) in _pictures


def show(canvas, frame: int) -> bool:
    """Put the picture of this frame on the canvas; False when there is none (or the canvas
    has another size since)."""
    item = _pictures.get((id(canvas), int(frame)))
    if item is None:
        return False
    data, shape = item
    try:
        dst = np.asarray(canvas.get_renderer().buffer_rgba())
    except Exception:
        return False
    if dst.shape != shape:
        return False
    dst[...] = np.frombuffer(zlib.decompress(data), np.uint8).reshape(shape)
    canvas.blit()
    _pictures.move_to_end((id(canvas), int(frame)))
    return True


def render(canvas, frame: int, update) -> None:
    """Let the view draw the frame (update()), draw the canvas now and keep its picture."""
    global _busy, _size
    _busy += 1
    try:
        update()
        canvas.draw()
        pending = getattr(canvas, "_idle_draw_id", None)
        if pending:                            # the view asked for a draw later: done now
            try:
                canvas._tkcanvas.after_cancel(pending)
            except Exception:
                pass
            canvas._idle_draw_id = None
        buf = np.asarray(canvas.get_renderer().buffer_rgba())
        data = zlib.compress(buf.tobytes(), LEVEL)
    finally:
        _busy -= 1
    key = (id(canvas), int(frame))
    old = _pictures.pop(key, None)
    if old is not None:
        _size -= len(old[0])
    _pictures[key] = (data, buf.shape)
    _size += len(data)
    while _size > BUDGET_BYTES and len(_pictures) > 1:
        _size -= len(_pictures.popitem(last=False)[1][0])


def size_bytes() -> int:
    return _size


def count() -> int:
    return len(_pictures)
