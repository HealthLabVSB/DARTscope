"""
frame_marker.py

The dashed line that shows, in every plot drawn over time, where the frame chosen with the
Analysis frame slider is - the whole-recording plots (Vital Signs, RTI, HR, AI, Moving
person ...) are drawn once, so without it nothing in them follows the slider.

    install()      once at start-up, before the tabs build their figures: every
                   FigureCanvasTkAgg made from then on is registered (weakly)
    set_state(fn)  fn() -> FrameState or None: the frame and how it maps onto time
    refresh(root)  draw or move the line in the plots that are on screen now

Which axes get it: those labelled "Time ..." on x, the axes sharing x with them, and
"Chirp Index" axes (RTI: one column per chirp). Time axes are drawn either from the
start of the analysed segment or in recording time (segment start + frame time); which
one an axes uses is read off its data (its data starting before the segment start means
it counts from 0). Axes that draw their own frame indicator (a line with the gid below)
are left alone, as are axes that draw nothing.

After a plot is recomputed (its figure cleared and drawn again) the line is put back on
the next draw. Only plots of the visible tab are touched: moving the line redraws the
figure, and the hidden ones get theirs when their tab is shown.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass

GID = "frame-marker"
STYLE = dict(color="black", linestyle="--", linewidth=1.0, alpha=0.75, zorder=20)

_canvases: "weakref.WeakSet" = weakref.WeakSet()
_state_fn = None
_pending: "weakref.WeakSet" = weakref.WeakSet()    # canvases with a put-back queued
_installed = False


@dataclass
class FrameState:
    frame: int                 # index into the analysed segment
    frames: int                # frames in the segment
    period_s: float            # frame period
    t0_s: float = 0.0          # recording time of the segment's first frame


def install():
    """Register every Tk figure canvas created from now on."""
    global _installed
    if _installed:
        return
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    original = FigureCanvasTkAgg.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _canvases.add(self)
        try:
            self.mpl_connect("draw_event", _on_draw)
        except Exception:
            pass

    FigureCanvasTkAgg.__init__ = __init__
    _installed = True


def canvases() -> list:
    """Every Tk figure canvas made since install() that still exists."""
    return list(_canvases)


def set_state(fn):
    global _state_fn
    _state_fn = fn


def _state() -> FrameState | None:
    try:
        return _state_fn() if _state_fn is not None else None
    except Exception:
        return None


def _axis_kind(ax) -> str:
    label = (ax.get_xlabel() or "").strip().lower()
    if label.startswith("time"):
        return "time"
    if label.startswith("chirp index"):
        return "chirp"
    return ""


def _marked_axes(fig) -> list[tuple[object, str]]:
    """(axes, kind) of a figure that should show the frame."""
    out = []
    axes = [a for a in fig.axes if a.get_visible() and getattr(a, "name", "") == "rectilinear"]
    kinds = {a: _axis_kind(a) for a in axes}
    for a in axes:
        kind = kinds[a]
        if not kind:
            # an axes without a label of its own, sharing x with a time axes (stacked plots)
            for b in axes:
                if kinds[b] and b is not a and a.get_shared_x_axes().joined(a, b):
                    kind = kinds[b]
                    break
        if kind:
            out.append((a, kind))
    return out


def _has_data(ax) -> bool:
    return bool(ax.lines or ax.images or ax.collections or ax.patches) and \
        ax.dataLim.width > 0 and ax.dataLim.x0 == ax.dataLim.x0


def _x_for(ax, kind: str, st: FrameState) -> float | None:
    lim = ax.dataLim
    if kind == "chirp":
        if st.frames <= 0:
            return None
        per_frame = (lim.x1 - lim.x0 + 1) / st.frames
        return lim.x0 + st.frame * per_frame
    t_rel = st.frame * st.period_s
    if st.t0_s and lim.x0 < st.t0_s - max(st.period_s, 1e-3):
        return t_rel                      # this plot counts from the segment start
    return st.t0_s + t_rel


def _own_line(ax):
    for ln in ax.lines:
        if ln.get_gid() == GID:
            return ln
    return None


def _foreign_indicator(ax) -> bool:
    """A line with the gid that this module did not draw: the view has its own."""
    return any(ln.get_gid() == GID and not getattr(ln, "_frame_marker", False) for ln in ax.lines)


def mark_figure(fig, st: FrameState) -> bool:
    """Put or move the line in the figure's time axes; True when something changed."""
    changed = False
    for ax, kind in _marked_axes(fig):
        line = _own_line(ax)
        if line is not None and not getattr(line, "_frame_marker", False):
            continue                                   # the view's own indicator
        if not _has_data(ax) or (line is None and _foreign_indicator(ax)):
            continue
        x = _x_for(ax, kind, st)
        if x is None:
            continue
        if line is None:
            xlim = ax.get_xlim()
            autoscale = ax.get_autoscalex_on()
            line = ax.axvline(x, **STYLE)
            line.set_gid(GID)
            line._frame_marker = True
            ax.set_xlim(xlim)                          # the line must not widen the view
            ax.set_autoscalex_on(autoscale)
            changed = True
        elif line.get_xdata()[0] != x:
            line.set_xdata([x, x])
            changed = True
    return changed


def _visible(canvas) -> bool:
    try:
        w = canvas.get_tk_widget()
        return bool(w.winfo_exists() and w.winfo_ismapped() and w.winfo_viewable())
    except Exception:
        return False


def refresh(root=None):
    """Move (or draw) the line in every plot on screen."""
    st = _state()
    if st is None:
        return
    for canvas in list(_canvases):
        if not _visible(canvas):
            continue
        try:
            if mark_figure(canvas.figure, st):
                canvas.draw_idle()
        except Exception:
            pass


def _on_draw(event):
    """After a redraw: a cleared figure has lost its line - put it back once."""
    canvas = event.canvas
    if canvas in _pending:
        return
    st = _state()
    if st is None or not _visible(canvas):
        return
    need = any(_own_line(ax) is None and not _foreign_indicator(ax) and _has_data(ax)
               for ax, _ in _marked_axes(canvas.figure))
    if not need:
        return
    _pending.add(canvas)
    try:
        canvas.get_tk_widget().after_idle(_put_back, canvas)
    except Exception:
        _pending.discard(canvas)


def _put_back(canvas):
    _pending.discard(canvas)
    st = _state()
    if st is None:
        return
    try:
        if mark_figure(canvas.figure, st):
            canvas.draw_idle()
    except Exception:
        pass
