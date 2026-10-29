"""
Tests for frame_marker: the dashed frame line in the plots drawn over time.

Run with:  python -m pytest test_frame_marker.py -q
"""
import pytest

tk = pytest.importorskip("tkinter")

import frame_marker as fm


@pytest.fixture
def root(tk_root):
    """A window of its own for every test, in the session's one Tk (conftest.py): a new Tk
    for every test now and then could not read its own library ("Can't find a usable
    init.tcl") and the test was skipped; on macOS a second Tk root hangs."""
    w = tk.Toplevel(tk_root)
    w.geometry("400x300")
    fm.set_state(lambda: None)                        # nothing left over from another test
    yield w
    w.destroy()


def _canvas(root, build):
    fm.install()
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(4, 3))
    build(fig)
    c = FigureCanvasTkAgg(fig, master=root)
    c.get_tk_widget().pack(fill="both", expand=True)
    root.update()
    c.draw()
    return c


def _line(ax):
    return next((ln for ln in ax.lines if ln.get_gid() == fm.GID), None)


def test_time_axes_get_the_line_at_the_frame_time(root):
    def build(fig):
        a = fig.add_subplot(211)
        a.plot([0, 10], [0, 1])
        a.set_xlabel("Time [s]")
        b = fig.add_subplot(212)                      # no time axis: left alone
        b.plot([1, 2], [3, 4])
        b.set_xlabel("Frequency [Hz]")
    c = _canvas(root, build)
    fm.set_state(lambda: fm.FrameState(frame=40, frames=200, period_s=0.05))
    fm.refresh()
    a, b = c.figure.axes
    assert _line(a).get_xdata()[0] == pytest.approx(2.0)
    assert _line(b) is None
    xlim = a.get_xlim()
    fm.set_state(lambda: fm.FrameState(frame=100, frames=200, period_s=0.05))
    fm.refresh()
    assert _line(a).get_xdata()[0] == pytest.approx(5.0)
    assert a.get_xlim() == xlim                       # the line does not widen the view


def test_recording_time_or_from_the_segment_start(root):
    def build(fig):
        a = fig.add_subplot(211)
        a.plot([20, 30], [0, 1])                      # recording time: the segment from 20 s
        a.set_xlabel("Time [s]")
        b = fig.add_subplot(212)
        b.plot([0, 10], [0, 1])                       # counts from the segment start
        b.set_xlabel("Time [s]")
    c = _canvas(root, build)
    fm.set_state(lambda: fm.FrameState(frame=20, frames=200, period_s=0.05, t0_s=20.0))
    fm.refresh()
    a, b = c.figure.axes
    assert _line(a).get_xdata()[0] == pytest.approx(21.0)
    assert _line(b).get_xdata()[0] == pytest.approx(1.0)


def test_shared_axes_and_chirp_axes(root):
    def build(fig):
        top = fig.add_subplot(311)
        top.plot([0, 10], [0, 1])                     # no label, shares x with the one below
        low = fig.add_subplot(312, sharex=top)
        low.plot([0, 10], [1, 0])
        low.set_xlabel("Time [s]")
        rti = fig.add_subplot(313)
        rti.imshow([[0, 1], [1, 0]], extent=[0, 1999, 0, 5], aspect="auto")
        rti.set_xlabel("Chirp Index")
    c = _canvas(root, build)
    fm.set_state(lambda: fm.FrameState(frame=50, frames=200, period_s=0.05))
    fm.refresh()
    top, low, rti = c.figure.axes
    assert _line(top).get_xdata()[0] == pytest.approx(2.5)
    assert _line(low).get_xdata()[0] == pytest.approx(2.5)
    assert _line(rti).get_xdata()[0] == pytest.approx(500.0)   # 10 chirps a frame


def test_a_view_with_its_own_indicator_gets_no_second_line(root):
    def build(fig):
        a = fig.add_subplot(111)
        a.plot([0, 10], [0, 1])
        a.set_xlabel("Time [s]")
        own = a.axvline(3.0, color="red")
        own.set_gid(fm.GID)                           # Micro-Doppler marks its own this way
    c = _canvas(root, build)
    fm.set_state(lambda: fm.FrameState(frame=100, frames=200, period_s=0.05))
    fm.refresh()
    ax = c.figure.axes[0]
    marked = [ln for ln in ax.lines if ln.get_gid() == fm.GID]
    assert len(marked) == 1 and marked[0].get_xdata()[0] == 3.0


def test_a_recomputed_plot_gets_the_line_back(root):
    def build(fig):
        a = fig.add_subplot(111)
        a.plot([0, 10], [0, 1])
        a.set_xlabel("Time [s]")
    c = _canvas(root, build)
    fm.set_state(lambda: fm.FrameState(frame=20, frames=200, period_s=0.05))
    fm.refresh()
    fig = c.figure
    fig.clear()                                       # the view computed again
    a = fig.add_subplot(111)
    a.plot([0, 10], [1, 0])
    a.set_xlabel("Time [s]")
    c.draw()
    for _ in range(5):
        root.update()
    assert _line(fig.axes[0]).get_xdata()[0] == pytest.approx(1.0)


def test_nothing_loaded_draws_nothing(root):
    def build(fig):
        a = fig.add_subplot(111)
        a.plot([0, 10], [0, 1])
        a.set_xlabel("Time [s]")
    c = _canvas(root, build)
    fm.set_state(lambda: None)
    fm.refresh()
    assert _line(c.figure.axes[0]) is None
