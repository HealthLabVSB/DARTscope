"""
Tests for log_view - the Log tab's capture of everything the application prints.

Run with:  python -m pytest test_log_view.py -q
"""
import io

import log_view


def test_lines_are_classified():
    assert log_view.classify("[health]     ok   lanes  2") == "health"
    assert log_view.classify("[Fastcheck]   ok   clipping  no frame reaches full scale") == "health"
    assert log_view.classify("[Fastcheck]  WARN  zero-filled frames  6 of 50") == "warning"
    assert log_view.classify("[GUI] WARNING: 3 frames lost") == "warning"
    assert log_view.classify("Traceback (most recent call last):") == "error"
    assert log_view.classify("CFAR Detection error: boom") == "error"
    assert log_view.classify("RuntimeWarning: divide by zero", from_stderr=True) == "warning"
    assert log_view.classify("something on stderr", from_stderr=True) == "error"
    assert log_view.classify("[CFAR Detection] Frame 3: 2 targets detected") is None


def test_the_tee_keeps_the_console_and_feeds_the_log():
    log_view.drain()                         # start empty
    console = io.StringIO()
    tee = log_view._Tee(console, is_err=False)
    tee.write("first line\npartial")
    tee.write(" end\n")
    tee.flush()
    assert console.getvalue() == "first line\npartial end\n"
    chunks = log_view.drain()
    assert "".join(s for s, _ in chunks) == "first line\npartial end\n"
    assert all(err is False for _, err in chunks)


def test_no_console_is_fine():
    """The Windows build has no console: sys.stdout is None there."""
    log_view.drain()
    tee = log_view._Tee(None, is_err=True)
    assert tee.write("x\n") == 2
    tee.flush()
    assert log_view.drain() == [("x\n", True)]
