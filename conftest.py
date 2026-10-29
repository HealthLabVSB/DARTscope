"""
Shared pytest fixtures.

tk_root - one Tk for the whole test session, withdrawn; the tests that need Tk put their
windows and frames into it. On macOS a second Tk root in the same process hangs
(2026-10-06), and on this Windows PC a new Tk interpreter now and then could not read
its own library ("Can't find a usable init.tcl") - so the Tk tests of every file share
this one instead of making their own.
"""
import pytest


@pytest.fixture(scope="session")
def tk_root():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display: {e}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture(scope="session")
def tk_pump():
    """pump(root): handle pending window events for at most 50 ms. Not root.update(): on
    macOS an animated progress bar posts events faster than they are drawn and update()
    never returns (DCA_processing._pump_events, 2026-10-06)."""
    import time
    import _tkinter

    def pump(root, budget_s: float = 0.05):
        deadline = time.monotonic() + budget_s
        while time.monotonic() < deadline:
            if not root.tk.dooneevent(_tkinter.DONT_WAIT):
                break
    return pump


@pytest.fixture(autouse=True)
def _no_real_cascade_calibration(tmp_path_factory, monkeypatch):
    """Decoding a cascade take reads the board's calibration from the user's configuration
    folder (cascade/calibration.py): no test may see or overwrite the real one."""
    monkeypatch.setenv("DARTSCOPE_CASCADE_CALIBRATION_DIR",
                       str(tmp_path_factory.mktemp("cascade_calibration")))
