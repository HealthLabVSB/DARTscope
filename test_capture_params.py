"""
Tests for capture_params - the capture-derived quantities every view has to agree on.

Run with:  python -m pytest test_capture_params.py -q
"""
import ast
import os
import subprocess
import sys

import pytest

import capture_params as cp

HERE = os.path.dirname(os.path.abspath(__file__))


def test_importing_it_does_not_start_a_gui():
    """
    The whole point of the module. DCA_processing builds its Tk root at import, so
    anything that lived there was unreachable from a script, a tool or a test.
    """
    code = "import capture_params, sys; print('tkinter' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=HERE, capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_the_gui_uses_this_rule_rather_than_a_copy_of_it():
    """A second copy of the frame-rate rule is how the tabs came to disagree before."""
    with open(os.path.join(HERE, "DCA_processing.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("_derive_period_ms", "_tx_selection_options", "_selected_chirp_positions"):
        assert name not in defined, f"DCA_processing defines its own {name} again"
    imported = {a.name for n in tree.body if isinstance(n, ast.ImportFrom)
                and n.module == "capture_params" for a in n.names}
    assert {"_derive_period_ms", "_tx_selection_options"} <= imported


# --- frame rate --------------------------------------------------------------------

@pytest.mark.parametrize("params, fps", [
    ({"frameConfig": {"framePeriodicity_ms": 10.0}}, 100.0),
    ({"frameConfig": {"framePeriodicity_ms": 40.0}}, 25.0),
    ({"frameConfig": {"framePeriodicity_ticks": 2_000_000}}, 100.0),     # 5 ns ticks
    ({"framePeriodicity_ms_selected": 50.0}, 20.0),
])
def test_frame_rate_from_the_log(params, fps):
    assert cp.derive_fps(params) == pytest.approx(fps)


def test_a_unit_slip_in_the_selected_period_is_caught():
    """2000 'ms' next to 2 000 000 ticks is 10 ms written in the wrong unit."""
    p = {"framePeriodicity_ms_selected": 2000.0,
         "frameConfig": {"framePeriodicity_ticks": 2_000_000}}
    assert cp.derive_period_ms(p) == pytest.approx(10.0)
    assert cp.derive_fps(p) == pytest.approx(100.0)


def test_a_genuinely_long_frame_is_left_alone():
    p = {"framePeriodicity_ms_selected": 200.0,
         "frameConfig": {"framePeriodicity_ticks": 40_000_000}}     # also 200 ms
    assert cp.derive_period_ms(p) == pytest.approx(200.0)


@pytest.mark.parametrize("params", [{}, {"frameConfig": None},
                                    {"frameConfig": {"framePeriodicity_ms": 0}}])
def test_the_fallback_is_reported_not_silent(params):
    seen = []
    fps = cp.derive_fps(params, "unit test", on_fallback=seen.append)
    assert fps == cp.DEFAULT_FPS_FALLBACK
    assert len(seen) == 1 and "unit test" in seen[0] and "assuming" in seen[0]


def test_a_failing_fallback_callback_does_not_break_the_answer():
    def boom(_msg):
        raise RuntimeError("status bar gone")
    assert cp.derive_fps({}, on_fallback=boom) == cp.DEFAULT_FPS_FALLBACK


def test_no_callback_on_the_normal_path():
    seen = []
    cp.derive_fps({"frameConfig": {"framePeriodicity_ms": 10.0}}, on_fallback=seen.append)
    assert seen == []


# --- which chirps belong to which transmitter --------------------------------------

def test_tx_choices_follow_the_log_order():
    p = {"tx_order_names": ["TX1", "TX3", "TX1", "TX3"]}
    opts, idx = cp.tx_selection_options(p)
    assert opts == ["All", "TX:TX1", "TX:TX3"] and idx == 0


def test_tx_choices_fall_back_to_the_chirp_list():
    p = {"chirps": [{"txActive": "TX2"}, {"txActive": None}, {"txActive": "TX2"}]}
    assert cp.tx_selection_options(p)[0] == ["All", "TX:TX2"]


def test_chirp_positions_for_one_transmitter():
    p = {"tx_order_names": ["TX1", "TX3", "TX1", "TX3"]}
    assert cp.selected_chirp_positions(p, "TX:TX3") == {"positions": [1, 3], "pattern_len": 4}
    assert cp.selected_chirp_positions(p, "All") is None
    assert cp.selected_chirp_positions(p, "TX:TX9") is None
    assert cp.selected_chirp_positions(p, "nonsense") is None


def test_old_names_still_resolve():
    assert cp._derive_period_ms is cp.derive_period_ms
    assert cp._tx_selection_options is cp.tx_selection_options
    assert cp._selected_chirp_positions is cp.selected_chirp_positions
