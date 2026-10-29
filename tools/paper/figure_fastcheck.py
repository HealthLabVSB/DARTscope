"""
The integrity figure of the article, drawn by the Fastcheck tab itself.

    python -m tools.paper.figure_fastcheck FIG_DIR [--stem test_D] [--folder DIR]

The recording is decoded and checked exactly as the Fastcheck tab does it
(radar_processing.load_raw_cube, fastcheck.run), and the four panels are drawn by the
tab's own methods (FastcheckTab._redraw and the _plot_* it calls) on a figure of the
size of an IEEE page, without a Tk window. Only the units are put in round brackets.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fastcheck
import fastcheck_gui
from capture_params import derive_fps
from radar_processing import derive_capture_dimensions, load_raw_cube, range_fft_size
from tools.captures import find_captures
from tools.paper import dartscope_style as ds

VITALS_DIR = os.environ.get("DARTSCOPE_PAPER_DATA", "data")


class _Value:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v


def _tab_view(fig, res):
    """The parts of a FastcheckTab its drawing uses, without the Tk widgets."""
    tab = fastcheck_gui.FastcheckTab.__dict__
    view = type("FastcheckView", (), {name: tab[name] for name in
                                      ("_redraw", "_spans", "_plot_rti", "_plot_integrity",
                                       "_plot_range_energy", "_plot_time_energy", "_not_judged")})()
    view.fig = fig
    view.result = res
    view.motion_var = _Value(False)
    view.cmap = lambda default="viridis": default
    view.canvas = SimpleNamespace(draw_idle=lambda: None)
    return view


def check(stem: str, folder: str = VITALS_DIR):
    cap = next(c for c in find_captures(folder) if c.stem == stem)
    params = dict(cap.params)
    with contextlib.redirect_stdout(io.StringIO()):
        cube = load_raw_cube(cap.bin_paths[0], params, derive_capture_dimensions(dict(params)))
        fps = derive_fps(params)
        res = fastcheck.run(stem, params, cube, fps=fps, bin_paths=cap.bin_paths,
                            n_fft_range=range_fft_size(int(params.get("numAdcSamples") or cube.shape[-1])))
    return res


def figure(res, out: str, width: float = ds.COL2, height: float = 4.6) -> dict:
    ds.apply()
    fig = plt.figure(figsize=(width, height))
    view = _tab_view(fig, res)
    view._redraw()
    # the tab's margins are for a 10 x 6 in window; on a printed page the labels need room
    fig.axes[0].get_gridspec().update(left=0.075, right=0.915, top=0.94, bottom=0.09,
                                      hspace=0.42, wspace=0.34)
    ds.paper_units(fig)
    # the two stacked axes at the bottom right are short on paper: their labels on two lines
    for ax in fig.axes:
        lab = ax.get_ylabel()
        if lab in ("Breathing (/min)", "Over noise (dB)"):
            ax.set_ylabel(lab.replace(" (", "\n("), fontsize=8)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=200)
    plt.close(fig)
    return dict(stem=res.stem, verdict=res.verdict(), numbers=res.key_numbers(),
                damaged=int(res.damaged.sum()), frames=int(res.damaged.size))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fig_dir")
    ap.add_argument("--stem", default="test_D")
    ap.add_argument("--folder", default=VITALS_DIR)
    a = ap.parse_args(argv)
    res = check(a.stem, a.folder)
    info = figure(res, os.path.join(a.fig_dir, "fig_fastcheck"))
    for k, v in info.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
