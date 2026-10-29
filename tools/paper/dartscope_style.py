"""
The figures of the article in DARTscope's own look.

Every figure of the paper is drawn with the matplotlib style of the application
(theme.style_matplotlib: Apple's system palette, quiet grey axes, a faint grid, the
interface font, viridis maps) and, where a view of DARTscope shows the same thing, by
that view's own drawing code. Only what the journal requires differs from the screen:
the size of an IEEE column, fonts embedded as TrueType, and units in round brackets
("Time (s)") where the interface writes "Time [s]".

    apply()             theme.style_matplotlib() plus the print settings
    paper_units(fig)    "[s]" -> "(s)" in every label, title and legend of a figure
    BLUE, ORANGE, ...   the data colours of the interface (theme.PLOT_COLORS)

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import re

import theme

COL1, COL2 = 3.5, 7.16           # IEEE column and page width (inches)

BLUE, ORANGE, GREEN, RED, PURPLE, TEAL = theme.PLOT_COLORS[:6]
GREY = theme.PLOT_COLORS[-1]
TEXT = theme.LABEL
SECONDARY = theme.SECONDARY

_BRACKETS = re.compile(r"\[([^\[\]]+)\]")


def apply() -> None:
    """DARTscope's plot style with the settings a printed IEEE figure needs."""
    import matplotlib as mpl
    theme.style_matplotlib()
    mpl.rcParams.update({
        "axes.titlesize": 9,          # the views title their panels at 9 pt
        "axes.titlepad": 5,
        "pdf.fonttype": 42,           # TrueType in the PDF, not Type 3
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "image.interpolation": "nearest",
    })


def units(text: str) -> str:
    return _BRACKETS.sub(r"(\1)", text) if text else text


def paper_units(fig) -> None:
    """Units in round brackets everywhere in <fig>, as T-MTT asks."""
    for ax in fig.axes:
        ax.set_xlabel(units(ax.get_xlabel()))
        ax.set_ylabel(units(ax.get_ylabel()))
        for loc in ("left", "center", "right"):
            t = ax.get_title(loc=loc)
            if t:
                ax.set_title(units(t), loc=loc)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_text(units(t.get_text()))
        for t in ax.texts:
            t.set_text(units(t.get_text()))


def letter(ax, s: str, dx: float = -0.14, dy: float = 1.03) -> None:
    """Panel letter (a), (b), ... in the interface's title weight."""
    ax.text(dx, dy, f"({s})", transform=ax.transAxes, fontsize=9, fontweight="semibold",
            color=TEXT, va="bottom", ha="left")
