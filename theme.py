"""
theme.py

The look of DARTscope in one place: fonts, colours, the classic Tk widgets and the
matplotlib style. It follows Apple's Human Interface Guidelines where Tk allows it:

    - system fonts (SF on macOS, Segoe UI on Windows) and one monospaced face,
      hierarchy by weight and colour rather than by many sizes
    - Apple's semantic colours: label / secondary / tertiary text, one accent blue, and
      the increased-contrast variants of green, orange and red for coloured text so it
      stays readable on the grey window (about 4.5:1)
    - content over chrome: thin light separators instead of black boxes, quiet plots
      with light axes and grid, the data in the system palette

The widgets stay native (aqua on macOS, vista on Windows) and nothing is moved: the
layout of the tabs is untouched, only fonts, colours and backgrounds change.

On macOS aqua paints its surfaces in fixed greys. A theme derived from it
(_tint_aqua) swaps only the surface elements for ones that take a colour, so the
window gets a faint blue-grey tint (SURFACE) while buttons, checkboxes, fields and
tabs stay native. The classic tk widgets mixed in are given the same background
when they are first mapped (harmonize), or they would show as patches.

Nothing here knows about radar data; importable wherever Tk is.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform.startswith("win")


# ---------------- Display scaling (Windows) ----------------
SCALE = 1.0                 # the main display's scale once enable_dpi_awareness() ran


def px(n: float) -> int:
    """A size given in 96-dpi pixels, in the display's pixels."""
    return int(round(n * SCALE))


def enable_dpi_awareness() -> float:
    """
    Call before the Tk root is made. Windows draws a program that does not declare
    itself DPI aware at 96 dpi and stretches the picture to the display's scaling: on
    a 4K display at 125 % (and a laptop at 200 %) the text and the plots of DARTscope
    came out blurred. Declared system-DPI aware, as IDLE is, Tk draws at the scaling of
    the main display - fonts in points and matplotlib's figures grow with it; on a
    second display with another scaling Windows still stretches the window.
    Returns the scale of the main display (1.25 at 125 %), 1.0 elsewhere.
    DARTSCOPE_DPI_AWARE=0 keeps the old stretched window.
    """
    global SCALE
    import os
    if not IS_WIN or os.environ.get("DARTSCOPE_DPI_AWARE", "1") == "0":
        return 1.0
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)     # system aware
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()           # before Windows 8.1
        dpi = ctypes.windll.user32.GetDpiForSystem() if hasattr(ctypes.windll.user32, "GetDpiForSystem") else 96
        SCALE = max(1.0, dpi / 96.0)
        return SCALE
    except Exception:
        return 1.0


def work_area(root: tk.Misc) -> tuple[int, int, int, int]:
    """(x, y, width, height) of the main display without the taskbar (Windows), else the
    screen."""
    if IS_WIN:
        try:
            import ctypes
            from ctypes import wintypes
            r = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(r), 0):   # SPI_GETWORKAREA
                return r.left, r.top, r.right - r.left, r.bottom - r.top
        except Exception:
            pass
    return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()


def initial_geometry(root: tk.Misc, width: int, height: int, scale: float = 1.0) -> None:
    """Open the window at width x height (96-dpi pixels, times the display's scale) -
    within the work area: on a laptop at 200 % (1440 x 900 of those pixels) a
    1400 x 900 window with its title bar did not fit and the frame slider and status
    bar at the bottom were cut off. A window that does not fit is opened maximised."""
    x, y, aw, ah = work_area(root)
    w, h = int(width * scale), int(height * scale)
    chrome = int(40 * scale)                                     # title bar and borders
    if w > aw or h + chrome > ah:
        w, h = min(w, aw), min(h, ah - chrome)
        root.geometry(f"{w}x{h}+{x}+{y}")
        if IS_WIN:
            try:
                root.state("zoomed")
            except tk.TclError:
                pass
        return
    root.geometry(f"{w}x{h}")

# ---------------- Colours (Apple system colours, light appearance) ----------------
ACCENT = "#007AFF"          # systemBlue: selection, focus, prominent buttons
ACCENT_PRESSED = "#0062CC"
LINK = "#0068DA"            # linkColor

LABEL = "#1D1D1F"           # primary text
SECONDARY = "#6E6E73"       # secondary text: status lines, hints, units
TERTIARY = "#8E8E93"        # tertiary text: examples, disabled rows
SEPARATOR = "#D1D1D6"       # thin borders and plot axes
GRID = "#E5E5EA"            # plot grid
FILL = "#F2F2F7"            # quiet fill: code, read-only boxes

# Coloured text: the increased-contrast variants of the system colours
BLUE = "#0040DD"
GREEN = "#248A3D"
ORANGE = "#C93400"
RED = "#D70015"
INDIGO = "#3634A3"

LEVEL = {"ok": GREEN, "info": BLUE, "warning": ORANGE, "error": RED}

# Row tints for tables (the system colours at low opacity on white)
TINT_YELLOW = "#FFF5D1"
TINT_GREEN = "#E4F6E8"
TINT_GREEN_STRONG = "#C5EBCF"
TINT_RED = "#FDE6E7"
TINT_ORANGE = "#FFEEDB"
TINT_BLUE = "#E6F0FF"
HIGHLIGHT = "#FFE58A"       # search hits

# Surfaces (macOS): a faint blue-grey tint in place of aqua's flat greys. One colour for
# the window and everything in it; tabs and groups are told apart by a thin border.
SURFACE = "#E8EEF7"
SURFACE_BORDER = "#C5D1E2"

TOOLTIP_BG = "#FBFBFD"
PENDING_BG = "#FFD60A"      # systemYellow: a Run button with unapplied settings (non-macOS)
PROMINENT_BG = "#34C759"    # systemGreen: the START button (non-macOS)

# Data colours for plots: Apple's system palette in an order that keeps neighbours apart
PLOT_COLORS = ["#007AFF", "#FF9500", "#34C759", "#FF3B30", "#AF52DE",
               "#30B0C7", "#FF2D55", "#5856D6", "#A2845E", "#8E8E93"]

# ---------------- Fonts ----------------
# Sizes in the code are Windows points (Segoe UI 9 is the Windows default). Tk on
# macOS draws a point as one pixel, so the same number came out a quarter smaller
# and the 9 pt status texts were hard to read; scale them to the Mac's 13 pt body.
# Menlo is wider than Consolas, so it is scaled less (9 -> 11, the Mac's fixed font).
if IS_MAC:
    UI_FAMILY, MONO_FAMILY, _SCALE, _MONO_SCALE = ".AppleSystemUIFont", "Menlo", 4 / 3, 1.2
elif IS_WIN:
    UI_FAMILY, MONO_FAMILY, _SCALE, _MONO_SCALE = "Segoe UI", "Consolas", 1.0, 1.0
else:
    UI_FAMILY, MONO_FAMILY, _SCALE, _MONO_SCALE = "DejaVu Sans", "DejaVu Sans Mono", 1.0, 1.0


def _size(size: float, scale: float = _SCALE) -> int:
    return max(1, round(size * scale))


def font(size: float = 9, *style: str) -> tuple:
    """The interface font at <size> Windows points, e.g. font(10, "bold")."""
    return (UI_FAMILY, _size(size), *[s for s in style if s])


def mono(size: float = 9, *style: str) -> tuple:
    """The monospaced font for logs, tables and code."""
    return (MONO_FAMILY, _size(size, _MONO_SCALE), *[s for s in style if s])


# ---------------- Buttons ----------------
# A ttk button can be drawn as the blue default button on macOS, which is the
# platform's way to say "press this one"; a classic tk button there ignores its
# background colour. On Windows it is the other way round: the native ttk button
# cannot be coloured, the tk one can. So the buttons that change their emphasis are
# made by Button() and switched with set_prominent().
Button = ttk.Button if IS_MAC else tk.Button


def set_prominent(button, on: bool, colour: str = PENDING_BG) -> None:
    """Emphasise <button> (blue default button on macOS, <colour> elsewhere) or not."""
    try:
        if isinstance(button, ttk.Button):
            button.configure(default="active" if on else "normal")
        else:
            default = getattr(button, "_theme_bg", None)
            if default is None:
                default = button._theme_bg = button.cget("bg")
            button.configure(bg=colour if on else default)
    except tk.TclError:
        pass


# ---------------- Slider ----------------
class Slider(ttk.Frame):
    """
    A native ttk slider with the interface of tk.Scale, as DARTscope uses it.

    tk.Scale draws its thumb in the widget's background colour, so once that matches
    the window the thumb disappears; ttk.Scale is the platform's own slider (the round
    knob on macOS) but has no label, no value read-out, no resolution and does not
    call its command from set(). This keeps tk.Scale's behaviour on top of it:

        get()                the value rounded to the resolution (an int for 1)
        set(v)               moves the thumb and calls command(str) when the value
                             changed, after idle, as tk.Scale does
        command              called with the value as a string when it changes
        variable             kept in step both ways (writing it moves the thumb)
        label, showvalue     a small caption and read-out above the track
        config/cget          from_, to, state, command, label, length; others go to
                             the inner ttk.Scale
        bind                 goes to the track, so drags and releases arrive
    """

    _OWN = ("from_", "to", "command", "variable", "label", "showvalue", "resolution",
            "state", "length", "orient")

    def __init__(self, master, from_=0, to=100, orient="horizontal", command=None,
                 variable=None, label="", showvalue=True, resolution=1, length=None,
                 state="normal", **_ignored):
        super().__init__(master)
        self._from, self._to = float(from_), float(to)
        self._res = float(resolution) if resolution else 0.0
        self._command = command
        self._uservar = variable
        self._showvalue = bool(showvalue)
        self._value = self._snap(self._from)
        self._pending = None
        self._var = tk.DoubleVar(self, value=self._value)

        self._head = ttk.Frame(self)
        self._label = ttk.Label(self._head, text=label or "", font=font(8),
                                foreground=SECONDARY)
        self._readout = ttk.Label(self._head, font=font(8, "bold"), foreground=LABEL)
        self._label.pack(side="left")
        self._readout.pack(side="left", padx=(6, 0))
        if label or self._showvalue:
            self._head.pack(side="top", fill="x")
        kw = {"length": length} if length else {}
        self.scale = ttk.Scale(self, from_=self._from, to=self._to, orient=orient,
                               variable=self._var, command=self._on_drag, **kw)
        self.scale.pack(side="top", fill="x", expand=True)
        self._set_state(state)

        if variable is not None:
            try:
                self._value = self._snap(float(variable.get()))
            except (tk.TclError, ValueError):
                pass
            self._var.set(self._value)
            variable.trace_add("write", self._on_uservar)
        self._show()

    # --- value -----------------------------------------------------------------------
    def _snap(self, v: float) -> float:
        lo, hi = min(self._from, self._to), max(self._from, self._to)
        v = max(lo, min(hi, v))
        if self._res > 0:
            v = round(v / self._res) * self._res
        return v

    def _fmt(self, v: float) -> str:
        if self._res > 0 and float(self._res).is_integer():
            return str(int(round(v)))
        return f"{v:g}"

    def _show(self):
        self._readout.configure(text=self._fmt(self._value) if self._showvalue else "")

    def _apply(self, v: float):
        """Set the value; True when it changed."""
        v = self._snap(v)
        changed = v != self._value
        self._value = v
        if float(self._var.get()) != v:
            self._var.set(v)
        if self._uservar is not None and changed:
            try:
                cur = self._uservar.get()
            except (tk.TclError, ValueError):
                cur = None
            if cur != self.get():
                self._writing = True
                try:
                    self._uservar.set(self.get())
                finally:
                    self._writing = False
        self._show()
        return changed

    def _on_drag(self, raw):
        if self._apply(float(raw)) and self._command is not None:
            self._command(self._fmt(self._value))
        elif float(self._var.get()) != self._value:
            self._var.set(self._value)        # hold the thumb on the resolution step

    def _on_uservar(self, *_):
        if getattr(self, "_writing", False):
            return
        try:
            self._apply(float(self._uservar.get()))
        except (tk.TclError, ValueError):
            pass

    def _schedule_command(self):
        if self._command is None or self._pending is not None:
            return

        def fire():
            self._pending = None
            if self._command is not None:
                self._command(self._fmt(self._value))
        self._pending = self.after_idle(fire)

    def get(self):
        if self._res > 0 and float(self._res).is_integer():
            return int(round(self._value))
        return self._value

    def set(self, value):
        if self._apply(float(value)):
            self._schedule_command()

    # --- options ---------------------------------------------------------------------
    def _set_state(self, state):
        self._state = str(state)
        self.scale.state(["disabled"] if self._state == "disabled" else ["!disabled"])

    def configure(self, cnf=None, **kw):
        if cnf:
            kw.update(cnf)
        if not kw:
            return super().configure()
        if "from_" in kw or "to" in kw:
            self._from = float(kw.pop("from_", self._from))
            self._to = float(kw.pop("to", self._to))
            self.scale.configure(from_=self._from, to=self._to)
            if self._apply(self._value):
                self._schedule_command()
        if "state" in kw:
            self._set_state(kw.pop("state"))
        if "command" in kw:
            self._command = kw.pop("command")
        if "label" in kw:
            self._label.configure(text=kw.pop("label") or "")
        if "showvalue" in kw:
            self._showvalue = bool(kw.pop("showvalue"))
            self._show()
        if "resolution" in kw:
            self._res = float(kw.pop("resolution") or 0)
        if "length" in kw:
            self.scale.configure(length=kw.pop("length"))
        if kw:
            self.scale.configure(**kw)

    config = configure

    def cget(self, key):
        if key in ("from", "from_"):
            return self._from
        if key == "to":
            return self._to
        if key == "state":
            return self._state
        if key == "label":
            return self._label.cget("text")
        if key == "resolution":
            return self._res
        return self.scale.cget(key)

    __getitem__ = cget

    def bind(self, sequence=None, func=None, add=None):
        return self.scale.bind(sequence, func, add)


# ---------------- Tk ----------------
_BG_CLASSES = ("Frame", "Label", "Checkbutton", "Radiobutton", "Button", "Scale",
               "Labelframe", "Canvas", "Message")
_RING_CLASSES = ("Entry", "Spinbox", "Button", "Checkbutton", "Radiobutton", "Scale",
                 "Frame", "Label", "Canvas")
_DEFAULT_BGS = {"systemwindowbackgroundcolor", "systembuttonface", "#d9d9d9", "#ececec",
                "systemtransparent"}


def _depth(widget) -> int:
    """How many notebooks and labelframes <widget> sits in, within its toplevel."""
    n = 0
    w = widget.master
    while w is not None and not isinstance(w, (tk.Tk, tk.Toplevel)):
        if w.winfo_class() in ("TNotebook", "TLabelframe"):
            n += 1
        w = w.master
    return n


def level_background(widget) -> str:
    """The background the ttk widgets around <widget> are drawn with."""
    if _tinted:
        return SURFACE
    d = min(_depth(widget), 7)
    return "systemWindowBackgroundColor" + (str(d) if d else "")


_tinted = False


_AQUA_STYLES = (".", "TButton", "TLabel", "TFrame", "TLabelframe", "TLabelframe.Label",
                "TNotebook", "TNotebook.Tab", "TEntry", "TCombobox", "TSpinbox", "Treeview",
                "Heading", "Treeview.Heading", "TProgressbar", "Toolbutton", "TMenubutton",
                "TCheckbutton", "TRadiobutton", "TScale", "TScrollbar", "TSeparator")


def _tint_aqua(style: ttk.Style, root: tk.Misc) -> None:
    """
    Replace aqua's grey surfaces by SURFACE, keeping the native controls.

    aqua draws every background itself in fixed greys, ignores -background, and its
    own elements cannot be redefined. So a theme derived from it takes over aqua's
    settings and swaps only the surface elements (the widget background, frames,
    labelframes, the notebook pane) for clam's, which take colours; buttons,
    checkboxes, fields, combo boxes, sliders and the tab strip stay aqua's.
    """
    global _tinted
    if "dartscope" not in style.theme_names():
        settings = {name: (style.configure(name) or {}, style.map(name) or {})
                    for name in _AQUA_STYLES}
        style.theme_create("dartscope", parent="aqua")
        style.theme_use("dartscope")
        for el in ("background", "Frame.border", "Labelframe.border", "Notebook.client"):
            style.element_create(el, "from", "clam")
        for name, (conf, maps) in settings.items():
            if conf:
                style.configure(name, **conf)
            if maps:
                style.map(name, **maps)
    else:
        style.theme_use("dartscope")
    _tinted = True
    style.configure(".", background=SURFACE)
    # Fields keep their white box; the area around them takes the surface
    for name in ("TEntry", "TSpinbox", "TCombobox"):
        style.configure(name, background=SURFACE, fieldbackground="white")
    style.configure("Horizontal.TProgressbar", background=SURFACE)
    style.configure("TLabelframe", background=SURFACE, bordercolor=SURFACE_BORDER,
                    lightcolor=SURFACE, darkcolor=SURFACE, relief="solid", borderwidth=1)
    # aqua's pane had an inset of its own; the padding grows by it so the tabs'
    # contents stay where they were
    style.configure("TNotebook", background=SURFACE, bordercolor=SURFACE_BORDER,
                    lightcolor=SURFACE, darkcolor=SURFACE, padding=(25, 11, 25, 20))
    style.configure("Heading", background=SURFACE)
    root.configure(bg=SURFACE)


def _harmonize_one(widget) -> None:
    if getattr(widget, "_theme_done", False):
        return
    widget._theme_done = True
    cls = widget.winfo_class()
    try:
        bg = level_background(widget)
        if cls in _BG_CLASSES and str(widget.cget("bg")).lower() in _DEFAULT_BGS:
            widget.configure(bg=bg)
            if cls in ("Checkbutton", "Radiobutton", "Button", "Scale", "Label"):
                widget.configure(activebackground=bg if cls != "Scale" else ACCENT_PRESSED)
        if cls in _RING_CLASSES:
            widget.configure(highlightbackground=bg)
        if cls == "Spinbox":
            widget.configure(buttonbackground=bg)
    except tk.TclError:
        pass


def harmonize(widget) -> None:
    """Match the classic tk widgets under <widget> to the ttk background of their level."""
    if not IS_MAC:
        return
    _harmonize_one(widget)
    for child in widget.winfo_children():
        harmonize(child)


def _options(root: tk.Misc) -> None:
    """Defaults for the classic tk widgets, through the option database."""
    add = root.option_add
    # Text and lists: a thin light border, the accent ring when focused, the system
    # selection colour - instead of the black sunken box
    for cls in ("Text", "Listbox"):
        add(f"*{cls}.relief", "flat")
        add(f"*{cls}.borderWidth", 0)
        add(f"*{cls}.highlightThickness", 1)
        add(f"*{cls}.highlightBackground", SEPARATOR)
        add(f"*{cls}.highlightColor", ACCENT)
        add(f"*{cls}.foreground", LABEL)
        add(f"*{cls}.selectBackground", "#B3D7FF")
        add(f"*{cls}.selectForeground", LABEL)
        add(f"*{cls}.inactiveSelectBackground", "#DCDCE0")
    add("*Listbox.activeStyle", "none")
    add("*Listbox.font", font(9))
    add("*Text.insertBackground", ACCENT)
    # Sliders: a flat accent-coloured thumb on a light track
    add("*Scale.troughColor", SEPARATOR)
    add("*Scale.sliderRelief", "flat")
    add("*Scale.borderWidth", 0)
    add("*Scale.highlightThickness", 0)
    add("*Scale.width", 10)
    add("*Scale.sliderLength", 22)
    add("*Scale.foreground", SECONDARY)
    add("*Scale.font", font(8))
    if not IS_MAC:
        add("*Scale.background", ACCENT)
        add("*Scale.activeBackground", ACCENT_PRESSED)
    for cls in ("Entry", "Spinbox"):
        add(f"*{cls}.selectBackground", "#B3D7FF")
        add(f"*{cls}.selectForeground", LABEL)
        add(f"*{cls}.insertBackground", ACCENT)
        add(f"*{cls}.highlightColor", ACCENT)
    add("*Label.foreground", LABEL)


def _ttk_styles(root: tk.Misc) -> None:
    style = ttk.Style(root)
    if not IS_MAC and not IS_WIN and style.theme_use() in ("default", "classic"):
        style.theme_use("clam")      # Linux: the closest to a current look
    if IS_MAC and style.theme_use() == "aqua":
        _tint_aqua(style, root)
    style.configure("TLabelframe.Label", font=font(10, "bold"), foreground=LABEL)
    style.configure("ParamHeader.TLabel", font=font(10, "bold"), foreground=LABEL)
    style.configure("Secondary.TLabel", foreground=SECONDARY)
    style.configure("Treeview", rowheight=_size(18) + 4)
    style.configure("Treeview.Heading", font=font(9, "bold"))
    if not IS_MAC:
        style.configure("Horizontal.TProgressbar", troughcolor=GRID, background=ACCENT)
        style.configure("TNotebook.Tab", padding=(12, 4))
        style.map("Treeview", background=[("selected", ACCENT)],
                  foreground=[("selected", "white")])


def apply(root: tk.Misc) -> None:
    """Set the DARTscope look on <root>; call it right after creating the Tk root."""
    _options(root)
    _ttk_styles(root)
    style_matplotlib()
    if IS_MAC:
        # Widgets created later (tabs built on demand, dialogs) are matched when
        # they first appear on screen.
        for cls in _BG_CLASSES + _RING_CLASSES + ("Spinbox",):
            root.bind_class(cls, "<Map>", lambda e: _harmonize_one(e.widget), add="+")


# ---------------- matplotlib ----------------
def style_matplotlib() -> None:
    """Quiet, light plots: thin grey axes, a faint grid, the system palette for data."""
    try:
        import matplotlib as mpl
        from cycler import cycler
    except ImportError:
        return
    fonts = ["Helvetica Neue", "SF Pro Text", "Segoe UI", "Arial", "DejaVu Sans"]
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": fonts + [f for f in mpl.rcParams["font.sans-serif"] if f not in fonts],
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.titlecolor": LABEL,
        "axes.titlepad": 8,
        "axes.labelsize": 9,
        "axes.labelcolor": "#3A3A3C",
        "axes.edgecolor": "#C7C7CC",
        "axes.linewidth": 0.8,
        "axes.prop_cycle": cycler(color=PLOT_COLORS),
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "xtick.color": "#AEAEB2",
        "ytick.color": "#AEAEB2",
        "xtick.labelcolor": SECONDARY,
        "ytick.labelcolor": SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.3,
        "lines.solid_capstyle": "round",
        "patch.edgecolor": "none",
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#E5E5EA",
        "legend.fancybox": True,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "figure.titlesize": 11,
        "figure.titleweight": "semibold",
        "savefig.facecolor": "white",
        "image.cmap": "viridis",
    })
