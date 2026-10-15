"""
vital_bands.py

The breathing and heart-rate bands every vital-signs view searches (roadmap 9.1).

The Vital Signs tab searched 48-120 bpm for the heart (42-102 with "Resting"), the AI
Vital and HR tabs 48-180 bpm; breathing was 5-40 /min in one and 6-36 /min in the
others. One recording therefore gave different numbers on different tabs, and a heart
above 120 bpm could not be found on the Vital Signs tab at all. All three read the
bands from here now, and the GUI edits one shared set.

The bands are only where the peak is looked for; they cannot tell a heart from a
breathing harmonic inside them. Measured in the Vital Signs pipeline against the somno
Pulse on six recordings (Pulse 66-102 bpm), mean absolute error of the heart rate over
all windows / over the confident ones:

    heart band     MAE all   MAE confident
    42-102 bpm      15.7        17.8        the old "Resting"
    45-150          12.9        12.3
    48-120          11.6        11.8        the old Vital Signs default
    48-180          11.2        11.0        the old AI Vital / HR default
    50-110          10.3        12.4
    50-150          10.2         8.6        the new default
    54-150           9.2         7.7

The LOW edge decides it: every bpm below ~54 is room for the 3rd and 4th harmonic of
breathing, and the narrow tops did not help. 54 measured best only because nobody here
was below 66 bpm; it would cut off an athlete or a sleeper, hence 50. For a slow heart
use the preset that goes lower and expect more harmonic picks.

tools/vital_range_scan.py keeps its own bands: they score how much a range bin looks
like a breathing person, and were tuned for that.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

from dataclasses import dataclass

MAX_BPM = 300.0            # sanity limit for either band


@dataclass(frozen=True)
class Bands:
    resp_bpm: tuple            # (low, high) breaths per minute
    heart_bpm: tuple           # (low, high) beats per minute

    @property
    def resp_hz(self) -> tuple:
        return (self.resp_bpm[0] / 60.0, self.resp_bpm[1] / 60.0)

    @property
    def heart_hz(self) -> tuple:
        return (self.heart_bpm[0] / 60.0, self.heart_bpm[1] / 60.0)

    def describe(self) -> str:
        return (f"breathing {self.resp_bpm[0]:g}-{self.resp_bpm[1]:g} /min, "
                f"heart {self.heart_bpm[0]:g}-{self.heart_bpm[1]:g} bpm")


PRESETS = {
    "Adult": Bands((5.0, 40.0), (50.0, 150.0)),
    "Slow heart (athlete, sleep)": Bands((5.0, 40.0), (40.0, 120.0)),
    "Children, exercise": Bands((10.0, 60.0), (60.0, 200.0)),
}
DEFAULT_PRESET = "Adult"
CUSTOM = "Custom"

_current = PRESETS[DEFAULT_PRESET]
_listeners = []


def _check(pair, what: str) -> tuple:
    lo, hi = (float(v) for v in pair)
    if not (0.0 < lo < hi <= MAX_BPM):
        raise ValueError(f"{what} band {lo:g}-{hi:g} must satisfy 0 < low < high <= {MAX_BPM:g}")
    return (lo, hi)


def current() -> Bands:
    return _current


def set_bands(resp_bpm=None, heart_bpm=None) -> Bands:
    """Change one or both bands for every view; returns the new set."""
    global _current
    new = Bands(_check(resp_bpm, "breathing") if resp_bpm is not None else _current.resp_bpm,
                _check(heart_bpm, "heart") if heart_bpm is not None else _current.heart_bpm)
    if new != _current:
        _current = new
        for fn in list(_listeners):
            try:
                fn(new)
            except Exception as e:                    # a broken view must not block the rest
                print(f"[vital_bands] listener failed: {e}")
    return _current


def set_preset(name: str) -> Bands:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {list(PRESETS)}")
    b = PRESETS[name]
    return set_bands(b.resp_bpm, b.heart_bpm)


def preset_of(bands: Bands | None = None) -> str:
    b = bands or _current
    for name, p in PRESETS.items():
        if p == b:
            return name
    return CUSTOM


def add_listener(fn) -> None:
    """fn(bands) is called whenever the bands change."""
    if fn not in _listeners:
        _listeners.append(fn)


def remove_listener(fn) -> None:
    if fn in _listeners:
        _listeners.remove(fn)


def from_config(config: dict | None) -> Bands:
    """
    The bands a pipeline should use for this run.

    config['resp_bpm'] / config['heart_bpm'] win when given, otherwise the shared
    bands.
    """
    c = config or {}
    b = _current
    resp = _check(c["resp_bpm"], "breathing") if c.get("resp_bpm") is not None else b.resp_bpm
    heart = _check(c["heart_bpm"], "heart") if c.get("heart_bpm") is not None else b.heart_bpm
    return Bands(resp, heart)


# --- Tk controls shared by every vital-signs tab ------------------------------------

_tk_vars = None


def tk_controls(parent, on_change=None, vertical: bool = False):
    """
    A preset box and the four limits, bound to ONE set of Tk variables, so every tab
    that shows them shows the same values. Edits take effect on Enter or on leaving
    the field. An invalid pair (low >= high) is left as typed and not applied - moving
    a band from 48-120 to 130-200 passes through 130-120 - so the views keep the last
    valid bands until it is fixed. on_change(bands) is registered as a listener.
    vertical=True stacks it for a narrow side panel. Returns the frame - or, when
    `parent` is a wrapping strip (flow_layout.FlowFrame), adds the preset and each band
    to it as groups of their own, so a narrow window wraps them instead of cutting the
    last field off, and returns the strip.
    """
    import tkinter as tk
    from tkinter import ttk
    from flow_layout import FlowFrame

    global _tk_vars
    if _tk_vars is None:
        _tk_vars = {k: tk.StringVar() for k in ("preset", "r_lo", "r_hi", "h_lo", "h_hi")}

        def _show(b: Bands):
            _tk_vars["preset"].set(preset_of(b))
            _tk_vars["r_lo"].set(f"{b.resp_bpm[0]:g}")
            _tk_vars["r_hi"].set(f"{b.resp_bpm[1]:g}")
            _tk_vars["h_lo"].set(f"{b.heart_bpm[0]:g}")
            _tk_vars["h_hi"].set(f"{b.heart_bpm[1]:g}")

        _show(_current)
        add_listener(_show)
    v = _tk_vars

    def _commit(_evt=None):
        try:
            set_bands((v["r_lo"].get(), v["r_hi"].get()), (v["h_lo"].get(), v["h_hi"].get()))
        except (ValueError, TypeError) as e:
            print(f"[vital_bands] not applied, still {_current.describe()}: {e}")
            try:
                parent.bell()
            except Exception:
                pass
            return
        v["preset"].set(preset_of())

    def _pick(_evt=None):
        name = v["preset"].get()
        if name in PRESETS:
            set_preset(name)

    flow = isinstance(parent, FlowFrame) and not vertical
    frame = parent if flow else ttk.Frame(parent)
    head = parent.group() if flow else frame
    box = ttk.Combobox(head, textvariable=v["preset"], values=list(PRESETS) + [CUSTOM],
                       state="readonly", width=24)
    box.bind("<<ComboboxSelected>>", _pick)
    rows = (("breath /min", "r_lo", "r_hi"), ("heart bpm", "h_lo", "h_hi"))
    if vertical:
        ttk.Label(frame, text="Bands:").grid(row=0, column=0, sticky="w")
        box.grid(row=0, column=1, columnspan=3, sticky="ew", pady=(0, 2))
        for r, (label, lo, hi) in enumerate(rows, start=1):
            ttk.Label(frame, text=label).grid(row=r, column=0, sticky="w")
            for c, key in ((1, lo), (3, hi)):
                e = ttk.Entry(frame, textvariable=v[key], width=5)
                e.grid(row=r, column=c, sticky="w")
                e.bind("<Return>", _commit)
                e.bind("<FocusOut>", _commit)
            ttk.Label(frame, text="-").grid(row=r, column=2)
    else:
        ttk.Label(head, text="Bands:").pack(side="left")
        box.pack(side="left", padx=(2, 6))
        for label, lo, hi in rows:
            part = parent.group() if flow else frame
            ttk.Label(part, text=label).pack(side="left", padx=(4, 2))
            for key in (lo, hi):
                e = ttk.Entry(part, textvariable=v[key], width=4)
                e.pack(side="left")
                e.bind("<Return>", _commit)
                e.bind("<FocusOut>", _commit)
    if on_change is not None:
        add_listener(on_change)
    return frame
