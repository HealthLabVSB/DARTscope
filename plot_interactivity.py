"""
Reusable helpers for interactive Matplotlib canvases embedded in Tkinter GUIs.

Provides:
- create_interactive_canvas(parent, fig): FigureCanvasTkAgg + NavigationToolbar2Tk
- normalize_for_plot(signal): robust normalization for visual comparison of signals
- bin_extent(x_axis, y_axis): imshow extent with every pixel centred on its axis value
- few_bins_interpolation(n): 'nearest' for an image axis of a handful of bins
- PlotInteractivityManager: X-axis SpanSelector zoom (synchronized across time axes),
  storage of original view limits, Reset View / Reset X Zoom
- add_plot_control_bar(parent, manager): "X Zoom Mode" / "Reset X Zoom" / "Reset View" controls
"""

import numpy as np
import tkinter as tk
from tkinter import ttk
import theme
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector

from flow_layout import FlowFrame


def normalize_for_plot(sig):
    """Normalizace signálu pro vizuální porovnání v jednom grafu.

    - odstraní NaN/Inf,
    - odečte medián,
    - vydělí robustním rozsahem (IQR), případně směrodatnou odchylkou.
    """
    sig = np.asarray(sig, dtype=float).flatten()
    if sig.size == 0:
        return sig
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    sig = sig - np.median(sig)
    q75, q25 = np.percentile(sig, [75, 25])
    scale = q75 - q25
    if not np.isfinite(scale) or scale < 1e-12:
        scale = np.std(sig)
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return sig / scale


def bin_extent(x_axis, y_axis):
    """
    imshow extent that puts every pixel's centre on its axis value.

    Spanning just the first to the last value squeezes N pixels into N-1 bins and moves
    each one by up to half a bin - unnoticed with 64 Doppler bins, but with the 4 chirps
    of a vital-signs frame it drew the zero-velocity bin at -1.6 m/s.
    """
    def edges(a):
        a = np.asarray(a, dtype=float)
        half = (a[-1] - a[0]) / (2 * (a.size - 1)) if a.size > 1 else 0.5
        return float(a[0] - half), float(a[-1] + half)
    return [*edges(x_axis), *edges(y_axis)]


# Below this an axis is drawn bin by bin. Matplotlib's "auto" interpolation smooths an
# image whose other axis is barely enlarged (256 range bins in ~330 px), and with 32
# Doppler bins that drew horizontal streaks - velocities the map does not resolve
# (AWR2243 cascade, 2026-10-09).
FEW_BINS = 64


def few_bins_interpolation(n_bins: int):
    """
    imshow interpolation for an axis of `n_bins` bins: 'nearest' for a handful - smoothed,
    4 velocity bins 12.9 m/s wide drew a blob that suggested a resolution they do not
    have - and Matplotlib's default (None) otherwise.
    """
    return "nearest" if n_bins < FEW_BINS else None


# No screen shows more columns than this; an image wider than it is drawn averaged down.
MAX_IMAGE_COLUMNS = 4096


class ColumnPooledImage:
    """
    An image too wide to draw whole at every redraw: the RTI of a 3-min take at 100 fps
    has 180 000 chirps x 256 range bins, and matplotlib colour-maps and resamples every
    one of them on each draw - 3.4 s, so each resize of the window, each step of the
    frame line and each zoom froze the application.

    It is shown averaged over groups of columns, at most MAX_IMAGE_COLUMNS for the part in
    view - the mean is what matplotlib's own anti-aliasing showed at that size anyway.
    Zoomed in, the columns in view are averaged again, finer, down to single columns, so
    no detail is lost. The full data stay in .data; .im is the AxesImage (colour limits,
    colormap and colorbar as for any image).

        pooled = ColumnPooledImage(ax, img, extent, aspect="auto", origin="lower")
        pooled.im.set_clim(lo, hi)
    """

    def __init__(self, ax, data, extent, max_cols: int = MAX_IMAGE_COLUMNS, **imshow_kw):
        self.ax = ax
        self.data = np.asarray(data)
        self.x0, self.x1, self.y0, self.y1 = (float(v) for v in extent)
        self.max_cols = max(1, int(max_cols))
        self.n = self.data.shape[1]
        self.shown = None                       # (first column, end column, group size)
        self._updating = False
        img, ext = self._pooled(0, self.n)
        self.im = ax.imshow(img, extent=ext, **imshow_kw)
        self.im.column_pool = self              # the axes' callbacks hold methods weakly
        if self.n > self.max_cols:
            ax.callbacks.connect("xlim_changed", self._on_xlim)

    def _pooled(self, c0: int, c1: int):
        """Columns c0..c1 averaged in groups to at most max_cols, and their extent."""
        k = max(1, -(-(c1 - c0) // self.max_cols))
        self.shown = (c0, c1, k)
        part = self.data[:, c0:c1]
        if k > 1:
            m = (c1 - c0) // k * k
            groups = [part[:, :m].reshape(part.shape[0], -1, k).mean(axis=2)] if m else []
            if m < c1 - c0:
                groups.append(part[:, m:].mean(axis=1, keepdims=True))
            part = groups[0] if len(groups) == 1 else np.concatenate(groups, axis=1)
        dx = (self.x1 - self.x0) / self.n
        return part, [self.x0 + c0 * dx, self.x0 + c1 * dx, self.y0, self.y1]

    def _on_xlim(self, ax):
        if self._updating:
            return
        lo, hi = sorted(ax.get_xlim())
        dx = (self.x1 - self.x0) / self.n
        c0 = int(np.clip(np.floor((lo - self.x0) / dx), 0, self.n))
        c1 = int(np.clip(np.ceil((hi - self.x0) / dx), 0, self.n))
        if c1 <= c0:
            return
        s0, s1, k = self.shown
        if s0 <= c0 and c1 <= s1 and k == max(1, -(-(c1 - c0) // self.max_cols)):
            return                              # in view and at the detail it needs
        self._updating = True
        try:
            img, ext = self._pooled(c0, c1)
            self.im.set_data(img)
            # set_extent would also move the limits of an autoscaling axis
            auto = ax.get_autoscalex_on(), ax.get_autoscaley_on()
            ax.set_autoscalex_on(False)
            ax.set_autoscaley_on(False)
            self.im.set_extent(ext)
            ax.set_autoscalex_on(auto[0])
            ax.set_autoscaley_on(auto[1])
        finally:
            self._updating = False


def figure_rows(fig) -> int:
    """Rows of subplots of a figure: the distinct first grid rows of its axes (a colorbar
    or an axis spanning rows adds none)."""
    starts = set()
    for ax in fig.axes:
        ss = ax.get_subplotspec()
        if ss is not None:
            starts.add(ss.get_topmost_subplotspec().rowspan.start)
    return max(1, len(starts))


RESIZE_SETTLE_MS = 250
_resize_installed = False


def install_settled_resize():
    """
    Redraw a plot when its size has settled, not at every pixel of a window drag: each
    Configure of a TkAgg canvas drew the whole figure (0.1-0.2 s for the Vitals or RTI
    view), so dragging the window edge stuttered from one redraw to the next. The canvas
    is resized at once - the old picture stays in it - and drawn RESIZE_SETTLE_MS after
    the last size change.
    """
    global _resize_installed
    if _resize_installed:
        return
    original = FigureCanvasTkAgg.resize

    def resize(self, event):
        original(self, event)                      # new figure size; asks for a draw at idle
        pending = getattr(self, "_idle_draw_id", None)
        widget = self.get_tk_widget()
        if pending:
            widget.after_cancel(pending)
            self._idle_draw_id = None
        later = getattr(self, "_settled_draw_id", None)
        if later:
            widget.after_cancel(later)

        def settled():
            self._settled_draw_id = None
            self.draw_idle()
        self._settled_draw_id = widget.after(RESIZE_SETTLE_MS, settled)

    FigureCanvasTkAgg.resize = resize
    _resize_installed = True


def image_percentiles(img, q, max_values: int = 2_000_000):
    """np.percentile of an image from at most max_values of its values (2 M of the RTI's
    46 M give the same display limits in 1/20 of the time). Drawn at random with a fixed
    seed, not every n-th: a stride that shares a factor with the last axis (24 with 256
    range bins) would only ever look at some of the bins."""
    a = np.ravel(img)
    if a.size <= max_values:
        return np.percentile(a, q)
    idx = np.random.default_rng(0).integers(0, a.size, max_values)
    return np.percentile(a[idx], q)


def create_interactive_canvas(parent, fig):
    """Vytvoří FigureCanvasTkAgg s NavigationToolbar2Tk (Home/Back/Forward/Pan/Zoom/Save).

    Toolbar je zabalen dole, canvas vyplní zbytek rodiče.
    Vrací (canvas, toolbar).
    """
    canvas = FigureCanvasTkAgg(fig, master=parent)
    try:
        toolbar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
    except TypeError:
        # Older matplotlib versions do not support pack_toolbar
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.pack_forget()
    toolbar.update()
    toolbar.pack(side="bottom", fill="x")
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
    return canvas, toolbar


class PlotInteractivityManager:
    """Správa interaktivity jedné Matplotlib figury:

    - SpanSelector pro výběr úseku v ose X (synchronizovaný pro všechny časové osy),
    - uložení výchozích rozsahů (original_xlim / original_ylim),
    - Reset View / Reset X Zoom,
    - zapnutí/vypnutí X Zoom módu.

    Po každém překreslení grafů zavolejte register_time_axes(axes) -- znovu
    vytvoří SpanSelectory a uloží výchozí rozsahy, takže interaktivita
    funguje i po novém výpočtu.
    """

    def __init__(self, canvas, autoscale_y=True):
        self.canvas = canvas
        self.fig = canvas.figure
        self.autoscale_y = autoscale_y
        self.time_axes = []
        self.span_selectors = []
        self.original_limits = {}
        self.x_zoom_enabled = True

    # ------------------------------------------------------------------
    # Setup / redraw support
    # ------------------------------------------------------------------
    def register_time_axes(self, axes):
        """Registrace časových os (znovu volat po každém překreslení).

        Vytvoří SpanSelector pro každou časovou osu a uloží výchozí rozsahy
        všech os ve figuře.
        """
        self._clear_selectors()
        self.time_axes = [ax for ax in axes if ax is not None]
        for ax in self.time_axes:
            sel = self._make_selector(ax)
            if sel is not None:
                sel.set_active(self.x_zoom_enabled)
                self.span_selectors.append(sel)
        self.save_original_limits()

    def _make_selector(self, ax):
        try:
            return SpanSelector(
                ax, self._on_x_span, "horizontal", useblit=True,
                props=dict(alpha=0.2, facecolor="tab:blue"), button=1)
        except TypeError:
            # Older matplotlib versions use rectprops instead of props
            try:
                return SpanSelector(
                    ax, self._on_x_span, "horizontal", useblit=True,
                    rectprops=dict(alpha=0.2, facecolor="tab:blue"), button=1)
            except Exception:
                return None
        except Exception:
            return None

    def _clear_selectors(self):
        for sel in self.span_selectors:
            try:
                sel.set_active(False)
                sel.disconnect_events()
            except Exception:
                pass
        self.span_selectors = []

    def save_original_limits(self):
        """Uloží výchozí rozsahy (original_xlim, original_ylim) všech os figury."""
        self.original_limits = {}
        for ax in self.fig.get_axes():
            self.original_limits[ax] = (ax.get_xlim(), ax.get_ylim())

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def set_x_zoom_mode(self, enabled):
        """Zapne/vypne výběr úseku v ose X myší."""
        self.x_zoom_enabled = bool(enabled)
        for sel in self.span_selectors:
            try:
                sel.set_active(self.x_zoom_enabled)
            except Exception:
                pass

    def _toolbar_active(self):
        toolbar = getattr(self.canvas, "toolbar", None)
        mode = getattr(toolbar, "mode", "") if toolbar is not None else ""
        return bool(str(mode))

    def _on_x_span(self, xmin, xmax):
        # Ignore selection while toolbar pan/zoom is active or X zoom disabled
        if not self.x_zoom_enabled or self._toolbar_active():
            return
        if not np.isfinite(xmin) or not np.isfinite(xmax):
            return
        if abs(xmax - xmin) < 1e-9:
            return
        self.apply_x_range(min(xmin, xmax), max(xmin, xmax))

    def apply_x_range(self, xmin, xmax):
        """Nastaví stejný xlim všem registrovaným časovým osám (synchronizace X)."""
        for ax in self.time_axes:
            ax.set_xlim(xmin, xmax)
            if self.autoscale_y:
                self._autoscale_y(ax, xmin, xmax)
        self.canvas.draw_idle()

    def _autoscale_y(self, ax, xmin, xmax):
        ymin, ymax = np.inf, -np.inf
        for line in ax.get_lines():
            # Skip axhline/axvline etc. (not in pure data coordinates)
            try:
                if line.get_transform() is not ax.transData:
                    continue
            except Exception:
                pass
            xd = np.asarray(line.get_xdata(), dtype=float)
            yd = np.asarray(line.get_ydata(), dtype=float)
            if xd.size == 0 or yd.size != xd.size:
                continue
            mask = (xd >= xmin) & (xd <= xmax) & np.isfinite(yd)
            if not np.any(mask):
                continue
            ymin = min(ymin, float(np.min(yd[mask])))
            ymax = max(ymax, float(np.max(yd[mask])))
        for coll in getattr(ax, "collections", []):
            try:
                offsets = np.asarray(coll.get_offsets(), dtype=float)
            except Exception:
                continue
            if offsets.ndim != 2 or offsets.shape[0] == 0:
                continue
            xd, yd = offsets[:, 0], offsets[:, 1]
            mask = (xd >= xmin) & (xd <= xmax) & np.isfinite(yd)
            if not np.any(mask):
                continue
            ymin = min(ymin, float(np.min(yd[mask])))
            ymax = max(ymax, float(np.max(yd[mask])))
        if np.isfinite(ymin) and np.isfinite(ymax):
            if ymax <= ymin:
                pad = max(abs(ymin) * 0.05, 1e-6)
            else:
                pad = 0.05 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset_x_zoom(self):
        """Vrátí pouze osu X (a auto Y) časových grafů na výchozí rozsah."""
        for ax in self.time_axes:
            limits = self.original_limits.get(ax)
            if limits is None:
                continue
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
        self.canvas.draw_idle()

    def reset_view(self):
        """Obnoví výchozí rozsahy všech grafů ve figuře."""
        for ax, (xlim, ylim) in self.original_limits.items():
            try:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            except Exception:
                pass
        toolbar = getattr(self.canvas, "toolbar", None)
        if toolbar is not None:
            try:
                toolbar.update()  # reset navigation (Back/Forward) stack
            except Exception:
                pass
        self.canvas.draw_idle()


def add_plot_control_bar(parent, manager):
    """Vytvoří lištu s ovládáním interaktivity grafu.

    Obsahuje: checkbox 'X Zoom Mode' (výběr úseku v ose X myší),
    tlačítka 'Reset X Zoom' a 'Reset View'.
    Vrací (frame, x_zoom_var). Lišta se v úzkém okně zalomí (flow_layout.FlowFrame);
    volající ji zabalí s fill="x". Je-li parent sám FlowFrame (lišta nastavení
    záložky), přidají se ovládací prvky do něj jako další skupiny a vrátí se parent,
    který už zabalený je.
    """
    bar = parent if isinstance(parent, FlowFrame) else FlowFrame(parent)
    zoom, resets, hint = bar.group(), bar.group(), bar.group()
    x_zoom_var = tk.BooleanVar(value=manager.x_zoom_enabled)
    ttk.Checkbutton(
        zoom, text="X Zoom Mode", variable=x_zoom_var,
        command=lambda: manager.set_x_zoom_mode(x_zoom_var.get())
    ).pack(side="left", padx=(4, 2))
    ttk.Button(resets, text="Reset X Zoom", command=manager.reset_x_zoom).pack(side="left", padx=2)
    ttk.Button(resets, text="Reset View", command=manager.reset_view).pack(side="left", padx=2)
    ttk.Label(
        hint, text="(drag in the plot to select a range on the X axis)",
        font=theme.font(8), foreground=theme.TERTIARY
    ).pack(side="left", padx=6)
    return bar, x_zoom_var
