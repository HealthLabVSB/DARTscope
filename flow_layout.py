"""
flow_layout.py

Layout widgets that keep every control reachable in a small window.

    FlowFrame    a control strip whose groups wrap onto further rows when the window is
                 too narrow for one row. It was written for the Angle of Arrival tab,
                 whose single row of about eighteen widgets needed a 1900 px wide window;
                 every control strip of the application uses it now.
    ScrollFrame  a vertically scrolling panel for the tall settings columns (HR, AI,
                 Helper, the parameter panel of the Decode tab). With fit= it is the
                 safety net of a whole tab instead: it fills the window and only scrolls
                 when the window is too low for the controls and a minimum plot (the
                 Analysis tab: at 800 x 600 for five of its views, at 1024 x 700 for
                 none).
    cap_plot_heights  lowers the height the plots of a tab ask for, so what a tab
                 asks for is its controls plus a small plot - the figure sizes the
                 plots were created with (up to 1200 px) are not a minimum.

The control rows of the other tabs were plain frames packed left to right. Tk does not
wrap those: in a window narrower than the row it gives the space to the widgets packed
first and hides the rest, so at 1024 x 700 the CFAR tab lost its Run button, the
Vitals tab its band limits and the HR tab everything below the filter buttons -
72 hidden or cut controls over all tabs, 96 at 800 x 600.

Nothing here knows about radar data; importable wherever Tk is.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class FlowFrame(tk.Frame):
    """
    A control strip whose contents wrap onto further rows when the window is narrow.

    Groups are laid out left to right and moved to the next row when they no longer
    fit, so the same controls stay usable at any width. Children are positioned with
    place(), so anything added has to go through add(); group() makes an inner frame
    for widgets that belong together and must not be split (a label and its field).

    The strip is laid out again when its own width changes and when one of its groups
    changes size - a label whose text grows after a computation, a field that gets
    enabled with a longer text - so nothing ends up hidden under its neighbour.
    """

    def __init__(self, master, hgap=10, vgap=4, **kw):
        super().__init__(master, **kw)
        self._items = []
        self._hgap, self._vgap = hgap, vgap
        self._height = 0
        self._pending = False
        self.bind("<Configure>", lambda _e: self._reflow())

    def group(self, **kw):
        """An inner frame for widgets that must stay on one row together."""
        return self.add(ttk.Frame(self, **kw))

    def add(self, widget):
        self._items.append(widget)
        widget.place(x=0, y=0)
        # Placed without a width, a group follows its own requested size; when that
        # changes it is resized, which is a <Configure> on it.
        widget.bind("<Configure>", self._schedule, add="+")
        self._schedule()
        return widget

    def _schedule(self, _evt=None):
        if not self._pending:
            self._pending = True
            self.after_idle(self._reflow)

    def _reflow(self):
        self._pending = False
        try:
            width = self.winfo_width()
        except tk.TclError:
            return                                        # destroyed meanwhile
        if width <= 1:
            return
        x = y = row_h = 0
        for w in self._items:
            try:
                ww, wh = w.winfo_reqwidth(), w.winfo_reqheight()
            except tk.TclError:
                continue
            if x > 0 and x + ww > width:
                x, y, row_h = 0, y + row_h + self._vgap, 0
            w.place(x=x, y=y)
            x += ww + self._hgap
            row_h = max(row_h, wh)
        total = y + row_h
        # Only resize when the wrapping actually changed, otherwise configuring the
        # frame inside its own <Configure> handler loops.
        if total and total != self._height:
            self._height = total
            self.configure(height=total)
            # One more or one less row changes what the tab needs; a ScrollFrame with
            # fit= listens for this.
            try:
                self.event_generate("<<FlowHeight>>")
            except tk.TclError:
                pass


class ScrollFrame(ttk.Frame):
    """
    A vertically scrolling panel. Put the children into `.inner`.

    - The scrollbar is packed before the canvas, so a narrow pane squeezes the canvas
      and never the scrollbar (the AI tab's panel had a scrollbar that a 340 px canvas
      packed first pushed out of sight).
    - The inner frame is kept as wide as the visible area, so children packed with
      fill="x" stretch to it; the panel asks for the width of its content, so a
      paned window starts it wide enough.
    - The mouse wheel scrolls the panel under the pointer, and only that one - on
      Windows and macOS (<MouseWheel>) and on X11 (buttons 4 and 5). Over a panel,
      text or list inside it that scrolls on its own, the wheel is left to that one.
    - xscroll=True adds a horizontal scrollbar for content wider than the view (the
      inner frame then keeps at least its own width instead of being squeezed).
    - fit=callable makes it the frame of a whole tab: the inner frame fills the view
      both ways, as a plain frame would, until the view is lower than fit() - the
      height the content needs. Then the content keeps that height, the scrollbar
      appears and the rest is scrolled to. The callable may return None for "not
      settled yet, ask again".
    """

    def __init__(self, master, width: int | None = None, xscroll: bool = False,
                 fit=None, **kw):
        super().__init__(master, **kw)
        self._fixed_width = width
        self._xscroll = xscroll
        self._fit_fn = fit
        self._fit_pending = False
        self._fit_later = None
        self._fit_size = None
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                width=width if width else 1, height=1)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        if fit is None:
            self.vbar.pack(side="right", fill="y")      # with fit=, only when needed
        if xscroll:
            self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
            self.canvas.configure(xscrollcommand=self.hbar.set)
            self.hbar.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        if fit is None:
            self.inner.bind("<Configure>", self._on_inner, add="+")
            self.canvas.bind("<Configure>", self._on_canvas, add="+")
        else:
            self.inner.bind("<Configure>", self._schedule_fit, add="+")
            self.canvas.bind("<Configure>", self._schedule_fit, add="+")
            # What the content needs changes when a strip wraps into more or fewer
            # rows and when another tab is shown.
            for seq in ("<<FlowHeight>>", "<<NotebookTabChanged>>"):
                self.bind_all(seq, self._schedule_fit, add="+")
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind_all(seq, self._on_wheel, add="+")

    def _on_inner(self, _evt=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        if not self._fixed_width and not self._xscroll:
            want = self.inner.winfo_reqwidth()
            if want > 1 and int(self.canvas.cget("width")) != want:
                self.canvas.configure(width=want)
        elif self._xscroll:
            self._fit_width(self.canvas.winfo_width())

    def _on_canvas(self, evt):
        self._fit_width(evt.width)

    def _fit_width(self, view_width: int):
        width = max(view_width, self.inner.winfo_reqwidth()) if self._xscroll else view_width
        if int(float(self.canvas.itemcget(self._window, "width") or 0)) != width:
            self.canvas.itemconfigure(self._window, width=width)

    def _schedule_fit(self, _evt=None):
        if not self._fit_pending:
            self._fit_pending = True
            self.after_idle(self._fit)
        # A new request climbs one level of frames per idle pass, so on a tab's first
        # showing what it needs is only known a moment later: look once more then.
        if self._fit_later is not None:
            self.after_cancel(self._fit_later)
        self._fit_later = self.after(80, self._fit_again)

    def _fit_again(self):
        self._fit_later = None
        self._fit()

    def _fit(self):
        self._fit_pending = False
        try:
            view_w, view_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if view_w <= 1 or view_h <= 1:
                return
            need = self._fit_fn()
            if need is None:
                self._schedule_fit()
                return
            need = int(need)
            # Shown or hidden, the scrollbar changes the width and so the wrapping,
            # which is measured again. The rule cannot swing back and forth: the bar
            # comes when the content needs more than the view without it and goes
            # only when it needs no more than the view even with it.
            if not self.vbar.winfo_manager() and need > view_h:
                self.vbar.pack(side="right", fill="y", before=self.canvas)
                self._schedule_fit()
                return
            if self.vbar.winfo_manager() and need <= view_h:
                self.vbar.pack_forget()
                self._schedule_fit()
                return
            height = max(view_h, need)
            if self._fit_size != (view_w, height):
                self._fit_size = (view_w, height)
                self.canvas.itemconfigure(self._window, width=view_w, height=height)
                self.canvas.configure(scrollregion=(0, 0, view_w, height))
            if height <= view_h:
                self.canvas.yview_moveto(0)
        except tk.TclError:
            return                                        # destroyed meanwhile

    def _can_scroll(self) -> bool:
        top, bottom = self.canvas.yview()
        return top > 0.0 or bottom < 1.0

    def _wheel_is_mine(self) -> bool:
        """The pointer is over this panel and not over something inside it that
        scrolls on its own - a nested panel with more than it shows, a text, a list."""
        try:
            w = self.winfo_containing(*self.winfo_pointerxy())
        except (KeyError, tk.TclError):
            return False                                  # e.g. a combobox drop-down
        while w is not None and w is not self:
            if isinstance(w, ScrollFrame) and w._can_scroll():
                return False
            if isinstance(w, (tk.Text, tk.Listbox, ttk.Treeview)):
                return False
            w = w.master
        return w is self

    def _on_wheel(self, evt):
        try:
            if not self.winfo_viewable() or not self._wheel_is_mine():
                return None
            if not self._can_scroll():
                return None                               # everything fits
        except tk.TclError:
            return None                                   # destroyed; the binding stays
        if getattr(evt, "num", None) == 4:
            step = -1
        elif getattr(evt, "num", None) == 5:
            step = 1
        else:
            delta = int(getattr(evt, "delta", 0) or 0)
            if delta == 0:
                return None
            step = -max(1, abs(delta) // 120) if delta > 0 else max(1, abs(delta) // 120)
        self.canvas.yview_scroll(step, "units")
        return None


def cap_plot_heights(widget, height: int, min_for=None) -> bool:
    """
    Lower the height that the plot canvases under `widget` ask for to `height` px.

    A matplotlib canvas asks for its figure size (the Vitals figure 1200 px) and a
    tab asks for the sum of what is in it, so without this every tab would seem to
    need a screen and a half. Where the plots are - the stretching row of each tab -
    they get all the room left anyway, whatever they ask for. The canvases of the
    scrolling panels themselves are left alone. Returns True if anything changed.

    min_for(canvas widget) -> px or None gives a plot that needs more its own minimum,
    which it then asks for: the six rows of subplots of the Vitals view squeezed into
    the 160 px of a one-row plot came out as six flat lines with their labels on top of
    each other. The tab scrolls when the window is lower than that.
    """
    changed = False
    stack = [widget]
    while stack:
        w = stack.pop()
        try:
            children = w.winfo_children()
        except tk.TclError:
            continue
        for c in children:
            stack.append(c)
            try:
                if c.winfo_class() != "Canvas" or isinstance(c.master, ScrollFrame):
                    continue
                own = min_for(c) if min_for is not None else None
                cap = max(height, int(own)) if own else height
                if getattr(c, "_flow_cap", None) is None:
                    # On a scaled display (125 %, 150 %) matplotlib sets the request
                    # back to the figure size when the canvas is first shown; this
                    # runs right after its <Map> handler and lowers it again.
                    c.bind("<Map>", lambda _e, c=c: _recap(c), add="+")
                c._flow_cap = cap
                req = c.winfo_reqheight()
                if req > cap or (own and req < cap):
                    c.configure(height=cap)
                    changed = True
            except tk.TclError:
                pass
    return changed


def _recap(canvas):
    try:
        if canvas.winfo_reqheight() > canvas._flow_cap:
            canvas.configure(height=canvas._flow_cap)
            canvas.event_generate("<<FlowHeight>>")    # what the tab needs changed
    except tk.TclError:
        pass
