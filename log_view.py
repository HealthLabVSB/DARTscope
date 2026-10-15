"""
log_view.py

What the application prints, shown inside the application: the Log tab.

DARTscope reports a lot on the console - the capture health report after every load,
the CFAR target list, calibration and IQ-correction summaries, lost-packet warnings.
The Windows build has no console (DARTscope.spec: console=False), so all of that was
invisible there. install() copies everything written to stdout and stderr into a
queue (the console, when there is one, still gets it); LogView shows the queue.

Messages come from worker threads too - the heavy computations run in the
background - so the Tk widget is only ever touched from the Tk thread, which drains
the queue on a timer.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import queue
import sys
from datetime import datetime

_queue: "queue.Queue[tuple[str, bool]]" = queue.Queue()
_installed = False


class _Tee:
    """A stream that writes to the original stream and to the log queue."""

    def __init__(self, original, is_err: bool):
        self._orig = original
        self._err = is_err

    def write(self, s):
        if self._orig is not None:
            try:
                self._orig.write(s)
            except Exception:
                pass
        if s:
            _queue.put((str(s), self._err))
        return len(s) if s else 0

    def flush(self):
        if self._orig is not None:
            try:
                self._orig.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self._orig, "encoding", None) or "utf-8"


def install() -> None:
    """Start copying stdout/stderr into the log. Safe to call more than once."""
    global _installed
    if _installed:
        return
    sys.stdout = _Tee(sys.stdout, False)
    sys.stderr = _Tee(sys.stderr, True)
    _installed = True


def classify(line: str, from_stderr: bool = False) -> str | None:
    """'error', 'warning', 'health' or None - how a log line is highlighted."""
    low = line.lower()
    if "warn" in low:
        return "warning"
    if from_stderr or "traceback" in low or "error" in low:
        return "error"
    if low.lstrip().startswith(("[health]", "[fastcheck]")):
        return "health"
    return None


def drain(max_items: int = 2000) -> list:
    """Everything queued so far, as (text, from_stderr) chunks."""
    out = []
    for _ in range(max_items):
        try:
            out.append(_queue.get_nowait())
        except queue.Empty:
            break
    return out


class LogView:
    """The Log tab: the text, and Clear / Save / Copy / Auto-scroll."""

    MAX_LINES = 20000
    POLL_MS = 150

    def __init__(self, parent, notebook=None, tab=None, title: str = "Log"):
        import tkinter as tk
        from tkinter import ttk
        import theme

        self._tk = tk
        self.notebook, self.tab, self.title = notebook, tab, title
        self._partial = {False: "", True: ""}
        self._flagged = False

        from flow_layout import FlowFrame
        bar = FlowFrame(parent)
        bar.pack(side="top", fill="x", padx=6, pady=(6, 0))
        g = bar.group()
        ttk.Button(g, text="Clear", command=self.clear).pack(side="left")
        ttk.Button(g, text="Save...", command=self.save).pack(side="left", padx=(4, 0))
        ttk.Button(g, text="Copy all", command=self.copy_all).pack(side="left", padx=(4, 0))
        self.autoscroll = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar.group(), text="Auto-scroll", variable=self.autoscroll).pack(side="left")
        ttk.Label(bar.group(), text="Everything the application reports: capture health, detections, "
                                    "calibration, warnings and errors.",
                  foreground=theme.SECONDARY, wraplength=480).pack(side="left")

        body = ttk.Frame(parent)
        body.pack(side="top", fill="both", expand=True, padx=6, pady=6)
        self.text = tk.Text(body, wrap="none", font=theme.mono(9), state="disabled")
        ys = ttk.Scrollbar(body, orient="vertical", command=self.text.yview)
        xs = ttk.Scrollbar(body, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side="right", fill="y")
        xs.pack(side="bottom", fill="x")
        self.text.pack(side="left", fill="both", expand=True)
        self.text.tag_configure("error", foreground=theme.RED)
        self.text.tag_configure("warning", foreground=theme.ORANGE)
        self.text.tag_configure("health", foreground=theme.BLUE)

        if notebook is not None:
            notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed, add="+")
        self.text.after(self.POLL_MS, self._poll)

    # --- actions ------------------------------------------------------------------
    def clear(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def contents(self) -> str:
        return self.text.get("1.0", "end-1c")

    def save(self):
        from tkinter import filedialog, messagebox
        name = f"dartscope_log_{datetime.now():%Y%m%d_%H%M%S}.txt"
        path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile=name,
                                            filetypes=[("Text files", "*.txt")])
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.contents())
            except OSError as e:
                messagebox.showerror("Log", f"Could not save the log: {e}")

    def copy_all(self):
        self.text.clipboard_clear()
        self.text.clipboard_append(self.contents())

    # --- plumbing -----------------------------------------------------------------
    def _on_tab_changed(self, _evt=None):
        try:
            if self.notebook.nametowidget(self.notebook.select()) is self.tab and self._flagged:
                self.notebook.tab(self.tab, text=self.title)
                self._flagged = False
        except Exception:
            pass

    def _flag(self):
        """Mark the tab when a warning or error arrives while it is not shown."""
        if self.notebook is None or self._flagged:
            return
        try:
            if self.notebook.nametowidget(self.notebook.select()) is not self.tab:
                self.notebook.tab(self.tab, text=f"{self.title} (!)")
                self._flagged = True
        except Exception:
            pass

    def _append(self, line: str, tag):
        self.text.insert("end", line, tag or ())

    def _poll(self):
        chunks = drain()
        if chunks:
            self.text.configure(state="normal")
            for s, err in chunks:
                s = self._partial[err] + s
                parts = s.split("\n")
                self._partial[err] = parts.pop()          # an unfinished line waits
                for line in parts:
                    tag = classify(line, err)
                    if tag in ("error", "warning"):
                        self._flag()
                    self._append(line + "\n", tag)
            n = int(self.text.index("end-1c").split(".")[0])
            if n > self.MAX_LINES:
                self.text.delete("1.0", f"{n - self.MAX_LINES}.0")
            self.text.configure(state="disabled")
            if self.autoscroll.get():
                self.text.see("end")
        self.text.after(self.POLL_MS, self._poll)
