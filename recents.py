"""
recents.py — independent recent-files manager for GUI
Keeps last N paths in a dedicated JSON file so decoding/parsing modules
(dca1000_decode, processing backends) stay unaware of GUI state.
"""
from __future__ import annotations
import json
from pathlib import Path

try:
    from platformdirs import user_config_dir  # type: ignore
except Exception:
    user_config_dir = None  # fallback


class RecentFilesManager:
    def __init__(self, app_name: str = "bme_radar_gui", limit: int = 5, filename: str = "recent_gui.json"):
        self.limit = max(1, int(limit))
        if user_config_dir:
            cfgdir = Path(user_config_dir(app_name, appauthor=False))
        else:
            cfgdir = Path.home() / f".{app_name}"
        cfgdir.mkdir(parents=True, exist_ok=True)
        self.store_path = cfgdir / filename
        self._items = self._load()

    # ---- persistence ----
    def _load(self):
        try:
            if self.store_path.exists():
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return [p for p in data if isinstance(p, str)]
        except Exception:
            pass
        return []

    def _save(self):
        try:
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(self._items[: self.limit], f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ---- API ----
    def add(self, path: str):
        if not path:
            return
        try:
            p = str(Path(path).resolve())
        except Exception:
            p = str(path)
        self._items = [i for i in self._items if i != p]
        self._items.insert(0, p)
        if len(self._items) > self.limit:
            self._items = self._items[: self.limit]
        self._save()

    def remove(self, path: str):
        self._items = [i for i in self._items if i != path]
        self._save()

    def clear(self):
        self._items = []
        self._save()

    def list(self):
        return list(self._items[: self.limit])


# Global singleton for GUI
RECENTS = RecentFilesManager()


# ---- Optional Tk helper (kept here so GUI can import without touching decode) ----
def tk_pick_recent_and_open(parent=None, on_open=None):
    """Small Tk dialog to pick from recent files; returns selected path or calls on_open."""
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        return None

    recents = RECENTS.list()
    if not recents:
        if parent is not None:
            try:
                messagebox.showinfo("Recently opened", "History is empty.")
            except Exception:
                pass
        return None

    win = tk.Toplevel(parent) if parent else tk.Tk()
    win.title("Recently opened files")
    win.geometry("720x240")
    lb = tk.Listbox(win)
    lb.pack(fill="both", expand=True)
    for p in recents:
        lb.insert("end", p)

    def _choose(evt=None):
        sel = lb.curselection()
        if not sel:
            return
        path = lb.get(sel[0])
        try:
            win.destroy()
        except Exception:
            pass
        if callable(on_open):
            on_open(path)
        else:
            win.selected_path = path

    lb.bind("<Double-Button-1>", _choose)
    lb.bind("<Return>", _choose)

    btn_frame = tk.Frame(win)
    btn_frame.pack(fill="x", padx=6, pady=6)
    tk.Button(btn_frame, text="Open", command=_choose).pack(side="right")
    tk.Button(btn_frame, text="Close", command=win.destroy).pack(side="right", padx=(0, 6))

    win.mainloop() if parent is None else None
    return getattr(win, "selected_path", None)