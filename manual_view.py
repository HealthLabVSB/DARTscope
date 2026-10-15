"""
manual_view.py

The Manual tab: docs/MANUAL.md with a collapsible table of contents.

Left, the chapters, sections and sub-sections (click the arrows to open or close a
chapter); right, the selected part - a chapter shows everything under it, a section
just itself. The search box narrows the table of contents to the sections that
mention every word typed and highlights the words. Links inside the manual jump to
the section they name. "Save as HTML" writes the whole manual as one page to share.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import os
import tkinter as tk
import theme
import webbrowser
from tkinter import filedialog, messagebox, ttk

import manual as mdoc
from flow_layout import FlowFrame


class ManualView:
    def __init__(self, parent, path: str):
        self.path = path
        self.root_section = None
        self._items = {}                 # tree item id -> Section
        self._links = {}                 # text tag -> link target
        self._current = None

        bar = FlowFrame(parent)
        bar.pack(side="top", fill="x", padx=6, pady=(6, 0))
        g = bar.group()
        ttk.Label(g, text="Search:").pack(side="left")
        self.query = tk.StringVar()
        entry = ttk.Entry(g, textvariable=self.query, width=32)
        entry.pack(side="left", padx=(4, 2))
        entry.bind("<Return>", lambda _e: self._first_match())
        self.query.trace_add("write", lambda *_: self._apply_search())
        ttk.Button(g, text="Clear", command=lambda: self.query.set("")).pack(side="left")
        g = bar.group()
        ttk.Button(g, text="Expand all", command=lambda: self._expand(True)).pack(side="left", padx=(0, 2))
        ttk.Button(g, text="Collapse all", command=lambda: self._expand(False)).pack(side="left")
        g = bar.group()
        ttk.Button(g, text="Save as HTML...", command=self.save_html).pack(side="left", padx=(0, 2))
        ttk.Button(g, text="Reload", command=self.load).pack(side="left")
        self.status = tk.StringVar(value="")
        ttk.Label(bar.group(), textvariable=self.status, foreground=theme.SECONDARY).pack(side="left")

        paned = ttk.PanedWindow(parent, orient="horizontal")
        paned.pack(side="top", fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(paned)
        self.tree = ttk.Treeview(left, show="tree", selectmode="browse")
        tys = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tys.set)
        tys.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        paned.add(left, weight=1)

        right = ttk.Frame(paned)
        self.text = tk.Text(right, wrap="word", font=theme.font(10), padx=14, pady=10,
                            state="disabled", cursor="arrow")
        ys = ttk.Scrollbar(right, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        paned.add(right, weight=3)
        self._configure_tags()
        self.load()

    # --- content ---------------------------------------------------------------------
    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.root_section = mdoc.parse(f.read())
        except OSError as e:
            self.root_section = mdoc.parse(f"# Manual\n\nThe manual could not be read: {e}\n")
        self._build_tree()
        first = self.tree.get_children("")
        if first:
            self.tree.selection_set(first[0])
            self.tree.see(first[0])

    def _build_tree(self, only=None):
        self.tree.delete(*self.tree.get_children(""))
        self._items.clear()

        def add(parent_id, node):
            for child in node.children:
                if only is not None and child not in only:
                    continue
                iid = self.tree.insert(parent_id, "end", text=mdoc.plain(child.title),
                                       open=only is not None)
                self._items[iid] = child
                add(iid, child)

        add("", self.root_section)

    def _expand(self, open_: bool):
        def walk(iid):
            self.tree.item(iid, open=open_)
            for c in self.tree.get_children(iid):
                walk(c)
        for iid in self.tree.get_children(""):
            walk(iid)

    def _on_select(self, _evt=None):
        sel = self.tree.selection()
        if sel and sel[0] in self._items:
            self._render(self._items[sel[0]])

    def show(self, key: str) -> bool:
        """Open the section named or anchored by key; False if there is none."""
        if self.root_section is None:
            return False
        target = mdoc.find(self.root_section, key)
        if target is None:
            return False
        if self.query.get():
            self.query.set("")
        for iid, sec in self._items.items():
            if sec is target:
                parent = self.tree.parent(iid)
                while parent:
                    self.tree.item(parent, open=True)
                    parent = self.tree.parent(parent)
                self.tree.selection_set(iid)
                self.tree.see(iid)
                self._render(target)            # now, not when the event loop gets to it
                return True
        return False

    # --- search ----------------------------------------------------------------------
    def _apply_search(self):
        q = self.query.get().strip()
        if not q:
            self._build_tree()
            self.status.set("")
            if self._current is not None:
                self.show(self._current.anchor)
            return
        hits = mdoc.search(self.root_section, q)
        keep = set()
        for h in hits:
            node = h
            while node is not None and node.level > 0:
                keep.add(node)
                node = node.parent
            keep.update(h.walk())
        self._build_tree(only=keep)
        self.status.set(f"{len(hits)} section(s) mention '{q}'" if hits else f"nothing mentions '{q}'")
        if self._current is not None:
            self._render(self._current)

    def _first_match(self):
        hits = mdoc.search(self.root_section, self.query.get())
        if hits:
            for iid, sec in self._items.items():
                if sec is hits[0]:
                    self.tree.selection_set(iid)
                    self.tree.see(iid)
                    break

    # --- export ----------------------------------------------------------------------
    def save_html(self):
        path = filedialog.asksaveasfilename(defaultextension=".html",
                                            initialfile="DARTscope_manual.html",
                                            filetypes=[("HTML files", "*.html")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(mdoc.to_html(self.root_section))
        except OSError as e:
            messagebox.showerror("Manual", f"Could not save: {e}")
            return
        if messagebox.askyesno("Manual", f"Saved to {path}.\nOpen it in the web browser?"):
            webbrowser.open("file:///" + os.path.abspath(path).replace("\\", "/"))

    # --- drawing ---------------------------------------------------------------------
    def _configure_tags(self):
        t = self.text
        t.tag_configure("h1", font=theme.font(18, "bold"), spacing1=6, spacing3=8)
        t.tag_configure("h2", font=theme.font(15, "bold"), spacing1=10, spacing3=6,
                        foreground=theme.LABEL)
        t.tag_configure("h3", font=theme.font(12, "bold"), spacing1=10, spacing3=4,
                        foreground=theme.LABEL)
        t.tag_configure("h4", font=theme.font(10, "bold"), spacing1=8, spacing3=2)
        t.tag_configure("para", spacing3=6)
        t.tag_configure("bold", font=theme.font(10, "bold"))
        t.tag_configure("italic", font=theme.font(10, "italic"))
        t.tag_configure("code", font=theme.mono(10), background=theme.FILL, foreground=theme.INDIGO)
        t.tag_configure("codeblock", font=theme.mono(9), background=theme.FILL,
                        lmargin1=16, lmargin2=16, spacing3=6)
        t.tag_configure("table", font=theme.mono(9), lmargin1=8, lmargin2=8)
        t.tag_configure("tablehead", font=theme.mono(9, "bold"), lmargin1=8, lmargin2=8)
        t.tag_configure("rowend", spacing3=6)          # a gap between table rows
        t.tag_configure("note", background=theme.TINT_BLUE, lmargin1=12, lmargin2=12,
                        rmargin=12, spacing1=4, spacing3=8)
        t.tag_configure("link", foreground=theme.LINK, underline=True)
        t.tag_configure("highlight", background=theme.HIGHLIGHT)
        for depth in range(5):
            t.tag_configure(f"li{depth}", lmargin1=12 + 22 * depth, lmargin2=28 + 22 * depth,
                            spacing3=2)
        t.tag_bind("link", "<Enter>", lambda _e: t.configure(cursor="hand2"))
        t.tag_bind("link", "<Leave>", lambda _e: t.configure(cursor="arrow"))

    def _inline(self, text: str, base_tags=()):
        for kind, s, target in mdoc.inline_tokens(text):
            tags = list(base_tags)
            if kind == "link":
                tag = f"link{len(self._links)}"
                self._links[tag] = target
                self.text.tag_bind(tag, "<Button-1>", lambda _e, tg=target: self._follow(tg))
                tags += ["link", tag]
            elif kind != "text":
                tags.append(kind)
            self.text.insert("end", s, tuple(tags))

    def _follow(self, target: str):
        if target.startswith("#"):
            if not self.show(target[1:]):
                self.status.set(f"no section '{target}'")
        elif target.startswith(("http://", "https://")):
            webbrowser.open(target)

    def _render(self, section):
        self._current = section
        t = self.text
        t.configure(state="normal")
        t.delete("1.0", "end")
        for tag in list(self._links):
            t.tag_delete(tag)
        self._links.clear()
        for b in mdoc.blocks(section.markdown()):
            kind = b[0]
            if kind == "heading":
                level = min(4, max(1, b[1] - (section.level - 2)))   # the section is an h2
                self._inline(b[2], (f"h{level}",))
                t.insert("end", "\n")
            elif kind == "para":
                self._inline(b[1], ("para",))
                t.insert("end", "\n")
            elif kind in ("bullet", "number"):
                depth = min(4, b[1])
                mark = "• " if kind == "bullet" else f"{b[2]}. "
                t.insert("end", mark, (f"li{depth}",))
                self._inline(b[-1], (f"li{depth}",))
                t.insert("end", "\n", (f"li{depth}",))
            elif kind == "code":
                t.insert("end", "\n".join(b[1]) + "\n", ("codeblock",))
            elif kind == "note":
                self._inline(b[1], ("note",))
                t.insert("end", "\n", ("note",))
            elif kind == "table":
                self._table(b[1])
            elif kind == "rule":
                t.insert("end", "\n")
        self._highlight()
        t.configure(state="disabled")
        t.yview_moveto(0.0)

    def _table(self, rows):
        if not rows:
            return
        cells = [[mdoc.plain(c) for c in r] for r in rows]
        n = max(len(r) for r in cells)
        cells = [r + [""] * (n - len(r)) for r in cells]
        # wrap long cells into lines so a wide table stays readable
        width_cap = max(12, 100 // n)
        widths = [min(width_cap, max(len(r[j]) for r in cells)) for j in range(n)]

        def wrap(s, w):
            words, lines, cur = s.split(), [], ""
            for word in words:
                if cur and len(cur) + 1 + len(word) > w:
                    lines.append(cur)
                    cur = word
                else:
                    cur = f"{cur} {word}" if cur else word
            lines.append(cur)
            return lines

        for i, r in enumerate(cells):
            wrapped = [wrap(c, widths[j]) for j, c in enumerate(r)]
            height = max(len(w) for w in wrapped)
            for k in range(height):
                line = " | ".join((w[k] if k < len(w) else "").ljust(widths[j])
                                  for j, w in enumerate(wrapped))
                tags = ("tablehead" if i == 0 else "table",) + (("rowend",) if k == height - 1 else ())
                self.text.insert("end", line.rstrip() + "\n", tags)
            if i == 0:
                self.text.insert("end", "-+-".join("-" * w for w in widths) + "\n", ("table",))
        self.text.insert("end", "\n")

    def _highlight(self):
        t = self.text
        t.tag_remove("highlight", "1.0", "end")
        for word in [w for w in self.query.get().split() if len(w) > 1]:
            start = "1.0"
            while True:
                pos = t.search(word, start, stopindex="end", nocase=True)
                if not pos:
                    break
                end = f"{pos}+{len(word)}c"
                t.tag_add("highlight", pos, end)
                start = end
