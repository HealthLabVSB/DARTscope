"""
manual.py

The user manual, docs/MANUAL.md, as a tree of sections - for the Manual tab, and as a
standalone HTML page to share.

One Markdown file is the only source: GitHub renders it, the Manual tab shows it with
a collapsible table of contents, and "Save as HTML" writes it out for anyone without
the application. Headings make the tree: "##" chapters, "###" sections, "####"
sub-sections. The Markdown used is a small, plain subset - headings, paragraphs,
bullet and numbered lists, **bold**, *italic*, `code`, fenced code blocks, tables,
"> " notes and [links](#anchor) - and this module is all it takes to read it.

Nothing here touches Tk; manual_view.py draws it.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
NUMBER_RE = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
INLINE_RE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\[[^\]]+\]\([^)\s]+\)|(?<![\w*])\*(?!\s)[^*]+?(?<!\s)\*(?![\w*]))")


def slug(title: str) -> str:
    """GitHub-style anchor of a heading: lower case, punctuation dropped, spaces to '-'."""
    s = re.sub(r"[`*_\[\]()]", "", title).strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"\s", "-", s)


@dataclass(eq=False)          # identity: sections go into sets and trees
class Section:
    title: str
    level: int                     # 1 = "#", 2 = "##", ...
    anchor: str = ""
    lines: list = field(default_factory=list)          # body until the next heading
    children: list = field(default_factory=list)
    parent: "Section | None" = field(default=None, repr=False)

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def markdown(self, with_children: bool = True, with_heading: bool = True) -> str:
        """The section as Markdown, with its sub-sections unless told otherwise."""
        out = []
        if with_heading and self.level > 0:
            out.append("#" * self.level + " " + self.title)
        out.extend(self.lines)
        if with_children:
            for c in self.children:
                out.append("")
                out.append(c.markdown())
        return "\n".join(out).strip("\n")

    def own_text(self) -> str:
        return self.title + "\n" + "\n".join(self.lines)

    def path(self) -> list:
        node, out = self, []
        while node is not None and node.level > 0:
            out.append(node.title)
            node = node.parent
        return out[::-1]


def parse(text: str) -> Section:
    """The manual as a tree. The '#' title becomes the root; code fences are skipped."""
    root = Section(title="", level=0)
    stack = [root]
    in_code = False
    seen = {}
    for raw in text.replace("\r\n", "\n").split("\n"):
        if raw.lstrip().startswith("```"):
            in_code = not in_code
            stack[-1].lines.append(raw)
            continue
        m = None if in_code else HEADING_RE.match(raw)
        if not m:
            stack[-1].lines.append(raw)
            continue
        level, title = len(m.group(1)), m.group(2)
        if level == 1 and root.level == 0 and not root.title:
            root.title, root.level, root.anchor = title, 1, slug(title)
            continue
        while stack[-1].level >= level and len(stack) > 1:
            stack.pop()
        base = slug(title)
        n = seen.get(base, 0)
        seen[base] = n + 1
        node = Section(title=title, level=level, anchor=base if n == 0 else f"{base}-{n}",
                       parent=stack[-1])
        stack[-1].children.append(node)
        stack.append(node)
    return root


def find(root: Section, key: str) -> "Section | None":
    """A section by anchor or title - exact first, then the first title containing key."""
    if not key:
        return None
    k = key.strip().lstrip("#").lower()
    for s in root.walk():
        if s.anchor == k or s.title.lower() == k:
            return s
    k_slug = slug(key)
    for s in root.walk():
        if s.anchor == k_slug:
            return s
    for s in root.walk():
        if s.level > 0 and k in s.title.lower():
            return s
    return None


def search(root: Section, query: str) -> list:
    """Sections whose own title or text contains every word of the query."""
    words = [w for w in query.lower().split() if w]
    if not words:
        return []
    return [s for s in root.walk()
            if s.level > 1 and all(w in s.own_text().lower() for w in words)]


# --- inline markup -------------------------------------------------------------------

def inline_tokens(text: str) -> list:
    """[(kind, text, target)] with kind in text / bold / italic / code / link."""
    out, pos = [], 0
    for m in INLINE_RE.finditer(text):
        if m.start() > pos:
            out.append(("text", text[pos:m.start()], None))
        tok = m.group(0)
        if tok.startswith("**"):
            out.append(("bold", tok[2:-2], None))
        elif tok.startswith("`"):
            out.append(("code", tok[1:-1], None))
        elif tok.startswith("["):
            label, target = re.match(r"\[([^\]]+)\]\(([^)\s]+)\)", tok).groups()
            out.append(("link", label, target))
        else:
            out.append(("italic", tok[1:-1], None))
        pos = m.end()
    if pos < len(text):
        out.append(("text", text[pos:], None))
    return out


def plain(text: str) -> str:
    """Inline markup removed - for table cells and search."""
    return "".join(t for _, t, _ in inline_tokens(text))


# --- blocks --------------------------------------------------------------------------

def blocks(markdown: str) -> list:
    """
    Block structure of a piece of the manual:
        ("heading", level, text)   ("para", text)        ("bullet", depth, text)
        ("number", depth, n, text) ("code", [lines])     ("table", [[cells]])
        ("note", text)             ("rule",)
    """
    lines = markdown.replace("\r\n", "\n").split("\n")
    out, i, para = [], 0, []

    def flush():
        if para:
            out.append(("para", " ".join(p.strip() for p in para)))
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            out.append(("code", code))
            i += 1
            continue
        m = HEADING_RE.match(line)
        if m:
            flush()
            out.append(("heading", len(m.group(1)), m.group(2)))
        elif not stripped:
            flush()
        elif stripped in ("---", "***", "___"):
            flush()
            out.append(("rule",))
        elif stripped.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            out.append(("table", rows))
            continue
        elif stripped.startswith(">"):
            flush()
            note = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                note.append(lines[i].strip()[1:].strip())
                i += 1
            out.append(("note", " ".join(n for n in note if n)))
            continue
        else:
            mb, mn = BULLET_RE.match(line), NUMBER_RE.match(line)
            if mb or mn:
                flush()
                indent = len((mb or mn).group(1).replace("\t", "    "))
                item = [mb.group(2) if mb else mn.group(3)]
                # continuation lines: indented further, not a new item or block
                while (i + 1 < len(lines) and lines[i + 1].strip()
                       and not BULLET_RE.match(lines[i + 1]) and not NUMBER_RE.match(lines[i + 1])
                       and len(lines[i + 1]) - len(lines[i + 1].lstrip()) > indent
                       and not lines[i + 1].strip().startswith(("|", ">", "```", "#"))):
                    i += 1
                    item.append(lines[i].strip())
                if mb:
                    out.append(("bullet", indent // 2, " ".join(item)))
                else:
                    out.append(("number", indent // 2, int(mn.group(2)), " ".join(item)))
            else:
                para.append(line)
        i += 1
    flush()
    return out


# --- HTML ------------------------------------------------------------------------------

_CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;
color:#1b1b1b;line-height:1.5}
h1{border-bottom:2px solid #ccc}h2{border-bottom:1px solid #ddd;margin-top:2em}
code{background:#f2f2f2;padding:1px 4px;border-radius:3px;font-family:Consolas,monospace}
pre{background:#f6f6f6;padding:10px;overflow-x:auto}
table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ccc;padding:4px 8px;
vertical-align:top}th{background:#f0f0f0}
blockquote{background:#eef5ff;border-left:4px solid #6a9be8;margin:8px 0;padding:6px 12px}
nav ul{list-style:none;padding-left:1em}nav>ul{padding-left:0}
"""


def _inline_html(text: str) -> str:
    out = []
    for kind, t, target in inline_tokens(text):
        e = html.escape(t)
        if kind == "bold":
            out.append(f"<b>{e}</b>")
        elif kind == "italic":
            out.append(f"<i>{e}</i>")
        elif kind == "code":
            out.append(f"<code>{e}</code>")
        elif kind == "link":
            out.append(f'<a href="{html.escape(target)}">{e}</a>')
        else:
            out.append(e)
    return "".join(out)


def _blocks_html(md: str, anchors: dict) -> str:
    out, lists = [], []

    def close_lists(depth=-1):
        while lists and (depth < 0 or len(lists) > depth + 1):
            out.append(f"</{lists.pop()}>")

    for b in blocks(md):
        kind = b[0]
        if kind in ("bullet", "number"):
            depth = b[1]
            tag = "ul" if kind == "bullet" else "ol"
            close_lists(depth)
            while len(lists) < depth + 1:
                out.append(f"<{tag}>")
                lists.append(tag)
            out.append(f"<li>{_inline_html(b[-1])}</li>")
            continue
        close_lists()
        if kind == "heading":
            level, text = b[1], b[2]
            anchor = anchors.get(text, slug(text))
            out.append(f'<h{level} id="{anchor}">{_inline_html(text)}</h{level}>')
        elif kind == "para":
            out.append(f"<p>{_inline_html(b[1])}</p>")
        elif kind == "code":
            out.append("<pre><code>" + html.escape("\n".join(b[1])) + "</code></pre>")
        elif kind == "table":
            rows = b[1]
            if rows:
                out.append("<table><tr>" + "".join(f"<th>{_inline_html(c)}</th>" for c in rows[0])
                           + "</tr>")
                for r in rows[1:]:
                    out.append("<tr>" + "".join(f"<td>{_inline_html(c)}</td>" for c in r) + "</tr>")
                out.append("</table>")
        elif kind == "note":
            out.append(f"<blockquote>{_inline_html(b[1])}</blockquote>")
        elif kind == "rule":
            out.append("<hr>")
    close_lists()
    return "\n".join(out)


def to_html(root: Section) -> str:
    """The whole manual as one standalone HTML page with a table of contents."""
    anchors = {s.title: s.anchor for s in root.walk() if s.level > 0}

    def toc(node):
        items = [c for c in node.children if c.level <= 3]
        if not items:
            return ""
        return "<ul>" + "".join(
            f'<li><a href="#{c.anchor}">{html.escape(plain(c.title))}</a>{toc(c)}</li>'
            for c in items) + "</ul>"

    body = _blocks_html(root.markdown(), anchors)
    title = html.escape(plain(root.title) or "Manual")
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title>"
            f"<style>{_CSS}</style></head><body><nav><h2>Contents</h2>{toc(root)}</nav>"
            f"{body}</body></html>")
