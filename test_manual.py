"""
Tests for the user manual (docs/MANUAL.md) and its parser (manual.py).

The manual is only useful while it describes the application as it is, so besides the
parser these check the manual against the GUI source: every tab has a section, every
button, checkbox and label is mentioned, every link and every ⓘ target resolves.

Run with:  python -m pytest test_manual.py -q
"""
import os
import re

import pytest

import manual

HERE = os.path.dirname(os.path.abspath(__file__))
MANUAL_PATH = os.path.join(HERE, "docs", "MANUAL.md")
GUI_FILES = ("DCA_processing.py", "hr_gui.py", "ai_vital_gui.py", "gui_helper.py",
             "manual_view.py", "log_view.py", "plot_interactivity.py", "vital_bands.py",
             "fastcheck_gui.py", "moving_hr_gui.py", "measure_gui.py", "cascade_gui.py",
             "multi_vitals_gui.py", "radar_imaging_gui.py")


@pytest.fixture(scope="module")
def text():
    with open(MANUAL_PATH, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def root(text):
    return manual.parse(text)


def _source(name):
    with open(os.path.join(HERE, name), encoding="utf-8-sig") as f:
        return f.read()


# --- the manual against the application ------------------------------------------------

def test_every_analysis_view_has_its_section(root):
    views = re.findall(r'analysis_nb\.add\(\w+, text="([^"]+)"\)', _source("DCA_processing.py"))
    assert len(views) == 15
    chapter = manual.find(root, "Analysis views")
    titles = {c.title for c in chapter.children}
    assert set(views) <= titles, set(views) - titles


def test_every_main_tab_is_covered(root):
    tabs = re.findall(r'main_notebook\.(?:add|insert)\((?:\d+, )?\w+, text="([^"]+)"\)',
                      _source("DCA_processing.py"))
    assert {"Measure", "Decode", "Fastcheck", "Analysis", "Log", "Helper", "Manual"} <= set(tabs)
    for tab in tabs:
        if tab != "Manual":                 # the manual introduces itself
            assert manual.find(root, f"{tab} tab") is not None, tab


def test_every_control_is_mentioned(text):
    """Buttons, checkboxes, labels and frame titles of the GUI all appear in the manual."""
    low = text.lower()
    missing = []
    for name in GUI_FILES:
        pattern = r'(?:Button|Checkbutton|Label|LabelFrame|Menubutton)\([^)]*?text\s*=\s*"([^"{]+)"'
        for label in re.findall(pattern, _source(name)):
            label = label.strip()
            key = label.rstrip(":").strip()
            if len(key) < 3 or len(key) > 45 or key.startswith("("):
                continue                     # separators and explanatory sentences
            if key.lower() not in low:
                missing.append(f"{name}: {label}")
    assert not missing, missing


def test_every_info_icon_opens_a_section(root):
    src = _source("DCA_processing.py")
    calls = re.findall(r"(?<!def )add_card_header_extras\((.*?)\)\s*\n", src, flags=re.S)
    assert len(calls) >= 10
    for call in calls:
        name = re.search(r'"([^"]+)"', call).group(1)
        section = re.search(r'manual_section="([^"]+)"', call)
        target = section.group(1) if section else name
        assert manual.find(root, target) is not None, f"ⓘ of '{name}' opens '{target}'"


def test_internal_links_resolve_exactly(root):
    for s in root.walk():
        for label, target in re.findall(r"\[([^\]]+)\]\((#[^)]+)\)", "\n".join(s.lines)):
            found = manual.find(root, target)
            assert found is not None and found.anchor == target[1:], (s.title, target)


def test_the_quick_start_is_reachable_by_name(root):
    """The Decode tab's 'Quick start (Manual)' button opens this section."""
    assert manual.find(root, "Quick start").title == "Quick start"


def test_the_chapters(root):
    titles = [c.title for c in root.children]
    for expected in ("Getting started", "Decode tab", "Fastcheck tab", "Log tab",
                     "Analysis tab: global settings",
                     "Analysis views", "Workflows", "Command-line tools", "Troubleshooting",
                     "Glossary"):
        assert any(expected in t for t in titles), expected


# --- the parser ----------------------------------------------------------------------------

SAMPLE = """# Title
Intro.

## 1. First chapter
Text with **bold**, *italic*, `code` and a [link](#second).

### Second
- one
  continued
  - nested
- two

1. first
2. second

| A | B |
|---|---|
| a1 | **b1** |

> **Tip:** a note.

```
## not a heading
```

## 2. Other
### Second
"""


def test_tree_and_anchors():
    r = manual.parse(SAMPLE)
    assert r.title == "Title"
    assert [c.title for c in r.children] == ["1. First chapter", "2. Other"]
    assert r.children[0].anchor == "1-first-chapter"
    # a repeated heading gets a numbered anchor, as on GitHub
    assert [c.anchor for c in (r.children[0].children[0], r.children[1].children[0])] == \
        ["second", "second-1"]
    assert "## not a heading" in r.children[0].children[0].markdown()


def test_blocks():
    r = manual.parse(SAMPLE)
    kinds = [b[0] for b in manual.blocks(r.children[0].children[0].markdown())]
    assert kinds == ["heading", "bullet", "bullet", "bullet", "number", "number", "table",
                     "note", "code"]
    b = manual.blocks(r.children[0].children[0].markdown())
    assert b[1] == ("bullet", 0, "one continued") and b[2] == ("bullet", 1, "nested")
    assert b[6] == ("table", [["A", "B"], ["a1", "**b1**"]])


def test_inline_tokens():
    toks = manual.inline_tokens("Press **Run**, see `x`, [Vitals](#vitals), *this* - 2 * 3 * 4")
    kinds = [k for k, _, _ in toks if k != "text"]
    assert kinds == ["bold", "code", "link", "italic"]
    assert manual.plain("a **b** `c`") == "a b c"


def test_search_and_find():
    r = manual.parse(SAMPLE)
    assert [s.title for s in manual.search(r, "nested")] == ["Second"]
    assert manual.search(r, "") == []
    assert manual.find(r, "#1-first-chapter").title == "1. First chapter"
    assert manual.find(r, "other").title == "2. Other"
    assert manual.find(r, "nothing like it") is None


def test_html_is_complete(root):
    page = manual.to_html(root)
    assert page.startswith("<!DOCTYPE html>") and page.rstrip().endswith("</html>")
    for chapter in root.children:
        assert f'href="#{chapter.anchor}"' in page and f'id="{chapter.anchor}"' in page
    assert page.count("<table>") == page.count("</table>") > 0
