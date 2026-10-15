"""
acquisition/configfile.py

A radar configuration from a file, whatever kind: one of the lab's Lua scripts, the
_LogFile.txt of an earlier recording (to record again with the same settings), or a
.json sidecar / saved settings.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import json

from . import lua, studio_log
from .config import RadarConfig


def load_config(path: str) -> tuple[RadarConfig, list[str], str | None]:
    """(config, warnings, source text) from a Lua script, an API log or a JSON file."""
    low = path.lower()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if low.endswith(".lua"):
        s = lua.read_script(text, source=path)
        warnings = list(s.warnings)
        if s.uses_capture_card:
            warnings.append("the script's CaptureCardConfig_* calls set the DCA1000 fields; the "
                            "capture drives the DCA1000 itself and does not run them")
        return s.config, warnings, text
    if low.endswith(".json"):
        d = json.loads(text)
        return RadarConfig.from_dict(d.get("config", d)), [], text
    cfg = studio_log.read_config(path)
    return cfg, ["read from a log: the device set-up (COM port, firmware) is the default"], text
