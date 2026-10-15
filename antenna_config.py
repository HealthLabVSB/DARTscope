"""
antenna_config.py

Antenna (virtual array) configurations for AoA processing.

Why this exists: the antenna layout used to be guessed inside AOA.py from the RX/TX
channel counts, which cannot distinguish e.g. an IWR6843AOP (2D antenna-on-package)
from a non-AOP board that also runs 3 TX x 4 RX. A wrong layout silently produces
wrong angles, so the layout is now an explicit, user-selectable (and user-editable)
configuration.

A configuration describes the physical array in wavelengths (lambda):
    rx_positions : [[x, y], ...]      one entry per physical RX antenna
    tx_offsets   : [[x, y], ...]      offset added to the whole RX set per TX
    tx_masks     : [mask, ...]        TX enable mask each offset belongs to (1, 2, 4, ...)
    element_phase_deg : []            optional per-virtual-element phase correction

The virtual element for chirp t and antenna rx is  rx_positions[rx] + tx_offsets[t],
which is the standard TDM-MIMO construction.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import json
import os
import copy
import numpy as np

CONFIG_FILENAME = "antenna_configs.json"

AUTO_DETECT = "Auto-detect (from log)"

# Which configuration a chip name resolves to when the log does not say more.
#
# The chip name in an mmWave Studio log comes from the select_chip_version dropdown,
# which on an IWR6843AOPEVM still reads plain "IWR6843" - the package is not in the
# name. Guessing a linear array for it is the worst outcome: the AOP's RX antennas sit
# in a 2x2 grid spanning 0.5 lambda in azimuth, not a 1.5 lambda row, so every angle
# comes out computed on an array the board does not have.
#
# Override per capture by selecting a configuration explicitly, per installation by
# adding "chip_defaults" to antenna_configs.json (see load_user_configs).
CHIP_DEFAULTS = {
    "IWR6843AOP": "IWR6843AOP (3TX x 4RX, 2D)",
    "IWR6843": "IWR6843AOP (3TX x 4RX, 2D)",
    "AWR2544": "AWR2544LOPEVM (4TX x 4RX, waveguide)",
}

# Configuration the auto-detection falls back to for a 4 RX x 3 TX capture that looks
# like an antenna-on-package board. Verified against a real IWR6843AOP recording.
AOP_DEFAULT = "IWR6843AOP (3TX x 4RX, 2D)"

# The AWR2243 cascade: not picked automatically - its two devices need a calibration first
CASCADE_CONFIG = "AWR2243-2X-CAS-EVM (6TX x 8RX cascade)"


def _preset(name, rx_positions, tx_offsets, tx_masks, notes="", element_phase_deg=None):
    return {
        "name": name,
        "rx_positions": [list(map(float, p)) for p in rx_positions],
        "tx_offsets": [list(map(float, p)) for p in tx_offsets],
        "tx_masks": [int(m) for m in tx_masks],
        "element_phase_deg": list(element_phase_deg or []),
        "notes": notes,
        "builtin": True,
    }


# --- Built-in presets -------------------------------------------------------
# Positions are in wavelengths. A standard half-wavelength pitch is 0.5.
BUILTIN_CONFIGS = [
    _preset(
        AOP_DEFAULT,
        rx_positions=[[0.5, 0.0], [0.5, 0.5], [0.0, 0.0], [0.0, 0.5]],
        tx_offsets=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        tx_masks=[1, 2, 4],
        element_phase_deg=[180, 0, 180, 0, 180, 0, 180, 0, 180, 0, 180, 0],
        notes="Antenna-on-package: 2x2 RX grid, one TX extends azimuth, one extends "
              "elevation, giving an L-shaped 12-element virtual array. RX1 and RX3 are "
              "taken as rotated 180 deg on the package, so their phase is corrected.\n\n"
              "Checked against a real IWR6843AOP TDM-MIMO capture: the six virtual element "
              "pairs that share an x position measured 149 deg apart on average (i.e. "
              "anti-phase), and with this correction the strongest target at 1.11 m lands "
              "at -4 deg elevation with a 13.0 dB peak. Without it the same target is "
              "reported at +70 deg elevation with a 10.3 dB peak - physically implausible "
              "for a reflector at radar height, and a less coherent combination.",
    ),
    _preset(
        "IWR6843AOP - no phase correction",
        rx_positions=[[0.5, 0.0], [0.5, 0.5], [0.0, 0.0], [0.0, 0.5]],
        tx_offsets=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        tx_masks=[1, 2, 4],
        notes="Same geometry without the RX1/RX3 phase correction. Kept for comparison: "
              "on the capture used to verify the default it pushed the target to +70 deg "
              "elevation. Use it if your board turns out not to have the rotated elements "
              "- check with 'Compare all configurations' against a target at a known "
              "position.",
    ),
    _preset(
        "Azimuth ULA - 2TX x 4RX (8 virtual)",
        rx_positions=[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]],
        tx_offsets=[[0.0, 0.0], [2.0, 0.0]],
        tx_masks=[1, 2],
        notes="Classic linear array (e.g. IWR6843ISK / AWR1843 in 2TX mode): 8 elements "
              "at half-wavelength pitch, azimuth only - no elevation.",
    ),
    _preset(
        "Azimuth + elevation - 3TX x 4RX (1843/ISK)",
        rx_positions=[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]],
        tx_offsets=[[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]],
        tx_masks=[1, 2, 4],
        notes="Linear RX row; TX1/TX2 extend azimuth, TX3 sits one wavelength above and "
              "provides elevation. Typical ISK/ODS style board.",
    ),
    _preset(
        "AWR2544LOPEVM (4TX x 4RX, waveguide)",
        # RX1..RX4 in device order. On the board they read RX3, RX2, RX1, RX4 from left
        # to right, spaced lambda, 5*lambda/2, lambda - the two dimensions printed on
        # Figure 2-13 of the EVM user guide.
        rx_positions=[[3.5, 0.0], [1.0, 0.0], [0.0, 0.0], [4.5, 0.0]],
        # TX1..TX3 sit in a row 3*lambda/2 apart; TX4 repeats that step and is raised
        # by lambda/2, which is the only elevation the board has.
        tx_offsets=[[0.0, 0.0], [1.5, 0.0], [3.0, 0.0], [4.5, 0.5]],
        tx_masks=[1, 2, 4, 8],
        element_phase_deg=[180, 0, 0, 180] * 4,
        notes="AWR2544LOPEVM 3D waveguide antenna, 76-81 GHz, 15 dBi peak. Taken from "
              "the user guide (SPRUJB0): the virtual array of Figure 2-14 is reproduced "
              "exactly by these positions - 12 elements on the lower row at 0, 2, 3, 5, "
              "6, 7, 8, 9, 10, 12, 13, 15 half-wavelengths and 4 on the upper row at 9, "
              "11, 16, 18, giving 9 lambda of azimuth aperture (about 5.6 deg of "
              "beamwidth) and a single lambda/2 step in elevation.\n\n"
              "The user guide states that RX1 and RX4 are 180 degrees out of phase and "
              "must be compensated in post processing, which is what element_phase_deg "
              "does. Positions are in wavelengths, so the preset holds across the band.",
    ),
    _preset(
        CASCADE_CONFIG,
        # RX1..RX4 of the primary AWR2243 (RX0-3 of the data), then of the secondary
        # (RX4-7): half a wavelength apart, the secondary's RX1 8 lambda after the
        # primary's RX4
        rx_positions=[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0],
                      [9.5, 0.0], [10.0, 0.0], [10.5, 0.0], [11.0, 0.0]],
        # TX of the primary (masks 1, 2, 4) and of the secondary (masks 8, 16, 32 -
        # cascade/decode.py numbers them so). On the board each device's TX channels run
        # the other way than the user guide's antenna labels: the guide's TX1, TX3
        # (raised), TX2, TX4, TX5, TX6 at 0, 1, 2, 3.5, 5.5, 7.5 lambda are the masks 4, 2,
        # 1, 32, 16, 8 - measured, see the notes
        tx_offsets=[[2.0, 0.0], [1.0, 0.8], [0.0, 0.0], [7.5, 0.0], [5.5, 0.0], [3.5, 0.0]],
        tx_masks=[1, 2, 4, 8, 16, 32],
        notes="AWR2243-2X-CAS-EVM, two cascaded AWR2243 (6 TX, 8 RX). Positions from the "
              "user guide (SWRU639, Figures 2-3 and 2-4): TX at 0, 2, 3.5, 5.5 and 7.5 "
              "lambda and one raised 0.8 lambda at 1; RX at half-wavelength pitch on each "
              "device. They reproduce Figure 2-4 exactly: 40 azimuth elements on 38 "
              "distinct positions covering 0-18.5 lambda without a gap (3.5 and 13 are "
              "doubled) and a row of 8 0.8 lambda above at 1-2.5 and 10.5-12. The figure's "
              "caption measures the 8 lambda to the secondary's RX4, but only the "
              "secondary's RX1 at 8 lambda after the primary's RX4 gives the 18.5 lambda "
              "aperture (about 2.7 deg; TI measured 3.3 deg between two reflectors at "
              "9.5 m).\n\n"
              "Which chirp mask drives which antenna was measured on 2026-10-11 (a corner "
              "reflector on 8 marked floor positions at 1 and 2 m, 0.25 and 0.75 m to "
              "either side): within each device the TX run the other way than the guide's "
              "labels - masks 4, 2, 1 at 0, 1 (raised) and 2 lambda, masks 32, 16, 8 at "
              "3.5, 5.5 and 7.5. Split into a TX and an RX part, the echo puts the RX where "
              "listed (fitted 0.51, 0.99, 1.52, 9.47, 9.97, 10.44, 10.96 from RX1) and the "
              "TX at 1.0 and 2.0 lambda beside mask 1 on the other side than the guide's "
              "order, the secondary's 1.5 and 3.4 beyond it. With the guide's order a "
              "walking person fitted one point only 0.34-0.45 (the beam smeared over tens "
              "of degrees, people at mirrored angles near the centre); with this one "
              "0.71-0.91, and a person standing on the 8 marks lands on them. Takes with "
              "all six TX on one chirp (mask 63) do not see the TX layout at all.\n\n"
              "Azimuth sign: the reflector and a person on the user's left come out at "
              "positive angles, as on the IWR6843AOP. Elevation: mask 2 is raised, upwards "
              "- the reflector 25 cm above the radar 2 m away (+7.1 deg) turned mask 2 by "
              "+37 deg against the 8 takes at the radar's height (+36 expected), the other "
              "TX by 9 deg at most. The board sees only +-10 deg in elevation (SWRU639): 50 "
              "cm below at 1 and 2 m (-27, -14 deg) the reflector was not seen at all.",
    ),
    _preset(
        "SIMO - 4RX only (no MIMO)",
        rx_positions=[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]],
        tx_offsets=[[0.0, 0.0]],
        tx_masks=[1],
        notes="Single TX; angle resolution from the 4 physical RX antennas only.",
    ),
]


def config_path(base_dir: str | None = None) -> str:
    base = base_dir or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, CONFIG_FILENAME)


def _read_user_file(base_dir: str | None = None) -> dict:
    """
    Raw contents of antenna_configs.json, in either accepted shape.

    A bare list is the original format and stays valid. An object carries the same list
    under "configs" and may add "chip_defaults", so a new board can be taught to the
    auto-detection without editing this file:

        {
          "configs": [ {"name": "AWR2544LOP", "rx_positions": [...], ...} ],
          "chip_defaults": {"AWR2544": "AWR2544LOP"}
        }
    """
    path = config_path(base_dir)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[antenna_config] Could not read {path}: {e}")
        return {}
    if isinstance(data, list):
        return {"configs": data}
    if isinstance(data, dict):
        return data
    print(f"[antenna_config] {path} is neither a list nor an object; ignoring it.")
    return {}


def load_user_configs(base_dir: str | None = None) -> list[dict]:
    """User-defined configurations from disk (never raises)."""
    out = []
    for item in _read_user_file(base_dir).get("configs") or []:
        ok, msg = validate(item)
        if ok:
            item["builtin"] = False
            out.append(item)
        else:
            print(f"[antenna_config] Skipping configuration "
                  f"'{(item or {}).get('name', '?')}': {msg}")
    return out


def chip_defaults(base_dir: str | None = None) -> dict:
    """Chip name -> configuration name, built-ins overridden by the user file."""
    out = {str(k).upper(): str(v) for k, v in CHIP_DEFAULTS.items()}
    user = _read_user_file(base_dir).get("chip_defaults")
    if isinstance(user, dict):
        out.update({str(k).upper(): str(v) for k, v in user.items()})
    return out


def default_config_for_chip(chip_hint: str, base_dir: str | None = None) -> dict | None:
    """
    The configuration a chip name maps to, or None.

    Longest match wins, so "IWR6843AOP" is not shadowed by an "IWR6843" entry.
    """
    hint = str(chip_hint or "").upper()
    if not hint:
        return None
    table = chip_defaults(base_dir)
    for key in sorted(table, key=len, reverse=True):
        if key and key in hint:
            cfg = get_config(table[key], base_dir)
            if cfg is not None:
                return cfg
            print(f"[antenna_config] chip default for '{key}' names "
                  f"'{table[key]}', which does not exist.")
            return None
    return None


def save_user_configs(configs: list[dict], base_dir: str | None = None) -> bool:
    """Persist user configurations; built-ins are never written."""
    path = config_path(base_dir)
    payload = [c for c in configs if not c.get("builtin")]
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)   # atomic, so a crash cannot truncate the file
        return True
    except Exception as e:
        print(f"[antenna_config] Could not write {path}: {e}")
        try:
            os.unlink(tmp)
        except Exception:
            pass
        return False


def all_configs(base_dir: str | None = None) -> list[dict]:
    return [copy.deepcopy(c) for c in BUILTIN_CONFIGS] + load_user_configs(base_dir)


def config_names(base_dir: str | None = None) -> list[str]:
    return [AUTO_DETECT] + [c["name"] for c in all_configs(base_dir)]


def get_config(name: str, base_dir: str | None = None) -> dict | None:
    if not name or name == AUTO_DETECT:
        return None
    for c in all_configs(base_dir):
        if c["name"] == name:
            return c
    return None


def validate(cfg: dict) -> tuple[bool, str]:
    """Return (ok, message). Keeps a malformed config from reaching the beamformer."""
    if not isinstance(cfg, dict):
        return False, "Configuration is not a dictionary."
    if not str(cfg.get("name", "")).strip():
        return False, "Configuration needs a name."
    try:
        rx = np.asarray(cfg["rx_positions"], dtype=float)
        tx = np.asarray(cfg["tx_offsets"], dtype=float)
    except Exception:
        return False, "RX positions / TX offsets must be numeric [x, y] pairs."
    if rx.ndim != 2 or rx.shape[1] != 2 or rx.shape[0] < 1:
        return False, "RX positions must be a non-empty list of [x, y] pairs."
    if tx.ndim != 2 or tx.shape[1] != 2 or tx.shape[0] < 1:
        return False, "TX offsets must be a non-empty list of [x, y] pairs."
    masks = cfg.get("tx_masks") or []
    if len(masks) != tx.shape[0]:
        return False, f"tx_masks has {len(masks)} entries but there are {tx.shape[0]} TX offsets."
    phases = cfg.get("element_phase_deg") or []
    n_virt = rx.shape[0] * tx.shape[0]
    if phases and len(phases) != n_virt:
        return False, f"element_phase_deg has {len(phases)} entries, expected {n_virt}."
    return True, "OK"


def build_geometry(cfg: dict, masks=None, num_virt_rx: int | None = None,
                   num_phys_rx: int | None = None):
    """
    Virtual element positions for a configuration.

    masks: TX enable mask per unique chirp, as parsed from the log. When given, the TX
    blocks are ordered the way the radar actually transmitted them instead of assuming
    the configuration's own order.

    Returns (x, y) in wavelengths, centred on the array centroid.
    """
    rx = np.asarray(cfg["rx_positions"], dtype=float)
    tx = np.asarray(cfg["tx_offsets"], dtype=float)
    cfg_masks = [int(m) for m in (cfg.get("tx_masks") or [])]
    offs_by_mask = {m: tx[i] for i, m in enumerate(cfg_masks) if i < len(tx)}

    n_rx = rx.shape[0]
    if num_phys_rx and num_phys_rx != n_rx:
        print(f"[antenna_config] WARNING: config '{cfg['name']}' has {n_rx} RX but the "
              f"capture has {num_phys_rx}.")
    n_tx_needed = max(1, int(num_virt_rx // n_rx)) if num_virt_rx else tx.shape[0]

    if masks:
        order = [int(m) for m in masks[:n_tx_needed]]
    else:
        order = cfg_masks[:n_tx_needed]
    while len(order) < n_tx_needed:
        order.append(cfg_masks[len(order)] if len(order) < len(cfg_masks) else 0)

    xs, ys = [], []
    for m in order:
        off = offs_by_mask.get(int(m))
        if off is None:
            idx = min(len(tx) - 1, len(xs))
            off = tx[idx]
            print(f"[antenna_config] WARNING: TX mask {m} not in config '{cfg['name']}'; "
                  f"using offset {list(off)}.")
        xs.append(rx[:, 0] + off[0])
        ys.append(rx[:, 1] + off[1])

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if num_virt_rx:
        x, y = x[:num_virt_rx], y[:num_virt_rx]
    return x - np.mean(x), y - np.mean(y)


def element_phase_factors(cfg: dict, num_virt_rx: int) -> np.ndarray | None:
    """Complex per-element correction, or None when the configuration defines none."""
    phases = cfg.get("element_phase_deg") or []
    if not phases:
        return None
    ph = np.asarray(phases, dtype=float)
    if ph.size < num_virt_rx:
        ph = np.concatenate([ph, np.zeros(num_virt_rx - ph.size)])
    return np.exp(1j * np.deg2rad(ph[:num_virt_rx])).astype(np.complex64)


def describe(cfg: dict, masks=None) -> str:
    """Human-readable summary used by the GUI dialog."""
    ok, msg = validate(cfg)
    if not ok:
        return f"Invalid configuration: {msg}"
    n_rx = len(cfg["rx_positions"])
    n_tx = len(cfg["tx_offsets"])
    x, y = build_geometry(cfg, masks=masks)
    lines = [
        f"{cfg['name']}",
        f"{n_rx} RX x {n_tx} TX = {n_rx * n_tx} virtual elements",
        f"azimuth aperture: {x.max() - x.min():.2f} lambda ({len(np.unique(np.round(x, 3)))} distinct x)",
        f"elevation aperture: {y.max() - y.min():.2f} lambda ({len(np.unique(np.round(y, 3)))} distinct y)",
    ]
    if not np.any(np.abs(y - y[0]) > 1e-9):
        lines.append("no vertical spread - elevation is not resolvable")
    if cfg.get("element_phase_deg"):
        lines.append("per-element phase correction is active")
    if cfg.get("notes"):
        lines.append("")
        lines.append(cfg["notes"])
    return "\n".join(lines)
