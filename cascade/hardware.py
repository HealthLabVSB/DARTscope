"""
cascade/hardware.py

The cascade's own hardware: its XDS110 (JTAG to the AM273x) through TI's scripting
debugger (from UniFlash or CCS), and what the firmware reports back in gUserCfg.

    jtag("start", firmware, cfg.json)   CPU reset, load, write gUserCfg, run (jtag_cfg.js)
    jtag("stop", hex(&gUserCfg))        halt, read the status block and the test log, reset
    jtag("probe")                       connect and disconnect: does the AM273x answer?

The scripting debugger comes with TI UniFlash 8.x or Code Composer Studio 20 or newer
(scripting(); DARTSCOPE_TI_ROOT picks one installation). CCS has it complete. UniFlash
ships it without the ccs_base/cloudagent folder it expects; ccs_root() builds a folder of
links that completes it (ccs_base/* from UniFlash, cloudagent/node and cloudagent/src
from TICloudAgentHostApp) - on Windows without Developer Mode junctions and a hard link
(or a copy) of node.exe instead of symbolic links. UniFlash is tested on macOS, CCS 20.4
on Windows; on Windows CCS is taken first, as it needs no links.

Reads of a running R5F return the debugger's 0xBAD0BAD0 pattern, so the status is read
only by "stop", after the halt. "start" never halts the firmware once it runs: a halt
makes the R5F miss CSI ping-pong interrupts and the chirps of that moment are lost.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIRMWARE = HERE / "firmware" / "am273x_cascade_param.xer5f"
CCXML = HERE / "AM273x_XDS110.ccxml"
JTAG_SCRIPT = HERE / "jtag_cfg.js"
XDS110_IDS = {(0x0451, 0xBEF3), (0x0451, 0xBEF4)}

STATES = {0: "loaded", 1: "configuring", 2: "measuring", 3: "done", 4: "ERROR"}
ERR_STAGES = {0: "-", 1: "settings refused by the firmware's own check", 2: "MMWave_init",
              3: "MMWave_open (front-end boot, calibration)", 4: "profile refused",
              5: "chirp refused", 6: "MMWave_config (profile/chirp/frame refused by the radar)",
              7: "MMWave_start", 8: "MMWave_stop", 9: "LVDS/CBUFF"}
MMWAVE_ERRORS = {1: "EINVAL", 2: "EINIT", 3: "ENOTSUP", 4: "ECHCFG (channels)", 5: "EADCCFG (ADC format)",
                 6: "EPOWERCFG", 7: "ERFINIT (RF init/calibration)", 8: "EPROFILECFG (profile)",
                 9: "ECHIRPCFG (chirp)", 10: "EFRAMECFG (frame)", 11: "EPLLCFG", 12: "ESENSOR",
                 13: "EOS", 14: "EVERSION", 15: "EMSG", 23: "ENOMEM", 25: "EASYNCEVENT",
                 26: "ECALFAIL (calibration)", 31: "EDATAPATHCFG", 33: "EHSICLKCFG", 34: "ECSI2CFG",
                 38: "ERFEN", 40: "EPHASESHIFTCFG"}


class CascadeHardwareError(RuntimeError):
    pass


# --- TI's scripting debugger (UniFlash or CCS) -----------------------------------------------

UNIFLASH_GLOBS = ("/Applications/ti/uniflash_*", "~/ti/uniflash_*", "/opt/ti/uniflash_*", "C:/ti/uniflash_*")
CCS_GLOBS = ("/Applications/ti/ccs*/ccs", "~/ti/ccs*/ccs", "/opt/ti/ccs*/ccs", "C:/ti/ccs*/ccs")
TI_ROOT_ENV = "DARTSCOPE_TI_ROOT"      # one installation: a UniFlash folder or CCS's (ccs) folder
NOT_INSTALLED = ("Neither TI UniFlash nor Code Composer Studio 20+ is installed (they provide the JTAG "
                 "debugger): https://www.ti.com/tool/UNIFLASH")


def _win(plat: str | None = None) -> bool:
    return (plat or sys.platform).startswith("win")


def _os_dir(plat: str | None = None) -> str:
    plat = plat or sys.platform
    return "osx" if plat == "darwin" else ("win" if _win(plat) else "linux")


@dataclass(frozen=True)
class Scripting:
    """A TI installation whose scripting debugger runs jtag_cfg.js."""
    kind: str                  # "UniFlash" or "CCS"
    root: Path                 # uniflash_<version>, or CCS's ccs folder
    node: Path
    launcher: Path

    @property
    def name(self) -> str:
        """uniflash_8.7.0, ccs2040"""
        return self.root.parent.name if self.kind == "CCS" else self.root.name

    def ccs_root(self) -> Path:
        """The folder whose ccs_base the scripting runs from (initScripting's ccsRoot)."""
        return self.root if self.kind == "CCS" else ccs_root(self.root)


def _version(s: Scripting) -> tuple:
    """uniflash_8.10.0 after uniflash_8.9.1, ccs2040 after ccs1281"""
    return tuple(int(n) for n in re.findall(r"\d+", s.name))


def _uniflash(root: Path, plat: str | None = None) -> Scripting | None:
    if not (root / "deskdb/content/TICloudAgent" / _os_dir(plat)).is_dir():
        return None
    return Scripting("UniFlash", root, _node(root, plat), _launcher(root, plat))


def _ccs(root: Path, plat: str | None = None) -> Scripting | None:
    """CCS 20 or newer (Theia): scripting/launcher.mjs and a complete ccs_base/cloudagent."""
    agent = root / "ccs_base" / "cloudagent"
    node = agent / ("node.exe" if _win(plat) else "node")
    launcher = root / "scripting" / "launcher.mjs"
    if not (launcher.is_file() and node.is_file() and (agent / "src").is_dir()):
        return None
    return Scripting("CCS", root, node, launcher)


def _installed(globs, make, plat: str | None = None) -> list[Scripting]:
    roots = [Path(r) for g in globs for r in glob.glob(os.path.expanduser(g))]
    return sorted((s for s in (make(r, plat) for r in roots) if s is not None), key=_version)


def scripting(plat: str | None = None) -> Scripting | None:
    """The scripting debugger to use: the installation in DARTSCOPE_TI_ROOT if set, else the
    newest UniFlash, then the newest CCS (on Windows CCS first: it needs no links)."""
    env = os.environ.get(TI_ROOT_ENV)
    if env:
        p = Path(env).expanduser()
        for cand in (p, p / "ccs"):
            s = _uniflash(cand, plat) or _ccs(cand, plat)
            if s is not None:
                return s
        return None
    kinds = [(UNIFLASH_GLOBS, _uniflash), (CCS_GLOBS, _ccs)]
    for globs, make in (kinds[::-1] if _win(plat) else kinds):
        found = _installed(globs, make, plat)
        if found:
            return found[-1]
    return None


def uniflash_root() -> Path | None:
    """The newest UniFlash installation (its scripting debugger and node)."""
    found = _installed(UNIFLASH_GLOBS, _uniflash)
    return found[-1].root if found else None


def _node(root: Path, plat: str | None = None) -> Path:
    return root / "TICloudAgentHostApp" / ("node.exe" if _win(plat) else "node")


def _launcher(root: Path, plat: str | None = None) -> Path:
    return root / "deskdb/content/TICloudAgent" / _os_dir(plat) / "scripting" / "launcher.mjs"


def _link(target: Path, link: Path) -> None:
    """A symbolic link; on Windows without the right to make one (Developer Mode off) a
    junction for a folder and a hard link - or a copy - for a file."""
    try:
        os.symlink(target, link, target_is_directory=target.is_dir())
        return
    except OSError:
        if not _win():
            raise
    if target.is_dir():
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        try:
            os.link(target, link)
        except OSError:
            shutil.copy2(target, link)


def ccs_root(root: Path) -> Path:
    """A folder whose ccs_base is UniFlash's plus the cloudagent the scripting expects."""
    base = Path(os.environ.get("DARTSCOPE_CACHE", Path.home() / ".cache" / "dartscope")) / f"ccsroot_{root.name}"
    ccs_base = base / "ccs_base"
    agent = ccs_base / "cloudagent"
    node = agent / _node(root).name
    if (agent / "src").exists() and node.exists():
        return base
    agent.mkdir(parents=True, exist_ok=True)
    src_base = root / "deskdb/content/TICloudAgent" / _os_dir() / "ccs_base"
    for entry in src_base.iterdir():
        link = ccs_base / entry.name
        if not os.path.lexists(link):
            _link(entry, link)
    for link, target in ((node, _node(root)), (agent / "src", root / "TICloudAgentHostApp" / "src")):
        if not os.path.lexists(link):
            _link(target, link)
    return base


def jtag(mode: str, *args: str, timeout: float = 120) -> dict:
    """Run jtag_cfg.js; its 'RESULT {json}' line, or {'ok': False, 'error': ...}."""
    ti = scripting()
    if ti is None:
        return {"ok": False, "error": NOT_INSTALLED}
    try:
        ccs = ti.ccs_root()
    except OSError as e:
        return {"ok": False, "error": f"cannot prepare {ti.kind}'s scripting folder: {e}"}
    cmd = [str(ti.node), str(ti.launcher), str(JTAG_SCRIPT), mode, *args]
    env = {**os.environ, "CASCADE_CCS_ROOT": str(ccs)}
    extra = {"creationflags": subprocess.CREATE_NO_WINDOW} if _win() else {}   # no console window
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, cwd=HERE, env=env, **extra)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"JTAG ({mode}) did not answer within {timeout:.0f} s"}
    except OSError as e:
        return {"ok": False, "error": f"cannot run {ti.kind}'s node: {e}"}
    for line in p.stdout.splitlines()[::-1]:
        if line.startswith("RESULT "):
            try:
                return json.loads(line[7:])
            except ValueError:
                break
    tail = (p.stderr or p.stdout).strip().splitlines()
    return {"ok": False, "error": tail[-1] if tail else "no output from jtag_cfg.js"}


# --- the firmware --------------------------------------------------------------------------

def elf_symbol(path: Path, name: str) -> int:
    """Address of a symbol in a little-endian ELF32 file (the TI ARM linker's .xer5f)."""
    data = Path(path).read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 1 or data[5] != 1:
        raise ValueError(f"{path} is not a little-endian ELF32 file")
    shoff, = struct.unpack_from("<I", data, 0x20)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x2E)
    sections = [struct.unpack_from("<IIIIIIIIII", data, shoff + i * shentsize) for i in range(shnum)]
    want = name.encode()
    for sh in sections:
        if sh[1] != 2:                                   # SHT_SYMTAB
            continue
        strtab = sections[sh[6]]
        stroff = strtab[4]
        for off in range(sh[4], sh[4] + sh[5], sh[9] or 16):
            st_name, st_value = struct.unpack_from("<II", data, off)
            end = data.index(b"\0", stroff + st_name)
            if data[stroff + st_name:end] == want:
                return st_value
    raise KeyError(f"{name} not in {path}")


def gusercfg_address(firmware: Path = FIRMWARE) -> int:
    return elf_symbol(firmware, "gUserCfg")


def xds110_present(reader=None) -> bool:
    """The cascade's XDS110 (also on other TI EVMs - this checks presence, not the board):
    its COM ports, from the registry on Windows and from pyserial elsewhere (acquisition.ports)."""
    try:
        from acquisition import ports
        return any((p.vid, p.pid) in XDS110_IDS for p in ports.detect(reader).ports)
    except Exception:
        return False


def decode_mmwave_error(code: int) -> str:
    """MMWave_encodeError: [31:16] mmWave error (base -3100), [15:2] the subsystem's
    (mmWaveLink / front-end) error, [1:0] level."""
    u = code & 0xFFFFFFFF
    mmw = (u >> 16) - 0x10000 if (u >> 16) & 0x8000 else (u >> 16)
    sub = (u >> 2) & 0x3FFF
    sub = sub - 0x4000 if sub & 0x2000 else sub
    name = MMWAVE_ERRORS.get(-3100 - mmw, str(mmw)) if mmw < -3100 else str(mmw)
    return f"{name}, front-end error {sub}"


def describe_status(st: dict | None) -> str:
    if not st:
        return "?"
    txt = f"{STATES.get(st['state'], st['state'])}, {st['framesDone']} frames"
    if st["state"] == 4:
        code = st["errCode"]
        detail = decode_mmwave_error(code) if code < -0xFFFF else f"code {code}"
        txt += f" - {ERR_STAGES.get(st['errStage'], st['errStage'])}: {detail}"
    return txt
