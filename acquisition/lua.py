"""
acquisition/lua.py

The lab's mmWave Studio Lua scripts (C:/Data/Radar/config and one per dataset folder)
read into a RadarConfig, and a RadarConfig written back as the ar1.* commands that
configure the radar.

Reading
    The scripts are plain sequences of ar1.* calls, but not only: config_my.lua wraps
    every call in `if (ar1.X(...) == 0) then ... end` and chooses LVDSLaneConfig in an
    `if (partId == 6843) ... elseif ... end` chain. A regular expression over the text
    would take the last LVDSLaneConfig - the four-lane one meant for the 1243/1443 - so
    the text is walked statement by statement: assignments are remembered, simple
    conditions are evaluated, and only the branch that would run contributes calls.
    Anything the walker cannot evaluate is taken as executed and reported.

    Argument counts are checked per device family (xwr68: ProfileConfig 18, ChirpConfig
    10, FrameConfig 8, ChanNAdcConfig 10; xwr25: 23 / 11 / 7 / 11). A decimal comma -
    config_my.lua has `ProfileConfig(0, 60,25, ...)` - adds arguments, and Lua then reads
    60 and 25 as two values; that is reported with the line instead of being guessed at.

Writing
    radar_commands() gives the commands that configure the radar and nothing that
    touches the DCA1000: the acquisition layer drives the DCA1000 itself and binds its
    UDP ports, which mmWave Studio's CaptureCardConfig_* calls would also try to do.
    Physical values are printed with the fewest decimals that mmWave Studio turns back
    into the same code, so read -> write -> read is exact.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .config import (FREQ_LSB_GHZ, SLOPE_LSB_MHZ_US, TIME_LSB_US, Chirp, ConfigError,
                     Frame, Profile, RadarConfig, _round_code)


class LuaConfigError(ConfigError):
    """A script that cannot be turned into a configuration, with the line it happened on."""


@dataclass
class LuaCall:
    name: str                  # without the "ar1." prefix
    args: list
    line: int
    raw: str = ""


@dataclass
class LuaScript:
    config: RadarConfig
    calls: list[LuaCall]
    warnings: list[str] = field(default_factory=list)
    record_path: str | None = None          # CaptureCardConfig_StartRecord's file, if any
    uses_capture_card: bool = False


# --- tokenizer -----------------------------------------------------------------------------

_KEYWORDS = {"and", "break", "do", "else", "elseif", "end", "false", "for", "function", "if",
             "in", "local", "nil", "not", "or", "repeat", "return", "then", "true", "until",
             "while", "goto"}
_OPS = ("...", "..", "==", "~=", "<=", ">=", "::", "+", "-", "*", "/", "%", "^", "#", "<",
        ">", "=", "(", ")", "{", "}", "[", "]", ";", ":", ",", ".")
_NUM_RE = re.compile(r"0[xX][0-9a-fA-F]+|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'", "a": "\a",
            "b": "\b", "f": "\f", "v": "\v", "0": "\0"}


@dataclass
class _Tok:
    kind: str                  # "name", "kw", "num", "str", "op", "eof"
    value: object
    line: int
    pos: int


def _tokenize(text: str) -> list[_Tok]:
    toks: list[_Tok] = []
    i, n, line = 0, len(text), 1
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if c in " \t\r\f\v":
            i += 1
            continue
        if text.startswith("--", i):
            m = re.match(r"--\[(=*)\[", text[i:])
            if m:
                close = "]" + m.group(1) + "]"
                j = text.find(close, i)
                j = n if j < 0 else j + len(close)
            else:
                j = text.find("\n", i)
                j = n if j < 0 else j
            line += text.count("\n", i, j)
            i = j
            continue
        if c in "\"'":
            j, buf = i + 1, []
            while j < n and text[j] != c:
                if text[j] == "\\" and j + 1 < n:
                    nxt = text[j + 1]
                    # Lenient: an unknown escape keeps its character, as the lab's paths
                    # ("feb26\\\data_test.bin") rely on mmWave Studio accepting it.
                    buf.append(_ESCAPES.get(nxt, nxt))
                    j += 2
                    continue
                if text[j] == "\n":
                    line += 1
                buf.append(text[j])
                j += 1
            toks.append(_Tok("str", "".join(buf), line, i))
            i = j + 1
            continue
        m = re.match(r"\[(=*)\[", text[i:])
        if m:
            close = "]" + m.group(1) + "]"
            j = text.find(close, i)
            j = n if j < 0 else j
            body = text[i + len(m.group(0)):j]
            toks.append(_Tok("str", body, line, i))
            line += body.count("\n")
            i = j + len(close)
            continue
        if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            m = _NUM_RE.match(text, i)
            s = m.group(0)
            val = int(s, 16) if s[:2].lower() == "0x" else (float(s) if any(ch in s for ch in ".eE") else int(s))
            toks.append(_Tok("num", val, line, i))
            i = m.end()
            continue
        if c.isalpha() or c == "_":
            m = re.match(r"[A-Za-z_]\w*", text[i:])
            w = m.group(0)
            toks.append(_Tok("kw" if w in _KEYWORDS else "name", w, line, i))
            i += len(w)
            continue
        for op in _OPS:
            if text.startswith(op, i):
                toks.append(_Tok("op", op, line, i))
                i += len(op)
                break
        else:
            i += 1                                     # a stray character: skip it
    toks.append(_Tok("eof", None, line, n))
    return toks


# --- statement walker ------------------------------------------------------------------------

class _Unknown:
    """A value the walker could not work out."""

    def __repr__(self):
        return "<unknown>"


UNKNOWN = _Unknown()


class _Walker:
    def __init__(self, text: str):
        self.text = text
        self.toks = _tokenize(text)
        self.i = 0
        self.vars: dict[str, object] = {}
        self.calls: list[LuaCall] = []
        self.warnings: list[str] = []
        self.returned = False

    # tokens
    @property
    def tok(self) -> _Tok:
        return self.toks[self.i]

    def at(self, kind, value=None) -> bool:
        t = self.tok
        return t.kind == kind and (value is None or t.value == value)

    def take(self, kind=None, value=None) -> _Tok:
        t = self.tok
        if (kind and t.kind != kind) or (value is not None and t.value != value):
            raise LuaConfigError(f"line {t.line}: expected {value or kind}, found {t.value!r}")
        self.i += 1
        return t

    # statements
    def run(self):
        self.block(executing=True, until=())
        return self

    def block(self, executing: bool, until: tuple):
        while not self.at("eof") and not (self.tok.kind == "kw" and self.tok.value in until):
            self.statement(executing and not self.returned)

    def statement(self, ex: bool):
        t = self.tok
        if t.kind == "op" and t.value == ";":
            self.i += 1
        elif t.kind == "kw" and t.value == "if":
            self.if_stmt(ex)
        elif t.kind == "kw" and t.value == "local":
            self.i += 1
            if self.at("kw", "function"):
                self.skip_function()
            else:
                self.assignment_or_call(ex)
        elif t.kind == "kw" and t.value == "return":
            self.i += 1
            if not (self.tok.kind == "kw" and self.tok.value in ("end", "else", "elseif")) \
                    and not self.at("eof"):
                self.expr(False)
            if ex:
                self.returned = True
        elif t.kind == "kw" and t.value == "function":
            self.skip_function()
        elif t.kind == "kw" and t.value in ("do", "while", "for", "repeat"):
            self.skip_loop(ex)
        elif t.kind == "kw" and t.value == "break":
            self.i += 1
        elif t.kind == "name":
            self.assignment_or_call(ex)
        elif t.kind == "op" and t.value == "::":
            self.i += 1
            self.take("name")
            self.take("op", "::")
        else:
            raise LuaConfigError(f"line {t.line}: cannot read the script at {t.value!r}")

    def assignment_or_call(self, ex: bool):
        start = self.i
        targets = [self.suffixed(ex=False, want_target=True)]
        while self.at("op", ","):
            self.i += 1
            targets.append(self.suffixed(ex=False, want_target=True))
        if self.at("op", "="):
            self.i += 1
            values = [self.expr(ex)]
            while self.at("op", ","):
                self.i += 1
                values.append(self.expr(ex))
            if ex:
                for k, name in enumerate(targets):
                    if isinstance(name, str):
                        self.vars[name] = values[k] if k < len(values) else None
            return
        # a call statement: re-read it executing
        self.i = start
        self.suffixed(ex=ex)

    def if_stmt(self, ex: bool):
        self.take("kw", "if")
        taken = False
        cond = self.expr(ex)
        self.take("kw", "then")
        run = ex and self._truth(cond, self.toks[self.i - 1].line)
        self.block(run, ("elseif", "else", "end"))
        taken = run or taken
        while self.at("kw", "elseif"):
            self.i += 1
            cond = self.expr(ex and not taken)
            self.take("kw", "then")
            run = ex and not taken and self._truth(cond, self.toks[self.i - 1].line)
            self.block(run, ("elseif", "else", "end"))
            taken = taken or run
        if self.at("kw", "else"):
            self.i += 1
            self.block(ex and not taken, ("end",))
        self.take("kw", "end")

    def _truth(self, v, line) -> bool:
        if v is UNKNOWN:
            self.warnings.append(f"line {line}: a condition could not be evaluated; its branch "
                                 f"is taken as run")
            return True
        return v is not None and v is not False

    def _skip_body(self) -> int:
        """Skip to the matching end/until; returns how many ar1 references were passed.

        Blocks open with function, if, do and repeat - while and for open theirs with the
        do of their header - and close with end or until.
        """
        depth, calls = 1, 0
        while depth and not self.at("eof"):
            t = self.take()
            if t.kind == "name" and t.value == "ar1":
                calls += 1
            elif t.kind == "kw" and t.value in ("function", "if", "do", "repeat"):
                depth += 1
            elif t.kind == "kw" and t.value in ("end", "until"):
                depth -= 1
                if depth == 0 and t.value == "until":
                    self.expr(False)
        return calls

    def skip_function(self):
        self.take("kw", "function")
        self._skip_body()

    def skip_loop(self, ex: bool):
        first = self.tok.line
        opener = self.take().value
        calls = 0
        if opener in ("while", "for"):
            while not self.at("kw", "do") and not self.at("eof"):
                calls += self.take().value == "ar1"
            self.take("kw", "do")
        calls += self._skip_body()
        if calls and ex:
            self.warnings.append(f"line {first}: {calls} ar1 call(s) inside a loop were not read")

    # expressions
    _PREC = {"or": 1, "and": 2, "<": 3, ">": 3, "<=": 3, ">=": 3, "~=": 3, "==": 3, "..": 4,
             "+": 5, "-": 5, "*": 6, "/": 6, "%": 6}

    def expr(self, ex: bool, min_prec: int = 1):
        left = self.unary(ex)
        while True:
            t = self.tok
            op = t.value if t.kind in ("op", "kw") else None
            prec = self._PREC.get(op)
            if prec is None or prec < min_prec:
                return left
            self.i += 1
            right = self.expr(ex, prec + (0 if op == ".." else 1))   # .. is right-assoc
            left = self._binop(op, left, right)

    def unary(self, ex: bool):
        if self.at("kw", "not"):
            self.i += 1
            v = self.unary(ex)
            return UNKNOWN if v is UNKNOWN else (v is None or v is False)
        if self.at("op", "-"):
            self.i += 1
            v = self.unary(ex)
            return -v if isinstance(v, (int, float)) and not isinstance(v, bool) else UNKNOWN
        if self.at("op", "#"):
            self.i += 1
            v = self.unary(ex)
            return len(v) if isinstance(v, str) else UNKNOWN
        return self.simple(ex)

    def simple(self, ex: bool):
        t = self.tok
        if t.kind == "num" or t.kind == "str":
            self.i += 1
            return t.value
        if t.kind == "kw" and t.value in ("nil", "true", "false"):
            self.i += 1
            return {"nil": None, "true": True, "false": False}[t.value]
        if t.kind == "op" and t.value == "{":
            depth = 0
            while True:
                tt = self.take()
                if tt.kind == "op" and tt.value == "{":
                    depth += 1
                elif tt.kind == "op" and tt.value == "}":
                    depth -= 1
                    if depth == 0:
                        return UNKNOWN
        if t.kind == "kw" and t.value == "function":
            self.skip_function()
            return UNKNOWN
        return self.suffixed(ex)

    def suffixed(self, ex: bool, want_target: bool = False):
        """name(.name|[expr]|(args)|:name(args))* - records ar1.* calls when executing."""
        t = self.tok
        if t.kind == "op" and t.value == "(":
            self.i += 1
            val = self.expr(ex)
            self.take("op", ")")
            path = None
        else:
            name = self.take("name").value
            path = [name]
            val = self.vars.get(name, UNKNOWN if name not in ("ar1", "RSTD") else None)
        while True:
            if self.at("op", ".") and path is not None:
                self.i += 1
                path.append(self.take("name").value)
                val = UNKNOWN
            elif self.at("op", "["):
                self.i += 1
                self.expr(False)
                self.take("op", "]")
                path, val = None, UNKNOWN
            elif self.at("op", ":"):
                self.i += 1
                self.take("name")
                self.call_args(False)
                path, val = None, UNKNOWN
            elif self.at("op", "(") or self.at("str"):
                line = self.tok.line
                start_pos = self.tok.pos
                args = self.call_args(ex)
                raw = self.text[start_pos:self.toks[self.i - 1].pos + 1]
                if path and len(path) == 2 and path[0] == "ar1":
                    if ex:
                        self.calls.append(LuaCall(path[1], args, line, raw))
                    val = 0                              # ar1.* returns 0 on success
                elif path and len(path) == 2 and path[0] == "math":
                    val = _math_call(path[1], args)      # NUM_FRAMES = math.floor(60 * 1000 / 50)
                else:
                    val = UNKNOWN
                path = None
            else:
                break
        if want_target:
            return path[0] if path and len(path) == 1 else None
        return val

    def call_args(self, ex: bool) -> list:
        if self.at("str"):
            return [self.take().value]
        self.take("op", "(")
        args = []
        if not self.at("op", ")"):
            args.append(self.expr(ex))
            while self.at("op", ","):
                self.i += 1
                args.append(self.expr(ex))
        self.take("op", ")")
        return args

    @staticmethod
    def _binop(op, a, b):
        if op in ("and", "or"):
            if a is UNKNOWN:
                return UNKNOWN
            truthy = a is not None and a is not False
            return (b if truthy else a) if op == "and" else (a if truthy else b)
        if a is UNKNOWN or b is UNKNOWN:
            return UNKNOWN
        try:
            if op == "..":
                fmt = lambda v: (repr(v) if isinstance(v, float) else str(v))  # noqa: E731
                return fmt(a) + fmt(b)
            if op == "==":
                return a == b
            if op == "~=":
                return a != b
            return {"<": lambda: a < b, ">": lambda: a > b, "<=": lambda: a <= b,
                    ">=": lambda: a >= b, "+": lambda: a + b, "-": lambda: a - b,
                    "*": lambda: a * b, "/": lambda: a / b, "%": lambda: a % b}[op]()
        except Exception:
            return UNKNOWN


_MATH = {"floor": math.floor, "ceil": math.ceil, "abs": abs, "min": min, "max": max,
         "sqrt": math.sqrt}


def _math_call(name: str, args: list):
    """Lua's math.floor & co. on numbers; UNKNOWN for anything else."""
    fn = _MATH.get(name)
    if fn is None or not args or any(a is UNKNOWN or isinstance(a, bool)
                                     or not isinstance(a, (int, float)) for a in args):
        return UNKNOWN
    try:
        return fn(*args)
    except (TypeError, ValueError):
        return UNKNOWN


def read_calls(text: str) -> tuple[list[LuaCall], list[str]]:
    """The ar1.* calls a script would make, in order, and anything worth a warning."""
    w = _Walker(text).run()
    return w.calls, w.warnings


# --- calls -> configuration --------------------------------------------------------------------

ARG_COUNTS = {
    "xwr68": {"ProfileConfig": 18, "ChirpConfig": 10, "FrameConfig": 8, "ChanNAdcConfig": 10},
    "xwr25": {"ProfileConfig": 23, "ChirpConfig": 11, "FrameConfig": 7, "ChanNAdcConfig": 11},
}
_IGNORED = {"FullReset", "SOPControl", "Calling_IsConnected", "RfEnable", "RfInit",
            "DisableTestSource", "StartFrame", "StopFrame", "CaptureCardConfig_StopRecord",
            "StartMatlabPostProc", "GetBSSFwVersion", "GetMSSFwVersion", "SelectCaptureDevice",
            "CaptureCardConfig_StartRecord", "PowerOff", "Disconnect"}


def family_for_chip(chip: str) -> str | None:
    c = (chip or "").upper()
    if "2544" in c:
        return "xwr25"
    if any(k in c for k in ("6843", "68XX", "1642", "1843", "16XX", "18XX")):
        return "xwr68"
    return None


def _num(v, what, call) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise LuaConfigError(f"line {call.line}: ar1.{call.name}: {what} is not a number ({v!r})")
    return v


def _int(v, what, call) -> int:
    x = _num(v, what, call)
    if float(x) != int(x):
        raise LuaConfigError(f"line {call.line}: ar1.{call.name}: {what} must be whole ({x})")
    return int(x)


def _bits(values, call) -> int:
    m = 0
    for k, v in enumerate(values):
        if _int(v, f"enable flag {k + 1}", call):
            m |= 1 << k
    return m


def _check_count(call: LuaCall, family: str):
    want = ARG_COUNTS[family].get(call.name)
    if want is None or len(call.args) == want:
        return
    hint = ""
    if len(call.args) > want and re.search(r"\d,\d", call.raw.replace(" ", "")):
        hint = (" - a decimal comma? Lua reads '60,25' as two arguments; write 60.25")
    raise LuaConfigError(f"line {call.line}: ar1.{call.name} takes {want} arguments for the "
                         f"{family} family, the script passes {len(call.args)}{hint}")


def config_from_calls(calls: list[LuaCall], source: str = "") -> LuaScript:
    cfg = RadarConfig(chirps=[], source=source)
    warnings: list[str] = []
    record_path = None
    uses_card = False
    family = None
    profiles_seen: set[int] = set()

    for c in calls:
        if c.name == "SelectChipVersion":
            family = family_for_chip(str(c.args[0])) or family
            break
    if family is None:
        prof = next((c for c in calls if c.name == "ProfileConfig"), None)
        if prof is not None:
            family = next((f for f, n in ARG_COUNTS.items()
                           if n["ProfileConfig"] == len(prof.args)), None)
    if family is None:
        family = "xwr68"
        warnings.append("no SelectChipVersion: the xwr68 argument lists are assumed")
    cfg.device.lua_family = family

    for c in calls:
        a = c.args
        n = c.name
        if n in _IGNORED:
            if n.startswith("CaptureCardConfig") or n == "SelectCaptureDevice":
                uses_card = True
            if n == "CaptureCardConfig_StartRecord" and a and isinstance(a[0], str):
                record_path = a[0]
            continue
        _check_count(c, family)
        if n == "frequencyBandSelection":
            cfg.device.band = str(a[0]).upper()
        elif n == "SelectChipVersion":
            cfg.device.chip = str(a[0])
        elif n == "deviceVariantSelection":
            cfg.device.variant = str(a[0])
        elif n == "Connect":
            cfg.device.com_port = _int(a[0], "COM port", c)
            if len(a) > 1:
                cfg.device.baud_rate = _int(a[1], "baud rate", c)
        elif n in ("DownloadBSSFw", "DownloadMSSFw"):
            path = a[0] if a and isinstance(a[0], str) else None
            if path is None:
                warnings.append(f"line {c.line}: the firmware path of ar1.{n} could not be "
                                f"worked out; the default one is used")
                continue
            if n == "DownloadBSSFw":
                cfg.device.bss_firmware = path
            else:
                cfg.device.mss_firmware = path
            m = re.match(r"(.*?)[\\/]+rf_eval_firmware", path, re.IGNORECASE)
            if m:
                cfg.device.studio_dir = m.group(1)
        elif n == "PowerOn":
            cfg.device.power_on_arg = _int(a[0], "PowerOn mode", c)
        elif n == "ChanNAdcConfig":
            ntx = 3 if family == "xwr68" else 4
            cfg.channel.tx_mask = _bits(a[0:ntx], c)
            cfg.channel.rx_mask = _bits(a[ntx:ntx + 4], c)
            cfg.channel.adc_bits = _int(a[ntx + 4], "ADC bits", c)
            cfg.channel.adc_fmt = _int(a[ntx + 5], "ADC format", c)
            cfg.channel.iq_swap = _int(a[ntx + 6], "IQ swap", c)
        elif n == "LPModConfig":
            cfg.channel.lp_adc_mode = _int(a[1], "low-power ADC mode", c) if len(a) > 1 else 0
        elif n == "DataPathConfig":
            cfg.lvds.intf_sel = _int(a[0], "interface", c)
            cfg.lvds.transfer_fmt_pkt0 = _int(a[1], "packet 0 format", c)
            cfg.lvds.transfer_fmt_pkt1 = _int(a[2], "packet 1 format", c)
        elif n == "LvdsClkConfig":
            cfg.lvds.lane_clk = _int(a[0], "lane clock", c)
            cfg.lvds.data_rate = _int(a[1], "data rate", c)
        elif n == "LVDSLaneConfig":
            if len(a) != 8:
                raise LuaConfigError(f"line {c.line}: ar1.LVDSLaneConfig takes 8 arguments, "
                                     f"the script passes {len(a)}")
            cfg.lvds.lane_fmt_map = _int(a[0], "lane format", c)
            cfg.lvds.lane_mask = _bits(a[1:5], c)
            cfg.lvds.msb_first = _int(a[5], "MSB first", c)
            cfg.lvds.packet_header = _int(a[6], "packet header", c)
            cfg.lvds.crc = _int(a[7], "CRC", c)
        elif n == "ProfileConfig":
            cfg.profile = _profile(c, family, cfg.device.band, warnings)
            profiles_seen.add(cfg.profile.profile_id)
        elif n == "ChirpConfig":
            ch = _chirp(c, family, cfg.device.band)
            key = (ch.start_idx, ch.end_idx, ch.profile_id)
            cfg.chirps = [x for x in cfg.chirps
                          if (x.start_idx, x.end_idx, x.profile_id) != key] + [ch]
        elif n == "FrameConfig":
            cfg.frame = _frame(c, family, warnings)
        elif n == "CaptureCardConfig_EthInit":
            uses_card = True
            cfg.dca.host_ip, cfg.dca.dca_ip, cfg.dca.mac = str(a[0]), str(a[1]), str(a[2])
            cfg.dca.config_port = _int(a[3], "config port", c)
            cfg.dca.data_port = _int(a[4], "data port", c)
        elif n == "CaptureCardConfig_Mode":
            uses_card = True
            (cfg.dca.log_mode, cfg.dca.lvds_mode, cfg.dca.transfer_mode, cfg.dca.capture_mode,
             cfg.dca.data_format) = (_int(v, "mode", c) for v in a[:5])
            if len(a) > 5:
                cfg.dca.timer_s = _int(a[5], "timer", c)
        elif n == "CaptureCardConfig_PacketDelay":
            uses_card = True
            cfg.dca.packet_delay_us = _int(a[0], "packet delay", c)
        else:
            warnings.append(f"line {c.line}: ar1.{n} is not part of the model and was ignored")

    if "6843" not in cfg.device.chip:
        cfg.device.board = cfg.device.chip          # the scripts name the chip, not the board
    if len(profiles_seen) > 1:
        warnings.append(f"profiles {sorted(profiles_seen)} are configured; the loader reads one, "
                        f"the last ProfileConfig (profile {cfg.profile.profile_id}) is used")
    if not cfg.chirps:
        raise LuaConfigError("the script defines no ChirpConfig")
    cfg.chirps.sort(key=lambda ch: (ch.start_idx, ch.end_idx))
    return LuaScript(config=cfg, calls=calls, warnings=warnings, record_path=record_path,
                     uses_capture_card=uses_card)


def _profile(c: LuaCall, family: str, band: str, warnings: list) -> Profile:
    a = c.args
    ntx = 3 if family == "xwr68" else 4
    backoff = [_int(v, "TX backoff", c) for v in a[5:5 + ntx]]
    phase = [_num(v, "TX phase", c) for v in a[5 + ntx:5 + 2 * ntx]]
    k = 5 + 2 * ntx                                  # slope
    if any(phase):
        raise LuaConfigError(f"line {c.line}: a TX phase shift is set; how mmWave Studio packs "
                             f"it into the log is not known here, so it is refused")
    if any(backoff):
        warnings.append(f"line {c.line}: TX power backoff {backoff}: packed 8 bits per TX "
                        f"(inferred, no lab recording used it)")
    hpf1_arg, hpf2_arg, rx_gain = _int(a[k + 4], "HPF1", c), _int(a[k + 5], "HPF2", c), a[k + 6]
    if band not in FREQ_LSB_GHZ:
        raise LuaConfigError(f"line {c.line}: band {band!r} unknown - frequencyBandSelection "
                             f"must come first")
    return Profile.from_physical(
        band, profile_id=_int(a[0], "profile", c), start_ghz=_num(a[1], "start frequency", c),
        idle_us=_num(a[2], "idle time", c), adc_start_us=_num(a[3], "ADC start", c),
        ramp_end_us=_num(a[4], "ramp end", c), slope_mhz_us=_num(a[k], "slope", c),
        tx_start_us=_num(a[k + 1], "TX start", c), num_adc_samples=_int(a[k + 2], "samples", c),
        fs_ksps=_num(a[k + 3], "sample rate", c), rx_gain=_int(rx_gain, "RX gain", c),
        hpf1=hpf1_arg & 0xFF, hpf2=hpf2_arg & 0xFF, tx_backoff=backoff,
        hpf1_lua_flags=hpf1_arg & ~0xFF, hpf2_lua_flags=hpf2_arg & ~0xFF)


def _chirp(c: LuaCall, family: str, band: str) -> Chirp:
    a = c.args
    ntx = 3 if family == "xwr68" else 4
    return Chirp(
        start_idx=_int(a[0], "start index", c), end_idx=_int(a[1], "end index", c),
        profile_id=_int(a[2], "profile", c),
        start_freq_var=_round_code(_num(a[3], "start frequency variation", c) / FREQ_LSB_GHZ[band]),
        freq_slope_var=_round_code(_num(a[4], "slope variation", c) / SLOPE_LSB_MHZ_US[band]),
        idle_time_var=_round_code(_num(a[5], "idle variation", c) / TIME_LSB_US),
        adc_start_time_var=_round_code(_num(a[6], "ADC start variation", c) / TIME_LSB_US),
        tx_mask=_bits(a[7:7 + ntx], c))


def _frame(c: LuaCall, family: str, warnings: list) -> Frame:
    a = c.args
    delay = _num(a[5], "trigger delay", c)
    if delay:
        warnings.append(f"line {c.line}: trigger delay {delay}: kept as given (unverified units)")
    return Frame(chirp_start=_int(a[0], "chirp start", c), chirp_end=_int(a[1], "chirp end", c),
                 num_frames=_int(a[2], "frames", c), num_loops=_int(a[3], "loops", c),
                 periodicity=Frame.ticks(_num(a[4], "periodicity", c)),
                 trigger_delay=int(delay), trigger_select=_int(a[-1], "trigger", c))


def read_script(text: str, source: str = "") -> LuaScript:
    calls, warnings = read_calls(text)
    script = config_from_calls(calls, source=source)
    script.warnings[:0] = warnings
    return script


def read_script_file(path: str) -> LuaScript:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return read_script(f.read(), source=path)


# --- configuration -> commands --------------------------------------------------------------

def _shortest(code: int, lsb: float) -> str:
    """
    A decimal that mmWave Studio rounds back to `code`: three decimals as its GUI shows
    them (69.994, not the equally exact 70), more only when three are not enough.
    """
    for digits in range(3, 13):
        s = f"{code * lsb:.{digits}f}"
        if _round_code(float(s) / lsb) == code:
            return s.rstrip("0").rstrip(".") if "." in s else s
    return repr(code * lsb)


def _us(code: int) -> str:
    s = f"{code * TIME_LSB_US:.2f}".rstrip("0").rstrip(".")
    return s or "0"


def _q(path: str) -> str:
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def radar_commands(cfg: RadarConfig, boot: bool = True) -> list[str]:
    """
    The ar1.* commands that configure the radar for cfg, up to and including FrameConfig.

    boot=True starts from a reset: connect, firmware download, power up, RF init - what
    the lab's scripts do. Nothing here touches the DCA1000.
    """
    d, ch, lv, p, fr = cfg.device, cfg.channel, cfg.lvds, cfg.profile, cfg.frame
    if d.lua_family != "xwr68":
        # AWR2544 scripts read fine, but that device runs under mmWave Studio 3.x with other
        # argument meanings (DataPathConfig(513, ...)) and was never recorded this way.
        raise ConfigError(f"writing commands for the {d.lua_family} family is not supported; "
                          f"only xwr68 (IWR6843, IWR6843AOP)")
    ntx = 3
    band = d.band
    out = []
    if boot:
        bss, mss = d.firmware_paths()
        # the pauses of the lab's current script (C:/ti/MyRadar/VitalSigns_6843AOP.lua)
        out += [
            "ar1.FullReset()",
            "RSTD.Sleep(1000)",
            "ar1.SOPControl(2)",
            "RSTD.Sleep(1000)",
            f"ar1.Connect({d.com_port}, {d.baud_rate}, 1000)",
            f'ar1.frequencyBandSelection("{band}")',
            f'ar1.SelectChipVersion("{d.chip}")',
            f'ar1.deviceVariantSelection("{d.variant}")',
            f"ar1.DownloadBSSFw({_q(bss)})",
            f"ar1.DownloadMSSFw({_q(mss)})",
            f"ar1.PowerOn({d.power_on_arg}, 1000, 0, 0)",
            "ar1.RfEnable()",
        ]
    tx_flags = [(ch.tx_mask >> k) & 1 for k in range(ntx)]
    rx_flags = [(ch.rx_mask >> k) & 1 for k in range(4)]
    out.append("ar1.ChanNAdcConfig(" + ", ".join(map(str, tx_flags + rx_flags +
                                                    [ch.adc_bits, ch.adc_fmt, ch.iq_swap])) + ")")
    out.append(f"ar1.LPModConfig(0, {ch.lp_adc_mode})")
    if boot:
        out += ["ar1.RfInit()", "RSTD.Sleep(1000)"]
    out += [
        f"ar1.DataPathConfig({lv.intf_sel}, {lv.transfer_fmt_pkt0}, {lv.transfer_fmt_pkt1})",
        f"ar1.LvdsClkConfig({lv.lane_clk}, {lv.data_rate})",
        "ar1.LVDSLaneConfig(" + ", ".join(map(str, [lv.lane_fmt_map] +
                                              [(lv.lane_mask >> k) & 1 for k in range(4)] +
                                              [lv.msb_first, lv.packet_header, lv.crc])) + ")",
    ]
    if p.tx_phase_shifter:
        raise ConfigError("a TX phase shift cannot be written back (packing unknown)")
    backoff = [(p.tx_out_power_backoff >> (8 * k)) & 0xFF for k in range(ntx)]
    head = [str(p.profile_id), _shortest(p.start_freq_const, FREQ_LSB_GHZ[band]),
            _us(p.idle_time), _us(p.adc_start_time), _us(p.ramp_end_time)]
    head += [str(b) for b in backoff] + ["0"] * ntx
    tail = [_shortest(p.freq_slope_const, SLOPE_LSB_MHZ_US[band]), _us(p.tx_start_time),
            str(p.num_adc_samples), str(p.dig_out_sample_rate),
            str(p.hpf_corner_freq1 | p.hpf1_lua_flags), str(p.hpf_corner_freq2 | p.hpf2_lua_flags),
            str(p.rx_gain)]
    out.append("ar1.ProfileConfig(" + ", ".join(head + tail) + ")")
    for c in cfg.chirps:
        flags = [(c.tx_mask >> k) & 1 for k in range(ntx)]
        vals = [c.start_idx, c.end_idx, c.profile_id,
                _shortest(c.start_freq_var, FREQ_LSB_GHZ[band]) if c.start_freq_var else 0,
                _shortest(c.freq_slope_var, SLOPE_LSB_MHZ_US[band]) if c.freq_slope_var else 0,
                _us(c.idle_time_var), _us(c.adc_start_time_var)] + flags
        out.append("ar1.ChirpConfig(" + ", ".join(map(str, vals)) + ")")
    out.append("ar1.DisableTestSource(0)")
    period = f"{fr.period_ms:.6f}".rstrip("0").rstrip(".")
    out.append(f"ar1.FrameConfig({fr.chirp_start}, {fr.chirp_end}, {fr.num_frames}, "
               f"{fr.num_loops}, {period}, {fr.trigger_delay}, 0, {fr.trigger_select})")
    return out


START_COMMAND = "ar1.StartFrame()"
STOP_COMMAND = "ar1.StopFrame()"


def radar_script(cfg: RadarConfig, title: str = "") -> str:
    """A Lua script that configures the radar and nothing else, for mmWave Studio's Lua shell."""
    lines = [f"-- {title or 'DARTscope acquisition: radar configuration'}",
             "-- The DCA1000 is driven by DARTscope (python -m acquisition); this script",
             "-- must not call SelectCaptureDevice or CaptureCardConfig_*.", ""]
    return "\n".join(lines + radar_commands(cfg) + ["", "-- " + START_COMMAND + "  (sent by the capture)"]) + "\n"
