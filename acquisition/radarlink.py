"""
acquisition/radarlink.py

The 64-bit side of radarlink_host.py: starts the 32-bit helper that holds TI's
RadarLinkDLL.dll, calls its functions and collects the radar's asynchronous events.

The calls and their arguments are the ones mmWave Studio's own controller
(Clients/AR1xController/AR1xController.dll) makes for the lab's ar1.* commands - read
from its IL, see docs/NATIVE_RADAR_NOTES.md.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DLL_DIR = r"C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\Clients\AR1xController"


def find_python32() -> str:
    """
    The 32-bit Python the helper runs under: $DARTSCOPE_PY32 (the python.exe or its
    folder), inside DARTscope.exe (bundled as py32/), next to the exe (py32/), or the
    repository's .py32/. The last candidate when none exists, for the error message.
    """
    cands = []
    env = os.environ.get("DARTSCOPE_PY32")
    if env:
        cands.append(env if env.lower().endswith(".exe") else os.path.join(env, "python.exe"))
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        cands.append(os.path.join(bundle, "py32", "python.exe"))
    if getattr(sys, "frozen", False):
        cands.append(os.path.join(os.path.dirname(sys.executable), "py32", "python.exe"))
    cands.append(os.path.join(REPO, ".py32", "python.exe"))
    return next((c for c in cands if os.path.isfile(c)), cands[-1])


def find_host() -> str:
    """radarlink_host.py: next to this module, or where DARTscope.exe unpacked it."""
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        p = os.path.join(bundle, "acquisition", "radarlink_host.py")
        if os.path.isfile(p):
            return p
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "radarlink_host.py")


PY32 = find_python32()
HOST = find_host()


def native_status(dll_dir: str = DLL_DIR, firmware=None, windows: bool | None = None) -> list[str]:
    """What the radar path without mmWave Studio lacks on this computer; empty when it has all.
    Windows: the 32-bit helper, RadarLinkDLL, the firmware. macOS/Linux: pyftdi, pyserial,
    libusb, the DCA1000 on USB, the firmware (from an mmWave Studio in a Wine bottle)."""
    if windows is None:
        windows = sys.platform.startswith("win")
    if not windows:
        from .config import local_path
        from .mmwavelink import link_status
        out = link_status()
        for fw in firmware or ():
            if not os.path.isfile(local_path(fw)):
                out.append(f"radar firmware missing: {fw} - install mmWave Studio 2.1.1 (it only "
                           f"has to be installed) in a Whisky or CrossOver bottle, or set "
                           f"DARTSCOPE_STUDIO_DIR to its folder")
        return out
    out = []
    if not os.path.isfile(PY32):
        out.append(f"the 32-bit Python for the RadarLinkDLL helper is missing ({PY32}) - unpack "
                   f"python.org's 3.12 embeddable 32-bit zip there or set DARTSCOPE_PY32")
    if not os.path.isfile(HOST):
        out.append(f"radarlink_host.py is missing ({HOST})")
    if not os.path.isfile(os.path.join(dll_dir, "RadarLinkDLL.dll")):
        out.append(f"mmWave Studio 2.1.1 is not installed ({dll_dir}): the radar's library and "
                   f"firmware come from it (it only has to be installed, not running)")
    for fw in firmware or ():
        if not os.path.isfile(fw):
            out.append(f"radar firmware missing: {fw}")
    return out


class RadarLinkError(RuntimeError):
    pass


class RadarLinkClient:
    def __init__(self, dll_dir: str = DLL_DIR, python32: str = PY32, log=None):
        if not os.path.isfile(python32):
            raise RadarLinkError(f"32-bit Python not found at {python32}")
        if not os.path.isfile(os.path.join(dll_dir, "RadarLinkDLL.dll")):
            raise RadarLinkError(f"RadarLinkDLL.dll not found in {dll_dir}")
        self.log = log or (lambda s: None)
        self.info: dict = {}
        # no console window next to a windowed DARTscope
        self.proc = subprocess.Popen([python32, "-u", HOST, dll_dir], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                     bufsize=1, cwd=dll_dir,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.events: queue.Queue = queue.Queue()
        self.event_log: list[dict] = []
        self._replies: dict = {}
        self._cond = threading.Condition()
        self._next = 1
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self._ready = threading.Event()
        if not self._ready.wait(15):
            err = self.proc.stderr.read() if self.proc.poll() is not None else ""
            raise RadarLinkError(f"the RadarLinkDLL helper did not start {err}")

    def _read(self):
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("ready"):
                self.info = msg
                self._ready.set()
            elif "event" in msg:
                ev = dict(msg["event"], time=time.time())
                self.event_log.append(ev)
                self.events.put(ev)
                self.log(f"event msg={ev['msg']} sbid=0x{ev['sbid']:04x}")
            else:
                with self._cond:
                    self._replies[msg.get("id")] = msg
                    self._cond.notify_all()

    def call(self, fn: str, *args, timeout: float = 60.0) -> int:
        """fn(args) in the DLL; args are (type, value) pairs as radarlink_host takes them."""
        with self._cond:
            rid = self._next
            self._next += 1
        self.proc.stdin.write(json.dumps({"id": rid, "fn": fn, "args": list(args)}) + "\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + timeout
        with self._cond:
            while rid not in self._replies:
                left = deadline - time.monotonic()
                if left <= 0 or self.proc.poll() is not None:
                    raise RadarLinkError(f"{fn}: no answer from the helper within {timeout:g} s")
                self._cond.wait(min(left, 0.5))
            msg = self._replies.pop(rid)
        if "error" in msg:
            raise RadarLinkError(f"{fn}: {msg['error']}")
        ret = int(msg["ret"])
        self.log(f"{fn}{tuple(v for _, v in args)} -> {ret}")
        return ret

    def wait_event(self, sbids, timeout: float) -> dict | None:
        """The next event with one of these sub-block ids, or None after timeout."""
        want = {sbids} if isinstance(sbids, int) else set(sbids)
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            try:
                ev = self.events.get(timeout=left)
            except queue.Empty:
                return None
            if ev["sbid"] in want:
                return ev

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"id": 0, "fn": "__exit__"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(5)
            except Exception:
                self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def u32(v):
    return ("u32", int(v))


def i32(v):
    return ("i32", int(v))


def reset_into_sop(link: RadarLinkClient, sop: int = 2, device: int = 1) -> dict:
    """What ar1.SOPControl(sop) does: GPIO port, SOP lines, a reset pulse, ports closed."""
    r = {}
    r["setSystemType"] = link.call("RadarLinkImpl_setSystemType", i32(1))
    r["OpenGenericGpioIf"] = link.call("RadarLinkImpl_OpenGenericGpioIf", u32(device))
    r["SOPControl"] = link.call("RadarLinkImpl_SOPControl", u32(device), i32(sop))
    r["OpenBoardControlIf"] = link.call("RadarLinkImpl_OpenBoardControlIf", u32(device))
    time.sleep(0.5)
    r["FullReset(0)"] = link.call("RadarLinkImpl_FullReset", u32(device), i32(0))
    time.sleep(0.5)
    r["FullReset(1)"] = link.call("RadarLinkImpl_FullReset", u32(device), i32(1))
    r["CloseBoardControlIf"] = link.call("RadarLinkImpl_CloseBoardControlIf", u32(device))
    r["CloseGenericGpioIf"] = link.call("RadarLinkImpl_CloseGenericGpioIf", u32(device))
    return r
