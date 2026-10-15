"""
acquisition/radar.py

Who configures and starts the radar. The capture engine only needs four things from it:
configure(cfg), start(), stop(), info() - so where the commands come from can change
without touching the recording.

    StudioRemoteRadar  mmWave Studio as a headless radar driver: the ar1.* commands of
                       lua.radar_commands() sent over its RSTD network port (TCP 2777,
                       opened by mmWave Studio at start-up) through TI's
                       RtttNetClientAPI.dll. The radar keeps running mmWave Studio's
                       RF-eval firmware, so the LVDS data are exactly what every
                       recording so far contains. Needs pythonnet.
    ExternalRadar      someone else runs the radar: mmWave Studio's Lua shell with the
                       radar-only script (lua.radar_script), or a person at the GUI.
                       start() asks for the frames to be started and returns.
    NullRadar          no radar, for the DCA1000 simulator and the tests.

And in native.py, NativeRadar: no mmWave Studio at all - the same RadarLinkDLL calls
mmWave Studio makes, from a 32-bit helper process (docs/NATIVE_RADAR_NOTES.md). Flashing
the mmWave SDK demo instead was ruled out: it changes the LVDS layout (TI's own tool
turns reorderEnable on).

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import os
import sys

from . import lua
from .config import RadarConfig


class RadarError(RuntimeError):
    pass


class RadarController:
    name = "radar"

    def connect(self) -> None:
        pass

    def configure(self, cfg: RadarConfig) -> None:
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass

    def info(self) -> dict:
        return {"controller": self.name}


class NullRadar(RadarController):
    """No radar. on_start is called by start() - the simulator's start_stream, typically."""
    name = "none"

    def __init__(self, on_start=None, on_stop=None):
        self.on_start = on_start
        self.on_stop = on_stop
        self.configured: RadarConfig | None = None

    def configure(self, cfg: RadarConfig) -> None:
        self.configured = cfg

    def start(self) -> None:
        if self.on_start:
            self.on_start()

    def stop(self) -> None:
        if self.on_stop:
            self.on_stop()


class ExternalRadar(RadarController):
    """
    The radar is configured and started outside this program.

    prompt(message) is called by start() and must return once the frames are running
    (the command line waits for Enter). Without a prompt, start() returns at once and the
    capture simply waits for data.
    """
    name = "external"

    def __init__(self, prompt=None):
        self.prompt = prompt

    def configure(self, cfg: RadarConfig) -> None:
        self.cfg = cfg

    def start(self) -> None:
        if self.prompt:
            self.prompt("The DCA1000 is recording. Start the frames now (ar1.StartFrame() in "
                        "mmWave Studio's Lua shell, or the Trigger Frame button).")


# --- mmWave Studio over RSTD ---------------------------------------------------------------

RSTD_PORT = 2777
# SendCommand's status, measured against mmWave Studio 2.1.1 through pythonnet: 0 when the
# Lua ran - its return values in the result array - and 20000 when it raised, with the
# Lua error in GetLastError(). (TI's MATLAB example checks for 30000; that is not what
# this call returns.)
RSTD_OK = 0
RSTD_LUA_ERROR = 20000


class PythonNetRstd:
    """
    RtttNetClientAPI.dll from mmWave Studio's Clients folder, loaded through pythonnet.

    mmWave Studio serves one client at a time. A client that connects and closes without
    a command leaves the server waiting on it, and every later client hangs - so nothing
    here "tests" the port by connecting to it (status.tcp_listening looks instead).
    """

    def __init__(self, studio_dir: str):
        self.dll_dir = os.path.join(studio_dir, "mmWaveStudio", "Clients", "RtttNetClientController")
        self.client = None

    def connect(self, host: str, port: int) -> None:
        if not os.path.isfile(os.path.join(self.dll_dir, "RtttNetClientAPI.dll")):
            raise RadarError(f"RtttNetClientAPI.dll not found in {self.dll_dir}")
        try:
            import clr  # pythonnet
        except ImportError as e:
            raise RadarError("talking to mmWave Studio needs pythonnet: pip install pythonnet") from e
        if self.dll_dir not in sys.path:
            sys.path.append(self.dll_dir)
        clr.AddReference("RtttNetClientAPI")
        from RtttNetClientAPI import RtttNetClient  # type: ignore
        rc = RtttNetClient.Init()
        if rc != 0:
            raise RadarError(f"RtttNetClient.Init failed ({rc})")
        rc = RtttNetClient.Connect(host, int(port))
        if rc != 0:
            raise RadarError(f"cannot reach mmWave Studio at {host}:{port} ({rc}) - is it running? "
                             f"Its output window says 'RstdNet: Port {port}: Listening..' when ready")
        self.client = RtttNetClient

    def send(self, command: str, timeout_s: float = 90.0) -> tuple[int, list, str]:
        """
        (status, the Lua chunk's return values, error text).

        SendCommand never returns when mmWave Studio does not take the command - its
        server is still held by an earlier client that did not disconnect cleanly (seen
        after a failed ar1.Connect). A firmware download takes ~10 s; 90 s is a hang.
        """
        import threading
        box = {}

        def call():
            try:
                box["res"] = self.client.SendCommand(command, None)
            except Exception as e:                        # pragma: no cover - .NET errors
                box["exc"] = e

        th = threading.Thread(target=call, daemon=True)
        th.start()
        th.join(timeout_s)
        if th.is_alive():
            raise RadarError(f"mmWave Studio did not answer {command} within {timeout_s:g} s. Its "
                             f"network port is probably held by an earlier connection: restart "
                             f"mmWave Studio (or run RSTD.NetClose() and RSTD.NetStart() in its "
                             f"Lua shell) and try again")
        if "exc" in box:
            raise RadarError(f"{command}: {box['exc']}")
        res = box["res"]
        rc, arr = (res[0], res[1]) if isinstance(res, tuple) else (res, None)
        values = []
        for v in (arr if arr is not None else []):
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                values.append(str(v))
        err = "" if int(rc) == RSTD_OK else str(self.client.GetLastError())
        return int(rc), values, err

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.Disconnect()
            except Exception:
                pass
            self.client = None


class StudioRemoteRadar(RadarController):
    """
    mmWave Studio as the radar driver, over its network port.

    Every command is sent on its own, as `return ar1.X(...)`, so both ways it can fail are
    seen and named: the Lua raising (status 20000, mmWave Studio's error text) and the
    ar1 function returning something other than 0.
    """
    name = "mmwave-studio-remote"

    def __init__(self, host: str = "127.0.0.1", port: int = RSTD_PORT, boot: bool = True,
                 transport=None, studio_dir: str | None = None, on_command=None):
        self.host, self.port, self.boot = host, int(port), boot
        self.transport = transport
        self.studio_dir = studio_dir
        self.on_command = on_command
        self.sent: list[str] = []
        self.warnings: list[str] = []

    def connect(self) -> None:
        if self.transport is None:
            self.transport = PythonNetRstd(self.studio_dir or r"C:\ti\mmwave_studio_02_01_01_00")
        self.transport.connect(self.host, self.port)

    # ar1.FullReset() on its own fails with -7 on the lab's set-up in every session of
    # mmWave Studio's logs (Log0075, 77, 78: "Failed To give full reset, Error Value: -7",
    # the script prints "FullReset failure" and goes on). The reset that works is the one
    # ar1.SOPControl(2) does itself - it opens the GPIO control port, resets, sets SOP 2 -
    # so a failed FullReset is noted, as the lab's scripts do, and SOPControl must pass.
    # ar1.DisableTestSource(0) returns non-zero with "Test Source Already Disabled...!!!"
    # when there is nothing to disable - in the lab's logs, every time (Log0060-0077).
    TOLERATED = {"ar1.FullReset()", "ar1.DisableTestSource(0)"}

    def _run(self, command: str) -> None:
        try:
            self._run_once(command)
        except RadarError as e:
            if command not in self.TOLERATED:
                raise
            self.warnings.append(f"{e} (tolerated, as in the lab's scripts)")

    def _run_once(self, command: str) -> None:
        if self.transport is None:
            self.connect()
        is_ar1 = command.startswith("ar1.")
        if self.on_command:
            self.on_command(command)
        rc, values, err = self.transport.send(f"return {command}" if is_ar1 else command)
        self.sent.append(command)
        if rc != RSTD_OK:
            raise RadarError(f"mmWave Studio could not run {command}: {err or f'status {rc}'}")
        if is_ar1 and values and isinstance(values[0], float) and values[0] != 0:
            raise RadarError(f"{command} returned {values[0]:g} - mmWave Studio's output window "
                             f"says why")

    def configure(self, cfg: RadarConfig) -> None:
        if self.studio_dir is None:
            self.studio_dir = cfg.device.studio_dir
        for command in lua.radar_commands(cfg, boot=self.boot):
            self._run(command)

    def start(self) -> None:
        self._run(lua.START_COMMAND)

    def stop(self) -> None:
        self._run(lua.STOP_COMMAND)

    def close(self) -> None:
        if self.transport is not None:
            self.transport.close()

    def info(self) -> dict:
        return {"controller": self.name, "host": self.host, "port": self.port,
                "boot": self.boot, "commands_sent": len(self.sent), "warnings": list(self.warnings)}
