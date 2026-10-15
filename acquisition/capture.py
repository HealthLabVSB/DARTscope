"""
acquisition/capture.py

One recording, from configuration to verified files.

    1  check the configuration, the name and the free disk space
    2  DCA1000: CONFIG_FPGA, READ_FPGA_VERSION, CONFIG_PACKET_DATA (mmWave Studio's order)
    3  radar: everything up to FrameConfig (radar.RadarController); with a start gate,
       "ready" and wait for the operator's Start (nothing is written before it). The
       name, folder and notes are settled only then (at_start), so what the operator
       typed while the radar was waiting is what the recording is called
    4  files: <stem>_LogFile.txt, the recorder CSV header, <stem>_radar.lua, the sidecar
    5  receiver listening on the data port, then RECORD_START - the DCA1000 is armed
       before the radar sends its first chirp, as in the lab's scripts
    6  radar start (StartFrame); SensorStart goes into the log with its time
    7  wait until every expected byte is in, the DCA1000 reports the record completed, the
       stream has been idle for a while, the requested duration has passed, or a stop is
       asked for (stop_event)
    8  radar stop, RECORD_STOP, drain the socket, recorder CSV summary, log epilogue
    9  sidecar, then verification with the unchanged loader (verify.py)

Whatever fails, the radar and the DCA1000 are stopped, the files are closed and the
sidecar says what happened. Nothing is deleted: a broken recording is evidence.

The engine knows nothing about Tk. Progress goes to on_event(phase, info), so the command
line prints it and a future GUI tab can show it; the result is a Measurement that opens
through the loader like any recording chosen with Browse (the later "record, then open"
workflow needs nothing more from here).

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import os
import platform
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from . import lua, studio_log
from .config import RadarConfig
from .dca1000 import Dca1000Controller
from .measurement import (Measurement, artifact_paths, choose_stem, disk_free_bytes,
                          write_json)
from .radar import RadarController
from .receiver import UdpReceiver
from .recorder import RawRecorder

VERSION = "1.0"
SCHEMA = "dartscope-acquisition/1"


class CaptureError(RuntimeError):
    pass


class _Cancelled(Exception):
    """stop_event while waiting for the start gate: nothing was recorded."""


@dataclass
class CaptureRequest:
    config: RadarConfig
    folder: str
    name: str
    auto_increment: bool = False
    notes: dict = field(default_factory=dict)      # operator, subject, scene, free text
    duration_s: float | None = None                # for numFrames = 0 (until stopped)
    first_packet_timeout_s: float = 10.0
    idle_timeout_s: float = 2.0
    extra_wait_s: float = 5.0
    verify: bool = True
    bind_ip: str = ""
    source_text: str | None = None                 # the Lua/JSON the config came from
    # When given, the radar and the DCA1000 are configured first ("ready") and the
    # recording starts only once this is set - the operator's Start, after they have
    # taken their place. stop_event while waiting cancels without writing any file.
    start_gate: threading.Event | None = None
    # Called with the request once the start gate has opened, before any file is
    # written: it may change name, folder and notes. A take is often prepared before
    # the operator has typed its name (the GUI prepares the next take right after the
    # last one), so these are taken as they are at Start, not at Prepare.
    at_start: Callable[["CaptureRequest"], None] | None = None


@dataclass
class CaptureResult:
    measurement: Measurement
    ok: bool
    status: str
    stop_reason: str
    stats: dict
    verification: dict | None
    sidecar: dict
    error: str | None = None


def _dartscope_version() -> str | None:
    try:
        from version import __version__            # version.py at the top of the Radar folder
        return __version__
    except ImportError:
        return None


def _utc(t: float | None) -> str | None:
    return None if t is None else datetime.fromtimestamp(t, timezone.utc).isoformat()


class CaptureEngine:
    """
    close_radar=False keeps the radar's connection after the recording (a series of
    recordings: NativeRadar then reconfigures instead of booting again); the owner closes it.
    """

    def __init__(self, radar: RadarController, dca: Dca1000Controller | None = None,
                 on_event=None, stop_event: threading.Event | None = None,
                 close_radar: bool = True):
        self.radar = radar
        self.dca = dca
        self.on_event = on_event
        self.stop_event = stop_event or threading.Event()
        self.close_radar = close_radar

    def _emit(self, phase: str, **info):
        if self.on_event:
            try:
                self.on_event(phase, info)
            except Exception:
                pass

    def run(self, req: CaptureRequest) -> CaptureResult:
        cfg = req.config
        cfg.check()
        from .verify import loader_mismatches
        misread = loader_mismatches(cfg)
        if misread:
            raise CaptureError("the DARTscope loader would not read this recording as configured "
                               "(" + "; ".join(misread) + ") - change the frame count or the "
                               "chirp range")
        folder = os.path.abspath(req.folder)
        os.makedirs(folder, exist_ok=True)
        stem = choose_stem(folder, req.name, req.auto_increment)
        paths = artifact_paths(folder, stem)
        meas = Measurement(folder, stem)
        need = (cfg.total_bytes or 0) * 1.02 + 200e6
        free = disk_free_bytes(folder)
        if free < need:
            raise CaptureError(f"{free / 1e9:.1f} GB free in {folder}, the recording needs "
                               f"{need / 1e9:.1f} GB")
        self._emit("prepare", stem=stem, folder=folder, expected=cfg.expectation())

        dca = self.dca or Dca1000Controller(cfg.dca, bind_ip=req.bind_ip)
        timing: dict = {}
        side = {
            "schema": SCHEMA, "status": "preparing", "stem": stem, "folder": folder,
            "files": {"log": os.path.basename(paths["log"]), "csv": os.path.basename(paths["csv"]),
                      "radar_lua": os.path.basename(paths["radar_lua"]), "bins": []},
            "software": {"acquisition": VERSION, "dartscope": _dartscope_version(),
                         "python": sys.version.split()[0],
                         "platform": platform.platform(), "host": socket.gethostname()},
            "notes": dict(req.notes), "board": cfg.device.board,
            "radar_controller": self.radar.info(),
            "config": cfg.to_dict(), "physical": cfg.physical(), "expected": cfg.expectation(),
            "problems": [str(p) for p in cfg.problems()],
            "source_text": req.source_text, "timing": timing, "dca1000": {}, "stream": {},
            "verification": None, "error": None,
        }
        recorder = receiver = None
        record_started = radar_started = cancelled = False
        stop_reason = ""
        error = None
        try:
            dca.open()
            self._emit("dca1000", message="configuring the DCA1000")
            side["dca1000"]["fpga_version"] = dca.configure()

            self._emit("radar", message=f"configuring the radar ({self.radar.name})")
            self.radar.connect()
            self.radar.configure(cfg)
            side["radar_controller"] = self.radar.info()

            if req.start_gate is not None:
                self._emit("ready", message="radar configured - waiting for Start", stem=stem)
                while not req.start_gate.wait(0.05):
                    if self.stop_event.is_set():
                        cancelled = True
                        break
                if cancelled:
                    raise _Cancelled()
                if req.at_start is not None:
                    folder, stem, paths, meas = self._settle_name(req, cfg, side)

            t_cfg = time.time()
            timing["configured_utc"] = _utc(t_cfg)
            studio_log.write_log(paths["log"], cfg, datetime.fromtimestamp(t_cfg))
            with open(paths["radar_lua"], "x", encoding="utf-8") as f:
                f.write(lua.radar_script(cfg, f"DARTscope acquisition: {stem}"))
            recorder = RawRecorder(folder, stem, lvds_lanes=cfg.dca.container_lanes,
                                   max_file_mb=cfg.dca.max_file_size_mb).open()
            side["status"] = "recording"
            write_json(paths["sidecar"], side)

            receiver = UdpReceiver(recorder, port=cfg.dca.data_port, bind_ip=req.bind_ip).start()
            side["dca1000"]["receive_buffer_bytes"] = receiver.rcvbuf_effective
            t_rec = time.time()
            dca.start_record()
            record_started = True
            timing["record_start_utc"] = _utc(t_rec)
            self._emit("armed", message="DCA1000 recording, starting the radar")

            t_radar = time.time()
            self.radar.start()
            radar_started = True
            timing["radar_start_utc"] = _utc(t_radar)
            studio_log.append_event(paths["log"], studio_log.sensor_start_line(),
                                    datetime.fromtimestamp(t_radar))
            stop_reason = self._wait(req, cfg, dca, receiver, recorder, t_rec)
        except _Cancelled:
            stop_reason = "cancelled before the start"
        except BaseException as e:           # KeyboardInterrupt included: clean up, then raise
            error = e
            stop_reason = stop_reason or f"failed: {type(e).__name__}"
        finally:
            if radar_started:
                try:
                    self.radar.stop()
                except Exception as e:
                    side.setdefault("warnings", []).append(f"radar stop: {e}")
            if record_started:
                try:
                    dca.stop_record()
                except Exception as e:
                    side.setdefault("warnings", []).append(f"DCA1000 RECORD_STOP: {e}")
            if receiver is not None:
                receiver.stop()
                if receiver.error is not None and error is None:
                    error = receiver.error
            stats = recorder.close().to_dict() if recorder is not None else {}
            timing["stopped_utc"] = _utc(time.time())
            timing["stop_reason"] = stop_reason
            if stats.get("first_packet_ns"):
                timing["first_packet_utc"] = _utc(stats["first_packet_ns"] / 1e9)
                timing["first_packet_ns"] = stats["first_packet_ns"]
                timing["last_packet_ns"] = stats["last_packet_ns"]
            side["dca1000"]["async_events"] = [
                {"utc": _utc(ev.time), "status": ev.status, "names": ev.names}
                for ev in dca.async_events]
            side["dca1000"]["commands_sent"] = [p.hex(" ") for p in dca.sent]
            if self.dca is None:
                dca.close()
            if self.close_radar or error is not None:
                try:
                    self.radar.close()
                except Exception:
                    pass
            side["stream"] = stats
            side["files"]["bins"] = [os.path.basename(f["path"]) for f in stats.get("files", [])]
            if os.path.isfile(paths["log"]) and stats.get("first_seq") is not None:
                end = datetime.now()
                studio_log.append_event(paths["log"], studio_log.sensor_stop_line(), end)
                studio_log.append_event(paths["log"], studio_log.files_and_frames_line(
                    len(stats.get("files", [])), cfg.frame.num_frames), end)

        if cancelled:
            self._emit("done", status="cancelled", stop_reason=stop_reason)
            return CaptureResult(measurement=meas, ok=False, status="cancelled",
                                 stop_reason=stop_reason, stats={}, verification=None,
                                 sidecar=side, error=None)
        verification = None
        if error is None and req.verify and stats.get("bytes_written"):
            self._emit("verify", message="opening the recording with the DARTscope loader")
            from .verify import verify
            rep = verify(meas, cfg, recorder_stats=stats)
            verification = rep.to_dict()
            side["verification"] = verification
        if error is not None:
            status = "failed"
            side["error"] = f"{type(error).__name__}: {error}"
        elif not stats.get("bytes_written"):
            status = "failed"
            error = CaptureError("no data arrived from the DCA1000")
            side["error"] = str(error)
        elif verification is not None and not verification["ok"]:
            status = "incomplete"
        else:
            status = "complete"
        side["status"] = status
        if os.path.isdir(folder) and (os.path.exists(paths["sidecar"]) or recorder is not None):
            write_json(paths["sidecar"], side)
        self._emit("done", status=status, stop_reason=stop_reason)
        if isinstance(error, KeyboardInterrupt):
            raise error
        return CaptureResult(measurement=meas, ok=(status == "complete"), status=status,
                             stop_reason=stop_reason, stats=stats, verification=verification,
                             sidecar=side, error=side.get("error"))

    def _settle_name(self, req: CaptureRequest, cfg: RadarConfig, side: dict):
        """Apply req.at_start and choose the stem again; nothing has been written yet."""
        req.at_start(req)
        folder = os.path.abspath(req.folder)
        os.makedirs(folder, exist_ok=True)
        stem = choose_stem(folder, req.name, req.auto_increment)
        need = (cfg.total_bytes or 0) * 1.02 + 200e6
        free = disk_free_bytes(folder)
        if free < need:
            raise CaptureError(f"{free / 1e9:.1f} GB free in {folder}, the recording needs "
                               f"{need / 1e9:.1f} GB")
        paths = artifact_paths(folder, stem)
        side.update(stem=stem, folder=folder, notes=dict(req.notes))
        side["files"].update(log=os.path.basename(paths["log"]),
                             csv=os.path.basename(paths["csv"]),
                             radar_lua=os.path.basename(paths["radar_lua"]))
        self._emit("named", stem=stem, folder=folder)
        return folder, stem, paths, Measurement(folder, stem)

    def _wait(self, req: CaptureRequest, cfg: RadarConfig, dca: Dca1000Controller,
              receiver: UdpReceiver, recorder: RawRecorder, t_record: float) -> str:
        total = cfg.total_bytes
        planned = cfg.duration_s or req.duration_s
        # between frames the stream is silent for up to a frame period; only a longer
        # silence means the radar has finished
        idle_limit = max(req.idle_timeout_s, 2.0 * cfg.frame.period_ms / 1e3 + 0.5)
        t0 = time.monotonic()
        last_emit = 0.0
        while True:
            time.sleep(0.05)
            if receiver.error is not None:
                raise receiver.error
            now = time.monotonic()
            if self.stop_event.is_set():
                return "stop requested"
            first = receiver.first_rx_monotonic
            if first is None:
                if now - t0 > req.first_packet_timeout_s:
                    raise CaptureError(
                        f"no data from the DCA1000 {req.first_packet_timeout_s:g} s after the radar "
                        f"was started - is the radar running (FrameConfig, StartFrame), are the "
                        f"LVDS lanes set as in the DCA1000 mode, is port {cfg.dca.data_port} "
                        f"blocked by the firewall?")
                continue
            if now - last_emit > 0.5:
                last_emit = now
                st = recorder.stats
                self._emit("recording", bytes=recorder.stream_bytes, expected=total,
                           packets=st.received, zero_filled=st.zero_filled_packets,
                           elapsed_s=now - first)
            if total and recorder.stream_bytes >= total:
                return "all expected bytes received"
            # The FPGA reports 'record completed' once the LVDS data stop (1.9 s after the last
            # packet in the lab's Wireshark capture). Trusted for the end of a finite recording
            # only when at most its last frame is missing - a pause between long frames must
            # not end it.
            if dca.completed_since(t_record) is not None and (
                    not total or recorder.stream_bytes >= total - cfg.bytes_per_frame):
                return "DCA1000 reported the record completed"
            idle = receiver.idle_s() or 0.0
            if total and idle > idle_limit:
                return f"stream idle for {idle:.1f} s"
            if not total and req.duration_s and now - first >= req.duration_s:
                return f"duration {req.duration_s:g} s reached"
            if planned and now - first > planned + req.extra_wait_s:
                return "planned duration exceeded"
