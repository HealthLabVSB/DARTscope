"""
cascade/recording.py

One take with the cascade:

    1. the DCA1000 armed: CONFIG_FPGA (raw, 4 LVDS lanes, 16 bit, Ethernet), the packet gap
       of params.packet_delay_us, RECORD_START - acquisition.dca1000, the data datagrams
       recorded by acquisition.receiver/recorder into <name>_Raw_<n>.bin and
       <name>_Raw_LogFile.csv exactly as for the single-chip radar
    2. the firmware loaded over JTAG with the settings in gUserCfg, and run
    3. recorded until the expected bytes are in (or Stop, or the stream ends early)
    4. RECORD_STOP, the firmware halted: its status (frames done, or where it failed)

<name>_cascade.json describes the take: the settings, what they give, the data layout
([frame][loop][chirp in loop][RX 0..7][sample], int16 I/Q), the record statistics and the
firmware's status. If no data arrive, the firmware's status says why (e.g. the radar
refused the profile) and the take is removed.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from acquisition.config import Dca1000Config
from acquisition.dca1000 import Dca1000Controller, Dca1000Error
from acquisition.receiver import UdpReceiver
from acquisition.recorder import RawRecorder, csv_path

from . import hardware
from .params import NUM_RX, CascadeParams

NO_DATA_S = 12.0           # configuring both front-ends takes ~2-5 s after the JTAG start
IDLE_END_S = 3.0           # the stream stopped before the expected bytes


class CascadeRecordingError(RuntimeError):
    pass


def sidecar_path(folder, prefix) -> Path:
    return Path(folder) / f"{prefix}_cascade.json"


def dca1000_config(packet_delay_us: int) -> Dca1000Config:
    return Dca1000Config(lvds_mode=1, timer_s=30, packet_delay_us=int(packet_delay_us))


def check_dca1000(timeout_s: float = 1.5) -> str | None:
    """The DCA1000's FPGA version, or None when it does not answer (or the port is held)."""
    try:
        with Dca1000Controller(dca1000_config(50), timeout_s=timeout_s) as dca:
            return dca.read_fpga_version()
    except (Dca1000Error, OSError):
        return None


@dataclass
class CascadeRecording:
    params: CascadeParams
    folder: Path
    prefix: str
    notes: str = ""
    firmware: Path = hardware.FIRMWARE
    calibration: dict | None = None      # the board's calibration, kept with the take
    stop_event: threading.Event = field(default_factory=threading.Event)

    @property
    def first_part(self) -> Path:
        return Path(self.folder) / f"{self.prefix}_Raw_0.bin"

    @property
    def sidecar(self) -> Path:
        return sidecar_path(self.folder, self.prefix)

    def run(self, progress=lambda kind, **kw: None) -> dict:
        """progress(kind, **kw) with kind 'log' (msg), 'phase' (msg) or 'bytes' (done,
        total, lost). Returns the sidecar's content; raises CascadeRecordingError."""
        rep = self.params.check(Path(self.folder))
        if rep["errors"]:
            raise CascadeRecordingError("; ".join(rep["errors"]))
        if self.first_part.exists() or self.sidecar.exists():
            raise CascadeRecordingError(f"{self.prefix} already exists in {self.folder}")
        expected = rep["bytes_total"]
        log = lambda m: progress("log", msg=m)

        progress("phase", msg="DCA1000")
        dca = Dca1000Controller(dca1000_config(rep["packet_delay_us"]), timeout_s=2.0)
        try:
            dca.open()
            version = dca.configure()
        except (Dca1000Error, OSError) as e:
            dca.close()
            raise CascadeRecordingError(f"DCA1000: {e}")
        recorder = RawRecorder(str(self.folder), self.prefix, lvds_lanes=4).open()
        receiver = UdpReceiver(recorder).start()
        dca.start_record()
        t_armed = time.time()
        log(f"DCA1000 {version} recording (packet gap {rep['packet_delay_us']} µs), starting the firmware ...")

        meta = {"kind": "AWR2243-2X-CAS-EVM cascade", "prefix": self.prefix, "notes": self.notes,
                "params": self.params.to_dict(), "derived": rep["derived"], "warnings": rep["warnings"],
                "layout": {"dtype": "int16 I/Q little-endian", "num_samples": int(self.params.num_samples),
                           "num_rx": NUM_RX, "rx_order": "RX0-3 primary AWR2243, RX4-7 secondary",
                           "chirps_per_frame": rep["chirps_per_frame"],
                           "chirps_in_loop": len(self.params.chirps), "num_loops": int(self.params.num_loops),
                           "tx_chirps": self.params.chirps,
                           "index": "[frame][loop][chirp in loop][rx][sample][I,Q]",
                           "bytes_per_frame": rep["bytes_per_frame"]},
                "radar": {"range_res_m": rep["range_res_m"], "slope_mhz_us": rep["slope_mhz_us"],
                          "sample_rate_ksps": self.params.sample_rate_ksps, "wavelength_m": rep["wavelength_m"],
                          "chirp_time_us": rep["chirp_time_us"], "frame_rate_hz": rep["frame_rate_hz"],
                          "num_frames": self.params.num_frames},
                "dca1000": {"fpga": version, "packet_delay_us": rep["packet_delay_us"], "lvds_lanes": 4},
                "firmware": self.firmware.name, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        if self.calibration:
            # the calibration it was recorded with: a later one must not change this take silently
            meta["calibration"] = self.calibration
        status, error = None, None
        try:
            progress("phase", msg="Firmware")
            fd, cfg_file = tempfile.mkstemp(suffix=".json", prefix="cascade_cfg_")
            with os.fdopen(fd, "w") as f:
                json.dump({"words": self.params.words()}, f)
            res = hardware.jtag("start", str(self.firmware), cfg_file)
            os.unlink(cfg_file)
            if not res.get("ok"):
                raise CascadeRecordingError(f"JTAG: {res.get('error')} - is the cascade's USB connected "
                                            "and its 12 V on?")
            log("Firmware running, waiting for data ...")
            progress("phase", msg="Recording")
            t_start = time.time()
            limit = self.params.duration_s + 30
            while True:
                time.sleep(0.25)
                done = recorder.stream_bytes
                progress("bytes", done=done, total=expected, lost=recorder.stats.zero_filled_packets)
                if receiver.error is not None:
                    raise CascadeRecordingError(f"recording failed: {receiver.error}")
                if done >= expected:
                    break
                if self.stop_event.is_set():
                    log("Stopped by the user.")
                    break
                idle = receiver.idle_s()
                if idle is not None and idle > IDLE_END_S:
                    log("The data stopped before all frames were in.")
                    break
                if idle is None and time.time() - t_start > NO_DATA_S:
                    raise CascadeRecordingError("NODATA")
                if time.time() - t_start > limit:
                    log("Time limit of the take reached.")
                    break
        except CascadeRecordingError as e:
            error = str(e)
        finally:
            try:
                dca.stop_record()
            except (Dca1000Error, OSError):
                pass
            receiver.stop()
            stats = recorder.close()
            dca.close()
            progress("phase", msg="Stopping")
            fin = hardware.jtag("stop", hex(hardware.gusercfg_address(self.firmware)))
            if fin.get("ok"):
                status = fin.get("status")
                log(f"Firmware stopped: {hardware.describe_status(status)}")
            else:
                log(f"Could not stop the firmware: {fin.get('error')}")
            if error == "NODATA":
                if status and status.get("state") == 4:
                    error = f"The radar refused the settings: {hardware.describe_status(status)}"
                    lines = [l.strip() for l in (fin.get("log") or "").splitlines() if "Failed" in l or "Error" in l]
                    if lines:
                        error += "\nFirmware log: " + " | ".join(lines)[:500]
                else:
                    error = (f"No data from the DCA1000; firmware: {hardware.describe_status(status)}. "
                             "Is the Samtec cable in and are the DCA1000's switches set?")

        frames = stats.bytes_written // rep["bytes_per_frame"]
        meta["record"] = {"bytes": stats.bytes_written, "expected_bytes": expected, "frames_in_file": frames,
                          "complete": stats.bytes_written == expected,
                          "zero_filled_packets": stats.zero_filled_packets,
                          "zero_filled_bytes": stats.zero_filled_bytes, "received_packets": stats.received,
                          "files": [f["path"] for f in stats.files],
                          "dca_async": [", ".join(ev.names) for ev in dca.async_events],
                          "first_packet_ns": stats.first_packet_ns, "last_packet_ns": stats.last_packet_ns,
                          "firmware_status": status, "error": error,
                          "seconds_armed": round(time.time() - t_armed, 1)}
        if error and stats.bytes_written == 0:
            for f in stats.files:
                Path(f["path"]).unlink(missing_ok=True)
            Path(csv_path(str(self.folder), self.prefix)).unlink(missing_ok=True)
            raise CascadeRecordingError(error)
        self.sidecar.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        if error:
            raise CascadeRecordingError(error)
        return meta
