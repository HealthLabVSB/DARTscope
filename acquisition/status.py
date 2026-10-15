"""
acquisition/status.py

Before a recording: is everything where it has to be? Each check says what is wrong and
what to do, because the usual failures are set-up, not software:

    route          this PC reaches the DCA1000 from the configured host address
                   (192.168.33.30 on the adapter wired to the DCA1000)
    ports          UDP 4096 and 4098 are free (mmWave Studio's DCA1000 connection or a
                   stray DCA1000EVM_CLI_Record holds them otherwise)
    DCA1000        answers, and runs the recording FPGA image (its version)
    mmWave Studio  its RSTD port answers, when it is to configure the radar
    disk           room for the recording

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import socket
from dataclasses import dataclass

from .config import RadarConfig
from .dca1000 import Dca1000Controller, Dca1000Error, local_ip_towards
from .measurement import disk_free_bytes


@dataclass
class Status:
    name: str
    ok: bool
    detail: str

    def __str__(self):
        return f"{'OK  ' if self.ok else 'FAIL'} {self.name}: {self.detail}"


def port_free(port: int, bind_ip: str = "") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.bind((bind_ip, port))
            return True
        except OSError:
            return False


def tcp_listening(port: int) -> bool:
    """
    Whether something listens on a local TCP port - without connecting to it.

    Connecting is not harmless here: mmWave Studio's RSTD server serves one client at a
    time, and a client that connects and closes without sending a command leaves it
    waiting on that client for good - every later client, pythonnet included, then hangs.
    """
    import subprocess
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[1].endswith(f":{port}") and parts[3].upper().startswith("LISTEN"):
            return True
    return False


def preflight(cfg: RadarConfig, folder: str | None = None, bind_ip: str = "",
              query_dca: bool = True, studio_port: int | None = None) -> list[Status]:
    out: list[Status] = []
    d = cfg.dca
    src = local_ip_towards(d.dca_ip, d.config_port)
    out.append(Status("route", src == d.host_ip,
                      f"this PC sends to {d.dca_ip} from {src}" + (
                          "" if src == d.host_ip else
                          f"; set the adapter wired to the DCA1000 to {d.host_ip} / 255.255.255.0")))
    for port in (d.config_port, d.data_port):
        free = port_free(port, bind_ip)
        out.append(Status(f"UDP {port}", free, "free" if free else
                          "in use - close mmWave Studio's DCA1000 connection "
                          "(or DCA1000EVM_CLI_Record) first"))
    if query_dca and all(s.ok for s in out if s.name.startswith("UDP")):
        try:
            with Dca1000Controller(d, bind_ip=bind_ip, timeout_s=1.5) as dca:
                alive = dca.system_aliveness()
                ver = dca.read_fpga_version() if alive else None
            ok = bool(alive and ver and "playback" not in ver)
            out.append(Status("DCA1000", ok, f"FPGA {ver}" if ver else
                              f"no answer from {d.dca_ip}:{d.config_port} (power, Ethernet "
                              f"cable, switches as for mmWave Studio?)"))
        except Dca1000Error as e:
            out.append(Status("DCA1000", False, str(e)))
    if studio_port:
        ok = tcp_listening(studio_port)
        out.append(Status("mmWave Studio", ok, f"RSTD port {studio_port} " + (
            "listening" if ok else "closed - start mmWave Studio (it listens once started)")))
    if folder:
        free = disk_free_bytes(folder)
        need = (cfg.total_bytes or 0) * 1.02 + 200e6
        out.append(Status("disk", free >= need, f"{free / 1e9:.1f} GB free, "
                          f"{need / 1e9:.2f} GB needed" if cfg.total_bytes else
                          f"{free / 1e9:.1f} GB free (open-ended recording)"))
    return out
