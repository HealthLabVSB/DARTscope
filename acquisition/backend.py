"""
acquisition/backend.py

What runs a recording for the Measure tab and the series command, so they need not know
whether a radar is attached:

    HardwareBackend   the radar through NativeRadar (no mmWave Studio) and the DCA1000;
                      the radar stays powered between recordings, so after the first boot
                      (~20 s) each new configuration takes about a second
    SimulatorBackend  the DCA1000 simulator on localhost streaming a synthetic scene
                      (scene.py: a person swaying in front of the radar, two static
                      reflectors) - to try the tab and a whole series without hardware.
                      Recordings are cut to SIM_MAX_BYTES; their sidecar says "simulated".

Both: run(request, on_event, stop_event) -> CaptureResult; close() at the end.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import copy
import threading

from .capture import CaptureEngine, CaptureRequest, CaptureResult
from .serialport import com_number

SIM_MAX_BYTES = 40e6


class HardwareBackend:
    name = "native"

    def __init__(self, com_port=None, on_command=None, bind_ip: str = ""):
        from .native import NativeRadar
        self.com_port = com_port
        self.bind_ip = bind_ip
        self.radar = NativeRadar(boot=False, com_port=com_port, on_command=on_command)

    @property
    def booted(self) -> bool:
        return self.radar.booted

    def run(self, req: CaptureRequest, on_event=None, stop_event: threading.Event | None = None
            ) -> CaptureResult:
        if self.com_port and com_number(self.com_port) is not None:
            req.config.device.com_port = com_number(self.com_port)
        if self.bind_ip and not req.bind_ip:
            req.bind_ip = self.bind_ip
        engine = CaptureEngine(self.radar, on_event=on_event, stop_event=stop_event,
                               close_radar=False)
        return engine.run(req)

    def close(self):
        self.radar.close()


class SimulatorBackend:
    name = "simulator"

    def __init__(self, scene=None):
        from .scene import Scene
        self.scene = scene or Scene()
        self.booted = True

    def run(self, req: CaptureRequest, on_event=None, stop_event: threading.Event | None = None
            ) -> CaptureResult:
        from .dca1000 import Dca1000Controller
        from .layout import encode_for
        from .radar import NullRadar
        from .scene import scene_cube
        from .simulator import Dca1000Simulator, free_udp_port
        req = copy.copy(req)
        cfg = copy.deepcopy(req.config)
        frames = int(cfg.frame.num_frames) or int(round(10.0 * 1e3 / cfg.frame.period_ms))
        cap = max(1, int(SIM_MAX_BYTES // max(1, cfg.bytes_per_frame)))
        if frames > cap:
            frames = cap
            req.notes = dict(req.notes, simulated_frames_cut=f"to {cap} frames ({SIM_MAX_BYTES / 1e6:g} MB)")
        cfg.frame.num_frames = frames
        cfg.dca.host_ip = "127.0.0.1"
        cfg.dca.config_port, cfg.dca.data_port = free_udp_port(), free_udp_port()
        req.config = cfg
        req.bind_ip = "127.0.0.1"
        req.notes = dict(req.notes, simulated="DCA1000 simulator, synthetic scene")
        data = encode_for(cfg, scene_cube(cfg, frames, self.scene))
        sim = Dca1000Simulator(data, host_data_addr=("127.0.0.1", cfg.dca.data_port),
                               rate_mb_s=max(5.0, 2.0 * cfg.data_rate_mb_s))
        dca = Dca1000Controller(cfg.dca, bind_ip="127.0.0.1", dca_addr=sim.addr)
        engine = CaptureEngine(NullRadar(on_start=sim.start_stream), dca=dca, on_event=on_event,
                               stop_event=stop_event)
        try:
            return engine.run(req)
        finally:
            dca.close()
            sim.close()

    def close(self):
        pass
