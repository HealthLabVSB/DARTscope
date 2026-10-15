"""
cascade/data.py

A cascade take read back: <name>_cascade.json for the layout, the samples from the
<name>_Raw_<n>.bin parts (the recorder cuts them at 1 GiB, mid-frame) as one stream.

    take = CascadeTake.open("data/take1_Raw_0.bin")      # or the _cascade.json
    cube = take.frames(10, 2)          # complex64 [frame, chirp, rx, sample]
    first = take.first_chirps()        # complex64 [frame, rx, sample]: chirp 0 of every frame

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


class CascadeTake:
    def __init__(self, sidecar: Path):
        self.sidecar = Path(sidecar)
        self.meta = json.loads(self.sidecar.read_text(encoding="utf-8"))
        lay = self.meta["layout"]
        self.prefix = self.sidecar.name[: -len("_cascade.json")]
        self.folder = self.sidecar.parent
        self.num_samples = int(lay["num_samples"])
        self.num_rx = int(lay["num_rx"])
        self.chirps_per_frame = int(lay["chirps_per_frame"])
        self.chirps_in_loop = int(lay["chirps_in_loop"])
        self.num_loops = int(lay["num_loops"])
        self.chirp_bytes = self.num_rx * self.num_samples * 4
        self.frame_bytes = self.chirps_per_frame * self.chirp_bytes
        parts = sorted(self.folder.glob(f"{self.prefix}_Raw_*.bin"),
                       key=lambda p: int(re.search(r"_Raw_(\d+)\.bin$", p.name).group(1)))
        self.parts = [(p, p.stat().st_size) for p in parts]
        self.total_bytes = sum(s for _, s in self.parts)
        self.num_frames = self.total_bytes // self.frame_bytes
        rad = self.meta["radar"]
        self.range_res_m = float(rad["range_res_m"])
        self.frame_rate_hz = float(rad["frame_rate_hz"])
        self.wavelength_m = float(rad["wavelength_m"])
        self.chirp_time_s = float(rad["chirp_time_us"]) * 1e-6

    @classmethod
    def open(cls, path) -> "CascadeTake":
        p = Path(path)
        if p.name.endswith("_cascade.json"):
            return cls(p)
        m = re.match(r"(.*)_Raw_\d+\.bin$", p.name)
        if not m:
            raise ValueError(f"{p.name}: expected <name>_Raw_<n>.bin or <name>_cascade.json")
        side = p.with_name(m.group(1) + "_cascade.json")
        if not side.exists():
            raise FileNotFoundError(f"{side.name} is missing - not a cascade take")
        return cls(side)

    @staticmethod
    def is_cascade(bin_path) -> bool:
        m = re.match(r"(.*)_Raw_\d+\.bin$", Path(bin_path).name)
        return bool(m) and Path(bin_path).with_name(m.group(1) + "_cascade.json").exists()

    def read_bytes(self, offset: int, count: int) -> bytes:
        """count bytes of the stream from offset, across part boundaries."""
        out, start = [], 0
        for path, size in self.parts:
            end = start + size
            if offset < end and offset + count > start:
                a = max(offset, start) - start
                b = min(offset + count, end) - start
                with open(path, "rb") as f:
                    f.seek(a)
                    out.append(f.read(b - a))
            start = end
        return b"".join(out)

    def _complex(self, raw: bytes, shape) -> np.ndarray:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(*shape, 2)
        return x[..., 0] + 1j * x[..., 1]

    def frames(self, start: int, count: int = 1) -> np.ndarray:
        """complex64 [frame, chirp, rx, sample]"""
        count = max(0, min(count, self.num_frames - start))
        raw = self.read_bytes(start * self.frame_bytes, count * self.frame_bytes)
        return self._complex(raw, (count, self.chirps_per_frame, self.num_rx, self.num_samples))

    def first_chirps(self, chirp: int = 0) -> np.ndarray:
        """complex64 [frame, rx, sample]: one chirp of every frame (slow time for the
        range-time map and the displacement)."""
        out = np.empty((self.num_frames, self.num_rx, self.num_samples), np.complex64)
        for f in range(self.num_frames):
            raw = self.read_bytes(f * self.frame_bytes + chirp * self.chirp_bytes, self.chirp_bytes)
            out[f] = self._complex(raw, (self.num_rx, self.num_samples))
        return out
