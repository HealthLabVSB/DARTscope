"""
acquisition/verify.py

A recording counts as correct when the existing loader, unchanged, reads it as the
configuration says it should be - not when bytes have arrived. This runs that loader on a
recording and compares:

    names        find_log_file finds the log, get_all_bin_parts finds every part
    config       parse_logfile_full + derive_capture_dimensions give the configured RX,
                 samples, chirps per frame, frames, format, IQ order, lanes, TX sequence
    recorder     the CSV the loader reads states the packets this recording counted
    bytes        the file length against frames x bytes per frame; what is missing is
                 reported (the lab's DCA1000 keeps up to a few hundred bytes back at the
                 very end, and the loader then drops the incomplete last frame)
    decode       load_raw_cube on the first frames (copied beside the log in a temporary
                 folder, so a 1 GB file is not decoded in full) returns
                 (frames, chirps, RX, samples) complex64

and, as information only, two signs that the bytes are really complex LVDS samples in the
expected order: the int16 slot correlation of each LVDS group and how one-sided the
range spectrum is (docs/ACQUISITION.md, section 3). A scene with nothing in it can fail
those without anything being wrong, so they never fail the verification.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field

import numpy as np

from .config import RadarConfig
from .measurement import Measurement


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class VerifyReport:
    stem: str
    checks: list[Check] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append(Check(name, bool(ok), detail))
        return bool(ok)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "stem": self.stem, "checks": [asdict(c) for c in self.checks],
                "info": self.info}

    def lines(self) -> list[str]:
        out = [f"{'OK  ' if c.ok else 'FAIL'} {c.name}: {c.detail}" for c in self.checks]
        out += [f"     {k}: {v}" for k, v in self.info.items()]
        return out


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def verify(meas: Measurement, cfg: RadarConfig | None = None, recorder_stats: dict | None = None,
           decode_frames: int = 4) -> VerifyReport:
    from dca1000_decode import find_log_file, parse_logfile_full
    from radar_processing import derive_capture_dimensions, get_all_bin_parts, load_raw_cube

    rep = VerifyReport(meas.stem)
    first = meas.first_bin
    if not rep.add("data file", first is not None, first or f"no {meas.stem}_Raw_0.bin"):
        return rep
    parts = list(get_all_bin_parts(first))
    rep.add("parts", True, f"{len(parts)} file(s): " + ", ".join(os.path.basename(p) for p in parts))
    log = find_log_file(first)
    if not rep.add("log found by the loader", log is not None and os.path.samefile(log, meas.log_path),
                   str(log)):
        return rep
    with _quiet():
        params, _ = parse_logfile_full(log)
        dims = derive_capture_dimensions(params)
    rep.info["loader_dims"] = {k: dims[k] for k in ("num_frames", "chirps_per_frame", "num_rx",
                                                   "num_samples", "is_complex", "adc_fmt")}
    total = sum(os.path.getsize(p) for p in parts)
    rep.info["bytes_on_disk"] = total

    if cfg is not None:
        e = cfg.expectation()
        rep.add("RX", dims["num_rx"] == e["rx"], f"loader {dims['num_rx']}, configured {e['rx']}")
        rep.add("samples per chirp", dims["num_samples"] == e["samples"],
                f"loader {dims['num_samples']}, configured {e['samples']}")
        rep.add("chirps per frame", dims["chirps_per_frame"] == e["chirps_per_frame"],
                f"loader {dims['chirps_per_frame']}, configured {e['chirps_per_frame']}")
        rep.add("frames declared", dims["num_frames"] == max(1, e["frames"]) or e["frames"] == 0,
                f"loader {dims['num_frames']}, configured {e['frames']}")
        rep.add("ADC format", str(params.get("dataFmt_adcFmt")) == e["adc_format"],
                f"loader {params.get('dataFmt_adcFmt')}, configured {e['adc_format']}")
        rep.add("ADC bits", params.get("dataFmt_adcBits") == e["adc_bits"],
                f"loader {params.get('dataFmt_adcBits')}")
        rep.add("IQ order", str(params.get("dataFmt_iqOrder")) == e["iq_order"],
                f"loader {params.get('dataFmt_iqOrder')}, configured {e['iq_order']}")
        rep.add("RX layout", params.get("dataFmt_isInterleaved") is False,
                "non-interleaved" if params.get("dataFmt_isInterleaved") is False
                else f"loader reads interleaved={params.get('dataFmt_isInterleaved')}")
        rep.add("TX sequence", list(params.get("tx_order_masks") or []) == e["tx_pattern"],
                f"loader {params.get('tx_order_masks')}, configured {e['tx_pattern']}")
        rep.add("LVDS container", params.get("numLanes") == e["container_lanes"],
                f"loader {params.get('numLanes')} lanes ({params.get('numLanes_source')}), "
                f"recorder {e['container_lanes']}")
        rep.add("frame period", abs((params.get("framePeriodicity_ms_selected") or 0)
                                    - e["frame_period_ms"]) < 1e-6,
                f"loader {params.get('framePeriodicity_ms_selected')} ms")
        bpf = e["bytes_per_frame"]
        full = total // bpf if bpf else 0
        rep.info["complete_frames"] = int(full)
        if e["total_bytes"]:
            missing = e["total_bytes"] - total
            rep.info["bytes_missing_at_end"] = int(missing)
            # a short tail (the DCA1000 keeps some bytes back) costs the last frame only
            rep.add("byte count", 0 <= missing < bpf,
                    f"{total} of {e['total_bytes']} bytes, {full} complete frame(s)"
                    + (f"; the last {missing} bytes never arrived" if missing > 0 else ""))
        else:
            rep.add("byte count", full >= 1,
                    f"{total} bytes, {full} complete frame(s), open-ended recording")

    if recorder_stats is not None:
        rep.add("recorder log", params.get("dca_packets_received") == recorder_stats.get("received")
                and params.get("dca_packets_zero_filled") == recorder_stats.get("zero_filled_packets"),
                f"CSV: {params.get('dca_packets_received')} packets received, "
                f"{params.get('dca_packets_zero_filled')} zero-filled")

    # decode the first frames with load_raw_cube, as the GUI would
    bpf = (dims["chirps_per_frame"] * dims["num_rx"] * dims["num_samples"]
           * (2 if dims["is_complex"] else 1) * 2)
    n = int(min(decode_frames, total // bpf)) if bpf else 0
    if n < 1:
        rep.add("decode", False, "not one complete frame in the file")
        return rep
    tmp = tempfile.mkdtemp(prefix="acq_verify_")
    try:
        head = os.path.join(tmp, f"{meas.stem}_Raw_0.bin")
        with open(first, "rb") as src, open(head, "wb") as dst:
            dst.write(src.read(n * bpf))
        shutil.copy(meas.log_path, os.path.join(tmp, f"{meas.stem}_LogFile.txt"))
        if os.path.isfile(meas.csv_path):
            shutil.copy(meas.csv_path, os.path.join(tmp, f"{meas.stem}_Raw_LogFile.csv"))
        with _quiet():
            p2, _ = parse_logfile_full(os.path.join(tmp, f"{meas.stem}_LogFile.txt"))
            d2 = derive_capture_dimensions(p2)
            cube = load_raw_cube(head, p2, d2)
        want = (n, dims["chirps_per_frame"], dims["num_rx"], dims["num_samples"])
        rep.add("decode", tuple(cube.shape) == want and cube.dtype in (np.complex64, np.float32),
                f"load_raw_cube -> {tuple(cube.shape)} {cube.dtype}, expected {want}")
        if dims["is_complex"]:
            rep.info["iq_signs"] = layout_signs(head, int(params.get("numLanes") or 2), cube)
    except Exception as ex:
        rep.add("decode", False, f"{type(ex).__name__}: {ex}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rep


def loader_view(cfg: RadarConfig) -> tuple[dict, dict]:
    """(params, dims) the unchanged loader derives from the log this configuration writes."""
    from dca1000_decode import parse_logfile_full
    from radar_processing import derive_capture_dimensions

    from . import studio_log
    tmp = tempfile.mkdtemp(prefix="acq_preview_")
    try:
        log = os.path.join(tmp, "preview_LogFile.txt")
        studio_log.write_log(log, cfg)
        with _quiet():
            params, _ = parse_logfile_full(log)
            dims = derive_capture_dimensions(params)
        return params, dims
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def loader_mismatches(cfg: RadarConfig) -> list[str]:
    """
    Where the loader would read this configuration differently from what it is - checked
    before recording, so no recording is made that cannot be read back.

    The known case: a FrameConfig line whose two possible field layouts both fit the
    defined chirps (1 frame of chirps 0..1, 2 frames of 0..2): the parser prefers the
    layout mmWave Studio does not write and would decode the wrong chirps.
    """
    params, dims = loader_view(cfg)
    e = cfg.expectation()
    out = []
    pairs = [("RX", dims["num_rx"], e["rx"]), ("samples", dims["num_samples"], e["samples"]),
             ("chirps per frame", dims["chirps_per_frame"], e["chirps_per_frame"]),
             ("TX sequence", list(params.get("tx_order_masks") or []), e["tx_pattern"]),
             ("ADC format", params.get("dataFmt_adcFmt"), e["adc_format"]),
             ("IQ order", params.get("dataFmt_iqOrder"), e["iq_order"])]
    if e["frames"] > 0:
        pairs.append(("frames", dims["num_frames"], e["frames"]))
    for what, got, want in pairs:
        if got != want:
            out.append(f"{what}: the loader would read {got}, the configuration is {want}")
    return out


def layout_signs(path: str, lanes: int, cube: np.ndarray, n: int = 1 << 20) -> dict:
    """Slot correlation of the int16 groups and positive/negative beat energy (information)."""
    raw = np.fromfile(path, dtype="<i2", count=n).astype(np.float64)
    g = raw[: raw.size // (2 * lanes) * 2 * lanes].reshape(-1, 2 * lanes)
    out = {}
    if lanes >= 2 and g.shape[0] > 16:
        out["corr_I_I"] = float(np.corrcoef(g[:, 0], g[:, 1])[0, 1])
        out["corr_I_Q"] = float(np.corrcoef(g[:, 0], g[:, lanes])[0, 1])
    x = cube[: min(8, cube.shape[0])]
    x = x - x.mean(axis=-1, keepdims=True)
    spec = np.abs(np.fft.fft(x * np.hanning(x.shape[-1]), axis=-1)) ** 2
    ns = spec.shape[-1]
    pos, neg = spec[..., 1:ns // 2].sum(), spec[..., ns // 2 + 1:].sum()
    out["positive_vs_negative_beat_dB"] = float(10 * np.log10(pos / neg)) if pos > 0 and neg > 0 else None
    return out
