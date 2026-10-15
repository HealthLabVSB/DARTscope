"""
acquisition/measurement.py

Names, files and the metadata of a recording, and a handle that opens it with the
existing loader.

A recording is the set of files mmWave Studio leaves, plus two of this layer's own:

    <stem>_Raw_0.bin, _Raw_1.bin ...   the data                        (loader)
    <stem>_Raw_LogFile.csv             the recorder's log               (loader)
    <stem>_LogFile.txt                 the radar configuration          (loader)
    <stem>_acquisition.json            everything else: config model, host clock to the
                                       nanosecond, packet statistics, verification
    <stem>_radar.lua                   the radar commands, to repeat the recording in
                                       mmWave Studio by hand

The loader finds the first three by name (dca1000_decode.find_log_file,
radar_processing.get_all_bin_parts) and ignores the other two.

Stems are kept to what those name rules and mmWave Studio's Lua paths survive: ASCII
letters, digits, '_', '-', '.', no "_Raw" and no "_LogFile" inside. Czech names are
transliterated (Měření 1 -> Mereni_1) rather than refused. Nothing is ever overwritten.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass

SIDECAR_SUFFIX = "_acquisition.json"
RADAR_LUA_SUFFIX = "_radar.lua"
_STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,99}$")


class NamingError(ValueError):
    pass


def sanitize(name: str) -> str:
    """A usable stem from free text: accents dropped, spaces to '_', the rest removed."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_.\-]", "", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    return s


def check_stem(stem: str) -> str:
    if not _STEM_RE.match(stem or ""):
        raise NamingError(f"{stem!r}: use letters, digits, '_', '-' or '.', start with a letter "
                          f"or digit, at most 100 characters")
    low = stem.lower()
    if "_raw" in low or low.endswith("_logfile") or "_logfile" in low:
        raise NamingError(f"{stem!r}: '_Raw' and '_LogFile' are part of the file names the "
                          f"loader looks for and cannot be in the name")
    if stem.lower().endswith(SIDECAR_SUFFIX[:-5].lower()):
        raise NamingError(f"{stem!r}: ends like the metadata file")
    return stem


def artifact_paths(folder: str, stem: str) -> dict:
    return {
        "log": os.path.join(folder, f"{stem}_LogFile.txt"),
        "csv": os.path.join(folder, f"{stem}_Raw_LogFile.csv"),
        "bin0": os.path.join(folder, f"{stem}_Raw_0.bin"),
        "sidecar": os.path.join(folder, f"{stem}{SIDECAR_SUFFIX}"),
        "radar_lua": os.path.join(folder, f"{stem}{RADAR_LUA_SUFFIX}"),
    }


def existing_artifacts(folder: str, stem: str) -> list[str]:
    paths = [p for p in artifact_paths(folder, stem).values() if os.path.exists(p)]
    paths += glob.glob(os.path.join(glob.escape(folder), glob.escape(stem) + "_Raw_*.bin"))
    return sorted(set(paths))


def choose_stem(folder: str, name: str, auto_increment: bool = False) -> str:
    """The stem for a new recording; name_2, name_3 ... when asked to avoid a clash."""
    stem = check_stem(sanitize(name))
    if not existing_artifacts(folder, stem):
        return stem
    if not auto_increment:
        raise NamingError(f"a recording named {stem!r} already exists in {folder}")
    for k in range(2, 10000):
        cand = f"{stem}_{k}"
        if not existing_artifacts(folder, cand):
            return cand
    raise NamingError("no free name")


def disk_free_bytes(folder: str) -> int:
    probe = folder
    while probe and not os.path.exists(probe):
        probe = os.path.dirname(probe)
    return shutil.disk_usage(probe or ".").free


def write_json(path: str, data: dict) -> None:
    """Write atomically, so a crash never leaves half a sidecar."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


@dataclass
class Measurement:
    """One recording on disk. Opening it goes through the unchanged loader."""
    folder: str
    stem: str

    @classmethod
    def from_path(cls, path: str) -> "Measurement":
        """From a .bin, the log, the CSV, the sidecar, or folder/stem."""
        folder, name = os.path.split(os.path.abspath(path))
        for suffix in ("_Raw_LogFile.csv", "_LogFile.txt", SIDECAR_SUFFIX, RADAR_LUA_SUFFIX):
            if name.endswith(suffix):
                return cls(folder, name[: -len(suffix)])
        m = re.match(r"^(.*?)(?:_Raw)?_(\d+)\.bin$", name)
        if m:
            return cls(folder, m.group(1))
        if name.endswith(".bin"):
            return cls(folder, name[:-4])
        return cls(folder, name)

    @property
    def paths(self) -> dict:
        return artifact_paths(self.folder, self.stem)

    @property
    def log_path(self) -> str:
        return self.paths["log"]

    @property
    def csv_path(self) -> str:
        return self.paths["csv"]

    @property
    def sidecar_path(self) -> str:
        return self.paths["sidecar"]

    @property
    def first_bin(self) -> str | None:
        for name in (f"{self.stem}_Raw_0.bin", f"{self.stem}_0.bin", f"{self.stem}.bin"):
            p = os.path.join(self.folder, name)
            if os.path.isfile(p):
                return p
        return None

    @property
    def bin_paths(self) -> list[str]:
        from radar_processing import get_all_bin_parts
        first = self.first_bin
        return list(get_all_bin_parts(first)) if first else []

    def sidecar(self) -> dict | None:
        try:
            with open(self.sidecar_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def load_params(self, quiet: bool = True) -> dict:
        """parse_logfile_full on this recording's log, as the GUI's Decode tab does."""
        from dca1000_decode import find_log_file, parse_logfile_full
        log = find_log_file(self.first_bin) if self.first_bin else None
        if not log:
            raise FileNotFoundError(f"no _LogFile.txt for {self.stem} in {self.folder}")
        ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
        with ctx:
            params, _ = parse_logfile_full(log)
        return params

    def load_cube(self, quiet: bool = True):
        """(params, dims, cube[frames, chirps, RX, samples]) through load_raw_cube."""
        from radar_processing import derive_capture_dimensions, load_raw_cube
        params = self.load_params(quiet)
        ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
        with ctx:
            dims = derive_capture_dimensions(params)
            cube = load_raw_cube(self.first_bin, params, dims)
        return params, dims, cube
