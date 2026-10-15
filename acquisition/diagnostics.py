"""
acquisition/diagnostics.py

A check of a DARTscope build or installation, run by the program itself:

    DARTSCOPE_SELFTEST=<report.json> DARTscope.exe
        what the program finds - frozen or not, where it unpacked, the bundled 32-bit
        Python and helper script, mmWave Studio's library and firmware, a start of the
        RadarLinkDLL helper (loads the DLL, touches no hardware), the COM ports, whether the
        Measure tab was built - written to the JSON; then the program closes
    + DARTSCOPE_SELFTEST_TAKE=<folder>
        also one short take with the radar through the Measure tab, as a person would make
        it (the radar must be connected and mmWave Studio closed)

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from __future__ import annotations

import json
import os
import platform
import sys
import traceback


def build_info() -> dict:
    try:
        from version import __version__
    except ImportError:
        __version__ = None
    return {"dartscope": __version__, "frozen": bool(getattr(sys, "frozen", False)),
            "executable": sys.executable, "bundle": getattr(sys, "_MEIPASS", None),
            "python": sys.version.split()[0], "bits": 64 if sys.maxsize > 2 ** 32 else 32,
            "platform": platform.platform()}


def native_checks(start_helper: bool = True) -> dict:
    from .config import DeviceSetup
    from .radarlink import HOST, PY32, RadarLinkClient, native_status
    if not sys.platform.startswith("win"):
        dev = DeviceSetup()
        return {"link": "python mmWaveLink over the DCA1000 FTDI (no RadarLinkDLL)",
                "firmware": list(dev.local_firmware_paths()),
                "problems": native_status(firmware=dev.firmware_paths())}
    out = {"python32": PY32, "python32_exists": os.path.isfile(PY32), "host": HOST,
           "host_exists": os.path.isfile(HOST),
           "problems": native_status(firmware=DeviceSetup().firmware_paths())}
    if start_helper and out["python32_exists"] and out["host_exists"]:
        try:
            link = RadarLinkClient()
            out["helper"] = {"started": True, "bits": link.info.get("bits"), "dll": link.info.get("dll")}
            link.close()
        except Exception as e:
            out["helper"] = {"started": False, "error": f"{type(e).__name__}: {e}"}
    return out


def ports() -> dict:
    from .ports import detect
    det = detect()
    return {"radar": det.radar.name if det.radar else None,
            "dca1000": [p.name for p in det.dca1000_ports], "notes": det.notes}


def cascade_checks() -> dict:
    """What the cascade radar needs from a build: its firmware, JTAG script and target
    configuration packed in, TI's debugger (UniFlash or CCS) found, the XDS110 plugged in."""
    from cascade import hardware
    ti = hardware.scripting()
    return {"firmware": hardware.FIRMWARE.exists(), "jtag_script": hardware.JTAG_SCRIPT.exists(),
            "ccxml": hardware.CCXML.exists(), "ti_debugger": f"{ti.kind} {ti.name}" if ti else None,
            "xds110": hardware.xds110_present()}


def analysis_checks() -> dict:
    """scikit-learn (imported when the AI tab first needs it) and the display scale."""
    import ai_vital_analysis
    import theme
    return {"sklearn": ai_vital_analysis._load_sklearn(), "display_scale": theme.SCALE}


def selftest(measure_tab=None) -> dict:
    rep = {"build": build_info(), "measure_tab_built": measure_tab is not None}
    for key, fn in (("native", native_checks), ("ports", ports), ("cascade", cascade_checks),
                    ("analysis", analysis_checks)):
        try:
            rep[key] = fn()
        except Exception:
            rep[key] = {"error": traceback.format_exc()}
    try:
        import measure_gui  # noqa: F401
        rep["measure_gui_import"] = True
    except Exception as e:
        rep["measure_gui_import"] = f"{type(e).__name__}: {e}"
    return rep


def run_selftest(root, measure_tab, report_path: str, take_folder: str | None = None):
    """Scheduled from DARTscope's start-up when DARTSCOPE_SELFTEST is set; closes the program."""
    def write(rep):
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, default=str)

    def finish(rep):
        write(rep)
        root.after(200, root.destroy)

    def first():
        rep = selftest(measure_tab)
        write(rep)
        if take_folder and measure_tab is not None:
            os.makedirs(take_folder, exist_ok=True)

            def done(result):
                rep["take"] = result
                finish(rep)
            try:
                measure_tab.selftest_take(take_folder, done)
            except Exception:
                rep["take"] = {"ok": False, "error": traceback.format_exc()}
                finish(rep)
        else:
            finish(rep)

    root.after(1500, first)
