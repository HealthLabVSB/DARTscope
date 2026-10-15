"""
acquisition - recording TI mmWave radar data through the DCA1000 without mmWave Studio's
recorder, into exactly the files mmWave Studio produces, so the existing loader
(dca1000_decode, radar_processing) and everything after it read them unchanged.

    config.RadarConfig        the configuration, in mmWave Studio's log units
    lua.read_script_file      a lab Lua script -> RadarConfig
    studio_log                <stem>_LogFile.txt, written and read
    dca1000.Dca1000Controller the DCA1000's UDP command protocol
    recorder.RawRecorder      TI's recorder: _Raw_<n>.bin + _Raw_LogFile.csv
    receiver.UdpReceiver      the data port
    radar                     who configures and starts the radar
    capture.CaptureEngine     one recording, start to finish
    verify.verify             the recording read back by the unchanged loader
    measurement.Measurement   a recording on disk; load_cube() goes through the loader
    simulator                 a DCA1000 on localhost, for tests and demonstrations
    native.NativeRadar        the radar without mmWave Studio (RadarLinkDLL, radarlink*.py,
                              bootloader.py, winserial.py)
    ports.detect              the radar's COM port by its USB identity
    params.MeasureParams      the Measure tab's settings in physical units <-> RadarConfig
    quality                   the quick report on a take and its score for a movement
    series                    a parameter sweep: plan, takes, ranking, HTML/CSV report
    backend                   hardware or simulator behind the Measure tab (measure_gui.py)

Command line (from the Radar folder): python -m acquisition --help
Design and the analysis behind it: docs/ACQUISITION.md

Nothing in the processing code imports this package; this package imports the loader
only to verify recordings.

Authors: Daniel Barvik, Dan Hruby, and AI
"""
from .capture import VERSION, CaptureEngine, CaptureRequest, CaptureResult
from .config import (Chirp, ConfigError, Dca1000Config, DeviceSetup, Frame, Profile,
                     RadarConfig)
from .lua import read_script, read_script_file
from .measurement import Measurement

__all__ = ["VERSION", "CaptureEngine", "CaptureRequest", "CaptureResult", "Chirp",
           "ConfigError", "Dca1000Config", "DeviceSetup", "Frame", "Profile", "RadarConfig",
           "read_script", "read_script_file", "Measurement"]
