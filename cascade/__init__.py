"""
cascade - recording with the AWR2243-2X-CAS-EVM (two cascaded AWR2243 + AM2732R) and the
DCA1000, without mmWave Studio, on macOS or Windows (the JTAG debugger from TI UniFlash or
CCS 20+).

    params.py      the radar settings in physical units, their checks and derived values,
                   and the gUserCfg words the firmware reads
    hardware.py    JTAG through TI's scripting debugger from UniFlash or CCS 20+
                   (jtag_cfg.js): load the firmware, write gUserCfg, run; halt and read
                   its status; probe the connection
    recording.py   one take: DCA1000 (acquisition.dca1000) + UDP receiver and recorder
                   (acquisition.receiver/recorder) + the firmware; <name>_cascade.json
    data.py        reading a take back: frames across the 1 GiB _Raw_<n>.bin parts
    decode.py      a take for the Decode tab and the Analysis views: the params a mmWave
                   Studio log would give, and the radar cube without the LVDS demux

The firmware (firmware/am273x_cascade_param.xer5f, sources firmware/src/) is TI's
ti/utils/test/cascade test from the mmWave MCU+ SDK 04.04.00.01 with every radar parameter
taken from gUserCfg, which jtag_cfg.js writes before the firmware runs - a change of
settings needs no rebuild. The GUI is cascade_gui.py (the Measure tab's Cascade radar
sub-tab).

Authors: Daniel Barvik, Dan Hruby, and AI
"""
