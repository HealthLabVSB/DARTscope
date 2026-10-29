"""
version.py

DARTscope's version: the window title, the sidecar of every recording the Measure tab makes,
and the release tag on GitHub (v2.1).

    1.0   decoding and analysis of mmWave Studio + DCA1000 recordings
    2.0   recording without mmWave Studio: the Measure tab (the radar's COM port found by
          itself, takes started by the operator, a quick report, series over a range of
          settings ranked for a movement), the radar booted and configured through TI's
          RadarLinkDLL, and the Fastcheck, HR, AI and Helper work since 1.0
    2.1   opt-in heart-rate options of the T-MTT case study (breathing-harmonic comb notch,
          prewhitening, band-edge index Q_edge), the article's analysis scripts (tools/paper,
          tools/paper_array) and the test suite

Authors: Daniel Barvik, Dan Hruby, and AI
"""
__version__ = "2.1"
