"""
The parsed radar parameters have to match what TI's documented LSBs give.

Every case here is recomputed from the raw ProfileConfig / FrameConfig integers rather
than compared against a number someone once read off a screen.

Run with:  python -m pytest test_parsing_values.py -q
"""
import math
import os
from datetime import datetime
import numpy as np
import pytest

from dca1000_decode import (
    C_LIGHT, _fill_velocity_and_range_limits, _merge_recorder_log,
    parse_frame_config_line, parse_logfile_full, parse_raw_log_csv,
    parse_recorder_time, decode_tx_enable,
)

# Three real captures: (name, fsynth GHz, ProfileConfig ints, numLoops, n_tx, adcFmt)
CAPTURES = [
    ("IWR6843 AoA", 2.7,
     dict(startFreqConst=1497521873, idleConst=10000, adcStartConst=600, rampEnd=4000,
          slopeConst=2486, nAdc=256, sampleRate_ksps=10000), 32, 3, "Complex1X"),
    ("IWR6843 vitals", 2.7,
     dict(startFreqConst=1497521873, idleConst=10000, adcStartConst=100, rampEnd=4000,
          slopeConst=2486, nAdc=256, sampleRate_ksps=10000), 5, 1, "Complex1X"),
    ("AWR1642 real", 3.6,
     dict(startFreqConst=1435384036, idleConst=10000, adcStartConst=600, rampEnd=3000,
          slopeConst=2071, nAdc=150, sampleRate_ksps=8000), 200, 1, "Real"),
]


def derive(fsynth, c):
    """TI LSBs: start frequency and slope scale with the synthesiser, times are 10 ns."""
    f0 = c["startFreqConst"] * fsynth / 2**26
    slope = c["slopeConst"] * (fsynth * 1e3 * 900) / 2**26
    idle, adc_start, ramp = c["idleConst"]/100, c["adcStartConst"]/100, c["rampEnd"]/100
    fs = c["sampleRate_ksps"] / 1000.0
    return dict(f0=f0, slope=slope, idle_us=idle, adc_start_us=adc_start,
                ramp_us=ramp, fs_msps=fs, f_eff=f0 + slope * adc_start / 1000.0)


def params_for(fsynth, c, loops, n_tx, adc_fmt):
    d = derive(fsynth, c)
    return {
        'startFreqGHz': d['f0'], 'effectiveStartFreqGHz': d['f_eff'],
        'freqSlopeMHz_us': d['slope'], 'idleTime_us': d['idle_us'],
        'rampEndTime_us': d['ramp_us'], 'digOutSampleRate_MHz': d['fs_msps'],
        'numAdcSamples': c['nAdc'], 'dataFmt_adcFmt': adc_fmt,
        'tx_order_masks': list(range(1, n_tx + 1)),
        'frameConfig': {'numLoops': loops},
    }, d


@pytest.mark.parametrize("name,fsynth,cfg,loops,n_tx,fmt", CAPTURES)
def test_velocity_limits_use_the_chirp_interval_and_bin_count(name, fsynth, cfg, loops, n_tx, fmt):
    """
    velocityRes_m_s was lambda / (2 * rampEndTime): no Doppler bin count, and the ramp
    time instead of the chirp repetition time. Both IWR6843 captures came out as the
    same 62.2 m/s although one has 32 Doppler bins and the other 5, and the AWR1642
    capture read 64.9 m/s where the real figure is 0.074.
    """
    p, d = params_for(fsynth, cfg, loops, n_tx, fmt)
    _fill_velocity_and_range_limits(p)

    # wavelength in the middle of the ADC window (roadmap 1.9)
    f_mid = d['f_eff'] * 1e9 + d['slope'] * 1e12 * (cfg['nAdc'] - 1) / (2 * d['fs_msps'] * 1e6)
    lam = C_LIGHT / f_mid
    tc = (d['idle_us'] + d['ramp_us']) * 1e-6 * n_tx
    assert p['velocityRes_m_s'] == pytest.approx(lam / (2 * loops * tc), rel=1e-9)
    assert p['maxVelocity_m_s'] == pytest.approx(lam / (4 * tc), rel=1e-9)
    assert p['chirpRepetitionTime_us'] == pytest.approx(tc * 1e6, rel=1e-9)
    assert p['dopplerBins'] == loops
    assert p['wavelength_mm'] == pytest.approx(lam * 1000, rel=1e-9)


@pytest.mark.parametrize("name,fsynth,cfg,loops,n_tx,fmt", CAPTURES)
def test_max_range_halves_for_real_sampling(name, fsynth, cfg, loops, n_tx, fmt):
    p, d = params_for(fsynth, cfg, loops, n_tx, fmt)
    _fill_velocity_and_range_limits(p)
    full = (d['fs_msps'] * 1e6) * C_LIGHT / (2 * d['slope'] * 1e12)
    expected = full / 2 if fmt.upper() == "REAL" else full
    assert p['maxRange_m'] == pytest.approx(expected, rel=1e-9)


def test_velocity_resolution_separates_captures_that_differ():
    """The old formula gave both IWR6843 captures the same number."""
    vals = []
    for name, fsynth, cfg, loops, n_tx, fmt in CAPTURES[:2]:
        p, _ = params_for(fsynth, cfg, loops, n_tx, fmt)
        _fill_velocity_and_range_limits(p)
        vals.append(p['velocityRes_m_s'])
    assert vals[0] != pytest.approx(vals[1], rel=0.01)


def test_velocity_axis_matches_the_parsed_resolution():
    """The number in the parameter list has to be the step of the axis on the plot."""
    from radar_processing import compute_range_doppler_map
    name, fsynth, cfg, loops, n_tx, fmt = CAPTURES[2]
    p, _ = params_for(fsynth, cfg, loops, n_tx, fmt)
    p['uniqueChirps_effective'] = 1
    p['unique_chirps_frame'] = 1
    p['tx_order_masks'] = [1]
    _fill_velocity_and_range_limits(p)
    cube = np.zeros((1, loops, 2, cfg['nAdc']), dtype=np.float32)
    cube[..., 10] = 1.0
    _, meta = compute_range_doppler_map(cube, p, n_fft_range=256)
    v = np.asarray(meta['velocity_axis'])
    assert float(v[1] - v[0]) == pytest.approx(p['velocityRes_m_s'], rel=1e-6)
    assert abs(float(v[0])) == pytest.approx(p['maxVelocity_m_s'], rel=1e-6)


# --- the recorder's own log ------------------------------------------------------

def test_recorder_csv_is_read(tmp_path):
    csv = tmp_path / "x_Raw_LogFile.csv"
    csv.write_text(
        "Start record configuration : \n,\nLog mode : Raw\nLVDS lane mode : 2 lane\n"
        "Record stop mode : Infinite\nMax file size (MB) : 1024,\n\nRaw Data :\n"
        "Out of sequence count - 74\nFirst Packet ID - 1\nLast Packet ID - 84396\n"
        "Number of received packets - 79871\nNumber of zero filled packets - 4525\n"
        "Number of zero filled bytes - 6588400\nDuration(sec) - 62\n")
    info = parse_raw_log_csv(str(csv))
    assert info['dca_lane_mode'] == 2
    assert info['dca_packets_zero_filled'] == 4525
    assert info['dca_out_of_sequence'] == 74
    assert info['dca_duration_s'] == 62
    assert info['dca_packet_loss_pct'] == pytest.approx(100 * (84396 - 79871) / 84396)


def test_missing_recorder_csv_is_not_an_error():
    assert parse_raw_log_csv(None) == {}
    assert parse_raw_log_csv("/nonexistent/x_Raw_LogFile.csv") == {}


def _capture(tmp_path, lane_mode, capture_start=None, log_text=""):
    """An API log with its recorder CSV beside it, named the way the pair is found."""
    log = tmp_path / "cap_LogFile.txt"
    log.write_text(log_text)
    times = ""
    if capture_start:
        times = (f"Capture start time - {capture_start}\n"
                 f"Capture end time - {capture_start}\n")
    (tmp_path / "cap_Raw_LogFile.csv").write_text(
        f"Log mode : Raw\nLVDS lane mode : {lane_mode} lane\n\nRaw Data :\n"
        "Out of sequence count - 0\nFirst Packet ID - 1\nLast Packet ID - 210165\n"
        "Number of received packets - 210165\nNumber of zero filled packets - 0\n"
        "Number of zero filled bytes - 0\nDuration(sec) - 12\n" + times)
    return str(log)


def test_recorder_lane_mode_beats_laneconfig(tmp_path):
    """
    The recorder's lane mode is the width of the container on disk, so it decides.

    A lane-sweep set recorded on a DCA1000 left in 2-lane mode carries LaneConfig masks
    of 1, 8 or 15 while every file is still a 2-lane container. Taking LaneConfig
    decoded those into the wrong shape - and since the byte count still worked out, it
    did so without raising.
    """
    log = _capture(tmp_path, 2)
    for radar_lanes in (1, 2, 4):
        params = {'numLanes': radar_lanes, 'numLanes_source': 'LaneConfig'}
        _merge_recorder_log(params, log)
        assert params['numLanes'] == 2
        assert params['numLanes_source'] == 'DCA1000 recorder log'
        assert params['numLanes_radar'] == radar_lanes
        assert params.get('lane_mode_mismatch', False) is (radar_lanes != 2)


def test_radar_lane_count_kept_when_recorder_log_is_absent(tmp_path):
    log = tmp_path / "solo_LogFile.txt"
    log.write_text("")
    params = {'numLanes': 4, 'numLanes_source': 'LaneConfig'}
    _merge_recorder_log(params, str(log))
    assert params['numLanes'] == 4
    assert params['numLanes_source'] == 'LaneConfig'


def test_unusable_recorder_lane_mode_is_ignored(tmp_path):
    """3 is not a container width the DCA1000 can report; fall back to the radar."""
    log = _capture(tmp_path, 3)
    params = {'numLanes': 2, 'numLanes_source': 'LaneConfig'}
    _merge_recorder_log(params, log)
    assert params['numLanes'] == 2
    assert params['numLanes_source'] == 'LaneConfig'


# --- when the capture actually started ------------------------------------------

def test_recorder_time_is_read_without_going_through_the_locale():
    """%a/%b follow LC_TIME; the recorder always writes English, so it is parsed by hand."""
    assert parse_recorder_time("Wed Oct 22 15:06:12 2025") == datetime(2025, 10, 22, 15, 6, 12)
    assert parse_recorder_time("Fri Jul 10 14:38:40 2026") == datetime(2026, 7, 10, 14, 38, 40)
    assert parse_recorder_time("Thu Jan  1 00:00:00 2026") == datetime(2026, 1, 1, 0, 0, 0)
    for junk in (None, "", "not a time", "Wed Xyz 22 15:06:12 2025", "Wed Feb 30 00:00:00 2025"):
        assert parse_recorder_time(junk) is None


def test_capture_start_is_taken_from_the_recorder_not_the_api_log(tmp_path):
    """
    mmWave Studio appends every capture to one log, so its first timestamp is when the
    session's first command was issued - on the vitals set that ran from 37 s to 28 min
    before the data actually starts. Only the recorder knows when the capture began, and
    that is the wall clock a reference recording gets lined up against.
    """
    log = _capture(tmp_path, 2, capture_start="Fri Jul 10 13:05:02 2026")
    log_start = datetime(2026, 7, 10, 12, 37, 4)
    params = {'radar_start_timestamp': log_start.timestamp(),
              'radar_start_time': log_start.strftime("%d-%m-%Y %H:%M:%S"),
              'radar_start_source': 'API log (first command)'}
    _merge_recorder_log(params, log)

    assert params['radar_start_time'] == "10-07-2026 13:05:02"
    assert params['radar_start_source'] == "DCA1000 recorder log"
    assert params['radar_start_timestamp'] == datetime(2026, 7, 10, 13, 5, 2).timestamp()
    # the log's own value stays reachable, and the gap is reported rather than hidden
    assert params['radar_start_timestamp_log'] == log_start.timestamp()
    assert params['capture_start_vs_log_s'] == pytest.approx(1678.0)


def test_api_log_start_is_kept_when_the_recorder_does_not_give_one(tmp_path):
    """Older recorder logs carry no capture time; the log's timestamp is all there is."""
    log = _capture(tmp_path, 2)          # CSV without Capture start time
    log_start = datetime(2026, 7, 10, 12, 37, 4)
    params = {'radar_start_timestamp': log_start.timestamp(),
              'radar_start_time': log_start.strftime("%d-%m-%Y %H:%M:%S"),
              'radar_start_source': 'API log (first command)'}
    _merge_recorder_log(params, log)

    assert params['radar_start_timestamp'] == log_start.timestamp()
    assert params['radar_start_source'] == "API log (first command)"
    assert 'capture_start_vs_log_s' not in params


def test_a_capture_start_matching_the_log_is_not_reported_as_a_gap(tmp_path):
    log = _capture(tmp_path, 2, capture_start="Fri Jul 10 14:33:42 2026")
    params = {'radar_start_timestamp': datetime(2026, 7, 10, 14, 33, 41).timestamp()}
    _merge_recorder_log(params, log)
    assert 'capture_start_vs_log_s' not in params        # 1 s is within the tolerance
    assert params['radar_start_source'] == "DCA1000 recorder log"


def test_recorder_csv_exposes_both_capture_edges(tmp_path):
    _capture(tmp_path, 2, capture_start="Wed Oct 22 15:06:12 2025")
    info = parse_raw_log_csv(str(tmp_path / "cap_Raw_LogFile.csv"))
    assert info['dca_capture_start'] == "22-10-2025 15:06:12"
    assert info['dca_capture_end'] == "22-10-2025 15:06:12"
    assert info['dca_capture_start_timestamp'] == datetime(2025, 10, 22, 15, 6, 12).timestamp()


@pytest.mark.parametrize("mask, expected, source", [
    (1, 1, 'LaneConfig'),          # lane1
    (3, 2, 'LaneConfig'),          # lane1+2
    (15, 4, 'LaneConfig'),         # all four
    (7, 4, 'LaneConfig (3->4)'),   # lane1+2+3 rides in a 4-lane container
    (13, 4, 'LaneConfig (3->4)'),  # lane1+3+4
    (14, 4, 'LaneConfig (3->4)'),  # lane2+3+4
])
def test_three_lane_mask_rides_in_a_four_lane_container(tmp_path, mask, expected, source):
    """
    There is no 3-lane LVDS mode, so a three-bit mask has to round up to four.

    These masks used to reach load_raw_cube as numLanes=3, which it refused outright.
    """
    log = tmp_path / "m_LogFile.txt"
    log.write_text(f"22-Oct-2025 15:00:31: API:LaneConfig,{mask},0,\n")
    params, _ = parse_logfile_full(str(log))
    assert params['numLanes'] == expected
    assert params['numLanes_source'] == source
    assert params['numLanes_popcount'] == bin(mask).count('1')


# --- the pieces the captures exercised ------------------------------------------

def test_frame_layout_of_each_capture():
    """Each of the three logs, with the chirps its ChirpConfig defines."""
    cases = [
        ("API:FrameConfig,0,0,1200,3,10000000,0,512,0,", [(0, 2)], 3, 3),
        ("API:FrameConfig,0,0,6000,5,2000000,0,512,0,", [(0, 0)], 5, 5),
        ("API:FrameConfig,0,0,150,200,8000000,0,150,0,", [(0, 0)], 200, 200),
    ]
    for line, ranges, loops, per_frame in cases:
        fc = parse_frame_config_line(line, chirp_ranges=ranges)
        assert fc['numLoops'] == loops, line
        assert fc['chirpsPerFrame'] == per_frame, line


def test_frame_periodicity_is_five_nanosecond_ticks():
    fc = parse_frame_config_line("API:FrameConfig,0,0,150,200,8000000,0,150,0,",
                                 chirp_ranges=[(0, 0)])
    assert fc['framePeriodicity_ms'] == pytest.approx(8000000 * 5e-6)
    assert fc['framePeriodicity_ms'] == 40.0


def test_tx_mask_names_start_at_one():
    assert decode_tx_enable(1) == "TX1"
    assert decode_tx_enable(12) == "TX3,TX4"


# --- the parts of a split capture ------------------------------------------------

def test_split_parts_are_collected_in_order(tmp_path):
    from radar_processing import get_all_bin_parts
    for i in range(3):
        (tmp_path / f"cap_Raw_{i}.bin").write_bytes(b"\x00\x00")
    got = [os.path.basename(p) for p in get_all_bin_parts(str(tmp_path / "cap_Raw_0.bin"))]
    assert got == ["cap_Raw_0.bin", "cap_Raw_1.bin", "cap_Raw_2.bin"]


def test_a_differently_named_file_is_not_taken_for_the_next_part(tmp_path):
    """
    feb26: data_test2_Raw_0.bin was a complete capture on its own, and the old fallback
    appended data_test2_1.bin - a 150 MB recording from the day before.
    """
    from radar_processing import get_all_bin_parts
    (tmp_path / "cap_Raw_0.bin").write_bytes(b"\x00\x00")
    (tmp_path / "cap_1.bin").write_bytes(b"\x00\x00")
    got = [os.path.basename(p) for p in get_all_bin_parts(str(tmp_path / "cap_Raw_0.bin"))]
    assert got == ["cap_Raw_0.bin"]


def test_parts_without_the_raw_infix_still_chain(tmp_path):
    from radar_processing import get_all_bin_parts
    for i in range(2):
        (tmp_path / f"cap_{i}.bin").write_bytes(b"\x00\x00")
    got = [os.path.basename(p) for p in get_all_bin_parts(str(tmp_path / "cap_0.bin"))]
    assert got == ["cap_0.bin", "cap_1.bin"]


# --- how many chirps a frame holds -----------------------------------------------

@pytest.mark.parametrize("chirps, frame_range, expected", [
    # vitals/data_test1: chirp 1 is defined but the frame only sends chirp 0
    ([(0, 0), (1, 1)], (0, 0), 1),
    # kleky_dataset1: 0..2 defined four times over, overlapping
    ([(0, 0), (0, 2), (1, 1), (2, 2)], (0, 2), 3),
    # a FrameConfig range wider than what is defined counts only the defined chirps
    ([(0, 0)], (0, 150), 1),
    ([(0, 0)], (0, 0), 1),
    ([(0, 1)], (0, 1), 2),
])
def test_a_frame_counts_each_chirp_it_sends_once(chirps, frame_range, expected):
    """
    Summing every ChirpConfig in a session log counted chirps the frame never sends and
    counted redefined ones twice; 92 captures decoded with every frame holding two.
    """
    from radar_processing import _effective_unique_chirps
    params = {"chirps": [{"chirpStartIdx": a, "chirpEndIdx": b} for a, b in chirps]}
    frame = {"chirpStartIdx": frame_range[0], "chirpEndIdx": frame_range[1]}
    assert _effective_unique_chirps(params, frame) == expected


def test_without_a_frame_range_the_defined_chirps_count_once():
    from radar_processing import _effective_unique_chirps
    params = {"chirps": [{"chirpStartIdx": 0, "chirpEndIdx": 2},
                         {"chirpStartIdx": 1, "chirpEndIdx": 1}]}
    assert _effective_unique_chirps(params, {}) == 3


def test_one_wavelength_for_every_view():
    """Roadmap 1.9: the parsed wavelength, the processing library and the vital-sign
    chain use the frequency in the middle of the ADC window; on the IWR6843 vitals
    capture the start of sampling would give displacements 1.89 % too large."""
    import capture_params
    import radar_processing as rp
    p, d = params_for(*CAPTURES[1][1:])
    _fill_velocity_and_range_limits(p)
    lam = capture_params.wavelength_m(p)
    assert rp.wavelength_mid_m(p) == pytest.approx(lam, rel=1e-12)
    assert p['wavelength_mm'] == pytest.approx(lam * 1000, rel=1e-12)
    lam_start = C_LIGHT / (d['f_eff'] * 1e9)
    assert lam_start / lam - 1 == pytest.approx(0.0189, abs=5e-4)
