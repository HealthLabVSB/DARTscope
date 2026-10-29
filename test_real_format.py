"""
Tests for captures recorded in REAL (not complex) ADC format.

Written against a real AWR1642 capture (adc_data, 77 GHz, 1 TX, 150 samples at 8 MSps,
25 fps) that was recorded with LaneConfig=1 into a DCA1000 running in 2-lane mode. Every
one of those defects showed up on that file.

Run with:  python -m pytest test_real_format.py -q
"""
import numpy as np
import pytest

import vitalsigns as vs
from dca1000_decode import decode_tx_enable, parse_frame_config_line
from radar_processing import _drop_zero_padded_slots, _real_container_is_doubled


# --- unused LVDS slots ----------------------------------------------------------

def test_zero_padded_slots_are_dropped():
    """
    The DCA1000 writes the slots its own lane mode reserves even when the radar drives
    fewer lanes, so the file reads sample, 0, sample, 0, ... Taken at face value that is
    a zero-stuffed signal: the spectrum mirrors and the range scale halves.
    """
    rng = np.random.default_rng(0)
    real = rng.integers(-500, 500, 20000).astype(np.int16)
    padded = np.zeros(2 * real.size, dtype=np.int16)
    padded[0::2] = real

    out, stride, keep = _drop_zero_padded_slots(padded, is_real_format=True)
    assert stride == 2 and keep == [0]
    assert np.array_equal(out, real)


def test_a_dense_real_stream_is_left_alone():
    rng = np.random.default_rng(1)
    dense = rng.integers(-500, 500, 20000).astype(np.int16)
    dense[dense == 0] = 1
    out, stride, _ = _drop_zero_padded_slots(dense, is_real_format=True)
    assert stride == 1 and np.array_equal(out, dense)


def test_complex_streams_are_never_touched():
    """Only REAL captures; a complex stream's Q half must not be mistaken for padding."""
    data = np.zeros(20000, dtype=np.int16)
    data[0::2] = np.arange(10000) % 500 - 250
    out, stride, _ = _drop_zero_padded_slots(data, is_real_format=False)
    assert stride == 1 and np.array_equal(out, data)


def test_an_all_zero_file_is_not_emptied():
    z = np.zeros(20000, dtype=np.int16)
    out, stride, _ = _drop_zero_padded_slots(z, is_real_format=True)
    assert stride == 1 and out.size == z.size


def test_zero_stuffing_halves_the_range_scale():
    """The consequence the fix exists to prevent, measured rather than asserted."""
    ns, fs_adc, slope, c = 150, 8e6, 99.987e12, 3e8
    t = np.arange(ns) / fs_adc
    truth_m = 1.44
    beat = 2 * truth_m * slope / c
    chirp = np.cos(2 * np.pi * beat * t)

    def peak_range(sig, n):
        spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        f = np.fft.rfftfreq(n, d=1 / fs_adc)
        return float(f[np.argmax(spec)] * c / (2 * slope))

    stuffed = np.zeros(2 * ns)
    stuffed[0::2] = chirp
    assert abs(peak_range(chirp, ns) - truth_m) < 0.1
    # Read as 150 samples, the zero-stuffed version puts the target at half the range.
    assert abs(peak_range(stuffed[:ns], ns) - truth_m / 2) < 0.1


# --- unused slots that are not zeros --------------------------------------------

def _beat_chirps(n_chirps, ns=150, fs=8e6, ranges=(0.9, 1.44, 2.6)):
    """A stack of chirps holding a few targets - smooth, the way ADC samples are."""
    t = np.arange(ns) / fs
    slope, c = 99.987e12, 3e8
    sig = sum(np.cos(2 * np.pi * (2 * r * slope / c) * t + i * 0.01) for i, r in enumerate(ranges))
    return np.tile(sig, n_chirps) * 300.0


def test_real_container_with_non_zero_surplus_slots_is_detected():
    """
    A 2-lane recorder writes groups of four int16 in REAL mode and fills only the first
    two. When the surplus slots hold something other than zeros they get past
    _drop_zero_padded_slots, and reading the stream straight through then doubles the
    frame count. On the capture this was found with it also mirrored the spectrum about
    its midpoint, though that depends on what the surplus slots happen to carry.
    """
    rng = np.random.default_rng(3)
    samples = _beat_chirps(400)
    surplus = rng.integers(-300, 300, samples.size).astype(np.float64)
    stream = np.empty(2 * samples.size, dtype=np.int16)
    stream.reshape(-1, 4)[:, 0:2] = samples.reshape(-1, 2)
    stream.reshape(-1, 4)[:, 2:4] = surplus.reshape(-1, 2)

    assert _real_container_is_doubled(stream, lanes=2, expected_samples=samples.size)
    # the size check alone must not carry it
    assert not _real_container_is_doubled(stream, lanes=2, expected_samples=samples.size // 2)


def test_a_genuinely_sequential_real_stream_is_left_alone():
    """Decimating a real stream never makes it smoother, so a long recording is safe."""
    samples = _beat_chirps(800).astype(np.int16)
    assert not _real_container_is_doubled(samples, lanes=2, expected_samples=samples.size // 2)
    assert not _real_container_is_doubled(samples, lanes=2, expected_samples=None)


def test_noise_is_not_re_cut():
    rng = np.random.default_rng(4)
    noise = rng.integers(-500, 500, 1 << 17).astype(np.int16)
    assert not _real_container_is_doubled(noise, lanes=2, expected_samples=noise.size // 2)


# --- damaged-frame detection on REAL data ---------------------------------------

def test_real_data_is_not_mistaken_for_packet_loss():
    """
    A real-valued ADC signal crosses zero constantly - the capture this was found with
    sits at 50 % exact zeros with no loss at all. Counting zeros flagged every frame;
    only a long unbroken run means a zero-filled packet.
    """
    rng = np.random.default_rng(2)
    cube = rng.integers(-400, 400, (20, 50, 4, 150)).astype(np.float32)
    assert (cube == 0).mean() > 0.001            # plenty of incidental zeros
    assert not vs.detect_damaged_frames(cube).any()


def test_a_zero_filled_packet_is_still_caught_in_real_data():
    rng = np.random.default_rng(3)
    cube = rng.integers(-400, 400, (20, 50, 4, 150)).astype(np.float32)
    cube[7, 10:20] = 0                            # a contiguous block, as a lost packet
    dmg = vs.detect_damaged_frames(cube)
    assert dmg.sum() == 1 and dmg[7]


# --- log parsing ----------------------------------------------------------------

@pytest.mark.parametrize("mask,expected", [
    (0, "None"), (1, "TX1"), (2, "TX2"), (3, "TX1,TX2"), (4, "TX3"),
    (7, "TX1,TX2,TX3"), (12, "TX3,TX4"), (15, "TX1,TX2,TX3,TX4"),
])
def test_tx_masks_are_named_the_way_ti_numbers_them(mask, expected):
    """Bit 0 is TX1, not TX0, and a four-transmitter mask must not lose a bit."""
    assert decode_tx_enable(mask) == expected


def test_frame_layout_uses_the_defined_chirps():
    """
    This capture logs FrameConfig as 0,0,150,200,8000000,0,150,0. Read with a leading
    frame index it claims chirps 0..150; ChirpConfig defines only chirp 0, which settles
    it - and the data agrees: the phase jump at the inter-frame gap repeats every 200
    chirps, so 200 is the loop count.
    """
    line = "API:FrameConfig,0,0,150,200,8000000,0,150,0,"
    fc = parse_frame_config_line(line, chirp_ranges=[(0, 0)])
    assert fc['chirpStartIdx'] == 0 and fc['chirpEndIdx'] == 0
    assert fc['numLoops'] == 200
    assert fc['chirpsPerFrame'] == 200
    assert fc['framePeriodicity_ms'] == 40.0


def test_frame_layout_without_chirp_ranges_prefers_a_sane_chirp_count():
    fc = parse_frame_config_line("API:FrameConfig,0,0,150,200,8000000,0,150,0,")
    assert fc['chirpsPerFrame'] == 200
