"""
Ground-truth tests for the AoA chain.

A synthetic target is placed at a known angle, the board described by the antenna
configuration is emulated (element positions AND any per-element phase), and the real
beamformers have to recover the angle.

The per-element phase matters: an IWR6843AOP has RX1/RX3 rotated 180 degrees on the
package. That was verified on a real TDM-MIMO capture - the six virtual element pairs
sharing an x position measured 149 degrees apart, and correcting for it put the target
at -4 degrees elevation instead of a physically impossible +70.

Run with:  python -m pytest test_aoa_geometry.py -q
"""
import json
import numpy as np
import pytest

import antenna_config as ac
from AOA import (
    _get_antenna_geometry,
    _resolve_antenna_config,
    compute_range_azimuth_map,
    compute_aoa_spectrum,
    compute_azimuth_elevation_map,
    compute_range_azimuth_elevation_map,
)

RX, NTX, FRAMES, LOOPS, RANGE_BINS, TARGET_BIN = 4, 3, 2, 8, 64, 20
ANGLES = np.arange(-90.0, 91.0, 1.0)
TRUTH_ANGLES = [-60, -40, -20, -10, 0, 10, 20, 40, 60]


def base_params(cfg=None):
    params = {
        'chip_hint': 'IWR6843',          # real logs carry no "AOP" suffix
        'is_frequency_domain': True,
        'uniqueChirps_effective': NTX,
        'unique_chirps_frame': NTX,
        'tx_order_masks': [1, 2, 4],
        'numAdcSamples': RANGE_BINS,
        'digOutSampleRate_MHz': 5.0,
        'freqSlopeMHz_us': 60.0,
    }
    if cfg is not None:
        params['antenna_config'] = cfg
    return params


def synth_cube(params, az_deg, el_deg=0.0):
    """
    Noiseless wavefront from one target, laid out as TDM-MIMO chirps.

    The board is emulated exactly as its configuration describes it, including the
    per-element phase, so the processing chain has to undo that phase to recover the
    true angle. Without this the test would only prove the code is consistent with
    itself.
    """
    x, y, _ = _get_antenna_geometry(params, NTX * RX, RX)
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    steer = np.exp(1j * 2 * np.pi * (x * np.sin(az) * np.cos(el) + y * np.sin(el)))

    cfg = _resolve_antenna_config(params, NTX * RX, RX)
    factors = ac.element_phase_factors(cfg, NTX * RX) if cfg else None
    if factors is not None:
        steer = steer * np.conj(factors)   # the hardware imprints the inverse of the fix

    cube = np.zeros((FRAMES, LOOPS * NTX, RX, RANGE_BINS), dtype=np.complex64)
    for tx in range(NTX):
        for rx in range(RX):
            cube[:, tx::NTX, rx, TARGET_BIN] = steer[tx * RX + rx]
    return cube


@pytest.mark.parametrize("truth", TRUTH_ANGLES)
def test_range_azimuth_recovers_known_angle(truth):
    params = base_params()          # auto-detect
    ra_map, meta = compute_range_azimuth_map(
        synth_cube(params, truth), params, angles_deg=ANGLES, remove_static=False)
    peak = float(meta['angles_deg'][int(np.argmax(ra_map[0, TARGET_BIN]))])
    assert abs(peak - truth) <= 2.0, f"expected {truth} deg, got {peak} deg"


@pytest.mark.parametrize("truth", TRUTH_ANGLES)
def test_1d_spectrum_recovers_known_angle(truth):
    params = base_params()
    spec, meta = compute_aoa_spectrum(
        synth_cube(params, truth), params, range_bin=TARGET_BIN,
        angles_deg=ANGLES, remove_static=False)
    peak = float(meta['angles_deg'][int(np.argmax(spec[0]))])
    assert abs(peak - truth) <= 2.0, f"expected {truth} deg, got {peak} deg"


@pytest.mark.parametrize("truth", [-30, 0, 30])
def test_explicit_aop_config_matches_auto_detection(truth):
    """Selecting the AOP preset explicitly must equal what auto-detection does."""
    cfg = ac.get_config(ac.AOP_DEFAULT)
    assert cfg is not None
    explicit = base_params(cfg)
    auto = base_params()
    peaks = []
    for params in (explicit, auto):
        ra_map, meta = compute_range_azimuth_map(
            synth_cube(params, truth), params, angles_deg=ANGLES, remove_static=False)
        peaks.append(float(meta['angles_deg'][int(np.argmax(ra_map[0, TARGET_BIN]))]))
    assert abs(peaks[0] - truth) <= 2.0
    assert peaks[0] == peaks[1]


@pytest.mark.parametrize("truth", [-30, 0, 30])
def test_board_without_rotated_elements(truth):
    """A board that does not have the rotated RX is served by the no-correction preset."""
    params = base_params(ac.get_config("IWR6843AOP - no phase correction"))
    ra_map, meta = compute_range_azimuth_map(
        synth_cube(params, truth), params, angles_deg=ANGLES, remove_static=False)
    peak = float(meta['angles_deg'][int(np.argmax(ra_map[0, TARGET_BIN]))])
    assert abs(peak - truth) <= 2.0, f"expected {truth} deg, got {peak} deg"


def test_wrong_phase_convention_ruins_elevation():
    """
    The RX phase convention is not cosmetic. On the real capture the wrong one moved the
    target from -4 to +70 degrees elevation; reproduced here in miniature.
    """
    correct = ac.get_config(ac.AOP_DEFAULT)
    wrong = ac.get_config("IWR6843AOP - no phase correction")
    cube = synth_cube(base_params(correct), 10.0, el_deg=0.0)

    elevations = []
    for cfg in (correct, wrong):
        spec, meta = compute_azimuth_elevation_map(
            cube, base_params(cfg), range_bin=TARGET_BIN, remove_static=False,
            angles_az_deg=np.arange(-60, 61, 2.0), angles_el_deg=np.arange(-80, 81, 2.0))
        el_i, _ = np.unravel_index(np.argmax(spec[0]), spec[0].shape)
        elevations.append(float(meta['angles_el_deg'][el_i]))

    assert abs(elevations[0]) <= 6.0, \
        f"correct convention should give ~0 deg elevation, got {elevations[0]}"
    assert abs(elevations[1]) > 40.0, \
        f"the wrong convention should throw elevation far off, got {elevations[1]}"


def test_elevation_is_recovered():
    params = base_params(ac.get_config(ac.AOP_DEFAULT))
    az_truth, el_truth = 10.0, 20.0
    cube = synth_cube(params, az_truth, el_truth)
    rae, meta = compute_range_azimuth_elevation_map(
        cube, params, angles_az_deg=np.arange(-60, 61, 2.0),
        angles_el_deg=np.arange(-40, 41, 2.0), remove_static=False, method="bartlett")
    frame = rae[0, TARGET_BIN - meta['r_start']]
    el_i, az_i = np.unravel_index(np.argmax(frame), frame.shape)
    assert abs(float(meta['angles_az_deg'][az_i]) - az_truth) <= 4.0
    assert abs(float(meta['angles_el_deg'][el_i]) - el_truth) <= 6.0


def test_builtin_configs_are_valid():
    for cfg in ac.BUILTIN_CONFIGS:
        ok, msg = ac.validate(cfg)
        assert ok, f"{cfg['name']}: {msg}"
        x, y = ac.build_geometry(cfg)
        assert len(x) == len(cfg['rx_positions']) * len(cfg['tx_offsets'])
        assert np.isclose(np.mean(x), 0.0) and np.isclose(np.mean(y), 0.0)


def test_ula_preset_has_no_elevation():
    cfg = ac.get_config("Azimuth ULA - 2TX x 4RX (8 virtual)")
    x, y = ac.build_geometry(cfg)
    assert np.allclose(y, 0.0)
    assert np.allclose(np.diff(np.sort(x)), 0.5)   # uniform half-wavelength pitch


def test_linear_platform_opts_out_of_aop_autodetection():
    """A capture explicitly flagged as linear must not get the 2D AOP layout."""
    params = base_params()
    params['platform'] = 'ULA'
    assert _resolve_antenna_config(params, NTX * RX, RX) is None
    _, y, _ = _get_antenna_geometry(params, NTX * RX, RX)
    assert np.allclose(y, 0.0)


@pytest.mark.parametrize("fn_name", ["range_az_el", "az_el"])
def test_clutter_removal_does_not_null_the_map(fn_name):
    """
    Clutter removal subtracts the mean over chirp loops. A Bartlett stage that then
    averages the complex snapshots takes that same mean, so the result is identically
    zero and the map collapses into numerical noise.

    On a real recording this showed up as a 30-million-voxel map spanning just 2.4 dB,
    which made every display threshold select the entire volume. The power has to be
    averaged over the snapshots instead.
    """
    params = base_params()
    # A moving target: its phase advances from loop to loop, so clutter removal must
    # keep it while removing anything static.
    x, y, _ = _get_antenna_geometry(params, NTX * RX, RX)
    cfg = _resolve_antenna_config(params, NTX * RX, RX)
    factors = ac.element_phase_factors(cfg, NTX * RX) if cfg else None
    steer = np.exp(1j * 2 * np.pi * x * np.sin(np.deg2rad(20.0)))
    if factors is not None:
        steer = steer * np.conj(factors)

    cube = np.zeros((1, LOOPS * NTX, RX, RANGE_BINS), dtype=np.complex64)
    for tx in range(NTX):
        for rx in range(RX):
            ch = steer[tx * RX + rx]
            for loop in range(LOOPS):
                doppler = np.exp(1j * 2 * np.pi * 0.25 * loop)      # moving target
                cube[0, loop * NTX + tx, rx, TARGET_BIN] = ch * doppler
                cube[0, loop * NTX + tx, rx, TARGET_BIN + 10] = ch  # static clutter

    if fn_name == "range_az_el":
        out, meta = compute_range_azimuth_elevation_map(
            cube, params, angles_az_deg=np.arange(-60, 61, 2.0),
            angles_el_deg=np.arange(-20, 21, 2.0), remove_static=True, method="bartlett")
        spread = float(out.max() - out.min())
        moving = out[0, TARGET_BIN - meta['r_start']].max()
        static = out[0, TARGET_BIN + 10 - meta['r_start']].max()
    else:
        out, meta = compute_azimuth_elevation_map(
            cube, params, range_bin=TARGET_BIN, remove_static=True,
            angles_az_deg=np.arange(-60, 61, 2.0), angles_el_deg=np.arange(-20, 21, 2.0))
        spread = float(out.max() - out.min())
        moving, static = out.max(), None

    assert spread > 20.0, f"map collapsed to a {spread:.1f} dB spread - clutter removal nulled it"
    if static is not None:
        assert moving - static > 20.0, (
            f"the moving target should survive clutter removal and the static one should "
            f"not; got {moving:.1f} dB vs {static:.1f} dB")


# --- which board the log is taken to describe ------------------------------------

def test_aop_is_detected_on_a_single_tx_capture():
    """
    A board is what it is whether it transmits on one TX or three.

    Auto-detection used to require 12 virtual elements, so a single-TX vital-signs
    capture on an AOP fell through to the linear fallback: 1.5 lambda of azimuth
    aperture instead of the 0.5 lambda the package actually has, and the per-element
    phase correction silently switched off.
    """
    params = base_params()
    params['tx_order_masks'] = [2]
    cfg = _resolve_antenna_config(params, RX, RX)
    assert cfg is not None and "AOP" in cfg['name']

    x, y, _ = _get_antenna_geometry(params, RX, RX)
    assert x.max() - x.min() == pytest.approx(0.5)     # 2x2 grid, not a 1.5 lambda row
    assert y.max() - y.min() == pytest.approx(0.5)
    ph = ac.element_phase_factors(cfg, RX)
    assert ph is not None
    assert np.allclose(np.rad2deg(np.angle(ph)) % 360, [180, 0, 180, 0])


def test_more_transmitters_widen_the_same_board():
    """The preset has to grow with the TX set, not change board."""
    span = {}
    for masks, n in (([2], 4), ([1, 2], 8), ([1, 2, 4], 12)):
        x, _ = ac.build_geometry(ac.get_config(ac.AOP_DEFAULT), masks=masks,
                                 num_virt_rx=n, num_phys_rx=RX)
        span[n] = x.max() - x.min()
    assert span[4] == pytest.approx(0.5)
    assert span[8] == pytest.approx(1.5)
    assert span[8] > span[4] and span[12] >= span[8]


def test_explicit_selection_still_beats_the_chip_default():
    cfg = ac.get_config("Azimuth ULA - 2TX x 4RX (8 virtual)")
    params = base_params(cfg)
    params['tx_order_masks'] = [1, 2]
    assert _resolve_antenna_config(params, 8, RX)['name'] == cfg['name']


def test_linear_platform_still_opts_out_with_one_tx():
    params = base_params()
    params['tx_order_masks'] = [1]
    params['platform'] = 'ULA'
    assert _resolve_antenna_config(params, RX, RX) is None


# --- teaching the auto-detection a new board -------------------------------------

def _write_user_file(tmp_path, payload):
    (tmp_path / ac.CONFIG_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def test_a_new_board_can_be_added_without_touching_the_code(tmp_path):
    board = {
        "name": "Bench board",
        "rx_positions": [[0.0, 0.0], [0.5, 0.0]],
        "tx_offsets": [[0.0, 0.0]],
        "tx_masks": [1],
    }
    _write_user_file(tmp_path, {"configs": [board], "chip_defaults": {"XWR9999": "Bench board"}})
    assert ac.get_config("Bench board", str(tmp_path))['name'] == "Bench board"
    got = ac.default_config_for_chip("XWR9999ABC", str(tmp_path))
    assert got is not None and got['name'] == "Bench board"


def test_the_longest_chip_match_wins(tmp_path):
    """An 'IWR6843' entry must not shadow a more specific 'IWR6843AOP' one."""
    _write_user_file(tmp_path, {"chip_defaults": {
        "IWR6843": "Azimuth ULA - 2TX x 4RX (8 virtual)",
        "IWR6843AOP": ac.AOP_DEFAULT,
    }})
    assert ac.default_config_for_chip("IWR6843AOP", str(tmp_path))['name'] == ac.AOP_DEFAULT
    assert "ULA" in ac.default_config_for_chip("IWR6843", str(tmp_path))['name']


def test_the_original_list_format_still_loads(tmp_path):
    board = {
        "name": "Legacy board",
        "rx_positions": [[0.0, 0.0], [0.5, 0.0]],
        "tx_offsets": [[0.0, 0.0]],
        "tx_masks": [1],
    }
    _write_user_file(tmp_path, [board])
    names = [c['name'] for c in ac.load_user_configs(str(tmp_path))]
    assert "Legacy board" in names


def test_a_malformed_entry_is_skipped_not_fatal(tmp_path):
    _write_user_file(tmp_path, {"configs": [{"name": "Broken"}]})
    assert ac.load_user_configs(str(tmp_path)) == []


def test_a_chip_default_naming_a_missing_config_is_not_fatal(tmp_path):
    _write_user_file(tmp_path, {"chip_defaults": {"XWR0000": "Nope"}})
    assert ac.default_config_for_chip("XWR0000", str(tmp_path)) is None


def test_awr2544_preset_reproduces_the_published_virtual_array():
    """
    Figure 2-14 of SPRUJB0, read off the drawing in half-wavelength cells.

    The board is 4 TX x 4 RX, so the 16 virtual elements are fully determined; if the
    RX spacings or the TX step were off, the two rows below could not both come out.
    """
    cfg = ac.get_config("AWR2544LOPEVM (4TX x 4RX, waveguide)")
    x, y = ac.build_geometry(cfg, masks=[1, 2, 4, 8], num_virt_rx=16, num_phys_rx=4)
    cells_x = np.round((x - x.min()) * 2).astype(int)      # in lambda/2
    cells_y = np.round((y - y.min()) * 2).astype(int)

    lower = sorted(cells_x[cells_y == 0])
    upper = sorted(cells_x[cells_y == 1])
    assert lower == [0, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 15]
    assert upper == [9, 11, 16, 18]
    assert set(cells_y) == {0, 1}                          # one lambda/2 elevation step
    assert cells_x.max() - cells_x.min() == 18             # 9 lambda of azimuth


def test_awr2544_compensates_the_out_of_phase_receivers():
    """The user guide says RX1 and RX4 are 180 degrees out of phase."""
    cfg = ac.get_config("AWR2544LOPEVM (4TX x 4RX, waveguide)")
    ph = np.rad2deg(np.angle(ac.element_phase_factors(cfg, 16))) % 360
    assert np.allclose(ph.reshape(4, 4), [[180, 0, 0, 180]] * 4)


def test_awr2544_is_auto_detected_from_the_chip_name():
    cfg = ac.default_config_for_chip("AWR2544")
    assert cfg is not None and cfg['name'].startswith("AWR2544LOPEVM")
