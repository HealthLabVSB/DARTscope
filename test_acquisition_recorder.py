"""
Tests for the recorder: DCA1000 datagrams -> _Raw_<n>.bin and _Raw_LogFile.csv as TI's
DCA1000EVM_CLI_Record writes them.

The CSV text is checked against a real one (dataset_square/M4_0test_Raw_LogFile.csv: 309
packets lost after packet 23), and read back with the loader's own parse_raw_log_csv.

Run with:  python -m pytest test_acquisition_recorder.py -q
"""
import os

import numpy as np
import pytest

from acquisition.dca1000 import encode_data_packet
from acquisition.recorder import RawRecorder, msvc_ctime
from acquisition.simulator import Faults, packetize
from dca1000_decode import parse_raw_log_csv
from radar_processing import get_all_bin_parts

T0 = 1_764_939_633 * 10 ** 9            # Fri Dec 05 2025, 14:00:33 local-ish; only the format matters


def stream(n_bytes: int, seed: int = 0) -> bytes:
    return np.random.default_rng(seed).integers(-2000, 2000, n_bytes // 2, dtype=np.int16).tobytes()


def record(tmp_path, data: bytes, faults: Faults | None = None, prefix="rec", **kw):
    rec = RawRecorder(str(tmp_path), prefix, **kw).open()
    for k, (seq, bc, payload) in enumerate(packetize(data, faults)):
        rec.feed(encode_data_packet(seq, bc, bytes(payload)), t_ns=T0 + k * 1000)
    st = rec.close()
    return rec, st


def read_all(rec) -> bytes:
    return b"".join(open(p, "rb").read() for p in rec.files)


def test_an_unbroken_stream_is_written_as_is(tmp_path):
    data = stream(1456 * 50 + 300)
    rec, st = record(tmp_path, data)
    assert read_all(rec) == data
    assert (st.first_seq, st.last_seq, st.received, st.zero_filled_packets) == (1, 51, 51, 0)
    info = parse_raw_log_csv(str(tmp_path / "rec_Raw_LogFile.csv"))
    assert info["dca_lane_mode"] == 2 and info["dca_log_mode"] == "Raw"
    assert info["dca_packets_received"] == 51 and info["dca_last_packet_id"] == 51
    assert info["dca_packets_zero_filled"] == 0 and info["dca_bytes_zero_filled"] == 0
    assert "dca_capture_start_timestamp" in info


def test_lost_packets_are_zero_filled_where_they_were(tmp_path):
    data = stream(1456 * 400)
    rec, st = record(tmp_path, data, Faults(drop=set(range(24, 333))))
    out = read_all(rec)
    assert len(out) == len(data)
    lost = slice(23 * 1456, 332 * 1456)
    assert out[lost] == bytes(309 * 1456)
    assert out[:23 * 1456] == data[:23 * 1456] and out[332 * 1456:] == data[332 * 1456:]
    assert (st.zero_filled_packets, st.zero_filled_bytes, st.out_of_sequence) == (309, 449904, 1)
    assert (st.oos_from, st.oos_to) == (23, 333)


def test_the_csv_is_ti_text(tmp_path):
    """Same event as dataset_square/M4_0test: packets 24..332 lost."""
    rec, st = record(tmp_path, stream(1456 * 400), Faults(drop=set(range(24, 333))))
    raw = open(tmp_path / "rec_Raw_LogFile.csv", "rb").read()
    text = raw.decode("ascii")
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")      # CRLF throughout
    head = ("Start record configuration : \r\n,\r\nLog mode : Raw\r\nLVDS lane mode : 2 lane\r\n"
            "Record stop mode : Infinite\r\nMax file size (MB) : 1024,\r\n,*DT 1,\r\n"
            ",33488,449904,\r\n\r\nRaw Data :\r\nOut of sequence count - 1\r\n"
            "Out of sequence seen from 23 to 333\r\nFirst Packet ID - 1\r\n"
            "Last Packet ID - 400\r\nNumber of received packets - 91\r\n"
            "Number of zero filled packets - 309\r\nNumber of zero filled bytes - 449904\r\n"
            "Capture start time - ")
    assert text.startswith(head)
    assert "\r\nCapture end time - " in text and text.rstrip().split("\r\n")[-1] == "Duration(sec) - 0"
    info = parse_raw_log_csv(str(tmp_path / "rec_Raw_LogFile.csv"))
    assert info["dca_packets_zero_filled"] == 309 and info["dca_bytes_zero_filled"] == 449904


def test_msvc_ctime():
    import time
    t = time.mktime((2025, 12, 5, 14, 0, 33, 0, 0, -1))
    assert msvc_ctime(t) == "Fri Dec 05 14:00:33 2025\n"


def test_a_late_packet_is_put_back(tmp_path):
    data = stream(1456 * 30)
    rec, st = record(tmp_path, data, Faults(swap={10, 20}))
    assert read_all(rec) == data
    assert (st.late_recovered, st.zero_filled_packets, st.zero_filled_bytes) == (2, 0, 0)
    assert st.received == 30


def test_duplicates_are_left_out(tmp_path):
    data = stream(1456 * 30)
    rec, st = record(tmp_path, data, Faults(duplicate={5, 6}))
    assert read_all(rec) == data and st.duplicates == 2 and st.zero_filled_packets == 0


def test_a_short_tail_is_just_shorter(tmp_path):
    """The lab's DCA1000 kept the last 128 bytes back; nothing is padded to make up for it."""
    data = stream(1456 * 20)
    rec, st = record(tmp_path, data, Faults(withhold_tail=128))
    assert read_all(rec) == data[:-128]


def test_parts_are_cut_after_whole_packets(tmp_path):
    """1 MiB parts end after 720 packets (1 048 320 bytes), as 1 GiB ones after 737 460."""
    data = stream(1456 * 1600 + 100)
    rec, st = record(tmp_path, data, max_file_mb=1)
    sizes = [os.path.getsize(p) for p in rec.files]
    assert sizes[:-1] == [720 * 1456] * (len(sizes) - 1)
    assert [os.path.basename(p) for p in rec.files] == [f"rec_Raw_{i}.bin" for i in range(len(sizes))]
    assert get_all_bin_parts(rec.files[0]) == rec.files
    assert read_all(rec) == data


def test_a_late_packet_can_go_back_into_an_earlier_part(tmp_path):
    """Packet 721 arrives before 720, whose place is the end of the first part."""
    data = stream(1456 * 1600)
    rec, st = record(tmp_path, data, Faults(swap={720}), max_file_mb=1)
    assert os.path.getsize(rec.files[0]) == 720 * 1456
    assert read_all(rec) == data and st.late_recovered == 1


def test_lost_first_packets_keep_the_file_frame_aligned(tmp_path):
    data = stream(1456 * 20)
    rec, st = record(tmp_path, data, Faults(drop={1, 2}))
    out = read_all(rec)
    assert len(out) == len(data) and out[:2 * 1456] == bytes(2 * 1456)
    assert out[2 * 1456:] == data[2 * 1456:]
    assert st.first_seq == 3 and st.leading_zero_bytes == 2 * 1456


def test_nothing_is_overwritten(tmp_path):
    record(tmp_path, stream(1456 * 3))
    with pytest.raises(FileExistsError):
        RawRecorder(str(tmp_path), "rec").open()


def test_no_packets_still_leaves_a_ti_csv(tmp_path):
    rec = RawRecorder(str(tmp_path), "empty").open()
    rec.close()
    info = parse_raw_log_csv(str(tmp_path / "empty_Raw_LogFile.csv"))
    assert info["dca_packets_received"] == 0 and rec.files == []
