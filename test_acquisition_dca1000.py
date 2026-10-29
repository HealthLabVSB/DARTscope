"""
Tests for the DCA1000 protocol and controller.

The byte strings are what mmWave Studio sent to the lab's DCA1000 and what the board
answered, taken from a Wireshark capture of a real session (Wireshark/dca1000.pcapng).

Run with:  python -m pytest test_acquisition_dca1000.py -q
"""
import os
import struct

import pytest

from acquisition import dca1000 as d
from acquisition.config import Dca1000Config
from acquisition.simulator import Dca1000Simulator, free_udp_port, packetize, Faults

DATA = os.environ.get("DARTSCOPE_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PCAP = os.path.join(DATA, "Wireshark", "dca1000.pcapng")

# host -> DCA1000 in the captured session, for CaptureCardConfig_Mode(1, 2, 1, 2, 3, 30)
# and CaptureCardConfig_PacketDelay(25)
SENT = {
    "config_fpga": "5a a5 03 00 06 00 01 02 01 02 03 1e aa ee",
    "read_version": "5a a5 0e 00 00 00 aa ee",
    "config_packets": "5a a5 0b 00 06 00 be 05 35 0c 00 00 aa ee",
    "record_start": "5a a5 05 00 00 00 aa ee",
    "record_stop": "5a a5 06 00 00 00 aa ee",
}
# DCA1000 -> host
VERSION_REPLY = "5a a5 0e 00 82 04 aa ee"
COMPLETED_ASYNC = "5a a5 0a 00 00 01 aa ee"


def hx(b: bytes) -> str:
    return b.hex(" ")


def test_commands_are_byte_identical_to_mmwave_studios():
    cfg = Dca1000Config()
    assert hx(d.encode_command(d.Cmd.CONFIG_FPGA, d.fpga_config_data(cfg))) == SENT["config_fpga"]
    assert hx(d.encode_command(d.Cmd.READ_FPGA_VERSION)) == SENT["read_version"]
    assert hx(d.encode_command(d.Cmd.CONFIG_PACKET_DATA, d.packet_config_data(25))) == SENT["config_packets"]
    assert hx(d.encode_command(d.Cmd.RECORD_START)) == SENT["record_start"]
    assert hx(d.encode_command(d.Cmd.RECORD_STOP)) == SENT["record_stop"]


def test_responses_decode_as_ti_does():
    r = d.decode_response(bytes.fromhex(VERSION_REPLY))
    assert r.cmd == d.Cmd.READ_FPGA_VERSION and d.decode_fpga_version(r.status) == (2, 9, False)
    a = d.decode_response(bytes.fromhex(COMPLETED_ASYNC))
    assert a.is_async and a.status == d.ASYNC_RECORD_COMPLETED
    assert d.async_status_names(a.status) == ["record completed"]
    assert d.decode_response(b"\x00" * 8) is None


def test_packet_delay_limits_follow_the_cli():
    assert struct.unpack("<HHH", d.packet_config_data(5)) == (1470, 625, 0)
    for bad in (4, 501):
        with pytest.raises(ValueError):
            d.packet_config_data(bad)


def test_data_header_round_trip():
    pkt = d.encode_data_packet(11, 14560, b"\x01\x02")
    assert d.parse_data_header(pkt) == (11, 14560) and pkt[10:] == b"\x01\x02"


def test_packetize_counts_bytes_before_each_packet_and_applies_faults():
    data = bytes(range(256)) * 20                     # 5120 bytes: 3 full packets + 752
    pk = packetize(data)
    assert [(s, bc, len(p)) for s, bc, p in pk] == [(1, 0, 1456), (2, 1456, 1456),
                                                    (3, 2912, 1456), (4, 4368, 752)]
    pk = packetize(data, Faults(drop={2}, swap={3}, duplicate={1}, withhold_tail=128))
    assert [s for s, _, _ in pk] == [1, 1, 4, 3]
    assert len(pk[2][2]) == 752 - 128


def _controller_and_sim(timeout_s=2.0, **sim_kw):
    host_cfg, host_data = free_udp_port(), free_udp_port()
    cfg = Dca1000Config(host_ip="127.0.0.1", config_port=host_cfg, data_port=host_data)
    sim = Dca1000Simulator(b"", host_data_addr=("127.0.0.1", host_data), **sim_kw)
    ctl = d.Dca1000Controller(cfg, bind_ip="127.0.0.1", dca_addr=sim.addr, timeout_s=timeout_s)
    return ctl, sim


def test_controller_configures_like_mmwave_studio():
    ctl, sim = _controller_and_sim()
    try:
        with ctl:
            assert ctl.configure() == "2.9"
            ctl.start_record()
            ctl.stop_record()
        assert [hx(p) for p in ctl.sent] == [SENT["config_fpga"], SENT["read_version"],
                                             SENT["config_packets"], SENT["record_start"],
                                             SENT["record_stop"]]
        assert sim.fpga_mode == bytes([1, 2, 1, 2, 3, 30])
    finally:
        sim.close()


def test_async_record_completed_is_seen():
    ctl, sim = _controller_and_sim(completed_after_s=0.05)
    try:
        with ctl:
            import time
            ctl.configure()
            t0 = time.time()
            ctl.start_record()
            sim.start_stream()
            sim.wait_stream(5)
            for _ in range(50):
                if ctl.completed_since(t0):
                    break
                time.sleep(0.02)
            ev = ctl.completed_since(t0)
            assert ev is not None and ev.names == ["record completed"]
    finally:
        sim.close()


def test_a_missing_board_times_out_with_advice():
    port = free_udp_port()
    cfg = Dca1000Config(host_ip="127.0.0.1", config_port=free_udp_port())
    ctl = d.Dca1000Controller(cfg, bind_ip="127.0.0.1", dca_addr=("127.0.0.1", port), timeout_s=0.3)
    with ctl:
        with pytest.raises(d.Dca1000Timeout, match="no answer"):
            ctl.configure_fpga()
        assert ctl.system_aliveness() is False


def test_a_rejected_mode_is_an_error():
    ctl, sim = _controller_and_sim()
    ctl.cfg.lvds_mode = 3
    try:
        with ctl:
            with pytest.raises(d.Dca1000Error, match="status 1"):
                ctl.configure_fpga()
    finally:
        sim.close()


@pytest.mark.skipif(not os.path.isfile(PCAP), reason="the Wireshark capture is not next to the repository")
def test_the_captured_session_matches_the_encoder():
    """Every config command in the capture is one this module produces for the same settings."""
    sent, got = [], []
    with open(PCAP, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            btype, blen = struct.unpack("<II", hdr)
            body = f.read(blen - 8)
            if btype != 6:
                continue
            frame = body[20:20 + struct.unpack("<I", body[12:16])[0]]
            if len(frame) < 42 or frame[23] != 17:
                continue
            u = 14 + (frame[14] & 0x0F) * 4
            sport, dport, ulen = struct.unpack(">HHH", frame[u:u + 6])
            payload = frame[u + 8:u + ulen]                  # short frames carry Ethernet padding
            if dport == 4096 and sport == 4096:
                sent.append(hx(payload))
            elif dport == 4096:
                got.append(hx(payload))
    assert set(sent) == {SENT[k] for k in ("config_fpga", "read_version", "config_packets",
                                            "record_start", "record_stop")}
    assert VERSION_REPLY in got and COMPLETED_ASYNC in got
