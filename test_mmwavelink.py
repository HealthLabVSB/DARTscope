"""
Tests for acquisition.mmwavelink (RadarLinkDLL's work in Python, for macOS/Linux):
the RHCP message bytes, the SPI and GPIO bytes sent to the DCA1000's FTDI, and a whole
command/response/event exchange against a simulated radar.

No hardware: the FTDI ports are replaced by fakes. The golden message is one the
IWR6843AOPEVM accepted on 2026-10-05 (RfEnable, sequence number 0).

Run with:  python -m pytest test_mmwavelink.py -q
"""
import struct

import pytest

from acquisition import mmwavelink as ml


# --- protocol bytes -------------------------------------------------------------------------------

def test_crc_is_ccitt_false():
    assert ml.crc16(b"123456789") == 0x29B1


def test_rf_enable_matches_what_the_radar_accepted():
    raw = ml.build_command(ml.MSG_DEV_POWERUP, [(ml.sbid(ml.MSG_DEV_POWERUP, 0), b"")], seq=0)
    assert raw.hex() == "3412214305801200000000000100e77f00400400678b"


def test_header_fields_padding_and_crc():
    raw = ml.build_command(ml.MSG_RF_STATIC_SET, [(ml.sbid(4, 3), struct.pack("<HH", 0, 1))], seq=5)
    assert raw[:4] == ml.H2D_SYNC
    opcode, length, flags, rem, nsbc, chk = struct.unpack_from("<6H", raw, 4)
    assert opcode & 0xF == ml.DIR_HOST_TO_BSS and (opcode >> 4) & 3 == ml.CLASS_CMD
    assert opcode >> 6 == 4
    assert (flags >> 12) == 5 and flags & 0xFFF == 0           # seq 5, ACK, CRC 16, no retry
    assert nsbc == 1 and rem == 0
    assert ml.checksum(raw[4:14]) == chk
    assert length == len(raw) - 4                              # header + payload + pad + CRC
    assert (length - 2) % 4 == 0
    body = raw[4:-2]
    assert struct.unpack("<H", raw[-2:])[0] == ml.crc16(body)


def test_directions():
    assert ml.cmd_direction(0x04) == ml.DIR_HOST_TO_BSS
    assert ml.cmd_direction(0x202) == ml.DIR_HOST_TO_MSS
    with pytest.raises(ml.LinkError):
        ml.cmd_direction(0x80)


def test_too_long_a_payload_is_refused():
    with pytest.raises(ml.LinkError):
        ml.build_command(8, [(ml.sbid(8, 1), bytes(240))], seq=0)


# --- the FTDI bytes ---------------------------------------------------------------------------------

class FakePort:
    def __init__(self, pins=0):
        self.written = []
        self.pin_value = pins
        self.reads = []

    def write(self, data):
        self.written.append(bytes(data))

    def read(self, n, timeout=1.0):
        out, self.reads = self.reads[:n], self.reads[n:]
        return bytes(out)

    def pins(self):
        return self.pin_value


def test_spi_write_is_one_cs_frame_per_word_msb_first():
    f = ml.Dca1000Ftdi(usb_dev=object())
    f.spi = FakePort()
    f.spi_write(bytes([0x34, 0x12, 0x21, 0x43]))
    sent = b"".join(f.spi.written)
    word = lambda hi, lo: bytes([0x80, 0xC0, 0x4B, 0x11, 0x01, 0x00, hi, lo, 0x80, 0xC8, 0x4B])
    assert sent == word(0x12, 0x34) + word(0x43, 0x21) + bytes([0x87])


def test_spi_read_swaps_the_words_back():
    f = ml.Dca1000Ftdi(usb_dev=object())
    f.spi = FakePort()
    f.spi.reads = list(bytes([0xDC, 0xBA, 0xAB, 0xCD]))
    assert f.spi_read(4) == bytes([0xBA, 0xDC, 0xCD, 0xAB])
    cmd = f.spi.written[0]
    assert cmd[:9] == bytes([0x80, 0xC2, 0x4B, 0x20, 0x01, 0x00, 0x80, 0xC8, 0x4B]) and cmd[-1] == 0x87


def test_sop_and_reset_bits():
    f = ml.Dca1000Ftdi(usb_dev=object())
    f.gpio = FakePort(pins=0xFF)
    f.sop(2)                                       # D2 0, D3 1, D4 1 (rlsSOPControl)
    assert f.gpio.written[-1] == bytes([0xFF & ~0x04])
    f.gpio.pin_value = 0x00
    f.sop(4)
    assert f.gpio.written[-1] == bytes([0x10])
    f.board = FakePort(pins=0x46)
    f.nreset(0)
    assert f.board.written[-1] == bytes([0x06])
    f.board.pin_value = 0x06
    f.nreset(1)
    assert f.board.written[-1] == bytes([0x46])


# --- a simulated radar --------------------------------------------------------------------------------

def device_message(msg_id, blocks, msg_class, seq=0):
    payload = b"".join(struct.pack("<HH", sb, len(d) + 4) + d for sb, d in blocks)
    length = ml.HEADER_LEN + len(payload)
    pad = (-length) % 4
    payload += b"\xff" * pad
    length += pad + 2
    opcode = 6 | (msg_class << 4) | (msg_id << 6)              # MSS to host
    head = struct.pack("<5H", opcode, length, (seq & 0xF) << 12, 0, len(blocks))
    head += struct.pack("<H", ml.checksum(head))
    body = head + payload
    return ml.D2H_SYNC + body + struct.pack("<H", ml.crc16(body))


class FakeRadarFtdi:
    """HOST_IRQ high while a message waits; CNYS drops it; reads return the message."""

    def __init__(self):
        self.outbox = []            # messages the radar wants to send
        self.tx = b""               # bytes being clocked out
        self.cnys = False
        self.commands = []
        self.reply = {}             # msg id -> callable(seq) -> list of messages
        self.opened = False

    def open_comm(self):
        self.opened = True

    def close_comm(self):
        self.opened = False

    def close(self):
        pass

    def host_irq(self):
        return 1 if (self.outbox and not self.cnys) else 0

    def spi_write(self, data):
        if data == ml.H2D_CNYS:
            self.cnys = True
            self.tx = self.outbox.pop(0)
            return
        opcode, length, flags = struct.unpack_from("<3H", data, 4)
        assert data[:4] == ml.H2D_SYNC
        assert struct.unpack("<H", data[-2:])[0] == ml.crc16(data[4:-2])
        msg_id, seq = opcode >> 6, flags >> 12
        self.commands.append((msg_id, data))
        self.outbox += self.reply.get(msg_id, lambda s: [device_message(msg_id, [], ml.CLASS_RSP, s)])(seq)

    def spi_read(self, n):
        out, self.tx = self.tx[:n], self.tx[n:]
        out += b"\xff" * (n - len(out))
        if not self.tx:
            self.cnys = False
        return out


def test_power_on_rf_enable_and_an_error_answer():
    dev = FakeRadarFtdi()
    dev.outbox.append(device_message(0x280, [(0x5000, bytes(16))], ml.CLASS_ASYNC))   # MSS up
    dev.reply[ml.MSG_DEV_POWERUP] = lambda s: [device_message(0x200, [], ml.CLASS_RSP, s),
                                               device_message(0x280, [(0x5001, bytes(16))], ml.CLASS_ASYNC)]
    # SensorStop after the frames ended: an error response carrying error type 21
    dev.reply[ml.MSG_RF_FRAME_TRIG] = lambda s: [device_message(0, [(0, struct.pack("<HH", 21, 0x140))],
                                                                ml.CLASS_RSP, s)]
    link = ml.MmwLink(ftdi=dev)
    assert link.call("RadarLinkImpl_PowerOn", ("u32", 1), ("char", 0), ("u32", 1000), ("char", 0),
                     ("u32", 0), ("callback", 0)) == 0
    assert link.wait_event(0x5000, 0.5)["sbid"] == 0x5000
    assert link.call("RadarLinkImpl_RfEnable", ("u32", 1)) == 0
    assert link.wait_event(0x5001, 0.5) is not None
    assert link.call("RadarLinkImpl_ChannelConfig", ("u8", 1), ("buf", "0f00070000000000")) == 0
    assert link.call("RadarLinkImpl_SensorStop", ("u32", 1)) == 21
    seqs = [struct.unpack_from("<H", raw, 8)[0] >> 12 for _, raw in dev.commands]
    assert seqs == [0, 1, 2]


def test_frame_config_sends_the_chirp_count_too():
    dev = FakeRadarFtdi()
    dev.outbox.append(device_message(0x280, [(0x5000, bytes(16))], ml.CLASS_ASYNC))
    link = ml.MmwLink(ftdi=dev)
    link.call("RadarLinkImpl_PowerOn", ("u32", 1), ("char", 0), ("u32", 1000), ("char", 0), ("u32", 0))
    ret = link.call("RadarLinkImpl_FrameConfig", ("u32", 1), ("u16", 0), ("u16", 1), ("u16", 250),
                    ("u16", 255), ("u32", 8000000), ("u32", 0), ("u32", 300), ("u8", 0), ("u16", 1),
                    ("u16", 0))
    assert ret == 0
    (m1, raw1), (m2, raw2) = dev.commands
    assert (m1, m2) == (ml.MSG_RF_DYNAMIC_SET, ml.MSG_DEV_CONFIG_APPLY)
    frame = raw1[4 + 12 + 4:4 + 12 + 4 + 24]
    assert struct.unpack("<HHHHHHIHBBI", frame) == (0, 0, 1, 255, 250, 300, 8000000, 1, 0, 0, 0)
    apply_ = raw2[4 + 12 + 4:4 + 12 + 4 + 8]
    assert struct.unpack("<IHH", apply_) == (2 * 255, 300, 0)


def test_no_answer_is_a_timeout():
    dev = FakeRadarFtdi()
    dev.outbox.append(device_message(0x280, [(0x5000, bytes(16))], ml.CLASS_ASYNC))
    link = ml.MmwLink(ftdi=dev, ack_timeout=0.05)
    link.call("RadarLinkImpl_PowerOn", ("u32", 1), ("char", 0), ("u32", 50), ("char", 0), ("u32", 0))
    dev.reply[ml.MSG_RF_INIT] = lambda s: []
    assert link.call("RadarLinkImpl_RfInit", ("u32", 1)) == ml.RL_RESP_TIMEOUT


def test_a_damaged_response_is_caught_by_the_crc():
    dev = FakeRadarFtdi()
    dev.outbox.append(device_message(0x280, [(0x5000, bytes(16))], ml.CLASS_ASYNC))
    link = ml.MmwLink(ftdi=dev, ack_timeout=0.05)
    link.call("RadarLinkImpl_PowerOn", ("u32", 1), ("char", 0), ("u32", 50), ("char", 0), ("u32", 0))

    def bad(seq):
        m = bytearray(device_message(4, [], ml.CLASS_RSP, seq))
        m[-1] ^= 0xFF
        return [bytes(m)]
    dev.reply[4] = bad
    assert link.call("RadarLinkImpl_LowPowerConfig", ("u32", 1), ("u16", 0)) == ml.RL_CRC_FAILED
    assert len(dev.commands) == 4                    # sent again with the retry flag, 3 times


def test_unknown_function_is_refused():
    link = ml.MmwLink(ftdi=FakeRadarFtdi())
    with pytest.raises(ml.LinkError):
        link.call("RadarLinkImpl_Something")
