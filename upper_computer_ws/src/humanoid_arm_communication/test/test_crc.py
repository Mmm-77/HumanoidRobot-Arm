"""Tests for CRC-16 implementation matching STM32 CRCs.c."""

from humanoid_arm_communication.crc import crc16_calculate, crc16_verify


def test_crc16_empty():
    """CRC-16 of empty data with init=0xFFFF should be 0xFFFF (or 0x0000? depends on impl)."""
    # With init=0xFFFF and no bytes processed, remains 0xFFFF
    assert crc16_calculate(b"") == 0xFFFF


def test_crc16_known_single_byte():
    """Verify CRC-16 for a single 0x00 byte."""
    # crc = 0xFFFF; byte=0x00
    # crc = (0xFFFF >> 8) ^ table[(0xFFFF ^ 0x00) & 0xFF]
    #     = 0x00FF ^ table[0xFF] = 0x00FF ^ 0x0F78 = 0x0F87
    assert crc16_calculate(b"\x00") == 0x0F87


def test_crc16_verify():
    """crc16_verify should return True for matching CRC."""
    data = b"\x01\x02\x03\x04"
    crc = crc16_calculate(data)
    assert crc16_verify(data, crc)
    assert not crc16_verify(data, 0x0000)


def test_crc16_deterministic():
    """Same input always produces same CRC."""
    data = b"\x38\x02" + b"\x00" * 200
    crc1 = crc16_calculate(data)
    crc2 = crc16_calculate(data)
    assert crc1 == crc2
