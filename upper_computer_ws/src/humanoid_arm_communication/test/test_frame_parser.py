"""Tests for frame_parser: fixed-size byte stream fragmentation."""

from humanoid_arm_communication.frame_parser import FrameParser
from humanoid_arm_communication.protocol import UPLOAD_FRAME_SIZE


def test_empty_feed():
    parser = FrameParser()
    frames = parser.feed(b"")
    assert frames == []


def test_exact_one_frame():
    parser = FrameParser()
    frame = b"\x00" * UPLOAD_FRAME_SIZE
    result = parser.feed(frame)
    assert len(result) == 1
    assert result[0] == frame


def test_partial_frame():
    parser = FrameParser()
    partial = b"\x00" * 50
    result = parser.feed(partial)
    assert result == []
    assert parser.buffered_bytes == 50


def test_accumulate_to_full_frame():
    parser = FrameParser()
    _ = parser.feed(b"\x00" * 50)
    result = parser.feed(b"\x00" * (UPLOAD_FRAME_SIZE - 50))
    assert len(result) == 1
    assert parser.buffered_bytes == 0


def test_multiple_frames():
    parser = FrameParser()
    raw = b"\xAA" * UPLOAD_FRAME_SIZE + b"\xBB" * UPLOAD_FRAME_SIZE
    result = parser.feed(raw)
    assert len(result) == 2
    assert result[0] == b"\xAA" * UPLOAD_FRAME_SIZE
    assert result[1] == b"\xBB" * UPLOAD_FRAME_SIZE


def test_reset():
    parser = FrameParser()
    _ = parser.feed(b"\x00" * 50)
    parser.reset()
    assert parser.buffered_bytes == 0
