#!/usr/bin/env python3
"""End-to-end test: encode a joint command, then simulate the lower computer
echoing back with the same angles.

Run without ROS or hardware:
    cd upper_computer_ws/src/humanoid_arm_communication
    python3 test/run_e2e.py
"""

import struct
import sys
import numpy as np

# Make the package importable from the parent directory
sys.path.insert(0, "..")

from humanoid_arm_communication.command_codec import encode_command
from humanoid_arm_communication.feedback_codec import decode_feedback, FeedbackFrame
from humanoid_arm_communication.crc import crc16_calculate
from humanoid_arm_communication.protocol import (
    ControlMode,
    HEADER_UPLOAD_UPPER,
    UPLOAD_FRAME_SIZE,
    UPLOAD_PAYLOAD_SIZE,
    DOWNLOAD_FRAME_SIZE,
)
from humanoid_arm_communication.feedback_codec import _UPLOAD_STRUCT


def build_echo_feedback(download_frame: bytes, add_error: int = 0) -> bytes:
    """Simulate the lower computer echoing back motor angles.

    Reads the target angles from the download frame and packs them
    into an upload frame as if the motors reached those positions.

    Args:
        download_frame: The 206-byte download frame.
        add_error: Bitmask to set on motor 1 (0 = no error).
    """
    # Unpack the pos array from the download frame at offset 4
    pos_floats = struct.unpack_from("<10f", download_frame, offset=4)

    # Build the upload payload
    errors = np.zeros(10, dtype=np.uint8)
    if add_error:
        errors[1] = add_error

    payload = _UPLOAD_STRUCT.pack(
        HEADER_UPLOAD_UPPER,               # B: head
        *([0.0, 0.0, 0.0, 1.0]),           # 4f: IMU quat (identity)
        *([0.0, 0.0, 0.0]),                # 3f: IMU gyro
        0x00,                               # B: IMU state
        *pos_floats,                        # 10f: Motor_Angle (echo target)
        *([0.0] * 10),                      # 10f: Motor_Speed (zero)
        *([0.0] * 10),                      # 10f: Motor_Torque (zero)
        *errors,                            # 10B: Motor_Error
    )

    assert len(payload) == UPLOAD_PAYLOAD_SIZE
    crc = struct.pack("<H", crc16_calculate(payload))
    return payload + crc


def main():
    passed = 0
    failed = 0

    # --- Test 1: basic encode ---
    print("=== Test 1: Basic encode ===")
    angles = np.radians([10, 20, -30, 45])
    velocities = np.radians([5, -3, 2, 0])
    cmd = encode_command(angles, velocities, ctrl_mode=ControlMode.MIT)

    assert len(cmd.data) == DOWNLOAD_FRAME_SIZE, f"Size mismatch: {len(cmd.data)}"
    assert cmd.data[0] == 0x38, f"Wrong header: {cmd.data[0]:#x}"
    assert cmd.data[1] == 0x02, f"Wrong ctrl mode: {cmd.data[1]}"
    print(f"  Frame size: {len(cmd.data)} bytes ✓")
    print(f"  Header: 0x{cmd.data[0]:02X} ✓")
    print(f"  CtrlMode: {cmd.data[1]} (MIT) ✓")
    print(f"  Target angles (deg): {cmd.joint_angles_deg}")

    # --- Test 2: roundtrip encode → simulate feedback → decode ---
    print("\n=== Test 2: Roundtrip encode → echo → decode ===")
    upload = build_echo_feedback(cmd.data)
    assert len(upload) == UPLOAD_FRAME_SIZE, f"Upload size: {len(upload)}"

    fb = decode_feedback(upload)
    assert fb is not None, "CRC failed on valid data!"

    tolerance_rad = np.deg2rad(0.1)  # 0.1 deg tolerance
    for i in range(4):
        diff = abs(fb.joint_angles_rad[i] - angles[i])
        assert diff < tolerance_rad, f"Joint {i+1}: {fb.joint_angles_rad[i]:.4f} != {angles[i]:.4f} (diff={diff:.6f})"

    print(f"  Decoded angles (rad): {fb.joint_angles_rad}")
    print(f"  Expected  angles (rad): {angles}")
    print(f"  Any motor error: {fb.any_motor_error} ✓ (expected False)")
    print(f"  All 4 joints match within 0.1° ✓")

    # --- Test 3: CRC corruption detection ---
    print("\n=== Test 3: CRC corruption detection ===")
    corrupted = bytearray(upload)
    corrupted[50] ^= 0xFF  # flip a bit in the payload
    fb_corrupt = decode_feedback(bytes(corrupted))
    assert fb_corrupt is None, "Should have detected CRC failure!"
    print(f"  Corrupted frame: decode returned None ✓")

    # --- Test 4: Motor error flags ---
    print("\n=== Test 4: Motor error flags ===")
    upload_err = build_echo_feedback(cmd.data, add_error=0x01)  # under-voltage on motor 1
    fb_err = decode_feedback(upload_err)
    assert fb_err is not None
    assert fb_err.any_motor_error, "Should report motor error"
    assert 1 in fb_err.error_motors, "Motor 1 should be flagged"
    print(f"  Motor error detected: motors {fb_err.error_motors} ✓")
    print(f"  Error bits: {fb_err.joint_errors}")

    # --- Test 5: Frame parser ---
    print("\n=== Test 5: Frame parser (粘包拆包) ===")
    upload2 = build_echo_feedback(cmd.data)
    from humanoid_arm_communication.frame_parser import FrameParser

    parser = FrameParser()
    # Feed 2 frames glued together + some junk after
    combined = upload + upload2 + b"\xAA" * 10
    frames = parser.feed(combined)

    assert len(frames) == 2, f"Expected 2 frames, got {len(frames)}"
    assert len(parser._buffer) == 10, f"Expected 10 leftover bytes, got {len(parser._buffer)}"

    fb1 = decode_feedback(frames[0])
    fb2 = decode_feedback(frames[1])
    assert fb1 is not None and fb2 is not None
    assert np.allclose(fb1.joint_angles_rad, fb2.joint_angles_rad)
    print(f"  2 glued frames + 10 junk bytes → 2 valid frames ✓")
    print(f"  {len(parser._buffer)} bytes left in buffer ✓")

    # --- Test 6: Reconnect manager ---
    print("\n=== Test 6: Reconnect manager state transitions ===")
    from humanoid_arm_communication.reconnect_manager import (
        ReconnectConfig, ReconnectManager, LinkState,
    )

    t = [0.0]

    def fake_clock():
        return t[0]

    def tick(dt):
        t[0] += dt

    mgr = ReconnectManager(
        ReconnectConfig(feedback_timeout_s=0.5, degraded_threshold_s=0.2),
        monotonic_clock=fake_clock,
    )

    assert mgr.link_state == LinkState.DISCONNECTED
    print("  Initial: DISCONNECTED ✓")

    mgr.on_feedback_received()
    tick(0.1)
    assert mgr.link_state == LinkState.CONNECTED
    print(f"  After feedback + 0.1s: {mgr.link_state.value} ✓")

    tick(0.2)  # total 0.3 > degraded
    assert mgr.link_state == LinkState.DEGRADED
    print(f"  After 0.3s total: {mgr.link_state.value} ✓")

    tick(0.3)  # total 0.6 > timeout
    assert mgr.link_state == LinkState.DISCONNECTED
    print(f"  After 0.6s total: {mgr.link_state.value} ✓")

    mgr.on_reconnect_failed()
    tick(0.6)
    assert mgr.should_reconnect
    print("  After backoff: should_reconnect = True ✓")

    print(f"\n{'='*50}")
    print(f"All tests passed!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
