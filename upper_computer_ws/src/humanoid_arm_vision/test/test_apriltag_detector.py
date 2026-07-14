import cv2
import numpy as np

from humanoid_arm_vision.apriltag_detector import AprilTagConfig, AprilTagDetector


def test_detects_only_configured_apriltag() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 7, 180)
    image = np.full((300, 300), 255, dtype=np.uint8)
    image[60:240, 60:240] = marker
    detector = AprilTagDetector(AprilTagConfig(family="tag36h11", target_id=7))
    detection = detector.detect(image)
    assert detection is not None
    assert detection.tag_id == 7
    assert detection.pixel_area > 30_000


def test_ignores_other_tag_ids() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 7, 180)
    image = np.full((300, 300), 255, dtype=np.uint8)
    image[60:240, 60:240] = marker
    detector = AprilTagDetector(AprilTagConfig(family="tag36h11", target_id=8))
    assert detector.detect(image) is None
