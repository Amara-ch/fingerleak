"""MediaPipe Hands wrapper for detecting hand landmarks in an image.

Yeh module Google ke MediaPipe Hands solution ko wrap karta hai.
Ek image deti ho, 21 hand landmarks (x, y, z) milte hain har detected hand ke liye.

Reference: https://google.github.io/mediapipe/solutions/hands.html
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class HandLandmarks:
    """Single hand ka detection result.

    Attributes:
        landmarks: (21, 3) numpy array — har point ka (x, y, z) normalized [0,1].
        landmarks_px: (21, 2) numpy array — pixel coordinates (x, y) in image.
        handedness: "Left" ya "Right".
        score: Detection confidence [0, 1].
    """
    landmarks: np.ndarray          # shape (21, 3), normalized
    landmarks_px: np.ndarray       # shape (21, 2), in pixels
    handedness: str
    score: float


class HandDetector:
    """MediaPipe Hands detector ka thin wrapper.

    Usage:
        detector = HandDetector()
        hands = detector.detect(image_bgr)
        for hand in hands:
            print(hand.handedness, hand.score)
    """

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = True,
    ) -> None:
        """Initialize MediaPipe Hands.

        Args:
            max_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence for detection.
            min_tracking_confidence: Minimum confidence for tracking (video).
            static_image_mode: True for single images, False for video stream.
        """
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

    def detect(self, image_bgr: np.ndarray) -> List[HandLandmarks]:
        """Detect hands in a BGR image (OpenCV format).

        Args:
            image_bgr: Image as numpy array in BGR color order (OpenCV default).

        Returns:
            List of HandLandmarks (empty if no hand detected).
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        h, w = image_bgr.shape[:2]
        # MediaPipe expects RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return []

        hands_out: List[HandLandmarks] = []
        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            # Normalized landmarks (0-1)
            lms_norm = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                dtype=np.float32,
            )
            # Pixel coordinates
            lms_px = np.stack(
                [lms_norm[:, 0] * w, lms_norm[:, 1] * h], axis=1
            ).astype(np.int32)

            # Handedness ("Left" / "Right") and score
            handedness = "Unknown"
            score = 0.0
            if results.multi_handedness and idx < len(results.multi_handedness):
                cls = results.multi_handedness[idx].classification[0]
                handedness = cls.label
                score = float(cls.score)

            hands_out.append(
                HandLandmarks(
                    landmarks=lms_norm,
                    landmarks_px=lms_px,
                    handedness=handedness,
                    score=score,
                )
            )
        return hands_out

    def draw(self, image_bgr: np.ndarray, hands: List[HandLandmarks]) -> np.ndarray:
        """Draw landmarks + connections on image (returns a copy).

        Args:
            image_bgr: Source BGR image.
            hands: List of HandLandmarks from detect().

        Returns:
            New BGR image with landmarks drawn.
        """
        out = image_bgr.copy()
        for hand in hands:
            # Re-create mediapipe-style landmark list for drawing utility
            from mediapipe.framework.formats import landmark_pb2
            proto = landmark_pb2.NormalizedLandmarkList()
            for lm in hand.landmarks:
                proto.landmark.add(x=float(lm[0]), y=float(lm[1]), z=float(lm[2]))
            self._mp_drawing.draw_landmarks(
                out,
                proto,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )
            # Add handedness label near wrist (landmark 0)
            wrist_x, wrist_y = hand.landmarks_px[0]
            cv2.putText(
                out,
                f"{hand.handedness} ({hand.score:.2f})",
                (int(wrist_x), int(wrist_y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        return out

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()