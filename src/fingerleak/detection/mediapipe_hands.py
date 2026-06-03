"""MediaPipe Hands wrapper — tuned for partial-hand and close-up detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class HandLandmarks:
    landmarks: np.ndarray          # (21, 3) normalized
    landmarks_px: np.ndarray       # (21, 2) pixel coords
    handedness: str
    score: float
    palm_facing: bool              # True = palm/finger pads visible (RISK), False = back of hand


class HandDetector:
    """Tuned MediaPipe wrapper.

    Key tweaks vs. defaults:
    - min_detection_confidence=0.3  → catches partial hands, close-ups, tilted poses
    - model_complexity=1            → full landmark model (vs lite), better on edge cases
    - palm_facing flag              → skip back-of-hand cases (no fingerprint risk)
    """

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.3,   # was 0.5 — too strict
        min_tracking_confidence: float = 0.3,
        static_image_mode: bool = True,
        model_complexity: int = 1,               # full model
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

    @staticmethod
    def _is_palm_facing(lms_norm: np.ndarray, handedness: str) -> bool:
        """Check if palm side (with fingerprints) faces the camera.

        Uses cross product of vectors WRIST→INDEX_MCP and WRIST→PINKY_MCP.
        Sign of z-component tells us palm vs back orientation.

        MediaPipe landmarks reference:
        - 0  = WRIST
        - 5  = INDEX_FINGER_MCP
        - 17 = PINKY_MCP
        """
        wrist = lms_norm[0]
        idx_mcp = lms_norm[5]
        pky_mcp = lms_norm[17]

        v1 = idx_mcp - wrist   # wrist → index base
        v2 = pky_mcp - wrist   # wrist → pinky base
        cross = np.cross(v1, v2)
        # cross.z sign depends on handedness:
        # Right hand: palm-facing → cross.z < 0
        # Left hand:  palm-facing → cross.z > 0
        if handedness == "Right":
            return cross[2] < 0
        elif handedness == "Left":
            return cross[2] > 0
        return True  # default: assume risk

    def detect(self, image_bgr: np.ndarray) -> List[HandLandmarks]:
        if image_bgr is None or image_bgr.size == 0:
            return []

        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return []

        hands_out: List[HandLandmarks] = []
        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            lms_norm = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                dtype=np.float32,
            )
            lms_px = np.stack(
                [lms_norm[:, 0] * w, lms_norm[:, 1] * h], axis=1
            ).astype(np.int32)

            handedness = "Unknown"
            score = 0.0
            if results.multi_handedness and idx < len(results.multi_handedness):
                cls = results.multi_handedness[idx].classification[0]
                handedness = cls.label
                score = float(cls.score)

            palm_facing = self._is_palm_facing(lms_norm, handedness)

            hands_out.append(
                HandLandmarks(
                    landmarks=lms_norm,
                    landmarks_px=lms_px,
                    handedness=handedness,
                    score=score,
                    palm_facing=palm_facing,
                )
            )
        return hands_out

    def draw(self, image_bgr: np.ndarray, hands: List[HandLandmarks]) -> np.ndarray:
        out = image_bgr.copy()
        for hand in hands:
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
            wrist_x, wrist_y = hand.landmarks_px[0]
            facing = "PALM" if hand.palm_facing else "BACK"
            color = (0, 255, 0) if hand.palm_facing else (128, 128, 128)
            cv2.putText(
                out,
                f"{hand.handedness} {facing} ({hand.score:.2f})",
                (int(wrist_x), int(wrist_y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )
        return out

    def close(self) -> None:
        self._hands.close()