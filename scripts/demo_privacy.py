"""Day 4 Demo: Detect fingertips → apply privacy filter → save before/after."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fingerleak.detection.mediapipe_hands import HandDetector
from fingerleak.detection.fingertip_crop import crop_fingertips
from fingerleak.privacy.filters import apply_filter


def xyxy_to_xywh(bbox):
    """Convert (x1, y1, x2, y2) to (x, y, w, h)."""
    x1, y1, x2, y2 = bbox
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/samples/sample.jpg")
    parser.add_argument("--mode", default="blur",
                        choices=["blur", "pixelate", "blackout", "emoji"])
    parser.add_argument("--out", default="outputs/privacy_demo.jpg")
    args = parser.parse_args()

    img_path = ROOT / args.image
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Cannot load image: {img_path}")
        return 1
    print(f"📷 Loaded: {img_path.name}  shape={img.shape}")

    detector = HandDetector(max_hands=2)
    hands = detector.detect(img)
    print(f"✋ Hands detected: {len(hands)}")

    if not hands:
        print("⚠️  No hands found — saving original.")
        cv2.imwrite(str(out_path), img)
        return 0

    # Crop fingertips → convert bbox format → collect
    bboxes = []
    for hand in hands:
        crops = crop_fingertips(img, hand)
        for c in crops:
            xywh = xyxy_to_xywh(c.bbox)
            bboxes.append(xywh)
            print(f"   • {c.finger:8s} bbox(x,y,w,h)={xywh}")

    print(f"🎯 Total fingertip regions: {len(bboxes)}")

    result = apply_filter(img, bboxes, mode=args.mode)
    print(f"🛡️  Filter applied: {result.mode} on {result.regions_filtered} regions")

    # Side-by-side comparison
    comparison = np.hstack([img, result.image])
    h, w = comparison.shape[:2]

    # Add labels with background bar
    bar_h = 60
    label_bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    cv2.putText(label_bar, "BEFORE", (w // 4 - 80, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
    cv2.putText(label_bar, f"AFTER ({args.mode.upper()})", (3 * w // 4 - 150, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 220, 0), 3)

    final = np.vstack([label_bar, comparison])
    cv2.imwrite(str(out_path), final)
    print(f"✅ Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())