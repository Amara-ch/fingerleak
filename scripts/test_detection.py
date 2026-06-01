"""Stage 0+1 baseline: hand detection -> peace sign -> distance -> fingertip crops.

Usage:
    python scripts/test_detection.py --img data/samples/sample.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fingerleak.detection.mediapipe_hands import HandDetector  # noqa: E402
from fingerleak.detection.peace_sign import detect_peace_sign  # noqa: E402
from fingerleak.detection.fingertip_crop import crop_fingertips  # noqa: E402
from fingerleak.geometry.distance_estimator import estimate_distance  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 0+1 baseline pipeline.")
    parser.add_argument("--img", type=str, default="data/samples/sample.jpg")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument(
        "--focal_px",
        type=float,
        default=None,
        help="Camera focal length in pixels (optional).",
    )
    args = parser.parse_args()

    img_path = Path(args.img)
    if not img_path.exists():
        print(f"❌ Image not found: {img_path}")
        return 1

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"❌ Could not read image: {img_path}")
        return 1

    h, w = image.shape[:2]
    print(f"📸 Loaded image: {img_path}  shape={image.shape}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "fingertip_crops"
    crops_dir.mkdir(exist_ok=True)

    detector = HandDetector(max_hands=2)
    hands = detector.detect(image)
    print(f"🖐️  Hands detected: {len(hands)}")

    if not hands:
        print("⚠️  No hands found.")
        detector.close()
        return 0

    report: dict = {"image": str(img_path), "image_shape": [h, w], "hands": []}

    for i, hand in enumerate(hands):
        print(f"\n--- Hand {i + 1} ---")
        print(f"  Handedness     : {hand.handedness}")
        print(f"  Det. score     : {hand.score:.3f}")

        # Peace sign
        peace = detect_peace_sign(hand)
        emoji = "✌️ YES" if peace.is_peace_sign else "❌ NO"
        print(f"  Peace sign     : {emoji}  (conf={peace.confidence:.3f})")
        print(f"  Finger states  : {peace.finger_states}")

        # Distance
        dist = estimate_distance(hand, w, h, focal_length_px=args.focal_px)
        print(
            f"  Distance       : {dist.distance_cm:.1f} cm  "
            f"(palm={dist.palm_width_px:.0f}px, conf={dist.confidence})"
        )
        print(f"  Note           : {dist.note}")

        # Fingertip crops
        crops = crop_fingertips(image, hand)
        print(f"  Fingertip crops: {len(crops)}")
        crop_records = []
        for c in crops:
            fname = f"hand{i+1}_{c.finger}.jpg"
            cv2.imwrite(str(crops_dir / fname), c.crop)
            print(
                f"    - {c.finger:<7}  tip=({c.tip_px[0]},{c.tip_px[1]})  "
                f"size={c.crop_size_px}px  -> {fname}"
            )
            crop_records.append({
                "finger": c.finger,
                "tip_px": c.tip_px.tolist(),
                "bbox": c.bbox.tolist(),
                "crop_size_px": int(c.crop_size_px),
                "saved_as": str(crops_dir / fname),
            })

        report["hands"].append({
            "handedness": hand.handedness,
            "detection_score": hand.score,
            "peace_sign": {
                "is_peace_sign": peace.is_peace_sign,
                "confidence": peace.confidence,
                "finger_states": peace.finger_states,
                "reason": peace.reason,
            },
            "distance": {
                "distance_cm": dist.distance_cm,
                "palm_width_px": dist.palm_width_px,
                "focal_length_px": dist.focal_length_px,
                "confidence": dist.confidence,
            },
            "fingertip_crops": crop_records,
        })

    # Annotated visualization
    annotated = detector.draw(image, hands)
    # Draw fingertip bboxes on annotated image too
    for hand in hands:
        for c in crop_fingertips(image, hand):
            x1, y1, x2, y2 = c.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(
                annotated, c.finger, (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1,
            )

    out_img = out_dir / "detection_result.jpg"
    cv2.imwrite(str(out_img), annotated)
    print(f"\n💾 Annotated image     : {out_img}")
    print(f"💾 Fingertip crops dir : {crops_dir}")

    # JSON report
    out_json = out_dir / "report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"💾 JSON report         : {out_json}")

    detector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())