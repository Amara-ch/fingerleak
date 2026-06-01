"""Full FingerLeak baseline pipeline (Stages 0, 1, 3, 4).

Hand detection -> peace sign -> distance -> fingertip crops
-> ridge enhancement -> FingerLeak-Score.

Usage:
    python scripts/test_detection.py --img data/samples/sample.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fingerleak.detection.mediapipe_hands import HandDetector  # noqa: E402
from fingerleak.detection.peace_sign import detect_peace_sign  # noqa: E402
from fingerleak.detection.fingertip_crop import crop_fingertips  # noqa: E402
from fingerleak.geometry.distance_estimator import estimate_distance  # noqa: E402
from fingerleak.ridge.gabor import enhance_ridges  # noqa: E402
from fingerleak.risk.score import compute_finger_score, aggregate_frame_score  # noqa: E402


CATEGORY_COLORS = {
    "Low": (0, 200, 0),         # green
    "Medium": (0, 215, 255),    # yellow
    "High": (0, 140, 255),      # orange
    "Critical": (0, 0, 255),    # red
}


def make_compare_grid(crop, enhanced, binarized):
    enh_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    bin_bgr = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)
    return np.hstack([crop, enh_bgr, bin_bgr])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default="data/samples/sample.jpg")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--focal_px", type=float, default=None)
    args = parser.parse_args()

    img_path = Path(args.img)
    if not img_path.exists():
        print(f"❌ Image not found: {img_path}")
        return 1

    image = cv2.imread(str(img_path))
    if image is None:
        return 1

    h, w = image.shape[:2]
    print(f"📸 Loaded image: {img_path}  shape={image.shape}")

    out_dir = Path(args.out_dir)
    crops_dir = out_dir / "fingertip_crops"
    enhanced_dir = out_dir / "ridge_enhanced"
    compare_dir = out_dir / "ridge_compare"
    for d in (out_dir, crops_dir, enhanced_dir, compare_dir):
        d.mkdir(parents=True, exist_ok=True)

    detector = HandDetector(max_hands=2)
    hands = detector.detect(image)
    print(f"🖐️  Hands detected: {len(hands)}")
    if not hands:
        detector.close()
        return 0

    annotated = detector.draw(image, hands)
    report = {"image": str(img_path), "image_shape": [h, w], "hands": []}

    for i, hand in enumerate(hands):
        print(f"\n--- Hand {i + 1} ---")
        print(f"  Handedness     : {hand.handedness}")
        print(f"  Det. score     : {hand.score:.3f}")

        peace = detect_peace_sign(hand)
        print(f"  Peace sign     : {'✌️ YES' if peace.is_peace_sign else '❌ NO'}  (conf={peace.confidence:.3f})")

        dist = estimate_distance(hand, w, h, focal_length_px=args.focal_px)
        print(f"  Distance       : {dist.distance_cm:.1f} cm  (palm={dist.palm_width_px:.0f}px, {dist.confidence})")

        crops = crop_fingertips(image, hand)
        print(f"  Fingertip crops: {len(crops)}")

        finger_scores = []
        crop_records = []
        for c in crops:
            base = f"hand{i+1}_{c.finger}"
            cv2.imwrite(str(crops_dir / f"{base}.jpg"), c.crop)

            ridge = enhance_ridges(c.crop)
            cv2.imwrite(str(enhanced_dir / f"{base}_enhanced.jpg"), ridge.enhanced)
            cv2.imwrite(str(enhanced_dir / f"{base}_binarized.jpg"), ridge.binarized)
            cv2.imwrite(str(compare_dir / f"{base}_compare.jpg"),
                        make_compare_grid(c.crop, ridge.enhanced, ridge.binarized))

            fs = compute_finger_score(
                finger=c.finger,
                crop_bgr=c.crop,
                crop_size_px=c.crop_size_px,
                ridge_strength=ridge.ridge_response,
                distance_cm=dist.distance_cm,
            )
            finger_scores.append(fs)

            print(
                f"    - {c.finger:<7} size={c.crop_size_px:>3}px "
                f"ridge={ridge.ridge_response:.3f} "
                f"score={fs.score:.3f} [{fs.category}]"
            )

            crop_records.append({
                "finger": c.finger,
                "tip_px": c.tip_px.tolist(),
                "bbox": c.bbox.tolist(),
                "crop_size_px": int(c.crop_size_px),
                "ridge_strength": float(ridge.ridge_response),
                "fingerleak_score": fs.score,
                "category": fs.category,
                "components": fs.components,
                "raw": fs.raw,
            })

            # Draw colored bbox + score on annotated frame
            x1, y1, x2, y2 = c.bbox
            color = CATEGORY_COLORS[fs.category]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{c.finger} {fs.score:.2f}"
            cv2.putText(annotated, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        frame_score = aggregate_frame_score(finger_scores)
        print(
            f"\n  🛡️  FRAME RISK   : {frame_score['category'].upper()}  "
            f"(max={frame_score['max_score']:.3f}, "
            f"mean={frame_score['mean_score']:.3f}, "
            f"worst={frame_score['worst_finger']})"
        )

        # Frame-level banner on annotated image
        banner_color = CATEGORY_COLORS[frame_score["category"]]
        cv2.rectangle(annotated, (0, 0), (w, 35), banner_color, -1)
        cv2.putText(
            annotated,
            f"FingerLeak Risk: {frame_score['category']} ({frame_score['max_score']:.2f})",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

        report["hands"].append({
            "handedness": hand.handedness,
            "detection_score": float(hand.score),
            "peace_sign": {
                "is_peace_sign": bool(peace.is_peace_sign),
                "confidence": float(peace.confidence),
                "finger_states": peace.finger_states,
            },
            "distance": {
                "distance_cm": float(dist.distance_cm),
                "palm_width_px": float(dist.palm_width_px),
                "confidence": dist.confidence,
            },
            "fingertips": crop_records,
            "frame_score": frame_score,
        })

    cv2.imwrite(str(out_dir / "detection_result.jpg"), annotated)
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 Annotated frame  : {out_dir / 'detection_result.jpg'}")
    print(f"💾 Compare grids    : {compare_dir}")
    print(f"💾 Report JSON      : {out_dir / 'report.json'}")
    detector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())