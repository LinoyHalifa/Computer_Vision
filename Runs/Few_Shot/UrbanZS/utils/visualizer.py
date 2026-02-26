import os
from pathlib import Path

import cv2

# Base dir = UrbanZH
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "annotated"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_annotated(image_path, img_bgr, bbox, label, severity_0_10=None):
    """Save annotated image with bbox and label."""
    img_vis = img_bgr.copy()

    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        text = label if label is not None else ""
        if severity_0_10 is not None:
            text += f" ({severity_0_10:.1f}/10)"

        cv2.putText(
            img_vis,
            text,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    out_name = Path(image_path).stem + "_annotated.png"
    out_path = OUTPUT_DIR / out_name
    cv2.imwrite(str(out_path), img_vis)
    return str(out_path)
