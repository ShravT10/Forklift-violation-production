#python scripts/view_roi.py --source "rtsps://admin:Kion%402024@10.102.10.230:554/video/live?channel=1&subtype=1"

import argparse
import ast
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_AREA_PATH = Path("Regions/area.txt")
DEFAULT_EXC_PATH = Path("Regions/exc_points.txt")


def load_points(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ROI file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "points" in data:
            return [(int(x), int(y)) for x, y in data["points"]]
        if isinstance(data, list):
            return [(int(x), int(y)) for x, y in data]
    except json.JSONDecodeError:
        pass

    candidate = text
    if "=" in candidate:
        candidate = candidate.split("=", 1)[1].strip()

    try:
        parsed = ast.literal_eval(candidate)
    except (ValueError, SyntaxError):
        start = min(candidate.find("["), candidate.find("(")) if "[" in candidate and "(" in candidate else None
        if start is not None and start != -1:
            if candidate[start] == "[":
                end = candidate.rfind("]")
            else:
                end = candidate.rfind(")")
            if end > start:
                candidate = candidate[start : end + 1]
                try:
                    parsed = ast.literal_eval(candidate)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError(f"Could not parse ROI file {path}: {exc}") from exc
            else:
                raise ValueError(f"Could not parse ROI file {path}")
        else:
            raise ValueError(f"Could not parse ROI file {path}")

    if isinstance(parsed, dict) and "points" in parsed:
        parsed = parsed["points"]
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Unsupported ROI format in {path}")

    return [(int(x), int(y)) for x, y in parsed]


def draw_rois(frame, rois):
    overlay = frame.copy()
    for idx, points in enumerate(rois):
        if not points:
            continue
        pts = np.array(points, dtype=np.int32)
        color = (0, 255, 0) if idx == 0 else (0, 0, 255)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
        centroid = np.mean(pts, axis=0).astype(int)
        cv2.putText(
            overlay,
            str(idx + 1),
            (centroid[0], centroid[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return overlay


def parse_args():
    parser = argparse.ArgumentParser(description="Display ROI overlays on a live video or RTSP stream")
    parser.add_argument("--source", default="0", help="Video source: RTSP URL, video file path, or camera index")
    parser.add_argument("--area", default=str(DEFAULT_AREA_PATH), help="Path to the main ROI points file")
    parser.add_argument("--exc", default=str(DEFAULT_EXC_PATH), help="Path to the exclusion ROI points file")
    parser.add_argument("--wait-ms", type=int, default=1, help="OpenCV wait key interval in milliseconds")
    return parser.parse_args()


def main():
    args = parse_args()

    area_points = load_points(args.area)
    exc_points = load_points(args.exc)
    rois = [area_points, exc_points]

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    cv2.namedWindow("ROI Viewer", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        displayed = draw_rois(frame, rois)
        cv2.putText(
            displayed,
            "q=quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("ROI Viewer", displayed)
        key = cv2.waitKey(args.wait_ms) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()