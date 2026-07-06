import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_OUT = Path("Regions/exc_points.txt")


class RTSPROIEditor:
    def __init__(self, source, out_path, close_shape=True):
        self.source = source
        self.out_path = Path(out_path)
        self.close_shape = close_shape
        self.pts = []
        self.current_frame = None
        self.capture = cv2.VideoCapture(source)

        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    def redraw(self):
        if self.current_frame is None:
            return None

        img = self.current_frame.copy()
        for i, p in enumerate(self.pts):
            cv2.circle(img, p, 4, (0, 255, 0), -1)
            cv2.putText(
                img,
                str(i),
                (p[0] + 5, p[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        if len(self.pts) > 1:
            cv2.polylines(img, [np.array(self.pts, np.int32)], self.close_shape, (0, 0, 255), 2)

        cv2.putText(
            img,
            "u=undo  s=save  q=quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        return img

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append((x, y))

    def save_points(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("w", encoding="utf-8") as f:
            f.write('{\n')
            f.write('  "points": [\n')
            for i, p in enumerate(self.pts):
                f.write('    (\n')
                f.write(f'      {p[0]},\n')
                f.write(f'      {p[1]}\n')
                f.write('    )')
                if i != len(self.pts) - 1:
                    f.write(',\n')
                else:
                    f.write('\n')
            f.write('  ]\n')
            f.write('}\n')
        print("Saved (Python-literal format):", self.pts)

    def run(self):
        cv2.namedWindow("ROI Tool (RTSP)", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("ROI Tool (RTSP)", self.mouse)

        while True:
            ok, frame = self.capture.read()
            if not ok:
                print("Failed to read frame from source. Check the RTSP URL or stream availability.")
                break

            self.current_frame = frame
            display = self.redraw()
            if display is None:
                display = frame

            cv2.imshow("ROI Tool (RTSP)", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("u") and self.pts:
                self.pts.pop()
            elif key == ord("s"):
                self.save_points()
            elif key == ord("q"):
                break

        self.capture.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Draw ROI points on a live RTSP stream")
    parser.add_argument(
        "--source",
        default="rtsp://YOUR_RTSP_URL",
        help="RTSP URL, webcam index, or video file path",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Path to save the ROI points file",
    )
    parser.add_argument(
        "--open-polyline",
        action="store_true",
        help="Draw an open polyline instead of a closed polygon",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    editor = RTSPROIEditor(args.source, args.out, close_shape=not args.open_polyline)
    editor.run()
