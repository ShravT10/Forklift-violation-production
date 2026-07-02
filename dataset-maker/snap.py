import cv2
import os
import time
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="RTSP snapshot collector for YOLO dataset")
    parser.add_argument("--rtsp", type=str, required=True, help="RTSP stream URL")
    parser.add_argument("--save_dir", type=str, default="captured_images", help="Directory to save images")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds to save frames after pressing 's'")
    parser.add_argument("--save_interval", type=float, default=0.2,
                        help="Time between saved frames in seconds (e.g. 0.2 = 5 images/sec)")
    parser.add_argument("--window_name", type=str, default="RTSP Live Feed", help="OpenCV window name")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.rtsp)

    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return

    print("Controls:")
    print("  s -> start saving snapshots")
    print("  q -> quit")

    saving = False
    save_start_time = 0
    last_save_time = 0
    session_dir = None
    image_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from stream.")
            break

        display_frame = frame.copy()

        # If saving mode is active
        if saving:
            elapsed = time.time() - save_start_time

            if elapsed <= args.duration:
                # Save frame only at the chosen interval
                if time.time() - last_save_time >= args.save_interval:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = os.path.join(session_dir, f"img_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    image_count += 1
                    last_save_time = time.time()

                status_text = f"SAVING... {elapsed:.1f}/{args.duration:.1f}s | Images: {image_count}"
                cv2.putText(display_frame, status_text, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                saving = False
                print(f"Done saving. {image_count} images saved in: {session_dir}")

        cv2.putText(display_frame, "Press 's' to save snapshots, 'q' to quit", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(args.window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and not saving:
            save_start_time = time.time()
            last_save_time = 0
            image_count = 0

            session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
            session_dir = os.path.join(args.save_dir, session_name)
            os.makedirs(session_dir, exist_ok=True)

            saving = True
            print(f"Started saving snapshots for {args.duration} seconds into: {session_dir}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()