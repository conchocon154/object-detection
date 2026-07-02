"""Object detection on a camera, video file, or image.

Examples:
    python main.py                        # default webcam (config.CAMERA_INDEX)
    python main.py --source 1             # webcam index 1
    python main.py --source clip.mp4      # a video file
    python main.py --source photo.jpg     # a single image
    python main.py --source clip.mp4 --save out.mp4 --no-show   # headless

Press 'q' in the window to quit (live/video mode).
"""

import argparse
import os
import time

import cv2

import config
from src.camera import Camera
from src.detector import Detector

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description="Camera / file object detection")
    p.add_argument(
        "--source",
        default=str(config.CAMERA_INDEX),
        help="Webcam index (e.g. 0) or path to a video/image file.",
    )
    p.add_argument("--save", metavar="PATH",
                   help="Write annotated output to this file (image or .mp4 video).")
    p.add_argument("--no-show", action="store_true",
                   help="Don't open a display window (useful for headless runs).")
    return p.parse_args()


def resolve_source(source):
    """Return an int for webcam indices, otherwise the path string."""
    return int(source) if source.isdigit() else source


def build_detector():
    return Detector(
        model_name=config.MODEL_NAME,
        conf=config.CONFIDENCE_THRESHOLD,
        iou=config.IOU_THRESHOLD,
        class_filter=config.CLASS_FILTER,
    )


def label_summary(result):
    labels = [result.names[int(b.cls[0])] for b in result.boxes]
    return labels


def run_image(path, detector, args):
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"Could not read image: {path}")
    result = detector.detect(frame)
    annotated = detector.annotate(frame, result, config.BOX_COLOR, config.TEXT_COLOR)

    print(f"Detections: {label_summary(result) or 'none'}")
    if args.save:
        cv2.imwrite(args.save, annotated)
        print(f"Saved -> {args.save}")
    if not args.no_show:
        cv2.imshow(config.WINDOW_NAME, annotated)
        print("Press any key in the window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_stream(source, detector, args):
    writer = None
    with Camera(source, config.FRAME_WIDTH, config.FRAME_HEIGHT) as cam:
        prev_t = time.time()
        for frame in cam.frames():
            result = detector.detect(frame)
            annotated = detector.annotate(
                frame, result, config.BOX_COLOR, config.TEXT_COLOR
            )

            if config.SHOW_FPS:
                now = time.time()
                fps = 1.0 / max(now - prev_t, 1e-6)
                prev_t = now
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if args.save:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save, fourcc, 30.0, (w, h))
                writer.write(annotated)

            if not args.no_show:
                cv2.imshow(config.WINDOW_NAME, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if writer is not None:
        writer.release()
        print(f"Saved -> {args.save}")
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    detector = build_detector()
    source = resolve_source(args.source)

    is_image = isinstance(source, str) and os.path.splitext(source)[1].lower() in IMAGE_EXTS
    if is_image:
        run_image(source, detector, args)
    else:
        run_stream(source, detector, args)


if __name__ == "__main__":
    main()
