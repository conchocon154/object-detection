# Camera Object Detection

Real-time object detection from a webcam using [Ultralytics YOLO](https://docs.ultralytics.com/) and OpenCV.

## Project structure

```
object-detection/
├── main.py            # entry point: capture → detect → display loop
├── config.py          # all tunable settings (model, thresholds, camera)
├── requirements.txt
├── src/
│   ├── camera.py      # OpenCV camera wrapper
│   └── detector.py    # YOLO inference + frame annotation
└── README.md
```

## Setup

```bash
cd object-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The YOLO weights (`yolov8n.pt`, ~6 MB) download automatically on first run.

## Run

```bash
python main.py
```

Press **`q`** in the video window to quit.

## Configuration

Edit `config.py` to change behavior:

- `MODEL_NAME` — swap `yolov8n.pt` for `yolov8s/m/l/x.pt` (larger = more accurate, slower).
- `CONFIDENCE_THRESHOLD` — raise to reduce false positives.
- `CLASS_FILTER` — e.g. `[0]` to detect people only ([COCO class ids](https://docs.ultralytics.com/datasets/detect/coco/)).
- `CAMERA_INDEX` — change if you have multiple cameras.

## Notes

- On macOS, grant camera permission to your terminal (System Settings → Privacy & Security → Camera) the first time you run it.
