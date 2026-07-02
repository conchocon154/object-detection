"""Central configuration for the object detection app."""

# --- Model ---
# Any Ultralytics-supported model name or local .pt path.
# "yolov8n.pt" (nano) is the smallest/fastest; use "yolov8s/m/l/x.pt" for more accuracy.
MODEL_NAME = "yolov8n.pt"

# --- Inference ---
CONFIDENCE_THRESHOLD = 0.5   # minimum score to keep a detection (0-1)
IOU_THRESHOLD = 0.45         # NMS overlap threshold
# Restrict to specific COCO class ids, e.g. [0] for "person" only. None = all classes.
CLASS_FILTER = None

# --- Camera ---
CAMERA_INDEX = 0             # 0 = default webcam
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- Display ---
WINDOW_NAME = "Object Detection"
BOX_COLOR = (0, 255, 0)      # BGR
TEXT_COLOR = (0, 0, 0)
SHOW_FPS = True
