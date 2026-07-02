"""YOLO-based object detector and frame annotator."""

import cv2
from ultralytics import YOLO


class Detector:
    def __init__(self, model_name, conf=0.5, iou=0.45, class_filter=None):
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou
        self.class_filter = class_filter

    def detect(self, frame):
        """Run inference on a single BGR frame. Returns the raw Ultralytics result."""
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=self.class_filter,
            verbose=False,
        )
        return results[0]

    def annotate(self, frame, result, box_color=(0, 255, 0), text_color=(0, 0, 0)):
        """Draw boxes + labels onto a copy of the frame and return it."""
        out = frame.copy()
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            label = f"{names[cls_id]} {score:.2f}"

            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 2, y1), box_color, -1)
            cv2.putText(out, label, (x1 + 1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        return out
