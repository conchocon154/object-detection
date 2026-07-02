"""Thin wrapper around an OpenCV VideoCapture with sane defaults."""

import cv2


class Camera:
    """Context-managed video source (webcam index or video-file path)."""

    def __init__(self, source=0, width=1280, height=720):
        self.source = source
        self.width = width
        self.height = height
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            hint = ""
            if isinstance(self.source, int):
                hint = (
                    "\n  On macOS this is usually a Camera permission block: grant your "
                    "terminal app camera access in System Settings > Privacy & Security > "
                    "Camera, then fully quit and reopen it. Also check no other app is "
                    "using the camera, and try a different index (--source 1)."
                )
            raise RuntimeError(f"Could not open video source {self.source!r}.{hint}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self

    def frames(self):
        """Yield frames until the stream ends or read fails."""
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def __exit__(self, *exc):
        if self.cap is not None:
            self.cap.release()
