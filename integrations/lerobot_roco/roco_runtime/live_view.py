"""Optional OpenCV live preview for bridge-rendered RGB frames."""

from typing import Any, Dict

import numpy as np


class LiveViewer:
    def __init__(self, window_name: str = "RoCo Bridge Live View", camera_alias: str = "front") -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for --live-view. Install opencv-python.") from exc
        self.cv2 = cv2
        self.window_name = window_name
        self.camera_alias = camera_alias
        self.closed = False
        self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)

    def show_observation(self, observation: Dict[str, Any]) -> None:
        if self.closed:
            return
        pixels = observation.get("pixels", {})
        image = None
        if isinstance(pixels, dict):
            image = pixels.get(self.camera_alias)
            if image is None and pixels:
                image = pixels[sorted(pixels.keys())[0]]
        else:
            image = pixels
        if image is None:
            return
        self.show_image(image)

    def show_image(self, image: Any) -> None:
        if self.closed:
            return
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            return
        bgr = self.cv2.cvtColor(arr.astype(np.uint8), self.cv2.COLOR_RGB2BGR)
        self.cv2.imshow(self.window_name, bgr)
        key = self.cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.cv2.destroyWindow(self.window_name)
        except Exception:
            pass
