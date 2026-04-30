import cv2 as cv
import mediapipe as mp
import time
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_image = mp.Image
mp_ImageFormat = mp.ImageFormat


class HandLandmarker:
    def __init__(self, model_path="hand_landmarker.task", num_hands=1, running_mode=vision.RunningMode.LIVE_STREAM):
        self.running_mode = running_mode
        self._latest_result = None
        self._latest_timestamp_ms = -1
        self._lock = threading.Lock()

        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.CPU,
        )

        options_kwargs = {
            "base_options": base_options,
            "running_mode": running_mode,
            "num_hands": num_hands,
        }

        if running_mode == vision.RunningMode.LIVE_STREAM:
            options_kwargs["result_callback"] = self._handle_live_stream_result

        options = vision.HandLandmarkerOptions(**options_kwargs)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _handle_live_stream_result(self, result, output_image, timestamp_ms):
        with self._lock:
            if timestamp_ms > self._latest_timestamp_ms:
                self._latest_timestamp_ms = timestamp_ms
                self._latest_result = result

    def detect(self, frame, timestamp_ms=None):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_img = mp_image(image_format=mp_ImageFormat.SRGB, data=rgb)
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        rgb.flags.writeable = False
        result = self.detector.detect_for_video(mp_img, timestamp_ms)
        rgb.flags.writeable = True
        return result

    def detect_async(self, frame, timestamp_ms):
        if self.running_mode != vision.RunningMode.LIVE_STREAM:
            raise ValueError("detect_async requires LIVE_STREAM running mode")

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_img = mp_image(image_format=mp_ImageFormat.SRGB, data=rgb)
        self.detector.detect_async(mp_img, timestamp_ms)

    def get_latest_result(self):
        with self._lock:
            return self._latest_result, self._latest_timestamp_ms