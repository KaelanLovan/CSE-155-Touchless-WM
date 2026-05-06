from __future__ import annotations

import threading

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_image = mp.Image
mp_ImageFormat = mp.ImageFormat


class HandLandmarker:
    """Thread-safe wrapper around MediaPipe's HandLandmarker.

    Runs in LIVE_STREAM mode with an internal callback. Use
    :meth:`detect_async` to push frames and :meth:`get_latest_result`
    to retrieve the most recent detection.
    """

    def __init__(
        self,
        model_path: str = "hand_landmarker.task",
        num_hands: int = 1,
        running_mode: vision.RunningMode = vision.RunningMode.LIVE_STREAM,
    ) -> None:
        """Initialise the landmarker.

        Args:
            model_path: Path to the MediaPipe hand landmarker ``.task`` file.
            num_hands: Maximum number of hands to detect.
            running_mode: MediaPipe running mode (LIVE_STREAM, VIDEO, or IMAGE).
        """
        self.running_mode = running_mode
        self._latest_result: vision.HandLandmarkerResult | None = None
        self._latest_timestamp_ms: int = -1
        self._lock = threading.Lock()

        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.CPU,
        )

        options_kwargs: dict = {
            "base_options": base_options,
            "running_mode": running_mode,
            "num_hands": num_hands,
        }

        if running_mode == vision.RunningMode.LIVE_STREAM:
            options_kwargs["result_callback"] = self._handle_live_stream_result

        options = vision.HandLandmarkerOptions(**options_kwargs)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _handle_live_stream_result(
        self,
        result: vision.HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        """Callback invoked by MediaPipe on each async detection result.

        Stores only the result with the latest timestamp.
        """
        with self._lock:
            if timestamp_ms > self._latest_timestamp_ms:
                self._latest_timestamp_ms = timestamp_ms
                self._latest_result = result

    def detect_async(self, frame: np.ndarray, timestamp_ms: int) -> None:
        """Submit a frame for asynchronous hand landmark detection.

        The result becomes available via :meth:`get_latest_result`.

        Args:
            frame: BGR image from OpenCV.
            timestamp_ms: Monotonically increasing frame timestamp.

        Raises:
            ValueError: If the detector was not created in LIVE_STREAM mode.
        """
        if self.running_mode != vision.RunningMode.LIVE_STREAM:
            raise ValueError("detect_async requires LIVE_STREAM running mode")

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_img = mp_image(image_format=mp_ImageFormat.SRGB, data=rgb)
        self.detector.detect_async(mp_img, timestamp_ms)

    def get_latest_result(self) -> tuple[vision.HandLandmarkerResult | None, int]:
        """Return the most recent detection result and its timestamp.

        Returns:
            A tuple of ``(result, timestamp_ms)``. ``result`` is None
            if no detection has completed yet.
        """
        with self._lock:
            return self._latest_result, self._latest_timestamp_ms
