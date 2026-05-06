from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Any

import cv2 as cv
import numpy as np
from mediapipe.tasks.python import vision

from classifier import GestureClassifier
from landmark_detection import HandLandmarker


def _detect_modifier_key() -> str:
    """Return the pyautogui modifier key name for the current platform.

    Returns:
        ``"command"`` on macOS, ``"alt"`` on Windows and Linux.
    """
    if sys.platform == "darwin":
        return "command"
    return "alt"


def _hotkey_failure_hint() -> str:
    """Return a platform-specific hint for pyautogui hotkey failures."""
    if sys.platform == "darwin":
        return "Grant Accessibility permission in System Settings."
    if sys.platform == "linux":
        return "pyautogui requires an X11 session; Wayland is not supported."
    return "Check that pyautogui is installed and your OS allows input simulation."


def put_latest(q: queue.Queue, item: Any) -> None:
    """Put *item* onto *q*, discarding the oldest entry if the queue is full.

    Designed for single-element queues where only the most recent value
    matters (e.g. a live camera frame).
    """
    try:
        q.put(item, block=False)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put(item, block=False)


class GestureBackend:
    """Coordinates camera capture, hand detection, gesture classification,
    and desktop actions.

    Runs two daemon threads:
    - Camera thread: reads frames from the webcam.
    - Process thread: runs detection + classification on each frame.
    """

    def __init__(
        self,
        cam_index: int = 0,
        perf_info: bool = False,
        modifier_key: str | None = None,
    ) -> None:
        """Initialise the backend.

        Args:
            cam_index: OpenCV camera device index.
            perf_info: If True, print per-frame timing to stdout.
            modifier_key: pyautogui modifier key name for window-switching
                shortcuts. Auto-detected from the current OS when None.
                Common values: ``"alt"`` (Windows/Linux), ``"command"``
                (macOS).
        """
        self.cam_index: int = cam_index
        self.perf_info: bool = perf_info
        self.modifier_key: str = (
            modifier_key if modifier_key is not None else _detect_modifier_key()
        )

        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.preview_queue: queue.Queue = queue.Queue(maxsize=1)
        self.result_queue: queue.Queue = queue.Queue(maxsize=1)

        self.app_stop_event = threading.Event()
        self.process_stop_event = threading.Event()

        self.camera_thread_ref: threading.Thread | None = None
        self.process_thread_ref: threading.Thread | None = None

        self.hand_detector = HandLandmarker(
            model_path="hand_landmarker.task",
            num_hands=1,
            running_mode=vision.RunningMode.LIVE_STREAM,
        )

        self.classifier = GestureClassifier()

        self.prev_hand_x: float | None = None
        self.gesture_label: str = "None"
        self.last_gesture_time: float = 0.0
        self.gesture_cooldown: float = 0.8
        self.swipe_threshold: float = 0.05

        self.controls_active: bool = True
        self.last_state_change_time: float = 0.0
        self.state_cooldown: float = 1.0

        self.timing_count: int = 0
        self.timing_total_ms: float = 0.0
        self.timing_detect_ms: float = 0.0
        self.timing_classify_ms: float = 0.0

        self._hotkey_broken: bool = False

    # --- lifecycle -----------------------------------------------------------

    def start_camera(self) -> None:
        """Start the camera capture thread if it is not already running."""
        if self.camera_thread_ref and self.camera_thread_ref.is_alive():
            return
        self.app_stop_event.clear()
        self.camera_thread_ref = threading.Thread(target=self._cam_thread, daemon=True)
        self.camera_thread_ref.start()

    def start_processing(self) -> None:
        """Start the gesture-processing thread if it is not already running."""
        if self.process_thread_ref and self.process_thread_ref.is_alive():
            return

        self.prev_hand_x = None
        self.gesture_label = "None"

        self.process_stop_event.clear()
        self.process_thread_ref = threading.Thread(
            target=self._process_thread, daemon=True
        )
        self.process_thread_ref.start()

    def stop_processing(self) -> None:
        """Signal the processing thread to stop and wait for it to exit."""
        self.process_stop_event.set()
        if self.process_thread_ref and self.process_thread_ref.is_alive():
            self.process_thread_ref.join(timeout=1.0)
        self.process_thread_ref = None

    def stop_all(self) -> None:
        """Stop processing and the camera, then wait for both threads."""
        self.stop_processing()
        self.app_stop_event.set()
        if self.camera_thread_ref and self.camera_thread_ref.is_alive():
            self.camera_thread_ref.join(timeout=1.0)
        self.camera_thread_ref = None

    def is_processing_running(self) -> bool:
        """Return True if the processing thread is currently alive."""
        return bool(self.process_thread_ref and self.process_thread_ref.is_alive())

    def get_latest_display_frame(self) -> np.ndarray | None:
        """Return the most recent annotated frame for display.

        Prefers the result (annotated) frame; falls back to the raw
        preview frame.

        Returns:
            An OpenCV BGR image, or None if no frame is available.
        """
        frame: np.ndarray | None = None

        try:
            frame = self.result_queue.get_nowait()
        except queue.Empty:
            pass

        if frame is None:
            try:
                frame = self.preview_queue.get_nowait()
            except queue.Empty:
                pass

        return frame

    # --- diagnostics ---------------------------------------------------------

    def print_average_timings(self) -> None:
        """Print average per-frame timing statistics to stdout."""
        if self.timing_count > 0:
            print("\n=== Average Timings ===")
            print(f"Frames measured: {self.timing_count}")
            print(f"Avg Total:     {self.timing_total_ms / self.timing_count:.2f}ms")
            print(f"Avg Detect:    {self.timing_detect_ms / self.timing_count:.2f}ms")
            print(f"Avg Classify:  {self.timing_classify_ms / self.timing_count:.2f}ms")
        else:
            print("\nNo frames with valid hand landmarks were measured.")

    # --- helpers -------------------------------------------------------------

    def _send_hotkey(self, *keys: str) -> None:
        """Send a keyboard shortcut via pyautogui.

        The import is deferred so pyautogui (which requires a display
        server) is only loaded when a swipe actually triggers.  Failures
        are logged once and silently ignored thereafter.
        """
        if self._hotkey_broken:
            return

        try:
            import pyautogui

            pyautogui.hotkey(*keys)
        except Exception as exc:
            self._hotkey_broken = True
            hint = _hotkey_failure_hint()
            print(f"Warning: pyautogui hotkey failed ({exc}).", hint)

    # --- threads -------------------------------------------------------------

    def _cam_thread(self) -> None:
        """Camera capture loop running in a daemon thread.

        Reads frames, mirrors them horizontally, and pushes them into
        the frame and preview queues.  If the camera cannot be opened
        the thread prints a warning and exits immediately.
        """
        cam = cv.VideoCapture(self.cam_index)
        if not cam.isOpened():
            print(f"Warning: Could not open camera at index {self.cam_index}.")
            print("Camera capture has been disabled.")
            self.app_stop_event.set()
            return

        try:
            cam.set(cv.CAP_PROP_BUFFERSIZE, 1)
        except AttributeError:
            pass
        cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

        while not self.app_stop_event.is_set():
            ret, frame = cam.read()
            if not ret:
                continue
            flip_frame = cv.flip(frame, 1)
            put_latest(self.frame_queue, flip_frame)
            put_latest(self.preview_queue, flip_frame)

        cam.release()

    def _process_thread(self) -> None:
        """Gesture-processing loop running in a daemon thread.

        For each camera frame:
        1. Runs async hand landmark detection.
        2. Classifies the static hand pose.
        3. Detects swipe gestures from horizontal hand movement.
        4. Triggers desktop actions (window-switch shortcut) on swipe.
        5. Annotates the frame and pushes it to the result queue.
        """
        frame_count: int = 0
        latest_result: vision.HandLandmarkerResult | None = None
        latest_result_timestamp: int = -1

        while not self.app_stop_event.is_set() and not self.process_stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            frame_count += 1
            frame_start = time.perf_counter()

            t0 = time.perf_counter()
            self.hand_detector.detect_async(frame, timestamp_ms=frame_count)
            result, result_timestamp = self.hand_detector.get_latest_result()
            if result is not None and result_timestamp > latest_result_timestamp:
                latest_result = result
                latest_result_timestamp = result_timestamp
            t1 = time.perf_counter()

            if (
                latest_result
                and latest_result.hand_landmarks
                and latest_result.hand_world_landmarks
            ):
                hand_2d = latest_result.hand_landmarks[0]

                t2 = time.perf_counter()
                hand_center_x = sum(lm.x for lm in hand_2d) / len(hand_2d)
                current_time = time.time()

                palm_open = self.classifier.is_open_palm(hand_2d)
                fist_closed = self.classifier.is_fist(hand_2d)

                # --- toggle controls on fist / open palm ---------------------
                if current_time - self.last_state_change_time > self.state_cooldown:
                    if fist_closed and self.controls_active:
                        self.controls_active = False
                        self.gesture_label = "Fist - Paused"
                        self.last_state_change_time = current_time
                        if self.perf_info:
                            print("Controls Paused")

                    elif palm_open and not self.controls_active:
                        self.controls_active = True
                        self.gesture_label = "Open Palm - Active"
                        self.last_state_change_time = current_time
                        if self.perf_info:
                            print("Controls Activated")

                # --- swipe detection -----------------------------------------
                if self.prev_hand_x is not None:
                    dx = hand_center_x - self.prev_hand_x

                    if (
                        self.controls_active
                        and current_time - self.last_gesture_time
                        > self.gesture_cooldown
                    ):
                        if dx > self.swipe_threshold:
                            self.gesture_label = "Swipe Right"
                            self.last_gesture_time = current_time
                            if self.perf_info:
                                print("Detected: Swipe Right")
                            self._send_hotkey(self.modifier_key, "tab")

                        elif dx < -self.swipe_threshold:
                            self.gesture_label = "Swipe Left"
                            self.last_gesture_time = current_time
                            if self.perf_info:
                                print("Detected: Swipe Left")
                            self._send_hotkey(self.modifier_key, "shift", "tab")

                # --- static gesture classification ---------------------------
                classify_idx = self.classifier.classify(hand_2d)
                t3 = time.perf_counter()

                if current_time - self.last_gesture_time > 0.6:
                    if not self.controls_active:
                        self.gesture_label = "Paused"
                    elif classify_idx is not None:
                        self.gesture_label = self.classifier.GESTURES[classify_idx]
                    else:
                        self.gesture_label = "Open Palm"

                self.prev_hand_x = hand_center_x

                # --- draw landmarks ------------------------------------------
                h, w, _ = frame.shape
                for lm in hand_2d:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cv.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                # --- timing --------------------------------------------------
                total_ms = (time.perf_counter() - frame_start) * 1000.0
                detect_ms = (t1 - t0) * 1000.0
                classify_ms = (t3 - t2) * 1000.0

                self.timing_count += 1
                self.timing_total_ms += total_ms
                self.timing_detect_ms += detect_ms
                self.timing_classify_ms += classify_ms

                if self.perf_info:
                    print(
                        f"[Frame {frame_count}] "
                        f"Total: {total_ms:.1f}ms | "
                        f"Detect: {detect_ms:.1f}ms | "
                        f"Classify: {classify_ms:.1f}ms"
                    )

            else:
                self.prev_hand_x = None
                self.gesture_label = "No Hand"

            # --- HUD overlay -------------------------------------------------
            cv.putText(
                frame,
                f"Gesture: {self.gesture_label}",
                (10, 40),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            status_text = "ACTIVE" if self.controls_active else "PAUSED"
            status_color = (0, 255, 0) if self.controls_active else (0, 0, 255)

            cv.putText(
                frame,
                f"Controls: {status_text}",
                (10, 75),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )

            put_latest(self.result_queue, frame)
