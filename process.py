import cv2 as cv
import numpy as np
from collections import deque
import torch
import threading
import queue
import time
import pyautogui


from mediapipe.tasks.python import vision

from vector_conversion import compute_hand_features
from landmark_detection import HandLandmarker
from classifier import GestureGRU

def put_latest(q, item):
    try:
        q.put(item, block=False)
    except queue.Full:
        try:
            q.get_nowait()

        except queue.Empty:
            pass
        try:
            q.put(item, block=False)
        except queue.Full:
            pass


class GestureBackend:
    def __init__(self, cam_index=0, perf_info=False):
        self.cam_index = cam_index
        self.perf_info = perf_info

        self.frame_queue = queue.Queue(maxsize=1)
        self.preview_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)

        self.app_stop_event = threading.Event()
        self.process_stop_event = threading.Event()

        self.camera_thread_ref = None
        self.process_thread_ref = None

        self.hand_detector = HandLandmarker(
            model_path="hand_landmarker.task",
            num_hands=1,
            running_mode=vision.RunningMode.LIVE_STREAM,
        )

        num_gestures = 5
        self.model = GestureGRU(input_size=85, hidden_size=64, num_layers=1, num_classes=num_gestures)
        self.model.eval()

        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        self.prev_tips = None

        # existing swipe gesture stuff
        self.prev_hand_x = None
        self.gesture_label = "None"
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.8
        self.swipe_threshold = 0.05

        # new state stuff
        self.controls_active = True
        self.last_state_change_time = 0
        self.state_cooldown = 1.0

        self.timing_count = 0
        self.timing_total_ms = 0.0
        self.timing_detect_ms = 0.0
        self.timing_features_ms = 0.0
        self.timing_gru_ms = 0.0

    def start_camera(self):
        if self.camera_thread_ref and self.camera_thread_ref.is_alive():
            return
        self.app_stop_event.clear()
        self.camera_thread_ref = threading.Thread(target=self._cam_thread, daemon=True)
        self.camera_thread_ref.start()

    def start_processing(self):
        if self.process_thread_ref and self.process_thread_ref.is_alive():
            return

        self.frame_buffer.clear()
        self.prev_tips = None
        self.prev_hand_x = None
        self.gesture_label = "None"

        self.process_stop_event.clear()
        self.process_thread_ref = threading.Thread(target=self._process_thread, daemon=True)
        self.process_thread_ref.start()

    def stop_processing(self):
        self.process_stop_event.set()
        if self.process_thread_ref and self.process_thread_ref.is_alive():
            self.process_thread_ref.join(timeout=1.0)
        self.process_thread_ref = None

    def stop_all(self):
        self.stop_processing()
        self.app_stop_event.set()
        if self.camera_thread_ref and self.camera_thread_ref.is_alive():
            self.camera_thread_ref.join(timeout=1.0)
        self.camera_thread_ref = None

    def is_processing_running(self):
        return bool(self.process_thread_ref and self.process_thread_ref.is_alive())

    def get_latest_display_frame(self):
        frame = None

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

    def print_average_timings(self):
        if self.timing_count > 0:
            print("\n=== Average Timings ===")
            print(f"Frames measured: {self.timing_count}")
            print(f"Avg Total: {self.timing_total_ms / self.timing_count:.2f}ms")
            print(f"Avg Detect: {self.timing_detect_ms / self.timing_count:.2f}ms")
            print(f"Avg Features: {self.timing_features_ms / self.timing_count:.2f}ms")
            print(f"Avg GRU: {self.timing_gru_ms / self.timing_count:.2f}ms")
        else:
            print("\nNo frames with valid hand landmarks were measured.")

    def count_extended_fingers(self, hand_2d):
        # count 4 main fingers by checking if tip is above pip
        finger_pairs = [
            (8, 6),    # index
            (12, 10),  # middle
            (16, 14),  # ring
            (20, 18),  # pinky
        ]

        extended = 0
        for tip_idx, pip_idx in finger_pairs:
            if hand_2d[tip_idx].y < hand_2d[pip_idx].y:
                extended += 1

        return extended

    def is_open_palm(self, hand_2d):
        return self.count_extended_fingers(hand_2d) >= 4

    def is_fist(self, hand_2d):
        return self.count_extended_fingers(hand_2d) <= 1

    def _cam_thread(self):
        cam = cv.VideoCapture(self.cam_index)
        cam.set(cv.CAP_PROP_BUFFERSIZE, 1)
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

    def _process_thread(self):
        frame_count = 0
        latest_result = None
        latest_result_timestamp = -1

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

            predicted_gesture = None

            if latest_result and latest_result.hand_landmarks and latest_result.hand_world_landmarks:
                hand_2d = latest_result.hand_landmarks[0]
                hand_3d = latest_result.hand_world_landmarks[0]

                # keep old feature pipeline
                t2 = time.perf_counter()
                feature_vector, self.prev_tips = compute_hand_features(hand_3d, self.prev_tips)
                self.frame_buffer.append(feature_vector)
                t3 = time.perf_counter()

                # keep old GRU pipeline
                t4 = time.perf_counter()
                if len(self.frame_buffer) == self.sequence_length:
                    input_seq = torch.tensor(np.array(self.frame_buffer)).unsqueeze(0).float()
                    with torch.no_grad():
                        output = self.model(input_seq)
                        predicted_gesture = torch.argmax(output, dim=1).item()
                t5 = time.perf_counter()

                hand_center_x = sum(lm.x for lm in hand_2d) / len(hand_2d)
                current_time = time.time()

                palm_open = self.is_open_palm(hand_2d)
                fist_closed = self.is_fist(hand_2d)

                # new fist / palm controls
                if current_time - self.last_state_change_time > self.state_cooldown:
                    if fist_closed and self.controls_active:
                        self.controls_active = False
                        self.gesture_label = "Fist - Paused"
                        self.last_state_change_time = current_time
                        if self.perf_info: print("Controls Paused")

                    elif palm_open and not self.controls_active:
                        self.controls_active = True
                        self.gesture_label = "Open Palm - Active"
                        self.last_state_change_time = current_time
                        if self.perf_info: print("Controls Activated")

                # keep old swipe logic basically the same, just require active mode
                if self.prev_hand_x is not None:
                    dx = hand_center_x - self.prev_hand_x

                    if self.controls_active and current_time - self.last_gesture_time > self.gesture_cooldown:
                        if dx > self.swipe_threshold:
                            self.gesture_label = "Swipe Right"
                            self.last_gesture_time = current_time
                            if self.perf_info: print("Detected: Swipe Right")
                            pyautogui.hotkey('command', 'tab')

                        elif dx < -self.swipe_threshold:
                            self.gesture_label = "Swipe Left"
                            self.last_gesture_time = current_time
                            if self.perf_info: print("Detected: Swipe Left")
                            pyautogui.hotkey('command', 'shift', 'tab')

                # only show idle labels if a recent swipe did not just happen
                if current_time - self.last_gesture_time > 0.6:
                    if not self.controls_active:
                        self.gesture_label = "Paused"
                    elif palm_open:
                        self.gesture_label = "Open Palm"

                self.prev_hand_x = hand_center_x

                h, w, _ = frame.shape
                for lm in hand_2d:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cv.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                total_ms = (time.perf_counter() - frame_start) * 1000.0
                detect_ms = (t1 - t0) * 1000.0
                features_ms = (t3 - t2) * 1000.0
                gru_ms = (t5 - t4) * 1000.0

                self.timing_count += 1
                self.timing_total_ms += total_ms
                self.timing_detect_ms += detect_ms
                self.timing_features_ms += features_ms
                self.timing_gru_ms += gru_ms

                if self.perf_info: print(
                    f"[Frame {frame_count}] "
                    f"Total: {total_ms:.1f}ms | "
                    f"Detect: {detect_ms:.1f}ms | "
                    f"Features: {features_ms:.1f}ms | "
                    f"GRU: {gru_ms:.1f}ms"
                )

            else:
                self.prev_hand_x = None
                self.gesture_label = "No Hand"

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