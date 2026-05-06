"""Tests for :mod:`process`."""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch

import pytest

from process import put_latest, GestureBackend


# ---------------------------------------------------------------------------
# put_latest
# ---------------------------------------------------------------------------


def test_put_latest_empty_queue():
    q: queue.Queue[int] = queue.Queue(maxsize=1)
    put_latest(q, 42)
    assert q.get_nowait() == 42


def test_put_latest_replaces_oldest():
    q: queue.Queue[int] = queue.Queue(maxsize=1)
    put_latest(q, 1)
    put_latest(q, 2)
    assert q.get_nowait() == 2


def test_put_latest_many_replacements():
    q: queue.Queue[int] = queue.Queue(maxsize=1)
    for i in range(100):
        put_latest(q, i)
    assert q.get_nowait() == 99


# ---------------------------------------------------------------------------
# GestureBackend lifecycle
# ---------------------------------------------------------------------------


@pytest.fixture
def backend():
    """Return a GestureBackend whose internal MediaPipe/classifier/camera
    deps are mocked so no real hardware or model files are required."""
    with (
        patch("process.HandLandmarker") as mock_hl,
        patch("process.GestureClassifier"),
        patch("cv2.VideoCapture") as mock_vc,
    ):
        mock_hl.return_value.detect_async = MagicMock()
        mock_hl.return_value.get_latest_result.return_value = (None, -1)
        mock_cam = MagicMock()
        mock_cam.isOpened.return_value = True
        mock_cam.read.return_value = (False, None)
        mock_vc.return_value = mock_cam
        yield GestureBackend()


class TestGestureBackendLifecycle:
    def test_initial_state(self, backend):
        assert backend.is_processing_running() is False
        assert backend.gesture_label == "None"
        assert backend.controls_active is True

    def test_start_stop_camera(self, backend):
        backend.start_camera()
        backend.stop_all()
        assert backend.camera_thread_ref is None

    def test_start_stop_processing(self, backend):
        backend.start_processing()
        assert backend.is_processing_running() is True
        backend.stop_processing()
        assert backend.is_processing_running() is False

    def test_stop_all_stops_both(self, backend):
        backend.start_camera()
        backend.start_processing()
        backend.stop_all()
        assert backend.camera_thread_ref is None
        assert backend.process_thread_ref is None

    def test_double_start_processing_noop(self, backend):
        backend.start_processing()
        first_thread = backend.process_thread_ref
        backend.start_processing()
        assert backend.process_thread_ref is first_thread

    def test_double_start_camera_noop(self, backend):
        backend.start_camera()
        first_thread = backend.camera_thread_ref
        backend.start_camera()
        assert backend.camera_thread_ref is first_thread

    def test_get_latest_display_frame_empty(self, backend):
        assert backend.get_latest_display_frame() is None

    def test_print_average_timings_no_data(self, capsys, backend):
        backend.print_average_timings()
        captured = capsys.readouterr()
        assert "No frames" in captured.out


# ---------------------------------------------------------------------------
# modifier_key auto-detection
# ---------------------------------------------------------------------------


def test_modifier_key_auto_detect_macos():
    with patch("sys.platform", "darwin"):
        from process import _detect_modifier_key

        assert _detect_modifier_key() == "command"


def test_modifier_key_auto_detect_windows():
    with patch("sys.platform", "win32"):
        from process import _detect_modifier_key

        assert _detect_modifier_key() == "alt"


def test_modifier_key_auto_detect_linux():
    with patch("sys.platform", "linux"):
        from process import _detect_modifier_key

        assert _detect_modifier_key() == "alt"


def test_modifier_key_explicit_override():
    with (
        patch("process.HandLandmarker"),
        patch("process.GestureClassifier"),
    ):
        backend = GestureBackend(modifier_key="ctrl")
        assert backend.modifier_key == "ctrl"
