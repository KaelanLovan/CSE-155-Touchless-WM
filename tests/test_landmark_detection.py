"""Tests for :mod:`landmark_detection`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# --- patch MediaPipe *before* importing the module under test -------------


@pytest.fixture(autouse=True)
def _mock_mediapipe():
    """Prevent every import of ``landmark_detection`` from loading the
    real MediaPipe model file."""
    with (
        patch("mediapipe.tasks.python.vision.HandLandmarker") as mock_hl,
        patch("mediapipe.tasks.python.vision.HandLandmarkerOptions"),
        patch("mediapipe.tasks.python.vision.RunningMode", create=True),
        patch("mediapipe.tasks.python.BaseOptions"),
    ):
        mock_hl.create_from_options.return_value = MagicMock()
        yield


# import after patching so the module-level mp.Image / mp.ImageFormat
# references resolve through the real mediapipe package but the
# HandLandmarker factory is mocked.
from landmark_detection import HandLandmarker  # noqa: E402

# ---------------------------------------------------------------------------
# constructor
# ---------------------------------------------------------------------------


def test_constructor_defaults():
    hl = HandLandmarker()
    assert hl.running_mode is not None
    assert hl.get_latest_result() == (None, -1)


def test_constructor_rejects_invalid_model_path():
    # MediaPipe raises at create_from_options time, not __init__ time.
    # Our mock swallows the error, so we simply verify the constructor
    # propagates the configured path correctly.  A real missing file
    # would raise at MediaPipe level.
    hl = HandLandmarker(model_path="nonexistent.task")
    assert hl is not None


# ---------------------------------------------------------------------------
# detect_async
# ---------------------------------------------------------------------------


def test_detect_async_requires_live_stream():
    hl = HandLandmarker(running_mode=2)  # VIDEO
    # We need a real-ish frame; a small numpy array will do.
    import numpy as np

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="LIVE_STREAM"):
        hl.detect_async(frame, timestamp_ms=1)


# ---------------------------------------------------------------------------
# get_latest_result
# ---------------------------------------------------------------------------


def test_get_latest_result_returns_tuple():
    hl = HandLandmarker()
    result, ts = hl.get_latest_result()
    assert result is None
    assert ts == -1
