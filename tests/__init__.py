"""Test utilities shared across the test suite."""

from __future__ import annotations


class FakeLandmark:
    """A lightweight stand-in for a MediaPipe NormalizedLandmark.

    Attributes:
        x: Normalised x coordinate (0.0--1.0).
        y: Normalised y coordinate (0.0--1.0).
        z: Normalised z coordinate.
    """

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.5, y: float = 0.5, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


def make_landmarks(finger_y_pairs: list[tuple[float, float]]) -> list[FakeLandmark]:
    """Build a 21-landmark list for one hand.

    Landmark indices 0--20 follow the MediaPipe hand topology.  Callers
    provide ``(tip_y, pip_y)`` tuples for fingers in order: *index*,
    *middle*, *ring*, *pinky*.  All other landmarks get the default
    ``y=0.5``.

    A finger is *extended* when ``tip_y < pip_y`` (tip is **above** pip
    in screen space).

    Args:
        finger_y_pairs: Four ``(tip_y, pip_y)`` tuples for
            ``[(8, 6), (12, 10), (16, 14), (20, 18)]``.

    Returns:
        A list of 21 FakeLandmark objects.
    """
    hand = [FakeLandmark() for _ in range(21)]

    tip_indices = [8, 12, 16, 20]
    pip_indices = [6, 10, 14, 18]

    for i, (tip_y, pip_y) in enumerate(finger_y_pairs):
        hand[tip_indices[i]].y = tip_y
        hand[pip_indices[i]].y = pip_y

    return hand
