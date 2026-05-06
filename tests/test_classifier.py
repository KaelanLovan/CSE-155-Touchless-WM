"""Tests for :mod:`classifier`."""

from __future__ import annotations

import pytest

from classifier import GestureClassifier

from . import FakeLandmark, make_landmarks


# ---------------------------------------------------------------------------
# count_extended_fingers
# ---------------------------------------------------------------------------

ALL_EXTENDED = [(0.2, 0.8), (0.2, 0.8), (0.2, 0.8), (0.2, 0.8)]
ALL_CLOSED = [(0.8, 0.2), (0.8, 0.2), (0.8, 0.2), (0.8, 0.2)]


@pytest.mark.parametrize(
    ("finger_pairs", "expected"),
    [
        (ALL_EXTENDED, 4),
        (ALL_CLOSED, 0),
        ([(0.2, 0.8), (0.8, 0.2), (0.8, 0.2), (0.8, 0.2)], 1),  # index only
        ([(0.2, 0.8), (0.2, 0.8), (0.8, 0.2), (0.8, 0.2)], 2),  # index + middle
        ([(0.2, 0.8), (0.2, 0.8), (0.2, 0.8), (0.8, 0.2)], 3),  # 3 fingers
        ([(0.2, 0.8), (0.2, 0.8), (0.2, 0.8), (0.01, 0.99)], 4),  # all extended
    ],
)
def test_count_extended_fingers(finger_pairs, expected):
    clf = GestureClassifier()
    hand = make_landmarks(finger_pairs)
    assert clf.count_extended_fingers(hand) == expected


def test_count_extended_fingers_tip_equals_pip():
    clf = GestureClassifier()
    hand = make_landmarks([(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)])
    assert clf.count_extended_fingers(hand) == 0


# ---------------------------------------------------------------------------
# is_open_palm / is_fist
# ---------------------------------------------------------------------------


def test_is_open_palm_true():
    clf = GestureClassifier()
    hand = make_landmarks(ALL_EXTENDED)
    assert clf.is_open_palm(hand) is True


def test_is_open_palm_false():
    clf = GestureClassifier()
    hand = make_landmarks(ALL_CLOSED)
    assert clf.is_open_palm(hand) is False


def test_is_fist_true_zero_extended():
    clf = GestureClassifier()
    hand = make_landmarks(ALL_CLOSED)
    assert clf.is_fist(hand) is True


def test_is_fist_true_one_extended():
    clf = GestureClassifier()
    hand = make_landmarks([(0.2, 0.8), (0.8, 0.2), (0.8, 0.2), (0.8, 0.2)])
    assert clf.is_fist(hand) is True


def test_is_fist_false_two_extended():
    clf = GestureClassifier()
    hand = make_landmarks([(0.2, 0.8), (0.2, 0.8), (0.8, 0.2), (0.8, 0.2)])
    assert clf.is_fist(hand) is False


# ---------------------------------------------------------------------------
# classify -- gesture indices
# ---------------------------------------------------------------------------


def test_classify_fist():
    clf = GestureClassifier(hold_frames=1)
    hand = make_landmarks(ALL_CLOSED)
    # need hold_frames consecutive frames
    for _ in range(2):
        idx = clf.classify(hand)
    assert idx == 0
    assert clf.GESTURES[idx] == "Fist"


def test_classify_point():
    clf = GestureClassifier(hold_frames=1)
    hand = make_landmarks([(0.2, 0.8), (0.8, 0.2), (0.8, 0.2), (0.8, 0.2)])
    for _ in range(2):
        idx = clf.classify(hand)
    assert idx == 1
    assert clf.GESTURES[idx] == "Point"


def test_classify_two():
    clf = GestureClassifier(hold_frames=1)
    hand = make_landmarks([(0.2, 0.8), (0.2, 0.8), (0.8, 0.2), (0.8, 0.2)])
    for _ in range(2):
        idx = clf.classify(hand)
    assert idx == 2
    assert clf.GESTURES[idx] == "Two"


def test_classify_three():
    clf = GestureClassifier(hold_frames=1)
    hand = make_landmarks([(0.2, 0.8), (0.2, 0.8), (0.2, 0.8), (0.8, 0.2)])
    for _ in range(2):
        idx = clf.classify(hand)
    assert idx == 3
    assert clf.GESTURES[idx] == "Three"


def test_classify_open():
    clf = GestureClassifier(hold_frames=1)
    hand = make_landmarks(ALL_EXTENDED)
    for _ in range(2):
        idx = clf.classify(hand)
    assert idx == 4
    assert clf.GESTURES[idx] == "Open"


# ---------------------------------------------------------------------------
# classify -- debounce
# ---------------------------------------------------------------------------


def test_debounce_requires_hold_frames():
    clf = GestureClassifier(hold_frames=3)

    open_hand = make_landmarks(ALL_EXTENDED)
    fist_hand = make_landmarks(ALL_CLOSED)

    for _ in range(6):
        clf.classify(open_hand)
    assert clf.confirmed_gesture == 4

    assert clf.classify(fist_hand) == 4

    assert clf.classify(fist_hand) == 4

    assert clf.classify(fist_hand) == 4

    assert clf.classify(fist_hand) == 0
    assert clf.confirmed_gesture == 0


def test_classify_returns_none_initially():
    clf = GestureClassifier(hold_frames=5)
    hand = make_landmarks(ALL_EXTENDED)
    assert clf.classify(hand) is None
    assert clf.confirmed_gesture is None


def test_confirmed_gesture_exposed():
    clf = GestureClassifier(hold_frames=1)
    hand = make_landmarks([(0.2, 0.8), (0.2, 0.8), (0.8, 0.2), (0.8, 0.2)])
    for _ in range(2):
        clf.classify(hand)
    assert clf.confirmed_gesture == 2
