from __future__ import annotations


class GestureClassifier:
    """Rule-based hand gesture classifier using MediaPipe 2D hand landmarks.

    Classifies static hand poses by counting extended fingers. Uses a
    simple hold-frame debounce to suppress flickering between gestures.

    A finger is *extended* when its tip is further from the wrist than
    its PIP joint along the axis orthogonal to the hand's natural
    direction.  The *rotation* parameter (0--3) accounts for camera
    orientation so that the comparison axis tracks the physical
    "up" direction:

    ========  ============  =================
    Rotation  OpenCV step   Comparison
    ========  ============  =================
    0         (none)        ``tip.y < pip.y``
    1         90° clockwise ``tip.x > pip.x``
    2         180°          ``tip.y > pip.y``
    3         270° cw       ``tip.x < pip.x``
    ========  ============  =================

    Attributes:
        GESTURES: Human-readable names for each gesture class index.
        confirmed_gesture: The last debounced gesture index, or None.
    """

    GESTURES: list[str] = ["Fist", "Point", "Two", "Three", "Open"]

    def __init__(self, hold_frames: int = 5, rotation: int = 0) -> None:
        """Initialise the classifier.

        Args:
            hold_frames: Number of consecutive identical frames required
                before a new gesture is confirmed.
            rotation: Camera rotation steps (0=normal, 1=90°cw,
                2=180°, 3=270°cw).
        """
        self._prev_gesture: int | None = None
        self._hold_count: int = 0
        self._hold_frames: int = hold_frames
        self.confirmed_gesture: int | None = None
        self.rotation: int = rotation % 4

    # -- comparison helpers ---------------------------------------------------

    def _tip_ahead_of_pip(self, tip, pip) -> bool:
        """Return True when *tip* is ahead of *pip* for the current rotation."""
        if self.rotation == 0:
            return tip.y < pip.y
        if self.rotation == 1:
            return tip.x > pip.x
        if self.rotation == 2:
            return tip.y > pip.y
        return tip.x < pip.x

    # -- public API -----------------------------------------------------------

    def count_extended_fingers(self, hand_2d: list) -> int:
        """Count how many of the four main fingers are extended.

        Args:
            hand_2d: List of 21 MediaPipe NormalizedLandmark objects.

        Returns:
            The number of extended fingers (0-4).
        """
        finger_pairs = [
            (8, 6),
            (12, 10),
            (16, 14),
            (20, 18),
        ]
        extended = 0
        for tip_idx, pip_idx in finger_pairs:
            if self._tip_ahead_of_pip(hand_2d[tip_idx], hand_2d[pip_idx]):
                extended += 1
        return extended

    def is_open_palm(self, hand_2d: list) -> bool:
        """Return True if the hand forms an open palm (4 fingers extended)."""
        return self.count_extended_fingers(hand_2d) >= 4

    def is_fist(self, hand_2d: list) -> bool:
        """Return True if the hand forms a fist (0-1 fingers extended)."""
        return self.count_extended_fingers(hand_2d) <= 1

    def classify(self, hand_2d: list) -> int | None:
        """Classify the current hand pose.

        Applies a hold-frame debounce: the gesture must be observed for
        ``hold_frames`` consecutive frames before it is confirmed.

        Args:
            hand_2d: List of 21 MediaPipe NormalizedLandmark objects.

        Returns:
            A gesture index (0-4) when confirmed, or the last confirmed
            index (which may be None) while debouncing.
        """
        extended = self.count_extended_fingers(hand_2d)

        if extended <= 0:
            gesture = 0
        elif extended == 1:
            gesture = 1
        elif extended == 2:
            gesture = 2
        elif extended == 3:
            gesture = 3
        else:
            gesture = 4

        if gesture != self._prev_gesture:
            self._hold_count = 0
        else:
            self._hold_count += 1

        self._prev_gesture = gesture

        if self._hold_count >= self._hold_frames:
            self.confirmed_gesture = gesture
            return gesture
        return self.confirmed_gesture
