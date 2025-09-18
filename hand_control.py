"""Utilities for translating hand movements into "flap" events.

This module encapsulates the logic for capturing webcam frames, running them
through MediaPipe Hands, and inferring when the player has performed a quick
upward motion that should trigger a flap inside the game.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Optional

import cv2

try:  # MediaPipe renamed the public package for solutions in newer releases.
    from mediapipe import solutions as mp_solutions
except ImportError:  # pragma: no cover - depends on mediapipe installation layout.
    try:
        from mediapipe.python import solutions as mp_solutions  # type: ignore[attr-defined]
    except ImportError as exc:  # pragma: no cover - easier to diagnose at runtime.
        raise ImportError(
            "The installed MediaPipe distribution does not expose the solutions API."
            " Install the official `mediapipe` package (version 0.9.x or 0.10.x)"
            " to enable hand tracking."
        ) from exc

mp_hands = mp_solutions.hands

try:
    HandLandmark = mp_hands.HandLandmark
except AttributeError:  # pragma: no cover
    from mediapipe.python.solutions.hands import HandLandmark  # type: ignore[attr-defined]


class HandFlapDetector:
    """Detect upward hand motions that should trigger a flap.

    Parameters
    ----------
    camera_index:
        Index of the camera that should be opened with ``cv2.VideoCapture``.
    history_length:
        Number of recent position samples used to smooth the motion signal.
        A shorter history increases responsiveness at the cost of more noise.
    rise_threshold:
        Minimum upward delta (in normalized coordinates) that must be observed
        within the smoothing window to count as a flap.
    velocity_threshold:
        Minimum upward velocity (in normalized units / second) detected across
        consecutive samples that should trigger a flap. This helps detect quick
        flicks even when the absolute travel distance is small.
    analysis_window:
        Maximum amount of history (in seconds) that should contribute to the
        flap decision. Limiting the window focuses the detector on recent
        movement so quick motions are easier to register.
    cooldown_s:
        Minimum number of seconds between two consecutive flap events. This
        prevents jitter when the hand remains in motion.
    debug:
        When ``True`` a debug window visualizing the detection pipeline will be
        shown. This is useful when tuning thresholds.
    """

    def __init__(
        self,
        *,
        camera_index: int = 0,
        history_length: int = 6,
        rise_threshold: float = 0.03,
        velocity_threshold: float = 0.75,
        analysis_window: float = 0.6,
        cooldown_s: float = 0.25,
        debug: bool = False,
    ) -> None:
        self.camera_index = camera_index
        self.history_length = history_length
        self.rise_threshold = rise_threshold
        self.velocity_threshold = velocity_threshold
        self.analysis_window = analysis_window
        self.cooldown_s = cooldown_s
        self.debug = debug

        self._capture = cv2.VideoCapture(self.camera_index)
        if not self._capture.isOpened():
            raise RuntimeError("Unable to open webcam. Ensure a camera is connected.")

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._capture.set(cv2.CAP_PROP_FPS, 30)

        self._mp_hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,
        )

        self._positions: Deque[float] = deque(maxlen=history_length)
        self._timestamps: Deque[float] = deque(maxlen=history_length)
        self._last_flap_time = 0.0
        self._flap_pending = False
        self._lock = threading.Lock()
        self._running = True
        self._frame: Optional[Any] = None
        self._closed = False

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._mp_hands.process(frame_rgb)

            y_position: Optional[float] = None
            if result.multi_hand_landmarks:
                # MediaPipe returns coordinates normalized in [0, 1]. We average
                # a handful of palm landmarks so a partially occluded wrist does
                # not prevent reliable tracking.
                landmarks = result.multi_hand_landmarks[0].landmark
                sample_indices = [
                    HandLandmark.WRIST,
                    HandLandmark.INDEX_FINGER_MCP,
                    HandLandmark.MIDDLE_FINGER_MCP,
                    HandLandmark.RING_FINGER_MCP,
                    HandLandmark.PINKY_MCP,
                ]
                y_position = sum(landmarks[idx.value].y for idx in sample_indices) / len(sample_indices)
                self._positions.append(y_position)
                self._timestamps.append(time.time())
                self._prune_history(self._timestamps[-1])
            else:
                self._positions.clear()
                self._timestamps.clear()

            if self.debug:
                self._frame = frame.copy()
                if result.multi_hand_landmarks:
                    mp_solutions.drawing_utils.draw_landmarks(
                        self._frame,
                        result.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS,
                    )

            self._update_flap_state()

            if self.debug and self._frame is not None:
                cv2.imshow("Hand Debug", self._frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    # Allow closing the debug window with escape.
                    self.debug = False
                    cv2.destroyWindow("Hand Debug")

        self._cleanup_resources()

    def _update_flap_state(self) -> None:
        if len(self._positions) < 2:
            return

        newest_t = self._timestamps[-1]
        self._prune_history(newest_t)
        if len(self._positions) < 2:
            return

        positions = list(self._positions)
        timestamps = list(self._timestamps)
        newest_y = positions[-1]

        max_delta = 0.0
        max_velocity = 0.0

        for idx, prev_y in enumerate(positions[:-1]):
            if newest_t - timestamps[idx] > self.analysis_window:
                continue
            delta = prev_y - newest_y
            if delta > max_delta:
                max_delta = delta

        for prev_idx in range(1, len(positions)):
            dt = timestamps[prev_idx] - timestamps[prev_idx - 1]
            if dt <= 0:
                continue
            if newest_t - timestamps[prev_idx] > self.analysis_window:
                continue
            velocity = (positions[prev_idx - 1] - positions[prev_idx]) / dt
            if velocity > max_velocity:
                max_velocity = velocity

        should_trigger = False
        if max_delta > self.rise_threshold:
            should_trigger = True
        elif max_velocity > self.velocity_threshold:
            should_trigger = True

        time_since_last = newest_t - self._last_flap_time

        if should_trigger and time_since_last >= self.cooldown_s:
            with self._lock:
                self._flap_pending = True
            self._last_flap_time = newest_t

    def poll_flap(self) -> bool:
        """Return ``True`` if a flap was detected since the last call."""

        with self._lock:
            detected = self._flap_pending
            self._flap_pending = False
        return detected

    def stop(self) -> None:
        """Stop the capture thread and release the webcam."""

        self._running = False
        self._thread.join(timeout=1.0)
        self._cleanup_resources()

    def __enter__(self) -> "HandFlapDetector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _cleanup_resources(self) -> None:
        if self._closed:
            return
        self._capture.release()
        self._mp_hands.close()
        if self.debug:
            cv2.destroyAllWindows()
        self._closed = True

    def _prune_history(self, newest_timestamp: float) -> None:
        while (
            len(self._timestamps) > 1
            and newest_timestamp - self._timestamps[0] > self.analysis_window
        ):
            self._timestamps.popleft()
            self._positions.popleft()


__all__ = ["HandFlapDetector"]
