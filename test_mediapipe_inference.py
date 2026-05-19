"""
Tests for mediapipe_inference.py

These tests cover the post-processing pipeline: gap interpolation, Butterworth
filtering, and continuous segment detection. They use synthetic landmark data
to verify correctness without requiring real video files or the MediaPipe model.

The actual extract_landmarks() function is not unit-tested here because it
requires a real video file and the pose_landmarker_heavy.task model. To verify
inference, run the script on a test video and visually inspect landmark_check.png.

Run with: pytest test_mediapipe_inference.py -v
"""

import numpy as np
import pytest
from mediapipe_inference import interpolate_gaps, butterworth_filter, get_continuous_segments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_landmarks(n_frames, fill_value=1.0):
    """Create a synthetic (N, 33, 5) landmark array filled with a constant.
    Columns are [x, y, z, visibility, presence]."""
    arr = np.full((n_frames, 33, 5), fill_value, dtype=np.float32)
    return arr


def make_sinusoidal_landmarks(n_frames, freq=2.0, fps=30.0):
    """Create landmarks where x/y/z follow a sine wave at the given frequency.
    Useful for testing that the Butterworth filter preserves low-frequency
    signals and removes high-frequency noise."""
    arr = np.zeros((n_frames, 33, 5), dtype=np.float32)
    t = np.arange(n_frames) / fps
    signal = np.sin(2 * np.pi * freq * t)
    for lm in range(33):
        for coord in range(3):
            arr[:, lm, coord] = signal
        arr[:, lm, 3] = 1.0  # visibility
        arr[:, lm, 4] = 1.0  # presence
    return arr


# ===========================================================================
# Tests for get_continuous_segments
# ===========================================================================

class TestGetContinuousSegments:
    """Tests for finding continuous runs of valid frames."""

    def test_all_valid(self):
        """A fully valid array should return one segment spanning the entire range."""
        valid = np.ones(100, dtype=bool)
        segments = get_continuous_segments(valid, min_length=5)
        assert segments == [(0, 100)]

    def test_all_invalid(self):
        """A fully invalid array should return no segments."""
        valid = np.zeros(100, dtype=bool)
        segments = get_continuous_segments(valid, min_length=5)
        assert segments == []

    def test_single_gap_in_middle(self):
        """A single invalid frame in the middle should split into two segments,
        provided both halves meet the minimum length."""
        valid = np.ones(100, dtype=bool)
        valid[50] = False
        segments = get_continuous_segments(valid, min_length=5)
        assert segments == [(0, 50), (51, 100)]

    def test_segments_below_min_length_are_excluded(self):
        """Short valid runs (below min_length) should be dropped."""
        valid = np.zeros(30, dtype=bool)
        # A 3-frame segment (too short for min_length=5)
        valid[10:13] = True
        # A 10-frame segment (long enough)
        valid[20:30] = True
        segments = get_continuous_segments(valid, min_length=5)
        assert segments == [(20, 30)]

    def test_multiple_gaps(self):
        """Multiple gaps should produce multiple segments."""
        valid = np.ones(50, dtype=bool)
        valid[10:12] = False  # gap at 10-11
        valid[30:33] = False  # gap at 30-32
        segments = get_continuous_segments(valid, min_length=5)
        assert segments == [(0, 10), (12, 30), (33, 50)]

    def test_gap_at_start(self):
        """Invalid frames at the beginning should be excluded from the first segment."""
        valid = np.ones(50, dtype=bool)
        valid[0:5] = False
        segments = get_continuous_segments(valid, min_length=5)
        assert segments == [(5, 50)]

    def test_gap_at_end(self):
        """Invalid frames at the end should be excluded from the last segment."""
        valid = np.ones(50, dtype=bool)
        valid[45:50] = False
        segments = get_continuous_segments(valid, min_length=5)
        assert segments == [(0, 45)]


# ===========================================================================
# Tests for interpolate_gaps
# ===========================================================================

class TestInterpolateGaps:
    """Tests for linear interpolation over short gaps in landmark data."""

    def test_no_gaps(self):
        """If all frames are valid, landmarks should be unchanged."""
        landmarks = make_landmarks(50, fill_value=1.0)
        valid = np.ones(50, dtype=bool)
        result, result_valid = interpolate_gaps(landmarks, valid)
        np.testing.assert_array_equal(result, landmarks)
        assert result_valid.all()

    def test_single_frame_gap_is_interpolated(self):
        """A single missing frame between two valid frames should be
        linearly interpolated for x, y, z coordinates."""
        landmarks = np.zeros((10, 33, 5), dtype=np.float32)
        valid = np.ones(10, dtype=bool)

        # Set up a linear ramp on landmark 0, x coordinate
        for i in range(10):
            landmarks[i, 0, 0] = float(i)  # x = frame index
            landmarks[i, :, 3] = 1.0  # visibility
            landmarks[i, :, 4] = 1.0  # presence

        # Remove frame 5
        valid[5] = False
        landmarks[5, :, :] = 0.0

        result, result_valid = interpolate_gaps(landmarks, valid)

        # Frame 5 should be interpolated to x=5.0 (linear between 4 and 6)
        assert result_valid[5] == True
        assert abs(result[5, 0, 0] - 5.0) < 0.01

    def test_gap_exceeding_max_gap_is_not_interpolated(self):
        """Gaps longer than max_gap should be left as zeros."""
        landmarks = make_landmarks(30, fill_value=1.0)
        valid = np.ones(30, dtype=bool)

        # Create a 10-frame gap (exceeds default max_gap=5)
        valid[10:20] = False
        landmarks[10:20] = 0.0

        result, result_valid = interpolate_gaps(landmarks, valid, max_gap=5)

        # Frames 10-19 should still be invalid
        assert not result_valid[10:20].any()

    def test_gap_at_max_gap_boundary(self):
        """A gap exactly equal to max_gap should be interpolated."""
        landmarks = np.zeros((20, 33, 5), dtype=np.float32)
        valid = np.ones(20, dtype=bool)

        for i in range(20):
            landmarks[i, 0, 0] = float(i)
            landmarks[i, :, 3] = 1.0
            landmarks[i, :, 4] = 1.0

        # Create a gap of exactly 5 frames (frames 5-9, gap length = 10-5 = 5)
        valid[6:10] = False
        landmarks[6:10] = 0.0

        result, result_valid = interpolate_gaps(landmarks, valid, max_gap=5)

        # These frames should now be valid and interpolated
        assert result_valid[6:10].all()
        # Frame 7 should be between frame 5 (x=5) and frame 10 (x=10)
        assert abs(result[7, 0, 0] - 7.0) < 0.1

    def test_gap_at_edges_is_not_interpolated(self):
        """Gaps at the very start or end of the array cannot be interpolated
        because there's no valid frame on one side."""
        landmarks = make_landmarks(20, fill_value=1.0)
        valid = np.ones(20, dtype=bool)

        valid[0:3] = False
        landmarks[0:3] = 0.0

        result, result_valid = interpolate_gaps(landmarks, valid)

        # Frames 0-2 should remain invalid (no valid frame to the left)
        assert not result_valid[0:3].any()

    def test_does_not_modify_original(self):
        """interpolate_gaps should not modify the input arrays."""
        landmarks = make_landmarks(20, fill_value=1.0)
        valid = np.ones(20, dtype=bool)
        valid[10] = False
        landmarks_copy = landmarks.copy()
        valid_copy = valid.copy()

        interpolate_gaps(landmarks, valid)

        np.testing.assert_array_equal(landmarks, landmarks_copy)
        np.testing.assert_array_equal(valid, valid_copy)

    def test_fewer_than_two_valid_frames(self):
        """With fewer than 2 valid frames, interpolation is impossible.
        Should return data unchanged."""
        landmarks = make_landmarks(10, fill_value=0.0)
        valid = np.zeros(10, dtype=bool)
        valid[5] = True
        landmarks[5] = 1.0

        result, result_valid = interpolate_gaps(landmarks, valid)
        assert result_valid.sum() == 1


# ===========================================================================
# Tests for butterworth_filter
# ===========================================================================

class TestButterworthFilter:
    """Tests for the zero-phase Butterworth low-pass filter."""

    def test_low_frequency_signal_preserved(self):
        """A signal well below the cutoff frequency should pass through
        the filter with minimal attenuation."""
        fps = 30.0
        n_frames = 300  # 10 seconds
        # 2 Hz signal, cutoff at 6 Hz — should be preserved
        landmarks = make_sinusoidal_landmarks(n_frames, freq=2.0, fps=fps)
        valid = np.ones(n_frames, dtype=bool)

        filtered = butterworth_filter(landmarks, valid, fps, cutoff=6.0)

        # The filtered signal should closely match the original
        original_signal = landmarks[:, 0, 0]
        filtered_signal = filtered[:, 0, 0]

        # Allow some edge effects but the middle should match well
        middle = slice(50, 250)
        correlation = np.corrcoef(original_signal[middle], filtered_signal[middle])[0, 1]
        assert correlation > 0.99, f"Low-freq signal was distorted: correlation={correlation}"

    def test_high_frequency_noise_removed(self):
        """High-frequency noise added to a low-frequency signal should be
        attenuated by the filter."""
        fps = 30.0
        n_frames = 300
        landmarks = make_sinusoidal_landmarks(n_frames, freq=2.0, fps=fps)
        valid = np.ones(n_frames, dtype=bool)

        # Add high-frequency noise (12 Hz, well above 6 Hz cutoff)
        t = np.arange(n_frames) / fps
        noise = 0.5 * np.sin(2 * np.pi * 12.0 * t)
        noisy_landmarks = landmarks.copy()
        for lm in range(33):
            for coord in range(3):
                noisy_landmarks[:, lm, coord] += noise

        filtered = butterworth_filter(noisy_landmarks, valid, fps, cutoff=6.0)

        # Filtered signal should be closer to the clean signal than the noisy one
        middle = slice(50, 250)
        clean = landmarks[:, 0, 0]
        noisy = noisy_landmarks[:, 0, 0]
        filt = filtered[:, 0, 0]

        error_before = np.mean((noisy[middle] - clean[middle]) ** 2)
        error_after = np.mean((filt[middle] - clean[middle]) ** 2)

        assert error_after < error_before * 0.1, (
            f"Filter didn't sufficiently remove noise: "
            f"MSE before={error_before:.4f}, after={error_after:.4f}"
        )

    def test_cutoff_above_nyquist_skips_filter(self):
        """If the cutoff frequency exceeds the Nyquist frequency, the filter
        should be skipped and landmarks returned unchanged."""
        fps = 10.0  # Nyquist = 5 Hz
        landmarks = make_landmarks(100, fill_value=3.14)
        valid = np.ones(100, dtype=bool)

        filtered = butterworth_filter(landmarks, valid, fps, cutoff=6.0)

        np.testing.assert_array_equal(filtered, landmarks)

    def test_does_not_modify_original(self):
        """butterworth_filter should not modify the input array."""
        landmarks = make_sinusoidal_landmarks(100, freq=2.0, fps=30.0)
        valid = np.ones(100, dtype=bool)
        original = landmarks.copy()

        butterworth_filter(landmarks, valid, 30.0)

        np.testing.assert_array_equal(landmarks, original)

    def test_visibility_and_presence_unchanged(self):
        """The filter should only modify x, y, z (indices 0-2) and leave
        visibility (index 3) and presence (index 4) untouched."""
        fps = 30.0
        landmarks = make_sinusoidal_landmarks(100, freq=2.0, fps=fps)
        landmarks[:, :, 3] = 0.95  # visibility
        landmarks[:, :, 4] = 0.99  # presence
        valid = np.ones(100, dtype=bool)

        filtered = butterworth_filter(landmarks, valid, fps)

        np.testing.assert_array_equal(filtered[:, :, 3], landmarks[:, :, 3])
        np.testing.assert_array_equal(filtered[:, :, 4], landmarks[:, :, 4])

    def test_segments_filtered_independently(self):
        """When there's a gap in valid frames, each continuous segment should
        be filtered independently. The gap frames should remain unchanged."""
        fps = 30.0
        n_frames = 100
        landmarks = make_sinusoidal_landmarks(n_frames, freq=2.0, fps=fps)
        valid = np.ones(n_frames, dtype=bool)

        # Create a gap
        valid[45:55] = False
        landmarks[45:55] = 0.0

        filtered = butterworth_filter(landmarks, valid, fps)

        # Gap frames should remain zero
        np.testing.assert_array_equal(filtered[45:55], 0.0)

    def test_short_segment_not_filtered(self):
        """Segments shorter than 3*order frames should not be filtered
        (filtfilt would fail). They should be returned unchanged."""
        fps = 30.0
        order = 4
        min_filterable = 3 * order + 1  # 13 frames

        landmarks = make_sinusoidal_landmarks(10, freq=2.0, fps=fps)
        valid = np.ones(10, dtype=bool)

        # 10 frames < 13, so this segment should be left untouched
        original = landmarks.copy()
        filtered = butterworth_filter(landmarks, valid, fps, order=order)

        np.testing.assert_array_equal(filtered[:, :, :3], original[:, :, :3])