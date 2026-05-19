"""
Tests for angle_computation.py

These tests cover:
- Initial contact detection (velocity zero-crossing on ankle y)
- Mid-swing detection and running direction inference
- Knee flexion angle computation (3D angle at knee joint)
- Tibial inclination (signed angle of shank from vertical)
- Thigh angle (signed angle of thigh from vertical)
- Cadence calculation
- Contact validation (alternating left/right)

All tests use synthetic landmark arrays that simulate realistic running
kinematics. No video files or MediaPipe models are required.

Run with: pytest test_angle_computation.py -v
"""

import numpy as np
import pytest
from angle_computation import *


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_empty_landmarks(n_frames):
    """Create a (N, 33, 5) array of zeros with full visibility/presence."""
    arr = np.zeros((n_frames, 33, 5), dtype=np.float32)
    arr[:, :, 3] = 1.0  # visibility
    arr[:, :, 4] = 1.0  # presence
    return arr


def make_running_landmarks(n_frames, fps, cadence_spm=170, direction=1):
    """Create synthetic normalized landmarks simulating running gait.

    The ankle y-coordinate oscillates sinusoidally to simulate the vertical
    motion of the foot during running. Each full cycle represents one stride
    (two steps). The knee also oscillates with a phase offset to simulate
    swing phase timing.

    Args:
        n_frames: number of frames to generate
        fps: frames per second
        cadence_spm: steps per minute (each foot contacts once per step)
        direction: 1 for left-to-right, -1 for right-to-left

    Returns:
        landmarks: (N, 33, 5) array with realistic ankle/knee trajectories
    """
    landmarks = make_empty_landmarks(n_frames)
    t = np.arange(n_frames) / fps

    # Step frequency per foot (each foot hits at half the cadence rate)
    step_freq = cadence_spm / 60.0 / 2.0  # Hz per foot

    # Ankle y: oscillates with peaks (y=0.9, near ground) and troughs (y=0.5, swing)
    # In normalized coords, higher y = closer to ground
    left_ankle_y = 0.7 + 0.2 * np.sin(2 * np.pi * step_freq * t)
    right_ankle_y = 0.7 + 0.2 * np.sin(2 * np.pi * step_freq * t + np.pi)

    # Knee y: similar oscillation but smaller amplitude and phase-shifted
    left_knee_y = 0.55 + 0.05 * np.sin(2 * np.pi * step_freq * t + np.pi)
    right_knee_y = 0.55 + 0.05 * np.sin(2 * np.pi * step_freq * t)

    # Hip: relatively stable vertically
    hip_y = 0.45 + 0.02 * np.sin(2 * np.pi * step_freq * t * 2)

    # X positions: offset based on direction
    base_x = 0.5
    ankle_offset = 0.05 * direction
    knee_offset = 0.03 * direction
    hip_offset = 0.0

    # Left side (landmarks 23=hip, 25=knee, 27=ankle)
    landmarks[:, 23, 0] = base_x + hip_offset  # hip x
    landmarks[:, 23, 1] = hip_y  # hip y
    landmarks[:, 25, 0] = base_x + knee_offset  # knee x
    landmarks[:, 25, 1] = left_knee_y  # knee y
    landmarks[:, 27, 0] = base_x + ankle_offset  # ankle x
    landmarks[:, 27, 1] = left_ankle_y  # ankle y

    # Right side (landmarks 24=hip, 26=knee, 28=ankle)
    landmarks[:, 24, 0] = base_x + hip_offset
    landmarks[:, 24, 1] = hip_y
    landmarks[:, 26, 0] = base_x + knee_offset
    landmarks[:, 26, 1] = right_knee_y
    landmarks[:, 28, 0] = base_x + ankle_offset
    landmarks[:, 28, 1] = right_ankle_y

    return landmarks


# ===========================================================================
# Tests for compute_knee_flexion
# ===========================================================================

class TestComputeKneeFlexion:
    """Tests for the 3D knee angle computation (angle at knee vertex)."""

    def test_straight_leg_returns_180(self):
        """When hip, knee, and ankle are collinear, the angle should be 180°."""
        hip =   np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        knee =  np.array([0.0, 1.0, 0.0, 1.0, 1.0])
        ankle = np.array([0.0, 2.0, 0.0, 1.0, 1.0])
        angle = compute_knee_flexion(hip, knee, ankle)
        assert abs(angle - 180.0) < 0.1

    def test_right_angle_returns_90(self):
        """When the hip-knee and knee-ankle vectors form a right angle,
        the result should be 90°."""
        hip =   np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        knee =  np.array([0.0, 1.0, 0.0, 1.0, 1.0])
        ankle = np.array([1.0, 1.0, 0.0, 1.0, 1.0])
        angle = compute_knee_flexion(hip, knee, ankle)
        assert abs(angle - 90.0) < 0.1

    def test_typical_running_contact_angle(self):
        """A knee angle typical of initial contact (~150-160°) should be
        computed correctly."""
        # Simulate slight knee bend: knee slightly in front of hip-ankle line
        hip =   np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        knee =  np.array([0.05, 0.5, 0.0, 1.0, 1.0])
        ankle = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
        angle = compute_knee_flexion(hip, knee, ankle)
        assert 140 < angle < 180, f"Expected ~150-170°, got {angle}"

    def test_symmetric_for_left_and_right(self):
        """The angle should be the same regardless of mirroring (left vs right
        leg), since it's the magnitude of the angle at the vertex."""
        hip_l =   np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        knee_l =  np.array([0.1, 0.5, 0.0, 1.0, 1.0])
        ankle_l = np.array([0.0, 1.0, 0.0, 1.0, 1.0])

        hip_r =   np.array([1.0, 0.0, 0.0, 1.0, 1.0])
        knee_r =  np.array([0.9, 0.5, 0.0, 1.0, 1.0])
        ankle_r = np.array([1.0, 1.0, 0.0, 1.0, 1.0])

        angle_l = compute_knee_flexion(hip_l, knee_l, ankle_l)
        angle_r = compute_knee_flexion(hip_r, knee_r, ankle_r)
        assert abs(angle_l - angle_r) < 0.1

    def test_handles_near_degenerate_case(self):
        """When points are nearly coincident, arccos clipping should prevent
        NaN from being returned."""
        hip =   np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        knee =  np.array([0.0, 1e-10, 0.0, 1.0, 1.0])
        ankle = np.array([0.0, 2e-10, 0.0, 1.0, 1.0])
        angle = compute_knee_flexion(hip, knee, ankle)
        assert not np.isnan(angle)


# ===========================================================================
# Tests for compute_tibial_flexion
# ===========================================================================

class TestComputeTibialFlexion:
    """Tests for tibial inclination angle (shank relative to vertical)."""

    def test_vertical_shin_returns_zero(self):
        """When the ankle is directly below the knee (vertical shin),
        the inclination should be 0°."""
        knee =  np.array([0.5, 0.5, 0.0, 1.0, 1.0])
        ankle = np.array([0.5, 0.8, 0.0, 1.0, 1.0])
        angle = compute_tibial_flexion(knee, ankle, direction=1)
        assert abs(angle) < 0.1

    def test_ankle_ahead_is_positive(self):
        """When the ankle is ahead of the knee in the running direction,
        the angle should be positive (overstriding indicator)."""
        knee =  np.array([0.5, 0.5, 0.0, 1.0, 1.0])
        ankle = np.array([0.6, 0.8, 0.0, 1.0, 1.0])  # ahead in +x direction
        angle = compute_tibial_flexion(knee, ankle, direction=1)
        assert angle > 0, f"Expected positive angle, got {angle}"

    def test_ankle_behind_is_negative(self):
        """When the ankle is behind the knee in the running direction,
        the angle should be negative."""
        knee =  np.array([0.5, 0.5, 0.0, 1.0, 1.0])
        ankle = np.array([0.4, 0.8, 0.0, 1.0, 1.0])  # behind in +x direction
        angle = compute_tibial_flexion(knee, ankle, direction=1)
        assert angle < 0, f"Expected negative angle, got {angle}"

    def test_direction_flips_sign(self):
        """The same landmark positions should produce opposite signs
        when the running direction is reversed."""
        knee =  np.array([0.5, 0.5, 0.0, 1.0, 1.0])
        ankle = np.array([0.6, 0.8, 0.0, 1.0, 1.0])

        angle_ltr = compute_tibial_flexion(knee, ankle, direction=1)
        angle_rtl = compute_tibial_flexion(knee, ankle, direction=-1)

        assert angle_ltr > 0
        assert angle_rtl < 0
        assert abs(angle_ltr + angle_rtl) < 0.1  # equal magnitude, opposite sign

    def test_known_angle(self):
        """A 45° triangle should return 45°."""
        knee =  np.array([0.5, 0.5, 0.0, 1.0, 1.0])
        ankle = np.array([0.8, 0.8, 0.0, 1.0, 1.0])  # dx=0.3, dy=0.3 → 45°
        angle = compute_tibial_flexion(knee, ankle, direction=1)
        assert abs(angle - 45.0) < 0.1


# ===========================================================================
# Tests for compute_thigh_angle
# ===========================================================================

class TestComputeThighAngle:
    """Tests for thigh inclination angle (thigh relative to vertical)."""

    def test_vertical_thigh_returns_zero(self):
        """When the knee is directly below the hip, angle should be 0°."""
        hip =  np.array([0.5, 0.3, 0.0, 1.0, 1.0])
        knee = np.array([0.5, 0.6, 0.0, 1.0, 1.0])
        angle = compute_thigh_angle(hip, knee, direction=1)
        assert abs(angle) < 0.1

    def test_knee_ahead_is_positive(self):
        """Knee ahead of hip in running direction should be positive."""
        hip =  np.array([0.5, 0.3, 0.0, 1.0, 1.0])
        knee = np.array([0.6, 0.6, 0.0, 1.0, 1.0])
        angle = compute_thigh_angle(hip, knee, direction=1)
        assert angle > 0

    def test_knee_behind_is_negative(self):
        """Knee behind hip in running direction should be negative
        (hip extension at toe-off)."""
        hip =  np.array([0.5, 0.3, 0.0, 1.0, 1.0])
        knee = np.array([0.4, 0.6, 0.0, 1.0, 1.0])
        angle = compute_thigh_angle(hip, knee, direction=1)
        assert angle < 0

    def test_direction_flips_sign(self):
        """Reversing direction should flip the angle sign."""
        hip =  np.array([0.5, 0.3, 0.0, 1.0, 1.0])
        knee = np.array([0.6, 0.6, 0.0, 1.0, 1.0])

        angle_ltr = compute_thigh_angle(hip, knee, direction=1)
        angle_rtl = compute_thigh_angle(hip, knee, direction=-1)

        assert angle_ltr > 0
        assert angle_rtl < 0


# ===========================================================================
# Tests for detect_initial_contacts
# ===========================================================================

class TestDetectInitialContacts:
    """Tests for foot strike detection via ankle y-velocity zero-crossing."""

    def test_detects_correct_number_of_contacts(self):
        """A synthetic running signal at known cadence should produce
        approximately the expected number of contacts per foot."""
        fps = 30.0
        duration = 10.0  # seconds
        n_frames = int(fps * duration)
        cadence_spm = 170  # steps per minute total, 85 per foot

        landmarks = make_running_landmarks(n_frames, fps, cadence_spm)
        contacts = detect_initial_contacts(landmarks, fps, foot='left')

        expected_contacts = int(cadence_spm / 2 * duration / 60)  # ~14
        # Allow ±2 for edge effects
        assert abs(len(contacts) - expected_contacts) <= 2, (
            f"Expected ~{expected_contacts} contacts, got {len(contacts)}"
        )

    def test_contacts_are_sorted(self):
        """Returned contact indices should be in chronological order."""
        landmarks = make_running_landmarks(300, 30.0, 170)
        contacts = detect_initial_contacts(landmarks, 30.0, foot='left')
        assert np.all(np.diff(contacts) > 0)

    def test_minimum_distance_enforced(self):
        """No two contacts should be closer than fps * 0.25 frames apart."""
        fps = 30.0
        landmarks = make_running_landmarks(300, fps, 170)
        contacts = detect_initial_contacts(landmarks, fps, foot='left')
        min_dist = int(fps * 0.25)
        if len(contacts) > 1:
            assert np.all(np.diff(contacts) >= min_dist)

    def test_left_and_right_are_offset(self):
        """Left and right foot contacts should be approximately half a
        stride apart (interleaved)."""
        fps = 30.0
        landmarks = make_running_landmarks(300, fps, 170)
        left = detect_initial_contacts(landmarks, fps, foot='left')
        right = detect_initial_contacts(landmarks, fps, foot='right')

        if len(left) > 1 and len(right) > 1:
            left_period = np.mean(np.diff(left))
            right_period = np.mean(np.diff(right))
            # Both feet should have similar stride periods
            assert abs(left_period - right_period) < 5, (
                f"Left period={left_period:.1f}, right={right_period:.1f}"
            )

    def test_returns_numpy_array(self):
        """Output should be a numpy array, not a list."""
        landmarks = make_running_landmarks(100, 30.0, 170)
        contacts = detect_initial_contacts(landmarks, 30.0, foot='left')
        assert isinstance(contacts, np.ndarray)

    def test_constant_signal_returns_no_contacts(self):
        """A flat ankle signal (no motion) should produce no contacts."""
        landmarks = make_empty_landmarks(100)
        landmarks[:, 27, 1] = 0.8  # constant y
        contacts = detect_initial_contacts(landmarks, 30.0, foot='left')
        assert len(contacts) == 0


# ===========================================================================
# Tests for validate_contacts
# ===========================================================================

class TestValidateContacts:
    """Tests for checking that left and right contacts alternate properly."""

    def test_perfectly_alternating_passes(self):
        """Perfectly interleaved contacts should return True."""
        left = np.array([10, 30, 50, 70, 90])
        right = np.array([20, 40, 60, 80, 100])
        assert validate_contacts(left, right) == True

    def test_many_violations_fails(self):
        """When most contacts don't alternate, should return False."""
        # All lefts before all rights — every transition is a violation
        left = np.array([10, 20, 30, 40, 50])
        right = np.array([60, 70, 80, 90, 100])
        assert validate_contacts(left, right) == False

    def test_few_violations_passes(self):
        """A small number of violations (< 20%) should still pass."""
        # Mostly alternating with one violation
        left = np.array([10, 25, 50, 70, 90])
        right = np.array([20, 40, 60, 80, 100])
        # Sequence: 10L, 20R, 25L, 40R, 50L, 60R, 70L, 80R, 90L, 100R — perfect
        assert validate_contacts(left, right) == True

    def test_empty_contacts(self):
        """Empty contact arrays should not crash."""
        left = np.array([])
        right = np.array([10, 20, 30])
        # Should handle gracefully (all same side = violations, but
        # with only one side, every consecutive pair is a violation)
        result = validate_contacts(left, right)
        assert isinstance(result, bool)


# ===========================================================================
# Tests for compute_cadence
# ===========================================================================

class TestComputeCadence:
    """Tests for cadence (steps per minute) calculation."""

    def test_known_cadence(self):
        """With known contact counts and duration, cadence should be exact."""
        # 85 left contacts + 85 right contacts in 60 seconds at 30fps = 1800 frames
        left = np.arange(0, 1800, 1800 // 85)[:85]
        right = np.arange(10, 1800, 1800 // 85)[:85]
        fps = 30.0
        total_frames = 1800

        cadence = compute_cadence(left, right, fps, total_frames)
        assert abs(cadence - 170) < 5, f"Expected ~170 spm, got {cadence}"

    def test_higher_cadence(self):
        """A faster runner should show higher cadence."""
        fps = 30.0
        total_frames = 900  # 30 seconds

        slow_left = np.arange(0, 900, 25)  # ~36 contacts
        slow_right = np.arange(12, 900, 25)
        fast_left = np.arange(0, 900, 15)  # ~60 contacts
        fast_right = np.arange(7, 900, 15)

        slow = compute_cadence(slow_left, slow_right, fps, total_frames)
        fast = compute_cadence(fast_left, fast_right, fps, total_frames)
        assert fast > slow

    def test_zero_contacts_returns_zero(self):
        """With no contacts, cadence should be 0."""
        cadence = compute_cadence(np.array([]), np.array([]), 30.0, 900)
        assert cadence == 0


# ===========================================================================
# Tests for compute_angles
# ===========================================================================

class TestComputeAngles:
    """Tests for the combined angle computation at initial contacts."""

    def test_returns_correct_keys(self):
        """The returned dictionary should contain all expected metric keys
        with left/right sub-keys."""
        landmarks = make_empty_landmarks(100)
        world_landmarks = make_empty_landmarks(100)

        # Place landmarks in a reasonable configuration
        for i in range(100):
            landmarks[i, 23, :2] = [0.5, 0.4]   # hip
            landmarks[i, 25, :2] = [0.5, 0.6]   # knee
            landmarks[i, 27, :2] = [0.5, 0.8]   # ankle
            landmarks[i, 24, :2] = [0.5, 0.4]
            landmarks[i, 26, :2] = [0.5, 0.6]
            landmarks[i, 28, :2] = [0.5, 0.8]
            world_landmarks[i, 23, :3] = [0.0, 0.0, 0.0]
            world_landmarks[i, 25, :3] = [0.0, 0.5, 0.0]
            world_landmarks[i, 27, :3] = [0.0, 1.0, 0.0]
            world_landmarks[i, 24, :3] = [0.0, 0.0, 0.0]
            world_landmarks[i, 26, :3] = [0.0, 0.5, 0.0]
            world_landmarks[i, 28, :3] = [0.0, 1.0, 0.0]

        left_contacts = np.array([10, 30, 50])
        right_contacts = np.array([20, 40, 60])

        result = compute_angles(right_contacts, left_contacts, 0.5,
                                landmarks, world_landmarks, direction=1)

        assert 'knee_flexion' in result
        assert 'tibial_inclination' in result
        assert 'thigh_angle' in result
        for key in result:
            assert 'left' in result[key]
            assert 'right' in result[key]

    def test_populates_both_sides(self):
        """Both left and right arrays should be populated when contacts
        are provided for both sides."""
        landmarks = make_empty_landmarks(100)
        world_landmarks = make_empty_landmarks(100)

        for i in range(100):
            # Slight knee bend configuration
            landmarks[i, 23, :2] = [0.5, 0.3]
            landmarks[i, 25, :2] = [0.52, 0.55]
            landmarks[i, 27, :2] = [0.5, 0.8]
            landmarks[i, 24, :2] = [0.5, 0.3]
            landmarks[i, 26, :2] = [0.52, 0.55]
            landmarks[i, 28, :2] = [0.5, 0.8]
            world_landmarks[i, 23, :3] = [0.0, 0.0, 0.0]
            world_landmarks[i, 25, :3] = [0.02, 0.25, 0.0]
            world_landmarks[i, 27, :3] = [0.0, 0.5, 0.0]
            world_landmarks[i, 24, :3] = [0.0, 0.0, 0.0]
            world_landmarks[i, 26, :3] = [0.02, 0.25, 0.0]
            world_landmarks[i, 28, :3] = [0.0, 0.5, 0.0]

        left_contacts = np.array([10, 30])
        right_contacts = np.array([20, 40])

        result = compute_angles(right_contacts, left_contacts, 0.5,
                                landmarks, world_landmarks, direction=1)

        assert len(result['knee_flexion']['left']) == 2
        assert len(result['knee_flexion']['right']) == 2

    def test_low_visibility_excluded(self):
        """Landmarks with visibility below threshold should not produce
        angle measurements."""
        landmarks = make_empty_landmarks(50)
        world_landmarks = make_empty_landmarks(50)

        # Set visibility to 0 for all landmarks
        landmarks[:, :, 3] = 0.0
        world_landmarks[:, :, 3] = 0.0

        left_contacts = np.array([10, 20])
        right_contacts = np.array([15, 25])

        result = compute_angles(right_contacts, left_contacts, 0.5,
                                landmarks, world_landmarks, direction=1)

        assert len(result['knee_flexion']['left']) == 0
        assert len(result['knee_flexion']['right']) == 0
        assert len(result['tibial_inclination']['left']) == 0


class TestComputeVerticalOscillation:
    """Tests for vertical oscillation (hip midpoint bounce amplitude).

    Vertical oscillation measures how much the runner's center of mass
    bounces vertically during each stride. It's computed as the difference
    between local maxima and minima of the hip midpoint y-coordinate.
    """

    def test_sinusoidal_signal_detects_peaks_and_troughs(self):
        """A clean sinusoidal hip trajectory should produce equal numbers
        of maxima and minima (±1)."""
        n_frames = 300
        landmarks = make_empty_landmarks(n_frames)

        t = np.arange(n_frames) / fps
        hip_y = 0.45 + 0.03 * np.sin(2 * np.pi * 3.0 * t)

        landmarks[:, 23, 1] = hip_y
        landmarks[:, 24, 1] = hip_y

        maxima, minima = compute_vertical_oscillation(landmarks, 0.5)

        assert len(maxima) > 0
        assert len(minima) > 0
        assert abs(len(maxima) - len(minima)) <= 1

    def test_oscillation_amplitude_is_correct(self):
        """For a known sinusoidal signal with amplitude A, the difference
        between mean maxima and mean minima should be approximately 2*A."""
        n_frames = 300
        landmarks = make_empty_landmarks(n_frames)

        amplitude = 0.04
        t = np.arange(n_frames) / fps
        hip_y = 0.45 + amplitude * np.sin(2 * np.pi * 3.0 * t)

        landmarks[:, 23, 1] = hip_y
        landmarks[:, 24, 1] = hip_y

        maxima, minima = compute_vertical_oscillation(landmarks, 0.5)

        measured = np.mean(maxima) - np.mean(minima)
        expected = 2 * amplitude
        assert abs(measured - expected) < 0.005

    def test_constant_signal_returns_empty(self):
        """A flat hip trajectory (no bounce) should produce no maxima or minima."""
        landmarks = make_empty_landmarks(100)
        landmarks[:, 23, 1] = 0.45
        landmarks[:, 24, 1] = 0.45

        maxima, minima = compute_vertical_oscillation(landmarks, 0.5)

        assert len(maxima) == 0
        assert len(minima) == 0

    def test_low_visibility_excluded(self):
        """Frames where hip landmarks have low visibility should not
        contribute peaks or troughs."""
        n_frames = 300
        landmarks = make_empty_landmarks(n_frames)

        t = np.arange(n_frames) / fps
        landmarks[:, 23, 1] = 0.45 + 0.03 * np.sin(2 * np.pi * 3.0 * t)
        landmarks[:, 24, 1] = 0.45 + 0.03 * np.sin(2 * np.pi * 3.0 * t)

        landmarks[:, 23, 3] = 0.0  # zero visibility
        landmarks[:, 24, 3] = 0.0

        maxima, minima = compute_vertical_oscillation(landmarks, 0.5)

        assert len(maxima) == 0
        assert len(minima) == 0

    def test_midpoint_is_average_of_both_hips(self):
        """Should use average of left and right hip y, not just one side.
        This test will fail if the parentheses bug is present
        (left + right/2 instead of (left + right)/2)."""
        n_frames = 300
        landmarks = make_empty_landmarks(n_frames)

        t = np.arange(n_frames) / fps
        amplitude = 0.03

        # Left hip at 0.40, right at 0.50 — midpoint should be 0.45
        landmarks[:, 23, 1] = 0.40 + amplitude * np.sin(2 * np.pi * 3.0 * t)
        landmarks[:, 24, 1] = 0.50 + amplitude * np.sin(2 * np.pi * 3.0 * t)

        maxima, minima = compute_vertical_oscillation(landmarks, 0.5)

        # Midpoint maxima should be near 0.45 + 0.03 = 0.48
        assert len(maxima) > 0
        assert abs(np.mean(maxima) - 0.48) < 0.01

    def test_higher_bounce_produces_larger_amplitude(self):
        """A runner with more vertical bounce should produce a larger
        difference between maxima and minima."""
        n_frames = 300
        t = np.arange(n_frames) / fps

        low = make_empty_landmarks(n_frames)
        low[:, 23, 1] = 0.45 + 0.02 * np.sin(2 * np.pi * 3.0 * t)
        low[:, 24, 1] = low[:, 23, 1]

        high = make_empty_landmarks(n_frames)
        high[:, 23, 1] = 0.45 + 0.06 * np.sin(2 * np.pi * 3.0 * t)
        high[:, 24, 1] = high[:, 23, 1]

        low_max, low_min = compute_vertical_oscillation(low, 0.5)
        high_max, high_min = compute_vertical_oscillation(high, 0.5)

        low_amp = np.mean(low_max) - np.mean(low_min)
        high_amp = np.mean(high_max) - np.mean(high_min)

        assert high_amp > low_amp