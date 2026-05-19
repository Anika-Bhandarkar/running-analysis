from mediapipe_inference import get_video_data
import numpy as np
import cv2

DIRECTION_CONFIDENCE_THRESHOLD = 0.7
VISIBILITY_THRESHOLD = 0.5

MODEL_PATH = 'pose_landmarker_heavy.task'
VIDEO_PATH = 'testingVideos/test.mp4'

cap = cv2.VideoCapture(VIDEO_PATH)
landmarks, world_landmarks, valid_frames, fps = get_video_data(VIDEO_PATH, MODEL_PATH)


"""
TO TRACK PER SIDE:
1. overstriding @ initial contact (shin angle)
2. thigh inclination @ initial contact
3. knee flexion @ initial contact
4. vertical oscillation
5. cadence
6. knee flexion during swing 
"""


def compute_cadence(left_contacts, right_contacts, fps, total_frames):
    """Return cadence in steps per minute"""
    total_contacts = len(left_contacts) + len(right_contacts)
    total_time = total_frames / fps / 60 # seconds / frame * frames * 1 min / 60 s = min
    return total_contacts / total_time; 
    
def compute_angles(right_contacts, left_contacts, visibility_threshold, landmarks, world_landmarks, direction):
    angles = {
        'knee_flexion': {'left': [], 'right': []},
        'tibial_inclination': {'left': [], 'right': []},
        'thigh_angle': {'left': [], 'right': []},
    }
    
    for i in range(len(left_contacts)):
        world_hip = world_landmarks[left_contacts[i], 23] 
        world_knee = world_landmarks[left_contacts[i], 25]
        world_ankle = world_landmarks[left_contacts[i], 27]

        hip = landmarks[left_contacts[i], 23]
        knee = landmarks[left_contacts[i], 25]
        ankle = landmarks[left_contacts[i], 27]

        if (world_hip[3] >= visibility_threshold and world_knee[3] >= visibility_threshold and world_ankle[3] >= visibility_threshold):
            angles['knee_flexion']['left'].append(compute_knee_flexion(world_hip, world_knee, world_ankle))
        if (knee[3] >= visibility_threshold and ankle[3] >= visibility_threshold):
            angles['tibial_inclination']['left'].append(compute_tibial_flexion(knee, ankle, direction))
        if (hip[3] >= visibility_threshold and knee[3] >= visibility_threshold):
            angles['thigh_angle']['left'].append(compute_thigh_angle(hip, knee, direction))

    for i in range(len(right_contacts)):
        world_hip = world_landmarks[right_contacts[i], 24] 
        world_knee = world_landmarks[right_contacts[i], 26]
        world_ankle = world_landmarks[right_contacts[i], 28]

        hip = landmarks[right_contacts[i], 24]
        knee = landmarks[right_contacts[i], 26]
        ankle = landmarks[right_contacts[i], 28]

        if (world_hip[3] >= visibility_threshold and world_knee[3] >= visibility_threshold and world_ankle[3] >= visibility_threshold):
            angles['knee_flexion']['right'].append(compute_knee_flexion(world_hip, world_knee, world_ankle))
        if (knee[3] >= visibility_threshold and ankle[3] >= visibility_threshold):
            angles['tibial_inclination']['right'].append(compute_tibial_flexion(knee, ankle, direction))
        if (hip[3] >= visibility_threshold and knee[3] >= visibility_threshold):
            angles['thigh_angle']['right'].append(compute_thigh_angle(hip, knee, direction))

    return angles

def compute_knee_flexion(world_hip, world_knee, world_ankle):
    "Computes angles between hip, knee and ankle at each initial contact."
    knee_ankle = world_knee[:3] - world_ankle[:3]
    knee_hip = world_knee[:3] - world_hip[:3]
    angle = np.dot(knee_ankle, knee_hip)/np.linalg.norm(knee_ankle)/np.linalg.norm(knee_hip)
    angle = np.clip(angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(angle))
    return angle
    
def compute_tibial_flexion(knee, ankle, direction):
    """Return angle between shin and vertical at initial contact, taking direction of motion as positive x.
    Positive angle indicates ankle in front of knee (overstriding)."""
    dx = (ankle[0] - knee[0]) * direction
    dy = ankle[1] - knee[1]
    return np.degrees(np.arctan2(dx, dy))

def compute_thigh_angle(hip, knee, direction):
    """Return angle formed by thigh and vertical at initial contact, taking direction of motion as positive x.
    Positive angle indicates knee ahead of hip."""
    dx = (knee[0] - hip[0]) * direction
    dy = knee[1] - hip[1]
    angle = np.degrees(np.arctan2(dx, dy))
    return angle

def detect_initial_contacts(landmarks, fps, foot='left'):
    """
        Return an array of indices corresponding to initial contact of specified foot. 
        1. calculate velocity of (normalized) ankle landmark 
        2. For t where v(t) = 0 and v changes from positive to negative, add t to contacts

        landmarks: normalized landmarks (N * 33 * 5)
        fps = frames per second of input video
    """
    ankle_idx = 27 if foot == "left" else 28
    contacts = [] #frame indices
    ankle_position = landmarks[:, ankle_idx, 1] #(N, 1)
    min_distance = int(fps * 0.25)

    ankle_velocity = np.gradient(ankle_position, 1.0/fps)
    for i in range(1, len(ankle_velocity) - 1):
        if ankle_velocity[i-1] > 0 and ankle_velocity[i] <= 0: 
            # at zero-crossing, confirm upward (negative) acceleration.
            if i < len(ankle_velocity) -1: 
                if ankle_velocity[i + 1] - ankle_velocity[i] < 0: 
                    contacts.append(i)


    # enforce minimum distance betwen contacts --> do not want to double-detect fluctuations in ankle when striking.
    if contacts: 
        filtered = [contacts[0]]
        for c in contacts[1:]:
            if c - filtered[-1] >= min_distance:
                filtered.append(c)
        contacts = filtered

    return np.array(contacts)

def validate_contacts(left_contacts, right_contacts):
    """Check that left and right contacts roughly alternate."""
    all_contacts = []
    for f in left_contacts:
        all_contacts.append((f, 'L'))
    for f in right_contacts:
        all_contacts.append((f, 'R'))
    all_contacts.sort(key=lambda x: x[0])

    violations = 0
    for i in range(1, len(all_contacts)):
        if all_contacts[i][1] == all_contacts[i - 1][1]:
            violations += 1

    if violations > len(all_contacts) * 0.2:
        print(f"Warning: {violations} non-alternating contacts out of {len(all_contacts)}. "
              f"Far-side leg detection may be unreliable.")
        return False
    else: 
        print("No violations. Reasonably alternating left and right contacts.")
        return True

def detect_mid_swing(world_landmarks, fps, visibility_threshold, direction_confidence_threshold):
    left_knee_drive_angle = []
    right_knee_drive_angle = []
    knee_ahead_count = 0
    knee_behind_count = 0

    left_knee_y = world_landmarks[:, 25, 1]
    velocity = np.gradient(left_knee_y, 1.0 / fps)

    # mid-swing occurs when velocity of knee goes from negative to positive
    for i in range (1, len(velocity) - 1):
        if velocity[i-1] < 0 and velocity[i] >= 0:
            hip = world_landmarks[i, 23]
            knee = world_landmarks[i, 25]
            ankle = world_landmarks[i, 27]

            if(hip[3] >= visibility_threshold and knee[3] >= visibility_threshold and ankle[3] >= visibility_threshold):
                angle = compute_knee_flexion(hip, knee, ankle)
                left_knee_drive_angle.append(angle)
                if knee[0] > ankle[0]:
                    knee_ahead_count += 1
                else:
                    knee_behind_count += 1

    right_knee_y = world_landmarks[:, 26, 1]
    velocity = np.gradient(right_knee_y, 1.0 / fps)

    for i in range (1, len(velocity) - 1):
        if velocity[i-1] < 0 and velocity[i] >= 0:
            hip = world_landmarks[i, 24]
            knee = world_landmarks[i, 26]
            ankle = world_landmarks[i, 28]

            if(hip[3] >= visibility_threshold and knee[3] >= visibility_threshold and ankle[3] >= visibility_threshold):
                angle = compute_knee_flexion(hip, knee, ankle)
                right_knee_drive_angle.append(angle)
                if knee[0] > ankle[0]:
                    knee_ahead_count += 1
                else:
                    knee_behind_count += 1
    
    total_counts = knee_ahead_count + knee_behind_count
    if total_counts == 0:
        raise ValueError("No mid-swing events detected.")
    
    ratio = knee_ahead_count / total_counts
    
    if ratio > direction_confidence_threshold:
        direction = 1
    elif ratio < (1 - direction_confidence_threshold):
        direction = -1
    else:
        raise ValueError("Uncertain direction based on knee positions.")


    return direction, left_knee_drive_angle, right_knee_drive_angle

def compute_vertical_oscillation(landmarks, visibility_threshold):
    """Compute vertical oscillation (amplitude) of midpoint of pelvis over time."""

    com_y = (landmarks[:, 23, 1] + landmarks[:, 24, 1]) / 2.0 #y coordinate of center of mass
    maxima = []
    minima = []

    for i in range(1, len(com_y) - 1):
        if landmarks[i, 23, 3] >= visibility_threshold and landmarks[i, 24, 3] >= visibility_threshold and com_y[i-1] < com_y[i] and com_y[i] > com_y[i+1]:
            maxima.append(com_y[i])
        elif landmarks[i, 23, 3] >= visibility_threshold and landmarks[i, 24, 3] >= visibility_threshold and com_y[i-1] > com_y[i] and com_y[i] < com_y[i+1]:
            minima.append(com_y[i])

    return maxima, minima

left_contacts = detect_initial_contacts(landmarks, fps, foot='left')
right_contacts = detect_initial_contacts(landmarks, fps, foot = 'right')
validate_contacts(left_contacts, right_contacts)

direction, left_knee_swing, right_knee_swing = detect_mid_swing(landmarks, fps, VISIBILITY_THRESHOLD, DIRECTION_CONFIDENCE_THRESHOLD)
metrics = compute_angles(right_contacts, left_contacts, VISIBILITY_THRESHOLD, landmarks, world_landmarks, direction)
maxima, minima = compute_vertical_oscillation(landmarks, VISIBILITY_THRESHOLD)

metrics["knee_swing"] = {'left': left_knee_swing, 'right': right_knee_swing}
metrics["vertical_oscillation"] = {'maxima': maxima, 'minima': minima}

cadence = compute_cadence(left_contacts, right_contacts, fps, len(landmarks))