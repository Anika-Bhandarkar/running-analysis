from mediapipe_inference import get_video_data
import numpy as np
import cv2

model_path = 'pose_landmarker_heavy.task'
video_path = 'testingVideos/test.mp4'
cap = cv2.VideoCapture(video_path)

landmarks, world_landmarks, valid_frames, fps = get_video_data(video_path, model_path)


"""
TO TRACK PER SIDE:
1. overstriding @ initial contact (shin angle)
2. thigh inclination @ initial contact
3. knee flexion @ initial contact
4. vertical oscillation
5. cadence
6. knee flexion during swing 
7. hip extension at toe-off 
"""

#TODO: add detection for which way the user is facing; angles should be computed relative to that. 

def compute_cadence(left_contacts, right_contacts, fps, total_frames):
    """Return cadence in steps per minute"""
    total_contacts = len(left_contacts) + len(right_contacts)
    total_time = total_frames / fps / 60 # seconds / frame * frames * 1 min / 60 s = min
    return total_contacts / total_time; 
    
def compute_angles(right_contacts, left_contacts, visibility_threshold, landmarks, world_landmarks):
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
            angles['tibial_inclination']['left'].append(compute_tibial_flexion(knee, ankle))
        if (hip[3] >= visibility_threshold and knee[3] >= visibility_threshold):
            angles['thigh_angle']['left'].append(compute_thigh_angle(hip, knee))

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
            angles['tibial_inclination']['right'].append(compute_tibial_flexion(knee, ankle))
        if (hip[3] >= visibility_threshold and knee[3] >= visibility_threshold):
            angles['thigh_angle']['right'].append(compute_thigh_angle(hip, knee))

    return angles


    

def compute_knee_flexion(world_hip, world_knee, world_ankle):
    "Computes angles between hip, knee and ankle at each initial contact."
    knee_ankle = world_knee[:3] - world_ankle[:3]
    knee_hip = world_knee[:3] - world_hip[:3]
    angle = np.dot(knee_ankle, knee_hip)/np.linalg.norm(knee_ankle)/np.linalg.norm(knee_hip)
    angle = np.clip(angle, -1.0, 1.0);
    angle = np.degrees(np.arccos(angle))
    return angle
    
def compute_tibial_flexion(knee, ankle):
    """Return angle between shin and vertical at initial contact."""
    dx = ankle[0] - knee[0]
    dy = ankle[1] - knee[1]
    return np.degrees(np.arctan2(dx, dy))

def compute_thigh_angle(hip, knee):
    """Return angle formed by thigh and vertical at initial contact."""
    dx = knee[0] - hip[0]
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

left_contacts = detect_initial_contacts(landmarks, fps, foot='left')
right_contacts = detect_initial_contacts(landmarks, fps, foot = 'right')
validate_contacts(left_contacts, right_contacts)