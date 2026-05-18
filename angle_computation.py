from mediapipe_inference import get_video_data
import numpy as np
import cv2

model_path = 'pose_landmarker_heavy.task'
video_path = 'testingVideos/test.mp4'
cap = cv2.VideoCapture(video_path)


landmarks, world_landmarks, valid_frames, fps = get_video_data(video_path, model_path)

def compute_cadence(left_contacts, right_contacts, fps, total_frames):
    """Return cadence in steps per minute"""
    total_contacts = len(left_contacts) + len(right_contacts)
    total_time = total_frames / fps / 60 # seconds / frame * frames * 1 min / 60 s = min
    return total_contacts / total_time; 
    

def compute_knee_flexion_at_contact():
    "Computes angles between hip, knee and ankle at each initial contact."
    visibility_threshold = 0.5
    left_angles = []
    right_angles = []
    for i in range(len(left_contacts)):
        hip = world_landmarks[left_contacts[i], 23] 
        knee = world_landmarks[left_contacts[i], 25]
        ankle = world_landmarks[left_contacts[i], 27]
        #only compute angle if all 3 landmarks are visible. 
        if (hip[3] >= visibility_threshold and knee[3] >= visibility_threshold and ankle[3] >= visibility_threshold):
            knee_ankle = knee[:3] - ankle[:3]
            knee_hip = knee[:3] - hip[:3]
            angle = np.dot(knee_ankle, knee_hip)/np.linalg.norm(knee_ankle)/np.linalg.norm(knee_hip)
            angle = np.arccos(angle)
            left_angles.append(angle)
    
    for i in range(len(right_contacts)):
        hip = world_landmarks[right_contacts[i], 24] 
        knee = world_landmarks[right_contacts[i], 26]
        ankle = world_landmarks[right_contacts[i], 28]
        #only compute angle if all 3 landmarks are visible. 
        if (hip[3] >= visibility_threshold and knee[3] >= visibility_threshold and ankle[3] >= visibility_threshold):
            knee_ankle = knee[:3] - ankle[:3]
            knee_hip = knee[:3] - hip[:3]
            angle = np.dot(knee_ankle, knee_hip)/np.linalg.norm(knee_ankle)/np.linalg.norm(knee_hip)
            angle = np.clip(angle, -1.0, 1.0);
            angle = np.degrees(np.arccos(angle))
            right_angles.append(angle)
    
    return left_angles, right_angles
    

def compute_tibial_flexion_at_contact():
    ...

def compute_thigh_flexion_at_contact():
    ...

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