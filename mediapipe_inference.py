import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt


model_path = 'pose_landmarker_heavy.task'
video_path = 'testingVideos/overstriding.mov'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def extract_landmarks(video_path='testingVideos/test.mp4', model_path='pose_landmarker_heavy.task'):
    """Extract pose landmarks from video using MediaPipe PoseLandmarker.
    Returns:
        all_landmarks: array of shape (N, 33, 5) with [x, y, z, visibility, presence]
        all_world_landmarks: array of shape (N, 33, 5) with world coordinates
        valid_frames: boolean array of shape (N,) indicating which frames had detections
        fps: frames per second of the video
    """ 
    def check_landmarks():
        """
        Draws landmarks on some frame and saves to png file for visual check.
        """
        # Quick visual check: draw landmarks on a middle frame
        check_idx = frame_idx // 2
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, check_idx)
        ret, frame = cap.read()
        cap.release()

        if ret and valid_frames[check_idx]:
            h, w = frame.shape[:2]
            for lm_idx in range(33):
                x = int(all_landmarks[check_idx, lm_idx, 0] * w)
                y = int(all_landmarks[check_idx, lm_idx, 1] * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            cv2.imwrite("landmark_check.png", frame)
            print("Saved landmark_check.png — visually verify keypoints look correct")

    # Create a pose landmarker instance with the video mode:
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="pose_landmarker_heavy.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.3,
        output_segmentation_masks=False,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 100 #over-allocate

    # Initialize empty arrays to store landmarks frame-by-frame.
    # Shape: (num_frames, 33, 5) — frames × landmarks × [x, y, z, visibility, presence]
    all_landmarks = np.zeros((total_frames, 33, 5), dtype=np.float32)
    all_world_landmarks = np.zeros((total_frames, 33, 5), dtype=np.float32) 
    valid_frames = np.zeros(total_frames, dtype = bool)

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR (OpenCV default) to RGB (MediaPipe expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Timestamp must be monotonically increasing, in milliseconds
            timestamp_ms = int(frame_idx * 1000 / fps)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # check if we've found a valid frame. If so, add detected landmarks to array.
            if result.pose_landmarks: 
                valid_frames[frame_idx] = True
                for lm_idx, lm in enumerate(result.pose_landmarks[0]):
                    all_landmarks[frame_idx, lm_idx] = [lm.x, lm.y, lm.z, lm.visibility, lm.presence]

                for lm_idx, lm in enumerate(result.pose_world_landmarks[0]):
                    all_world_landmarks[frame_idx, lm_idx] = [lm.x, lm.y, lm.z, lm.visibility, lm.presence]

            frame_idx += 1

    cap.release()

    # Trim to actual frames processed (in case total_frames was slightly off)
    all_landmarks = all_landmarks[:frame_idx]
    all_world_landmarks = all_world_landmarks[:frame_idx]
    valid_frames = valid_frames[:frame_idx]

    print(f"Processed {frame_idx} frames, {valid_frames.sum()} with detections "
        f"({valid_frames.mean()*100:.1f}% detection rate)")
    
    check_landmarks()
        
    return all_landmarks, all_world_landmarks, valid_frames, fps
    
def interpolate_gaps(landmarks, valid_frames, max_gap=5):
    """
    Linearly interpolate over short gaps in landmark data
    Fill in frames where data was not found by assuming linear relationship.
    
    Args:
        landmarks: array of shape (N, 33, 5)
        valid_frames: boolean array of shape (N,)
        max_gap: maximum gap length to interpolate (frames)
    
    Returns:
        interpolated landmarks, updated valid_frames
    """
    landmarks = landmarks.copy()
    valid = valid_frames.copy()
    n_frames = len(landmarks)
    
    # Find gaps
    valid_indices = np.where(valid)[0]
    
    if len(valid_indices) < 2:
        print("Warning: fewer than 2 valid frames, cannot interpolate")
        return landmarks, valid
    
    missing_indices = np.where(~valid)[0]
    interpolatable = set()

    for idx in missing_indices: 
        # Check gap length: find nearest valid frames on each side
        # slices element to only include valid indices whose index < idx (left side)
        left = valid_indices[valid_indices < idx]
        right = valid_indices[valid_indices > idx]
        if len(left) > 0 and len(right) > 0 and (right[0] - left[-1]) <= max_gap:
            interpolatable.add(idx)

    for lm_idx in range(33): 
        for coord_idx in range(3): 
            values = landmarks[:, lm_idx, coord_idx]
            # Build interpolator (scipy.interpolate.interp1d) from valid frames only
            interp_func = interp1d(valid_indices, values[valid_indices], kind='linear',
                bounds_error=False, fill_value="extrapolate")
            
            for idx in interpolatable: 
                landmarks[idx, lm_idx, coord_idx] = interp_func(idx)
    
    for idx in interpolatable: 
        valid[idx] = True

    return landmarks, valid

def butterworth_filter(landmarks, valid_frames, fps, cutoff=6.0, order=4):
    """
    Apply a zero-phase Butterworth low-pass filter to landmark trajectories.
    Essentially: remove noise. Butterworth filter is standard filter used in biomechanics. 
    
    Args:
        landmarks: array of shape (N, 33, 5)
        valid_frames: boolean array of shape (N,)
        fps: video frame rate
        cutoff: cutoff frequency in Hz (6 Hz is standard for running)
        order: filter order (4 is standard in biomechanics)
    
    Returns:
        filtered landmarks
    """
    landmarks = landmarks.copy()
    nyquist = fps / 2.0
    normalized_cutoff = cutoff / nyquist
    
    # Ensure cutoff is valid
    if normalized_cutoff >= 1.0:
        print(f"Warning: cutoff {cutoff} Hz >= Nyquist {nyquist} Hz, skipping filter")
        return landmarks
    
    b,a = butter(order, normalized_cutoff, btype='low') # type: ignore
    
    # Find longest continuous valid segment(s)
    # filtfilt needs continuous data, so filter each segment separately
    segments = get_continuous_segments(valid_frames, min_length=order * 3)
    
    for start, end in segments:
        for lm_idx in range(33):
            for coord_idx in range(3):  # only x, y, z
                signal = landmarks[start:end, lm_idx, coord_idx]
                # filtfilt requires segment length > padlen (3 * order)
                if len(signal) > 3 * order:
                    landmarks[start:end, lm_idx, coord_idx] = filtfilt(b, a, signal)
    
    return landmarks

def get_continuous_segments(valid_frames, min_length=12):
    """Find continuous runs of True in valid_frames."""
    segments = []
    start = None
    for i, v in enumerate(valid_frames):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_length:
                segments.append((start, i))
            start = None
    if start is not None and len(valid_frames) - start >= min_length:
        segments.append((start, len(valid_frames)))
    return segments

def get_video_data(video_path='testingVideos/test.mp4', model_path='pose_landmarker_heavy.task'):
    """
    Get final data from a video 
    Return: 
        landmarks: (number of frames, 33, 5) array of pose landmarks 
        world_landmarks: (number of frames, 33, 5) array of world landmarks
        valid_frames: (number of frames,) boolean array indicating valid frames
        fps: frames per second of the video
    """
    landmarks, world_landmarks, valid_frames, fps = extract_landmarks(video_path, model_path)
    landmarks, valid_frames = interpolate_gaps(landmarks, valid_frames)
    world_landmarks, valid_frames = interpolate_gaps(world_landmarks, valid_frames)

    landmarks = butterworth_filter(landmarks, valid_frames, fps)
    world_landmarks = butterworth_filter(world_landmarks, valid_frames, fps)

    return landmarks, world_landmarks, valid_frames, fps

get_video_data(video_path, model_path)