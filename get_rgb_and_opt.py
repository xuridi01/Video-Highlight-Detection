import cv2
import numpy as np

def extract_optical_flow(video_path, target_size=(224, 224), segment_length=50, overlap=25):
    # Get frames per second and normalize to 25 FPS
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(fps / 25))

    # Read first frame and resize
    ret, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, target_size)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    flows = []

    #Read and resize rest
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % frame_interval != 0:
            continue

        frame_resized = cv2.resize(frame, target_size)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prev_gray = gray

        # Normalize optical flow to [-1,1]
        flow = (flow - np.min(flow)) / (np.max(flow) - np.min(flow)) * 2 - 1
        flows.append(flow)

    cap.release()

    # Convert to overlapping 50-frame segments
    segments = []
    for i in range(0, len(flows) - segment_length + 1, overlap):
        segments.append(np.array(flows[i:i + segment_length]))

    np.save('selfie_day_opt.npy', segments)
    return


def extract_rgb_frames(video_path, target_size=(224, 224), segment_length=50, overlap=25):
    # Get frames per second and normalize to 25 FPS
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(fps / 25))  # Normalize to 25 FPS
    frames = []

    # Read and resize frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % frame_interval != 0:
            continue

        # Resize and normalize
        frame = cv2.resize(frame, target_size)
        frame = (frame / 127.5) - 1.0  # Normalize to [-1,1]
        frames.append(frame)

    cap.release()

    # Convert frames to overlapping 50-frame segments, overlap is 25
    segments = []
    for i in range(0, len(frames) - segment_length + 1, overlap):
        segments.append(np.array(frames[i:i + segment_length]))
    np.save('selfie_day.npy', segments)
    return
