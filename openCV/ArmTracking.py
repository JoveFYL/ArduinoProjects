from tkinter import W
import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Start capturing from webcam
cap = cv2.VideoCapture("./WhiteBigCover.MOV")

if not cap.isOpened():
    print("Error opening video")
  
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert color
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run pose estimation
    results = pose.process(rgb_frame)

    # Draw landmarks if any are detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract landmark coordinates
        landmarks = results.pose_landmarks.landmark

        # Helper to get pixel coordinates
        def get_coords(idx):
            h, w, _ = frame.shape
            lm = landmarks[idx]
            return int(lm.x * w), int(lm.y * h)

        # Arm points (left and right)
        # left_shoulder = get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)
        left_elbow = get_coords(mp_pose.PoseLandmark.LEFT_ELBOW)
        left_wrist = get_coords(mp_pose.PoseLandmark.LEFT_WRIST)

        # right_shoulder = get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        right_elbow = get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW)
        right_wrist = get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)

        # Draw arm lines
        # cv2.line(frame, left_shoulder, left_elbow, (0, 255, 0), 3)
        cv2.line(frame, left_elbow, left_wrist, (0, 255, 0), 3)

        # cv2.line(frame, right_shoulder, right_elbow, (255, 0, 0), 3)
        cv2.line(frame, right_elbow, right_wrist, (255, 0, 0), 3)

    # Show frame
    cv2.imshow('MediaPipe Arm Tracker', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
