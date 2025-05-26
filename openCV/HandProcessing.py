# Google's implementation
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np 
import cv2
import time

# Constants
MARGIN = 9  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1

# Drawing function
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # for each hand in detected list (hand_landmarks_list), 0 = first hand
    for idx in range(len(hand_landmarks_list)):
        # get 21 landmarks for hand #idx
        hand_landmarks = hand_landmarks_list[idx]

        # left/right hand?
        handedness = handedness_list[idx]

        # reformatting hand landmark data for mp's drawing utils
        # create buffer to hold normalised hand landmarks
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in hand_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Annotate handedness
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, (255, 0, 0), FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

# draw lines
def draw_start_line(rgb_image, x_start, x_end, y_start, y_end):
    cv2.line(rgb_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2) # green line 
    return rgb_image

def draw_end_line(rgb_image, x_start, x_end, y_start, y_end):
    cv2.line(rgb_image, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2) # red line 
    return rgb_image

def checkTime(minimumCountTime = 5):
    global lastCountTime
    current_time = time.time()
    return current_time - lastCountTime >= minimumCountTime 

def changeState(state, count, normalisedPixel, image, startLine, endLine, direction):
    global lastCountTime
    height, width = image.shape[:2]
    if direction == "vertical":
        pixel = normalisedPixel * height
    else:
        pixel = normalisedPixel * width

    if state == "start" and pixel <= startLine:
        state = "waiting to end"
    elif state == "waiting to end" and pixel >= endLine and not checkTime(): 
        print("COOLDOWN TIME")
    elif state == "waiting to end" and pixel >= endLine and checkTime():
        count += 1
        state = "counted"
        lastCountTime = time.time()
        print(f"Passed endLine → Counted! Total: {count}")
    elif state == "counted" and pixel < startLine:
        state = "start"
    print(f"Position: {int(pixel)}, State: {state}")
    return state, count

# Initialize MediaPipe HandLandmarker
def init_mediapipe():
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options, 
        num_hands=2,
        min_hand_detection_confidence=0.3,  # Lower from default 0.5
        min_hand_presence_confidence=0.3,   # Lower from default 0.5
        min_tracking_confidence=0.3         # Lower from default 0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return detector

# Video capture
cap = cv2.VideoCapture("./WhiteTubeAssembly2.MOV")
if not cap.isOpened():
    print("Error opening video")
    exit(0)

# variables
state = "start"
count = 0
detector = init_mediapipe()
startLine = 600
endLine = 1400
lastCountTime = 0

# Frame loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break

    # desiredFrame = frame[400:, 300:815] # given in BGR
    desiredFrame = frame

    # Convert BGR (OpenCV) to RGB
    frame_rgb = cv2.cvtColor(desiredFrame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detect hands
    detection_result = detector.detect(image)

    # Draw landmarks
    annotated_image = draw_landmarks_on_image(frame_rgb, detection_result)

    # uses RGB values 
    draw_start_line(annotated_image, startLine, startLine, 100, 1000)
    draw_end_line(annotated_image, endLine, endLine, 100, 1000)

    # if the wrist is in the frame
    if detection_result.hand_landmarks:
        for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
            wrist = hand_landmarks[0] # first hand, first landmark -> wrist
            if wrist: 
                state, count = changeState(state, count, wrist.x, desiredFrame, 600, 1400, "horizontal") # direction based on hand movement

    cv2.putText(annotated_image, "Count: " + str(count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display
    cv2.imshow('Hand Landmarks', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
