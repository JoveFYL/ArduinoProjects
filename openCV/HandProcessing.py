# Google's implementation 
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np 
import cv2
import time
from dataclasses import dataclass

@dataclass
class Config:
    videoPath: str 
    startLineX: int = 600
    startLineY: int = 100
    endLineX: int = 1400
    endLineY: int = 1000
    cooldownTime: float = 5.0  # seconds
    direction: str = "vertical"  # "horizontal" or "vertical"
    margin: int = 9 # pixels
    fontSize: int = 1
    fontThickness: int = 1

class Counter:
    def __init__(self, config: Config):
        self.state = "start"
        self.count = 0
        self.lastCountTime = 0
        self.config = config
        
    def checkTime(self):
        return time.time() - self.lastCountTime >= self.config.cooldownTime 

    def updateState(self, wrist_landmark, image):
        height, width = image.shape[:2]
        
        # Get the correct coordinate based on direction
        if self.config.direction == "horizontal":
            # For horizontal movement, track X coordinate (left to right)
            pixel_position = wrist_landmark.x * width
            start_line = self.config.startLineX
            end_line = self.config.endLineX
            
            # State machine logic for horizontal (left to right)
            if self.state == "start" and pixel_position <= start_line:
                self.state = "waiting to end"
            elif self.state == "waiting to end" and pixel_position >= end_line and not self.checkTime(): 
                print("COOLDOWN TIME")
            elif self.state == "waiting to end" and pixel_position >= end_line and self.checkTime():
                self.count += 1
                self.state = "counted"
                self.lastCountTime = time.time()
                print(f"Passed endLine → Counted! Total: {self.count}")
            elif self.state == "counted" and pixel_position <= start_line:
                self.state = "start"
                
        else:  # vertical movement (bottom to top)
            # For vertical movement, track Y coordinate (bottom to top)
            pixel_position = wrist_landmark.y * height
            start_line = self.config.startLineY  # Bottom line (higher Y value)
            end_line = self.config.endLineY      # Top line (lower Y value)
            
            # State machine logic for vertical (bottom to top) - reversed logic
            if self.state == "start" and pixel_position >= start_line:
                self.state = "waiting to end"
            elif self.state == "waiting to end" and pixel_position <= end_line and not self.checkTime(): 
                print("COOLDOWN TIME")
            elif self.state == "waiting to end" and pixel_position <= end_line and self.checkTime():
                self.count += 1
                self.state = "counted"
                self.lastCountTime = time.time()
                print(f"Passed endLine → Counted! Total: {self.count}")
            elif self.state == "counted" and pixel_position >= start_line:
                self.state = "start"
            
        print(f"Direction: {self.config.direction}, Position: {int(pixel_position)}, State: {self.state}")
        
# Drawing function
def draw_landmarks_on_image(rgb_image, detection_result, config: Config):
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
        text_y = int(min(y_coordinates) * height) - config.margin

        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    config.fontSize, (255, 0, 0), config.fontThickness, cv2.LINE_AA)

    return annotated_image

# Improved line drawing functions that adapt to direction
def draw_counting_lines(rgb_image, config: Config):
    height, width = rgb_image.shape[:2]
    
    if config.direction == "horizontal":
        # Draw vertical lines for horizontal movement detection
        # Start line (green)
        cv2.line(rgb_image, (config.startLineX, 0), (config.startLineX, height), (0, 255, 0), 2)
        # End line (red)
        cv2.line(rgb_image, (config.endLineX, 0), (config.endLineX, height), (255, 0, 0), 2)
        
        # Add labels
        cv2.putText(rgb_image, "START", (config.startLineX - 30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(rgb_image, "END", (config.endLineX - 20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    else:  # vertical direction (bottom to top movement)
        cv2.line(rgb_image, (0, config.startLineY), (width, config.startLineY), (0, 255, 0), 2)
        # End line (red) - at top
        cv2.line(rgb_image, (0, config.endLineY), (width, config.endLineY), (255, 0, 0), 2)
        
        # Add labels
        cv2.putText(rgb_image, "START (Bottom)", (10, config.startLineY - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(rgb_image, "END (Top)", (10, config.endLineY - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return rgb_image

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

def main():
    vid = "./WhiteTubeAssembly2.MOV" 
    # For horizontal movement (left to right)
    config_horizontal = Config(
        videoPath=vid, 
        direction="horizontal", 
        startLineX=450,    # Left boundary
        endLineX=1600,      # Right boundary
        startLineY=100,    # Not used for horizontal
        endLineY=600       # Not used for horizontal
    )
    
    # For vertical movement (bottom to top)
    config_vertical = Config(
        videoPath=vid, 
        direction="vertical", 
        startLineX=100,    # Not used for vertical
        endLineX=800,      # Not used for vertical
        startLineY=500,    # Bottom boundary (start line - higher Y value)
        endLineY=200       # Top boundary (end line - lower Y value)
    )
    
    # Choose which config to use
    config = config_horizontal # Change this to config_vertical for vertical tracking
    
    counter = Counter(config)
    detector = init_mediapipe()

    cap = cv2.VideoCapture(config.videoPath)
    if not cap.isOpened():
        print("Error opening video")
        exit(0)

    print(f"Tracking {config.direction} movement")
    if config.direction == "horizontal":
        print(f"Start line at X={config.startLineX}, End line at X={config.endLineX}")
    else:
        print(f"Start line (bottom) at Y={config.startLineY}, End line (top) at Y={config.endLineY}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        # Convert BGR (OpenCV) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect hands
        detection_result = detector.detect(image)

        # Draw landmarks
        annotated_image = draw_landmarks_on_image(frame_rgb, detection_result, config)

        # Draw counting lines based on direction
        annotated_image = draw_counting_lines(annotated_image, config)

        # Track hand movement
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                wrist = hand_landmarks[0]  # Wrist landmark
                if wrist: 
                    counter.updateState(wrist, frame)

        # Display count and direction info
        cv2.putText(annotated_image, f"Count: {counter.count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_image, f"Direction: {config.direction}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_image, f"State: {counter.state}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display
        cv2.imshow('Hand Counter', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()