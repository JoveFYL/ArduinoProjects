import numpy as np, cv2, mediapipe as mp, time, requests

# cap = cv2.VideoCapture("./BoardAssembly.MOV")
# height, width = 515, 320 # WorkingGuy
# pixels = {
#   "workingGuy": [515, 320],
#   "BoardAssembly": [515, 320],
#   "KnobAssembly": [515, 320]
# }

# desiredFrames = {
#   "workingGuy": [515, 320],
#   "BoardAssembly": [515, 320],
#   "KnobAssembly": [515, 320] 
# }
# # height, width = requests.get('http://192.168.1.4:804/api/workcenterstatus/1WCINJ-008').json()
# startLine, endLine = 235, 20
# lines = {
#   "workingGuy": [235, 20],
#   "BoardAssembly": [235, 20],
#   "KnobAssembly": [235, 20]
# }
# state = "start"
# tolerance = 5
# count = 0
# timeStart, timeEnd = 0, 0

# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands
# hand = mp_hands.Hands()

# # frame_count = 0
# # PROCESS_EVERY_N_FRAMES = 2

# if not cap.isOpened():
#   print("Error opening video")
#   exit(0)

# while True:
#   ret, frame = cap.read()
#   if not ret:
#     print("cannot read frame")
#     break

#   # frame_count += 1
#   # if frame_count % PROCESS_EVERY_N_FRAMES != 0:
#   #     continue

#   # desiredFrame = frame[400:, 300:815] # given in BGR
#   frame = cv2.GaussianBlur(frame, (5, 5), 0) # blur frame to remove noise
#   result = hand.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # convert to RGB to process

#   # Checking for hands
#   if result.multi_hand_landmarks:
#     for hand_landmarks in result.multi_hand_landmarks:
#       wrist = hand_landmarks.landmark[8]
#       # h, w, _ = desiredFrame.shape
#       h, w, _ = frame.shape
#       # landmark_x_pixel = int(wrist.x * h)
#       landmark_y_pixel = int(wrist.y * h)
#       # print(landmark_y_pixel)

#       if state == "start":
#         if abs(landmark_y_pixel - startLine) <= tolerance:
#           start = time.time()
#           state = "waiting to end"
#           print("Passed start")

#       elif state == "waiting to end":
#         if abs(landmark_y_pixel - endLine) <= tolerance:
#             end = time.time()
#             # if (end - start) >= 3:
#             count += 1
#             state = "counted"
#             print(f"Passed endLine → Counted! Total: {count}")
      
#       elif state == "counted":
#         if landmark_y_pixel > startLine + tolerance:
#           state = "start"
#           print("restart")

#       orb = cv2.ORB_create(nfeatures=1500)
#       keypoints, descriptors = orb.detectAndCompute(frame, None)
#       frame = cv2.drawKeypoints(frame, keypoints, None)
#       mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
  
#   # gray = cv2.cvtColor(desiredFrame, cv2.COLOR_BGR2GRAY)
#   # cv2.line(frame, (0, 250), (815, 250), (0, 255, 0), 2) # green line as start point
#   # cv2.line(frame, (0, 20), (815, 20), (0, 0, 255), 2) # red line as end point

#   cv2.line(frame, (800, 100), (800, 1000), (0, 255, 0), 2) # green line as start point
#   cv2.line(frame, (1400, 100), (1400, 1000), (0, 0, 255), 2) # red line as end point

#   cv2.imshow('frame', frame) 
#   if cv2.waitKey(10) == ord('q'):
#     break

# cap.release()
# cv2.destroyAllWindows()

cap = cv2.VideoCapture("./KnobAssembly.MOV")
height, width = 515, 320

# Initialize ORB detector outside the loop for better performance
orb = cv2.ORB_create(nfeatures=1000, patchSize=50, fastThreshold=30)  # Increase features for better detection

# Load reference image once
template = cv2.imread("./Doorknob.jpg", 0)
template = cv2.GaussianBlur(template, (5, 5), 10)
if template is None:
    print("Error: Could not load reference image 'doorknob.heic'")
    exit(0)

# Detect keypoints and descriptors for template once
kp1, des1 = orb.detectAndCompute(template, None)
print(f"Template keypoints detected: {len(kp1)}")

# Initialize matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Homography parameters
MIN_MATCH_COUNT = 10
GOOD_MATCH_RATIO = 0.7

# State tracking variables
state = "start"
tolerance = 5
count = 0
startLine, endLine = 235, 20

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

if not cap.isOpened():
    print("Error opening video")
    exit(0)

def draw_enhanced_keypoints_and_matches(frame, template, kp1, kp2, matches):
    # Create a copy for visualization
    vis_frame = frame.copy()
    
    # 1. Draw all keypoints in the frame
    frame_with_kp = cv2.drawKeypoints(vis_frame, kp2, None, 
                                     color=(0, 255, 255),  # Yellow keypoints
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # 2. Filter good matches based on distance
    good_matches = []
    if len(matches) > 0:
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Take only the best matches (you can adjust this threshold)
        distance_threshold = matches[0].distance * 2.5  # Adjust this multiplier
        good_matches = [m for m in matches if m.distance < distance_threshold]
    
    print(f"Total matches: {len(matches)}, Good matches: {len(good_matches)}")
    
    # 3. Draw matches side by side
    if len(good_matches) > 0:
        match_img = cv2.drawMatches(template, kp1, frame, kp2, 
                                   good_matches[:20],  # Show top 20 matches
                                   None,
                                   matchColor=(0, 255, 0),      # Green for good matches
                                   singlePointColor=(255, 0, 0), # Blue for keypoints
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Resize match image if it's too large
        h, w = match_img.shape[:2]
        if w > 1200:
            scale = 1200 / w
            new_w, new_h = int(w * scale), int(h * scale)
            match_img = cv2.resize(match_img, (new_w, new_h))
        
        cv2.imshow('Matches', match_img)
    
    # 4. Try to find object using homography if enough good matches
    if len(good_matches) >= MIN_MATCH_COUNT:
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        try:
            M, _ = cv2.findHomography(src_pts, dst_pts, 
                                       cv2.RANSAC, 5.0)
            
            if M is not None:
                # Get template dimensions
                h_temp, w_temp = template.shape
                
                # Define corners of template
                pts = np.float32([[0, 0], [w_temp, 0], [w_temp, h_temp], [0, h_temp]]).reshape(-1, 1, 2)
                
                # Transform corners to frame coordinates
                dst = cv2.perspectiveTransform(pts, M)
                
                # Draw bounding box around detected object
                frame_with_kp = cv2.polylines(frame_with_kp, [np.int32(dst)], 
                                            True, (0, 0, 255), 3, cv2.LINE_AA)  # Red box
                
                print(f"Object detected with {len(good_matches)} good matches!")
                
        except cv2.error as e:
            print(f"Homography calculation failed: {e}")
    
    return frame_with_kp

def create_match_visualization_window(frame, template, kp1, kp2, matches):
    """Create a detailed match visualization in a separate window"""
    
    if len(matches) == 0:
        return
        
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Create visualization with only the best matches
    best_matches = matches[:15]  # Top 15 matches
    
    # Draw detailed matches
    detailed_match_img = cv2.drawMatches(
        template, kp1, frame, kp2, best_matches, None,
        matchColor=(0, 255, 0),           # Green lines for matches
        singlePointColor=(255, 0, 0),     # Blue dots for keypoints
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # Add match quality information
    for i, match in enumerate(best_matches[:5]):  # Show info for top 5
        cv2.putText(detailed_match_img, f'Match {i+1}: dist={match.distance:.1f}',
                   (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Resize if too large
    h, w = detailed_match_img.shape[:2]
    if w > 1400:
        scale = 1400 / w
        new_w, new_h = int(w * scale), int(h * scale)
        detailed_match_img = cv2.resize(detailed_match_img, (new_w, new_h))
    
    cv2.imshow('Detailed Matches', detailed_match_img)

# Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        break
    
    # Convert frame to grayscale for ORB
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints and descriptors in current frame
    kp2, des2 = orb.detectAndCompute(frame_gray, None)
    
    matches = []
    if des1 is not None and des2 is not None:
        # Match descriptors
        matches = bf.match(des1, des2)
    
    # Enhanced visualization
    enhanced_frame = draw_enhanced_keypoints_and_matches(frame, template, kp1, kp2, matches)
    
    # Create detailed match window
    create_match_visualization_window(frame, template, kp1, kp2, matches)
    
    # Apply Gaussian blur for hand detection
    blurred_frame = cv2.GaussianBlur(enhanced_frame, (5, 5), 0)
    
    # Hand detection (your existing code)
    result = hand.process(cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB))
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[8]
            h, w, _ = blurred_frame.shape
            landmark_y_pixel = int(wrist.y * h)
            
            # Your existing state machine logic
            if state == "start":
                if abs(landmark_y_pixel - startLine) <= tolerance:
                    start = time.time()
                    state = "waiting to end"
                    print("Passed start")
            elif state == "waiting to end":
                if abs(landmark_y_pixel - endLine) <= tolerance:
                    end = time.time()
                    count += 1
                    state = "counted"
                    print(f"Passed endLine → Counted! Total: {count}")
            elif state == "counted":
                if landmark_y_pixel > startLine + tolerance:
                    state = "start"
                    print("restart")
            
            mp_drawing.draw_landmarks(blurred_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Draw reference lines
    # cv2.line(blurred_frame, (800, 100), (800, 1000), (0, 255, 0), 2)   # Green start line
    # cv2.line(blurred_frame, (1400, 100), (1400, 1000), (0, 0, 255), 2) # Red end line
    
    # Add information overlay
    cv2.putText(blurred_frame, f'Keypoints in frame: {len(kp2)}', 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(blurred_frame, f'Total matches: {len(matches)}', 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(blurred_frame, f'Count: {count}', 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Enhanced Frame', blurred_frame)
    
    # Press 'q' to quit, 's' to save current frame with matches
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f'saved_frame_{int(time.time())}.jpg', blurred_frame)
        print("Frame saved!")

cap.release()
cv2.destroyAllWindows()