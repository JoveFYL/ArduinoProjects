import numpy as np, cv2, mediapipe as mp, time, requests
cap = cv2.VideoCapture("./WorkingGuy.mp4")
height, width = 515, 320
# height, width = requests.get('http://192.168.1.4:804/api/workcenterstatus/1WCINJ-008').json()
startLine, endLine = 235, 20
state = "start"
tolerance = 5
count = 0
timeStart, timeEnd = 0, 0

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

if not cap.isOpened():
  print("Error opening video")
  exit(0)

while True:
  ret, frame = cap.read()
  if not ret:
    print("cannot read frame")
    break

  desiredFrame = frame[400:, 300:815] # given in BGR
  result = hand.process(cv2.cvtColor(desiredFrame, cv2.COLOR_BGR2RGB)) # convert to RGB to process

  # Checking for hands
  if result.multi_hand_landmarks:
    for hand_landmarks in result.multi_hand_landmarks:
      wrist = hand_landmarks.landmark[0]
      h, w, _ = desiredFrame.shape
      # landmark_x_pixel = int(wrist.x * h)
      landmark_y_pixel = int(wrist.y * h)
      # print(h)
      # print(landmark_y_pixel)

      if state == "start":
        if abs(landmark_y_pixel - startLine) <= tolerance:
          start = time.time()
          state = "waiting to end"
          print("Passed start")

      elif state == "waiting to end":
        if abs(landmark_y_pixel - endLine) <= tolerance:
            end = time.time()
            if (end - start) >= 3:
              count += 1
              state = "counted"
            print(f"Passed endLine → Counted! Total: {count}")
      
      elif state == "counted":
        if landmark_y_pixel > startLine + tolerance:
          state = "start"
          print("restart")

      mp_drawing.draw_landmarks(desiredFrame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
  
  # gray = cv2.cvtColor(desiredFrame, cv2.COLOR_BGR2GRAY)
  cv2.line(desiredFrame, (0, 250), (815, 250), (0, 255, 0), 2) # green line as start point
  cv2.line(desiredFrame, (0, 20), (815, 20), (0, 0, 255), 2) # red line as end point
  cv2.imshow('frame', desiredFrame) 
  if cv2.waitKey(10) == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()