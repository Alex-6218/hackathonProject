import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import time
import cv2


#define camera object and set window size for display
cam = cv2.VideoCapture(2)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))	

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

#create gesture recognizer object
options = vision.GestureRecognizerOptions(
    base_options=python.BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
recognizer = vision.GestureRecognizer.create_from_options(options)

start = time.monotonic()
last_ts = -1
last_name = None


running = True
while running:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        running = False
        break

    out.write(frame)
    cv2.imshow('Camera', frame)

    ts = int((time.monotonic() - start) * 1000)
    if ts <= last_ts:  # timestamps must strictly increase
        ts = last_ts + 1
        last_ts = ts

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # recognize gestures
    result = recognizer.recognize_for_video(mp_image, ts)
    if result.gestures:
        top = result.gestures[0][0]
        hand = result.handedness[0][0].category_name
        label = f"{top.category_name} ({hand} hand, {top.score:.2f})"
        name = top.category_name
    else:
        label, name = "No hand", "No hand"

    spell = "No spell"
    match name:
        case "Open_Palm":
            spell = "FIREBALL"
        case "Closed_Fist":
            spell = "RAISE EARTH"
        case "Pointing_Up":
            spell = "TELEPORT"
        case "ILoveYou":
            spell = "SUMMONNING SKELLIES"
    


    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
            print(spell)
            
    if key == ord('q'):
        running = False

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows() 