import cv2 as cv
import numpy as np
import Hand_Track as htm
import time
import autopy
import pyautogui

# Initialize camera and variables
cap = cv.VideoCapture(0)
camW, camH = 480, 720
cap.set(3, camW)
cap.set(4, camH)
ptime = 0

# Create Hand Detection object
Handtrack = htm.HandsDetection()

# Screen dimensions
screenW, screenH = autopy.screen.size()
# Variables for mouse clicking and drag
prev_x, prev_y = 0, 0
dragging = False
# print(screenW, screenH)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Detect hands and landmarks
    frame, results = Handtrack.findHands(frame)
    finalM = Handtrack.findPosition(frame)
    # Calculate and display FPS
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv.putText(frame, f'FPS: {fps:.2f}', (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    if len(finalM) > 8:  # Ensure there are enough landmarks
    # Get y-coordinates of each finger's tip and corresponding lower joint
        y_tip_index = finalM[8][2]
        y_joint_index = finalM[6][2]
        y_tip_middle = finalM[12][2]
        y_joint_middle = finalM[10][2]
        y_tip_ring = finalM[16][2]
        y_joint_ring = finalM[14][2]
        y_tip_pinky = finalM[20][2]
        y_joint_pinky = finalM[18][2]

        # Check if only the index finger is up
        if (
                y_tip_index < y_joint_index and  # Index finger is up
                y_tip_middle > y_joint_middle and  # Middle finger is down
                y_tip_ring > y_joint_ring and      # Ring finger is down
                y_tip_pinky > y_joint_pinky        # Pinky finger is down
):
            # Draw a circle only if the index finger is the only one up
            index_finger_tip = (finalM[8][1], finalM[8][2])  # (x, y) coordinates of the index finger tip
            cv.circle(frame, index_finger_tip, 5, (0, 255, 0), -1)


        # Display the video feed with cursor marker
    cv.imshow('Virtual Mouse', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv.destroyAllWindows()
