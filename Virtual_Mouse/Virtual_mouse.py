# import cv2 as cv
# import numpy as np
# import Hand_Track as htm
# import time
# import autopy
# cap = cv.VideoCapture(0)
# camW  = 480
# camH = 720
# cap.set(3, camW)
# cap.set(4, camH)
# ptime =0
# while True:
#     success, frame = cap.read()
#     if not success:
#         break
#     Handtrack = htm.HandsDetection()
#     frame, results = Handtrack.findHands(frame)
#     finalM = Handtrack.findPosition(frame)
#
#     # Check if finalM contains at least two points with (x, y) coordinates
#     if len(finalM) > 0 and len(finalM[0]) > 1:
#         x1, y1 = finalM[0][:2] # Extract x, y from the second point
#
#
#     # We Will BE getting the tip of index,middle finger
#     ctime = time.time()
#     fps = 1 / (ctime - ptime)
#     ptime = ctime
#
#     cv.putText(frame,f'FPS: {fps:.2f}',(30,50),1,1,(0,255,255),2)
#     cv.imshow('frame', frame)
#
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break
# cv.destroyAllWindows()
# cap.release()
#
#
import cv2 as cv
import numpy as np
import Hand_Track as htm
import time
import autopy

# Initialize camera and variables
cap = cv.VideoCapture(0)
camW = 480
camH = 720
cap.set(3, camW)
cap.set(4, camH)
ptime = 0

# Create Hand Detection object
Handtrack = htm.HandsDetection()

# Screen dimensions
screenW, screenH = autopy.screen.size()

while True:
    success, frame = cap.read()
    if not success:
        break

    frame, results = Handtrack.findHands(frame)
    finalM = Handtrack.findPosition(frame)

    # Check if any hand landmarks are detected
    if len(finalM) > 0:
        # Check if index (8) and middle (12) fingers are up
        index_finger_up = finalM[8][2] < finalM[7][2]  # Tip of index finger is above its base
        middle_finger_up = finalM[12][2] < finalM[11][2]  # Tip of middle finger is above its base

        # Debugging output
        # if index_finger_up:
        #     cv.putText(frame, "Index Finger Up", (30, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # if middle_finger_up:
        #     cv.putText(frame, "Middle Finger Up", (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if index_finger_up and middle_finger_up:
            x1, y1 = finalM[8][:2]  # Extract x, y from the index finger tip

            # Map hand coordinates to screen coordinates
            x3 = np.interp(x1, (0, camW), (0, screenW))
            y3 = np.interp(y1, (0, camH), (0, screenH))

            # Move the mouse cursor
            autopy.mouse.move(x3, y3)  # Invert x-coordinate if necessary
            cv.putText(frame, "Mouse Control Active", (20, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv.putText(frame, "Mouse Control Inactive", (20, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Calculate and display FPS
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv.putText(frame, f'FPS: {fps:.2f}', (30, 50), 1, 1, (0, 255, 255), 2)

    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv.destroyAllWindows()
cap.release()

