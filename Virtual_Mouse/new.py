import cv2 as cv
import numpy as np
import pyautogui
import autopy
import Hand_Track as htm
import time



# Initialize camera and variables
cap = cv.VideoCapture(0)
camW, camH = 640, 480  # Set camera resolution
cap.set(3, camW)
cap.set(4, camH)
ptime = 0

# Create Hand Detection object
Handtrack = htm.HandsDetection()

# Screen dimensions for cursor movement within the frame
screenW, screenH = camW, camH
while True:
    success, frame = cap.read()
    if not success:
        break
    frame_h,frame_w = frame.shape[:2]

    # Detect hands and landmarks
    frame, results = Handtrack.findHands(frame)
    finalM = Handtrack.findPosition(frame)

    # Calculate and display FPS
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv.putText(frame, f'FPS: {fps:.2f}', (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if len(finalM) > 8:  # Ensure there are enough landmarks
        # Get y-coordinates for index finger tip and joint
        y_tip_index = finalM[8][2]
        y_joint_index = finalM[6][2]

        # Check if the index finger is the only one up
        if (
                y_tip_index < y_joint_index  # Index finger is up
                and finalM[12][2] > finalM[10][2]  # Middle finger is down
                and finalM[16][2] > finalM[14][2]  # Ring finger is down
                and finalM[20][2] > finalM[18][2]  # Pinky finger is down
        ):
            # Draw circle at index fingertip if only index finger is raised

            x_tip_index, y_tip_index = finalM[8][1], finalM[8][2]
            screen_x = int(x_tip_index * screenW / camW)
            screen_y = int(y_tip_index * screenH / camH)
            # Draw the circle on the specified (x, y) location
            cv.circle(frame, (x_tip_index,y_tip_index), 5, (0, 233, 255), 3)  # Red circle on fingertip
            #For Moving the mouse
            autopy.mouse.move(screen_x,screen_y)

    if len(finalM) > 12:  # Ensure there are enough landmarks
        # Get y-coordinates for index finger tip and joint
        y_tip_index = finalM[8][2]
        y_joint_index = finalM[6][2]

        # Get y-coordinates for middle finger tip and joint
        y_tip_middle = finalM[12][2]
        y_joint_middle = finalM[10][2]

        # Check if both index and middle fingers are up
        if (
                y_tip_index < y_joint_index  # Index finger is up
                and y_tip_middle < y_joint_middle  # Middle finger is up
                and finalM[16][2] > finalM[14][2]  # Ring finger is down
                and finalM[20][2] > finalM[18][2]  # Pinky finger is down
        ):
            # Draw circle at index fingertip if only index and middle fingers are raised
            x_tip_index, y_tip_index = finalM[8][1], finalM[8][2]
            screen_x = int(x_tip_index * screenW / camW)
            screen_y = int(y_tip_index * screenH / camH)

            # Draw the circle on the specified (x, y) location
            cv.circle(frame, (x_tip_index, y_tip_index), 5, (255, 0, 255), 3)  # Red circle on fingertip

            # Move the cursor based on calculated screen coordinates
            pyautogui.click(screen_x, screen_y,clicks=1)
            pyautogui.sleep(1)






    # Display the video feed with cursor marker
    cv.imshow('Virtual Mouse', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv.destroyAllWindows()
