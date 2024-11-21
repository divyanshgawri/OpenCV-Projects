import cv2 as cv
import pyautogui
import autopy
import Hand_Track as htm
import time
import numpy as np

# Initialise camera and variables
cap = cv.VideoCapture(0)
camW, camH = 640, 480  # Set camera resolution
cap.set(3, camW)
cap.set(4, camH)
ptime = 0

# Screen dimensions for cursor movement
screenW, screenH = pyautogui.size()

# For scrolling
prev_y_scroll = 0
scroll_sensitivity = 10

# Create Hand Detection object
Handtrack = htm.HandsDetection()

# Smoothing factors for cursor movement
prev_x, prev_y = 0, 0
smooth_factor = 5  # Higher = more smoothing, lower = more responsive

while True:
    success, frame = cap.read()
    if not success:
        break
    frame_h, frame_w = frame.shape[:2]

    # Detect hands and landmarks
    frame, results = Handtrack.findHands(frame)
    finalM = Handtrack.findPosition(frame)

    # Calculate and display FPS
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv.putText(frame, f'FPS: {fps:.2f}', (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if len(finalM) > 20:  # Ensure there are enough landmarks
        # Extract finger landmarks
        x_tip_index, y_tip_index = finalM[8][1], finalM[8][2]
        y_joint_index = finalM[6][2]

        x_tip_middle, y_tip_middle = finalM[12][1], finalM[12][2]
        y_joint_middle = finalM[10][2]

        x_tip_ring, y_tip_ring = finalM[16][1], finalM[16][2]
        y_joint_ring = finalM[14][2]

        # Map webcam coordinates to screen coordinates
        screen_x = int(np.interp(x_tip_index, (0, camW), (0, screenW)))
        screen_y = int(np.interp(y_tip_index, (0, camH), (0, screenH)))

        # Smooth cursor movement
        curr_x = prev_x + (screen_x - prev_x) / smooth_factor
        curr_y = prev_y + (screen_y - prev_y) / smooth_factor
        prev_x, prev_y = curr_x, curr_y

        # Move the cursor based on index finger
        autopy.mouse.move(curr_x, curr_y)

        # Index Finger Gesture: Move the cursor
        if (
                y_tip_index < y_joint_index  # Index finger is up
                and y_tip_middle > y_joint_middle  # Middle finger is down
                and y_tip_ring > y_joint_ring  # Ring finger is down
        ):
            cv.circle(frame, (x_tip_index, y_tip_index), 10, (0, 255, 0), -1)

        # Click Gesture: Index and Middle Fingers Up
        elif (
                y_tip_index < y_joint_index  # Index finger is up
                and y_tip_middle < y_joint_middle  # Middle finger is up
                and y_tip_ring > y_joint_ring  # Ring finger is down
        ):
            cv.circle(frame, (x_tip_middle, y_tip_middle), 10, (255, 0, 0), -1)
            pyautogui.click()

        # Scroll Gesture: Index, Middle, and Ring Fingers Up
        elif (
                y_tip_index < y_joint_index  # Index finger is up
                and y_tip_middle < y_joint_middle  # Middle finger is up
                and y_tip_ring < y_joint_ring  # Ring finger is up
                and finalM[20][2] > finalM[18][2]  # Pinky finger is down
                and finalM[4][2] > finalM[3][2]  # Thumb is down
        ):
            # Visualise scrolling gesture
            cv.circle(frame, (x_tip_ring, y_tip_ring), 10, (255, 0, 255), -1)

            # Enable scrolling when three fingers are up
            current_y_scroll = y_tip_index
            if prev_y_scroll != 0:  # Skip the first frame
                scroll_amount = (prev_y_scroll - current_y_scroll) / scroll_sensitivity
                pyautogui.scroll(int(scroll_amount))

            prev_y_scroll = current_y_scroll
        else:
            prev_y_scroll = 0

    # Display the video feed with cursor marker
    cv.imshow('Virtual Mouse', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv.destroyAllWindows()
