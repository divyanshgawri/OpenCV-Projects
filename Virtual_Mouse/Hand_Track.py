import cv2 as cv
import mediapipe as mp
import time

class HandsDetection():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackingCon,
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  # Store results as a class attribute

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                # Get the coordinates of the landmarks
                h, w, _ = frame.shape
                x_min = int(w)  # Initialize to max width
                y_min = int(h)  # Initialize to max height
                x_max = 0       # Initialize to min width
                y_max = 0       # Initialize to min height

                for lm in handLms.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)  # Get landmark coordinates
                    # Update the bounding box coordinates
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                if draw:
                    # Draw landmarks on the frame
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
                    # Draw a rectangle around the detected hand
                    cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green rectangle

        return frame, self.results


    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 1, (255, 0, 255))
        return lmList

def main():
    cap = cv.VideoCapture(0)
    ptime = 0
    detector = HandsDetection()

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = detector.findHands(frame)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        # Displaying FPS
        cv.putText(frame, 'FPS: {:.2f}'.format(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow("Image", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
