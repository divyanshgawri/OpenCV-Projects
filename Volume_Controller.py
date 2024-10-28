import numpy as np
# import mediapipe as mp
import time
import cv2 as cv
import alsaaudio
import math
import Hand_tracking as ht
from Volume_hand_control import results

cap=cv.VideoCapture(0)
ptime =0
wcam,hcam =640,480
cap.set(3,wcam)
cap.set(4,hcam)
detector = ht.HandsDetection(detectionCon=0.7)
mixer = alsaaudio.Mixer()
max_volume = mixer.getvolume()[0]

def get_volume_level(distance):
    return int(np.interp(distance, [30, 150], [0, 100]))
while True:
    success, frame = cap.read()
    if not success:
        break
    frame,results  = detector.findHands(frame)
    lmlist = detector.findPosition(frame)
    if lmlist!=0:
        x1,y1 = lmlist[4][1],lmlist[4][2]
        x2,y2 = lmlist[8][1],lmlist[8][2]
        cv.circle(frame,(x1,y1),5,(255,0,0),cv.FILLED)
        cv.circle(frame,(x2,y2),5,(255,0,0),cv.FILLED)
        cv.line(frame,(x1,y1),(x2,y2),(255,0,0),2)
        length=math.hypot(x2-x1,y2-y1)
        volume=get_volume_level(length)
        mixer.setvolume(volume)
        print(f"Distance: {length}, Volume Level: {volume}%")  # Debug print

        # Display volume level on the frame
        cv.putText(frame, f'Volume: {volume}%', (50, 450), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)



    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
