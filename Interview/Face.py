from datetime import time
import time
import cv2 as cv
import cvzone
import numpy as np
from ultralytics import YOLO
import Face_detection as fd

######################################
ClassID = 0# 0 is Fake and 1 is Real
outputFolderpath = "/home/divyansh/Desktop/Computer_VIsion/OpenCV-Projects/Interview/DataSet/DataCollect"
offsetpercentageW = 10
save = False
blurThreshold = 35  # larger will be more focused
offsetpercentageH = 20
debug = True
######################################

cap = cv.VideoCapture(0)
ptime = 0
wcam, hcam = 640, 480
cap.set(3, wcam)
cap.set(4, hcam)

while True:
    success, img = cap.read()
    img_copy = img.copy()
    imgout = img.copy()
    if not success:
        break

    ctime = time.time()
    face_detector = fd.FaceDetector(minDetectionCon=0.8)
    face_detection, bboxs = face_detector.findFaces(img, draw=False)
    listblur = []  # True/False values indicating if the faces are blurred or not
    listInfo = []  # The Normalized value and the class name for the label txt file

    for bbox in bboxs:
        x, y, w, h = bbox[1]

        # Adding an offset to the face detected
        offsetW = (offsetpercentageW / 100) * w
        x = int(x - offsetW)
        w = int(w + offsetW * 2)
        offsetH = (offsetpercentageH / 100) * h
        y = int(y - offsetH * 2)
        h = int(h + offsetH * 3)

        # Avoid values less than 0
        x = max(0, x)
        y = max(0, y)
        w = max(0, w)
        h = max(0, h)

        # Normalize values
        ih, iw = img.shape[:2]
        xc, yc = x + w / 2, y + h / 2
        xcn, ycn = round(xc / iw, 6), round(yc / ih, 6)
        wn, hn = round(w / iw, 6), round(h / ih, 6)

        # Avoid values above 1
        xcn = min(1, xcn)
        ycn = min(1, ycn)
        wn = min(1, wn)
        hn = min(1, hn)

        listInfo.append([ClassID, xcn, ycn, wn, hn])

        # Find blurriness
        imgFace = img[y:y + h, x:x + w]
        # cv.imshow("Face", imgFace)
        blurValue = int(cv.Laplacian(imgFace, cv.CV_64F).var())
        listblur.append(blurValue > blurThreshold)
        # print(listblur)
        # Debug information
        if debug:
            ######################################### Do check this line #########################
            detection_score = round((bbox[2][0]) * 100, 2)
            cv.rectangle(img, (x, y, w, h), (255, 0, 0), 2) #Make Sure while saving in the whole image
            cvzone.putTextRect(img, f"Score:{int(detection_score)}% Blur:{blurValue}", (x, y - 20), scale=1, thickness=2)

    # Save data when conditions are met
    if save:
        if all(listblur) and listblur:
            current_time = time.time()
            current_time = str(int(current_time)).split('.')[0]  # Simplified time format

            # Save image
            cv.imwrite(f"{outputFolderpath}/{current_time}.jpg", img_copy) ##be careful while selecting this imgFace will only
            #and img will save the whole photo it will depend on you what you will do

            # Save normalized values (ClassID, xcn, ycn, wn, hn) to a text file
            with open(f"{outputFolderpath}/{current_time}.txt", "w") as f:
                for info in listInfo:
                    formatted_info = " ".join(map(str, info))  # Join values with a space
                    f.write(formatted_info + "\n")  # Add each detection to a new line

    # FPS calculation
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv.putText(img, f'FPS: {fps:.2f}', (10, 30), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv.imshow('Video', img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

