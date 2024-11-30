import mediapipe as mp
import time
import cv2 as cv

class FaceDetector():
    def __init__(self,minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon

        self.mpFacedetection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.facedetction = self.mpFacedetection.FaceDetection()

    def findFaces(self,frame,draw=True):
        imgRgb =cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.facedetction.process(imgRgb)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # print(id,detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                # mp_draw.draw_detection(imgRgb, detection)
                bboxC = detection.location_data.relative_bounding_box
                ih,iw,ic = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id,bbox,detection.score])
                cv.rectangle(frame, bbox, (0, 255, 0), 2)
                cv.putText(frame, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        return frame, bboxs


def main():

    cap = cv.VideoCapture(0)
    ptime = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        ctime = time.time()
        detector = FaceDetector()
        detector.findFaces(frame)

        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv.putText(frame,f"FPS :{int(fps)}",(10,30),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),4)
        cv.imshow('Video',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # print(results)
    cap.release()
    cv.destroyAllWindows()
if __name__ == "__main__" :
    main()