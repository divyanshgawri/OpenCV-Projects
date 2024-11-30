#Loading the libraries
import cv2 as cv
import cvzone
import numpy as np
from ultralytics import YOLO

confidence = 0.6
# Load YOLO model
model = YOLO("/home/divyansh/Desktop/Computer_VIsion/OpenCV-Projects/Interview/runs/detect/train12/weights/best.pt")
cap = cv.VideoCapture(0)
# For Checking  if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

metrics = model.val(data="/home/divyansh/Desktop/Computer_VIsion/OpenCV-Projects/Interview/Dataset/SplitData/data.yaml")

# Metrics output
print(metrics)

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    frame_copy = frame.copy()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Resize frame to match model input size
    img_resized = cv.resize(frame, (640, 480))

    # Run YOLO model on the frame
    results = model(img_resized)

    # Process detection results
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            filtered_boxes = []
            for box in boxes:
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class index
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

                # Only consider boxes above the confidence threshold
                if conf > confidence:
                    filtered_boxes.append((conf, cls, x1, y1, x2, y2))

            # Keep the box with the highest confidence only
            if filtered_boxes:
                filtered_boxes = sorted(filtered_boxes, key=lambda x: x[0], reverse=True)
                conf, cls, x1, y1, x2, y2 = filtered_boxes[0]

                # Label and color based on class
                label = model.names[cls]  # Get the class label
                color = (0, 255, 0) if label == 'Real' else (0, 0, 255)  # Green for 'Real', Red for 'Fake'

                # Draw bounding box and label on the frame
                cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv.putText(frame, f"{label} {conf*100:.2f}%", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



    # Display the frame with detections
    cv.imshow("Webcam Feed", frame)

    # Press 'q' to exit the loop
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
