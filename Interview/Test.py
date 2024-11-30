from ultralytics import YOLO
model1 = YOLO('yolov8n.pt')
model1.train(data='/home/divyansh/Desktop/Computer_VIsion/OpenCV-Projects/Interview/Dataset/SplitData/data.yaml',epochs=5)
