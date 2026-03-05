from ultralytics import YOLO

model = YOLO(model="yolo11n-pose.pt")

results = model.track("D:\\yolo11_test\\115.mp4", show=True, save=True)
