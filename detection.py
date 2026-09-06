from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("videos/test.mp4", show=True)