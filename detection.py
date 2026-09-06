from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.track(
    source="videos/test.mp4",
    show=True,
    tracker="bytetrack.yaml"
)