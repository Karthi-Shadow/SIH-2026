from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Your uploaded video
video = cv2.VideoCapture("videos/test1.mp4")

if not video.isOpened():
    print("❌ Cannot open video")
    exit()

fps = video.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

delay = int(1000 / fps)

while True:

    ret, frame = video.read()

    if not ret:
        break

    # YOLO detection + tracking
    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    # Process detected objects
    if results[0].boxes is not None:

        for box in results[0].boxes:

            # Class ID
            class_id = int(box.cls[0])

            # We only want people
            if class_id != 0:
                continue

            # Bounding box coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Tracking ID
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label
            label = f"Person ID: {track_id}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow(
        "AI Border Surveillance - Person Tracking",
        frame
    )

    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()