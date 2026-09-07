from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

video = cv2.VideoCapture("videos/test1.mp4")

while True:
    ret, frame = video.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    for box in results[0].boxes:

        class_id = int(box.cls[0])

        # Person = class 0
        if class_id != 0:
            continue

        bx1, by1, bx2, by2 = map(
            int,
            box.xyxy[0]
        )

        # Calculate person's center
        center_x = (bx1 + bx2) // 2
        center_y = (by1 + by2) // 2

        # Draw bounding box
        cv2.rectangle(
            frame,
            (bx1, by1),
            (bx2, by2),
            (0, 255, 0),
            2
        )

        # Draw center point
        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (255, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Person ({center_x}, {center_y})",
            (bx1, by1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Person Detection Test", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()