from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open video
video = cv2.VideoCapture("videos/test.mp4")

# Restricted zone
x1, y1 = 500, 200
x2, y2 = 800, 350
# YOLO COCO class
# 7 = truck
TRUCK_CLASS = 7

while True:

    ret, frame = video.read()

    if not ret:
        break

    # Detect and track
    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    # Draw restricted zone
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        "RESTRICTED ZONE",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    truck_intrusion = False

    # Check detected objects
    if results[0].boxes is not None:

        for box in results[0].boxes:

            # Get class ID
            class_id = int(box.cls[0])

            # Ignore everything except truck
            if class_id != TRUCK_CLASS:
                continue

            # Bounding box
            bx1, by1, bx2, by2 = map(
                int,
                box.xyxy[0]
            )

            # Truck center
            center_x = (bx1 + bx2) // 2
            center_y = (by1 + by2) // 2

            # Draw truck box
            cv2.rectangle(
                frame,
                (bx1, by1),
                (bx2, by2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "TRUCK",
                (bx1, by1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Check if truck enters restricted zone
            inside_zone = (
                x1 < center_x < x2
                and
                y1 < center_y < y2
            )

            if inside_zone:
                truck_intrusion = True

    # Alert
    if truck_intrusion:

        cv2.putText(
            frame,
            "!!! TRUCK INTRUSION !!!",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    else:

        cv2.putText(
            frame,
            "STATUS: SAFE",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Display
    cv2.imshow(
        "AI Border Surveillance",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()