from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open CCTV video
video = cv2.VideoCapture("videos/test1.mp4")

if not video.isOpened():
    print("Cannot open video")
    exit()

# Restricted zone
# Upper-center area
x1, y1 = 400, 480
x2, y2 = 950, 680

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

    intrusion = False

    # Check detections
    if results[0].boxes is not None:

        for box in results[0].boxes:

            # Person class = 0
            class_id = int(box.cls[0])

            if class_id != 0:
                continue

            # Bounding box
            bx1, by1, bx2, by2 = map(
                int,
                box.xyxy[0]
            )

            # Tracking ID
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1

            # Person center
            center_x = (bx1 + bx2) // 2
            center_y = (by1 + by2) // 2

            # Check whether person is inside zone
            inside_zone = (
                x1 < center_x < x2
                and
                y1 < center_y < y2
            )

            # Draw person
            cv2.rectangle(
                frame,
                (bx1, by1),
                (bx2, by2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Person ID: {track_id}",
                (bx1, by1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Draw center
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (255, 0, 0),
                -1
            )

            # Intrusion
            if inside_zone:
                intrusion = True

    # Alert
    if intrusion:

        cv2.putText(
            frame,
            "!!! INTRUSION DETECTED !!!",
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

    # Press Q to exit
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()