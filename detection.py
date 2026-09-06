from ultralytics import YOLO
import cv2

# -----------------------------
# 1. Load YOLO model
# -----------------------------
model = YOLO("yolo11n.pt")

# -----------------------------
# 2. Open CCTV/video
# -----------------------------
video = cv2.VideoCapture("videos/test.mp4")

# -----------------------------
# 3. Restricted Zone
# Change these values according
# to your video
# -----------------------------
x1, y1 = 500, 200
x2, y2 = 800, 350

# Vehicle classes in YOLO
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
vehicle_classes = [2, 3, 5, 7]

while True:

    # Read one frame
    ret, frame = video.read()

    if not ret:
        break

    # -----------------------------
    # 4. Detect + Track objects
    # -----------------------------
    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    # -----------------------------
    # 5. Draw Restricted Zone
    # -----------------------------
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

    intrusion_detected = False

    # -----------------------------
    # 6. Check detected objects
    # -----------------------------
    if results[0].boxes is not None:

        for box in results[0].boxes:

            # Get class ID
            class_id = int(box.cls[0])

            # Ignore non-vehicles
            if class_id not in vehicle_classes:
                continue

            # Get bounding box
            bx1, by1, bx2, by2 = map(
                int,
                box.xyxy[0]
            )

            # Find center of vehicle
            center_x = (bx1 + bx2) // 2
            center_y = (by1 + by2) // 2

            # -----------------------------
            # 7. Check restricted zone
            # -----------------------------
            inside_zone = (
                x1 < center_x < x2
                and
                y1 < center_y < y2
            )

            # -----------------------------
            # 8. Get vehicle name
            # -----------------------------
            vehicle_name = model.names[class_id]

            # -----------------------------
            # 9. Draw vehicle box
            # -----------------------------
            cv2.rectangle(
                frame,
                (bx1, by1),
                (bx2, by2),
                (0, 255, 0),
                2
            )

            # Display vehicle name
            cv2.putText(
                frame,
                vehicle_name,
                (bx1, by1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
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

            # -----------------------------
            # 10. Intrusion detected
            # -----------------------------
            if inside_zone:

                intrusion_detected = True

                cv2.putText(
                    frame,
                    "!!! VEHICLE INTRUSION !!!",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    frame,
                    "ALERT: Restricted Area",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

    # -----------------------------
    # 11. Normal status
    # -----------------------------
    if not intrusion_detected:

        cv2.putText(
            frame,
            "STATUS: SAFE",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # -----------------------------
    # 12. Show video
    # -----------------------------
    cv2.imshow(
        "AI Border Surveillance",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# -----------------------------
# 13. Release resources
# -----------------------------
video.release()
cv2.destroyAllWindows()