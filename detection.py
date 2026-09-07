from ultralytics import YOLO
import cv2
import os
import csv
from datetime import datetime

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open CCTV video
video = cv2.VideoCapture("videos/test1.mp4")

if not video.isOpened():
    print("Cannot open video")
    exit()

# Restricted zone
x1, y1 = 400, 480
x2, y2 = 950, 680

# ---------- Evidence capture setup ----------
EVIDENCE_DIR = "evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

LOG_FILE = os.path.join(EVIDENCE_DIR, "intrusion_log.csv")
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "track_id", "image_path"])

CAPTURE_COOLDOWN = 5.0  # seconds between captures per track_id
last_capture_time = {}  # track_id -> last capture timestamp

def save_evidence(frame, track_id):
    now = datetime.now()
    ts_str = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"intrusion_{track_id}_{ts_str}.jpg"
    filepath = os.path.join(EVIDENCE_DIR, filename)

    cv2.imwrite(filepath, frame)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([now.isoformat(), track_id, filepath])

    print(f"[EVIDENCE] Saved {filepath}")
# ---------------------------------------------

while True:

    ret, frame = video.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        frame, "RESTRICTED ZONE", (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    )

    intrusion = False
    intruding_ids = []

    if results[0].boxes is not None:

        for box in results[0].boxes:

            class_id = int(box.cls[0])
            if class_id != 0:
                continue

            bx1, by1, bx2, by2 = map(int, box.xyxy[0])

            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1

            center_x = (bx1 + bx2) // 2
            center_y = (by1 + by2) // 2

            inside_zone = (
                x1 < center_x < x2
                and
                y1 < center_y < y2
            )

            box_color = (0, 0, 255) if inside_zone else (0, 255, 0)

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2)
            cv2.putText(
                frame, f"Person ID: {track_id}", (bx1, by1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2
            )
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

            if inside_zone:
                intrusion = True
                intruding_ids.append(track_id)

    # Timestamp on frame (useful for evidence review)
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame, timestamp_str, (30, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )

    if intrusion:
        cv2.putText(
            frame, "!!! INTRUSION DETECTED !!!", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3
        )

        # ---- Automatic evidence capture (with cooldown per track) ----
        now_ts = datetime.now().timestamp()
        for track_id in intruding_ids:
            last_time = last_capture_time.get(track_id, 0)
            if now_ts - last_time >= CAPTURE_COOLDOWN:
                save_evidence(frame, track_id)
                last_capture_time[track_id] = now_ts
        # ----------------------------------------------------------------

    else:
        cv2.putText(
            frame, "STATUS: SAFE", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

    cv2.imshow("AI Border Surveillance", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()