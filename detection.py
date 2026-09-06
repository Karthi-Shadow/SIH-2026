import cv2

video = cv2.VideoCapture("videos/test.mp4")

fps = video.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

delay = int(1000 / fps)

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Restricted zone - center and upper area
    x1, y1 = 500, 200
    x2, y2 = 800, 350

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

    cv2.imshow("Border Surveillance", frame)

    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()