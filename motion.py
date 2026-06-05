import cv2
import time
import numpy as np
from ultralytics import YOLO

# ---------------- MODEL ----------------
model = YOLO("yolov8s.pt")

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opening")
    exit()

# ---------------- SETTINGS ----------------
INACTIVE_LIMIT = 5   # seconds
LOW_LIMIT = 2
MOTION_THRESHOLD = 4.5

track_data = {}

# ---------------- LOOP ----------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    active = low = inactive = 0

    # ---------------- DETECTION ----------------
    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes is not None:

        for box in results[0].boxes:

            if box.id is None:
                continue

            cls = int(box.cls.item())
            conf = float(box.conf.item())

            if cls != 0 or conf < 0.5:
                continue

            person_id = int(box.id.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            roi = gray[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            roi = cv2.GaussianBlur(roi, (5, 5), 0)

            # ---------------- INIT ----------------
            if person_id not in track_data:
                track_data[person_id] = {
                    "prev_roi": roi,
                    "last_move": time.time(),
                    "history": []
                }

            prev_roi = track_data[person_id]["prev_roi"]

            roi = cv2.resize(roi, (prev_roi.shape[1], prev_roi.shape[0]))

            diff = cv2.absdiff(prev_roi, roi)
            motion_score = np.mean(diff)

            # smooth motion
            track_data[person_id]["history"].append(motion_score)

            if len(track_data[person_id]["history"]) > 10:
                track_data[person_id]["history"].pop(0)

            avg_motion = np.mean(track_data[person_id]["history"])

            # ---------------- MOVEMENT CHECK ----------------
            if avg_motion > MOTION_THRESHOLD:
                track_data[person_id]["last_move"] = time.time()

            inactive_time = time.time() - track_data[person_id]["last_move"]

            # ---------------- STATUS ----------------
            if inactive_time >= INACTIVE_LIMIT:
                status = "INACTIVE"
                color = (0, 0, 255)
                inactive += 1

            elif inactive_time >= LOW_LIMIT:
                status = "LOW MOVEMENT"
                color = (0, 255, 255)
                low += 1

            else:
                status = "ACTIVE"
                color = (0, 255, 0)
                active += 1

            track_data[person_id]["prev_roi"] = roi

            # ---------------- DRAW ----------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame,
                        f"ID {person_id}: {status}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2)

    # ---------------- DASHBOARD ----------------
    cv2.putText(frame, f"ACTIVE: {active}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.putText(frame, f"LOW: {low}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.putText(frame, f"INACTIVE: {inactive}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    if inactive > 0:
        cv2.putText(frame, "ALERT: INACTIVE DETECTED",
                    (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,0,255), 3)

    cv2.imshow("Smart Monitoring System", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()