import cv2
import dlib
import numpy as np
import datetime
import json

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
capture_data = {}

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_center = midpoint(landmarks.part(36), landmarks.part(39))
        right_eye_center = midpoint(landmarks.part(42), landmarks.part(45))

        cv2.circle(frame, left_eye_center, 3, (0, 255, 0), -1)
        cv2.circle(frame, right_eye_center, 3, (0, 255, 0), -1)

        current_time = datetime.datetime.now().isoformat()
        capture_data[current_time] = {
            "left_eye": left_eye_center,
            "right_eye": right_eye_center
        }

    cv2.imshow("Eye Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

data = {"session_id": 1, "capture_data": capture_data}
with open("sessions/capture_data.json", "w") as f:
    json.dump(data, f, indent=2)

cap.release()
cv2.destroyAllWindows()