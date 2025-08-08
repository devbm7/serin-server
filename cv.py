import cv2
import supervision as sv
from rfdetr import RFDETRBase, RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES
import json
import datetime

model = RFDETRNano()

cap = cv2.VideoCapture(0)

capture_data = {}

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = frame[:, :, ::-1].copy()
    detections = model.predict(rgb_frame, threshold=0.5)

    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]
    # print(len(labels))
    count_of_person = 0
    count_of_devices = 0
    for label in labels:
        if 'person' in label:
            count_of_person += 1
        if 'laptop' in label or 'phone' in label or 'remote' in label or 'tv' in label:
            count_of_devices += 1
    print(count_of_person)
    print(count_of_devices)
    if count_of_person > 1 or count_of_devices > 0:
        current_time = datetime.datetime.now()
        capture_data.update({str(current_time): labels})
    annotated_frame = frame.copy()
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

    cv2.imshow("Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

data = {"session_id":1, "capture_data":capture_data}

with open("sessions/capture_data1.json", "w") as f:
    json.dump(data, f)
cap.release()
cv2.destroyAllWindows()