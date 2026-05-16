import cv2
from modules.face_detector import detect_faces
from modules.gender_predictor import predict_gender
from modules.age_predictor import predict_age
from utils.draw_utils import draw_label

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes = detect_faces(frame)

    for box in boxes:
        x1,y1,x2,y2 = box
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        face = cv2.resize(face, (227,227))

        gender = predict_gender(face)
        age = predict_age(face)

        label = f"{gender}, {age}"
        draw_label(frame, label, x1, y1-10)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow("Age Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()