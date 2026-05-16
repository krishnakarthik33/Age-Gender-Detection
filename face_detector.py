import cv2
from config import FACE_PROTO, FACE_MODEL

faceNet = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)

def detect_faces(frame, conf_threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300),
                                 [104,117,123], True, False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3] * w)
            y1 = int(detections[0,0,i,4] * h)
            x2 = int(detections[0,0,i,5] * w)
            y2 = int(detections[0,0,i,6] * h)
            boxes.append([x1,y1,x2,y2])

    return boxes