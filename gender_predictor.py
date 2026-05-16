import cv2
from config import *

genderNet = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

def predict_gender(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227,227),
                                 MODEL_MEAN_VALUES,
                                 swapRB=False)
    genderNet.setInput(blob)
    preds = genderNet.forward()
    return GENDER_LIST[preds[0].argmax()]