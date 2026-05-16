import cv2
from config import *

ageNet = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)

def predict_age(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227,227),
                                 MODEL_MEAN_VALUES,
                                 swapRB=False)
    ageNet.setInput(blob)
    preds = ageNet.forward()
    return AGE_LIST[preds[0].argmax()]