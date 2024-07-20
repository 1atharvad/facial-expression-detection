import cv2
import numpy as np
import base64
import uvicorn
from .facial_expression_model import FacialExpressionModel
from .prediction_on import PredictionOn
from fastapi import FastAPI

class PredictExpression(PredictionOn):
    def __init__(self):
        super(PredictionOn, self).__init__()
        self.capture_size = 64
        self.project_file_path = 'projects/facial_expression_detection/predict_expression'
        self.trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = FacialExpressionModel(f'model.h5')

    def get_prediction_on_frame(self, grayImg, co_ordinates):
        x, y, w, h = co_ordinates
        cropped_face = grayImg[y:y+h, x:x+w]
        roi = cv2.resize(cropped_face, (self.capture_size, self.capture_size))
        return self.model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

    def predict_on_img(self, img_base64_string: str):
        image_data = base64.b64decode(img_base64_string.split(',')[1])
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        grayScaledImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_co_ordinates = self.trained_face_data.detectMultiScale(grayScaledImg, 1.2)
        prediction_data = []

        for coordinates in face_co_ordinates:
            prediction_data.append({
                'face_coordinates': coordinates.tolist(),
                'predicted_emotion': self.get_prediction_on_frame(grayScaledImg / 255., coordinates)
            })

        return prediction_data
    
app = FastAPI()

@app.post("/api/test/")
def read_root(img_base64: str):
    return {"Hello": img_base64}

@app.get("/api/test-get/")
def read_root(img_base64: str):
    return {"Hello": img_base64}

# uvicorn predict_expression.prediction_api:app --port 8080 --reload
