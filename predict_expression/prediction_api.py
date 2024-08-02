import cv2
import numpy as np
import uvicorn
from .prediction_on import PredictionOn
from fastapi import FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

class PredictExpressionAPI(PredictionOn):
    def __init__(self):
        super(PredictionOn, self).__init__()
        self.version = '0.1.2'
        self.get_model_details()
        self.capture_size = 64
        self.trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = self.get_model()

    def get_face_co_ordinates(self, grayScaledImg):
        prediction_data = []
        face_co_ordinates = self.trained_face_data.detectMultiScale(grayScaledImg, 1.1)

        for coordinates in face_co_ordinates:
            prediction_data.append({
                'face_coordinates': coordinates.tolist(),
                'predicted_emotion': self.get_prediction_on_frame(grayScaledImg / 255., coordinates)
            })

        return prediction_data

    def predict_on_img(self, image_data: bytes, image_dim: dict):
        np_array = np.fromstring(image_data, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        resized_img = cv2.resize(image, (int(image_dim['img_width']), int(image_dim['img_height'])))
        grayScaledImg = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        return self.get_face_co_ordinates(grayScaledImg)
    
app = FastAPI()
api = PredictExpressionAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

@app.get('/')
def info():
    return 'Welcome to Emotion Prediction API'

@app.post('/api/predict-emotion/')
async def predict_emotion(img_file: UploadFile, img_height: str = Form(...), img_width: str = Form(...)):
    image_data = await img_file.read()
    image_dim = {
        'img_height': img_height,
        'img_width': img_width
    }
    return api.predict_on_img(image_data, image_dim)

if __name__ == '__main__':
    uvicorn.run(app, port=8080)

# uvicorn predict_expression.prediction_api:app --port 8080 --reload
