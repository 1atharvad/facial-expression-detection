import cv2
import numpy as np
from .facial_expression_model import FacialExpressionModel

class PredictionOn:
    def __init__(self):
        self.capture_size = 64
        self.trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = FacialExpressionModel('model.keras')
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def get_prediction_on_frame(self, grayImg, co_ordinates):
        x, y, w, h = co_ordinates
        cropped_face = grayImg[y:y+h, x:x+w]
        roi = cv2.resize(cropped_face, (self.capture_size, self.capture_size))
        return self.model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
    
    def draw_rect(self, prediction, img_frame, co_ordinates):
        x, y, w, h = co_ordinates
        text_size, _ = cv2.getTextSize(prediction, self.font, 1, 2)
        cv2.rectangle(img_frame, (x, y - 30 - text_size[1]), (x + text_size[0], y - 10), (180, 184, 176), -1)
        cv2.putText(img_frame, prediction, (x, y - 20), self.font, 1, (69, 74, 24), 2)
        cv2.rectangle(img_frame, (x, y), (x + w, y + h), (106, 118, 252), 4)

    def predict_on_img(self, img_file):
        img = cv2.imread(img_file)
        grayScaledImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_co_ordinates = self.trained_face_data.detectMultiScale(grayScaledImg, 1.15)

        for (x, y, w, h) in face_co_ordinates:
            prediction = self.get_prediction_on_frame(grayScaledImg / 255., (x, y, w, h))
            self.draw_rect(prediction, img, (x, y, w, h))

        cv2.imshow('Face Detector', img)
        cv2.waitKey(0)

    def predict_on_video(self, video_file):
        video = cv2.VideoCapture(video_file)

        while video.isOpened():
            _, frame = video.read()
            grayScaledImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_co_ordinates = self.trained_face_data.detectMultiScale(grayScaledImg, 1.3, 5)
            for (x, y, w, h) in face_co_ordinates:
                prediction = self.get_prediction_on_frame(grayScaledImg / 255., (x, y, w, h))
                self.draw_rect(prediction, frame, (x, y, w, h))

            cv2.imshow('Face Detector', frame)
            cv2.waitKey(1)

    def predict_on_webcam(self):
        webcam = cv2.VideoCapture(0)

        while True:
            _, frame = webcam.read()
            frame = cv2.flip(frame, 1)
            grayScaledImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_co_ordinates = self.trained_face_data.detectMultiScale(grayScaledImg, 1.3, 5)
            for (x, y, w, h) in face_co_ordinates:
                prediction = self.get_prediction_on_frame(grayScaledImg / 255., (x, y, w, h))
                self.draw_rect(prediction, frame, (x, y, w, h))

            cv2.imshow('Face Detector', frame)
            cv2.waitKey(1)
