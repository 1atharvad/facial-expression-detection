import numpy as np
from keras.models import load_model

class FacialExpressionModel:
    def __init__(self, model_file, emotions_list):
        self.emotions_list = emotions_list
        self.loaded_model = load_model(model_file)
        self.loaded_model.make_predict_function()

    def predict_emotion(self, img):
        predict_values = self.loaded_model.predict(img, verbose=False)
        return self.emotions_list[np.argmax(predict_values)]