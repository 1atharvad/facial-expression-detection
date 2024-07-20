import numpy as np
from keras.models import load_model

class FacialExpressionModel:
    def __init__(self, model_file):
        self.emotions_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.loaded_model = load_model(model_file)
        self.loaded_model.make_predict_function()
        print(self.loaded_model.compiled_metrics)

    def predict_emotion(self, img):
        predict_values = self.loaded_model.predict(img, verbose=False)
        print(predict_values)
        return self.emotions_list[np.argmax(predict_values)]