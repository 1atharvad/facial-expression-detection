import numpy as np
from keras.models import model_from_json

class FacialExpressionModel:
    def __init__(self, model_json_file, model_weights_file):
        self.emotions_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.get_model(model_json_file)
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def get_model(self, model_json_file):
        with open(model_json_file, 'r') as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

    def predict_emotion(self, img):
        predict_values = self.loaded_model.predict(img, verbose=False)
        return self.emotions_list[np.argmax(predict_values)]