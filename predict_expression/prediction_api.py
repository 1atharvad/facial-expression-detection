import os
import cv2
import numpy as np
import uvicorn
from .prediction_on import PredictionOn
from fastapi import FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

class PredictExpressionAPI(PredictionOn):
    def __init__(self):
        super().__init__()
        self.get_model_details('0.1.2')

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
    
    def predict_on_image(self, image: list):
        grayScaledImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        emotion_predictions = self.get_face_co_ordinates(grayScaledImg)

        for prediction in emotion_predictions:
            x, y, w, h = prediction['face_coordinates']
            self.draw_rect(prediction['predicted_emotion'], image, (x, y, w, h))

        return image
    
app = FastAPI()
api = PredictExpressionAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

@app.post('/api/predict-emotion/')
async def predict_emotion(img_file: UploadFile, img_height: str = Form(...), img_width: str = Form(...)):
    image_data = await img_file.read()
    image_dim = {
        'img_height': img_height,
        'img_width': img_width
    }
    return api.predict_on_img(image_data, image_dim)

description = '''
    This demo allows users to upload an image, and the system will predict the emotions of all faces detected in the image using a **Convolutional Neural Network (CNN)** model. By leveraging a multi-layered CNN architecture, the program accurately predicts facial expressions, categorizing them into seven distinct emotions: 😊&nbsp; happiness, 😢&nbsp; sadness, 😲&nbsp; surprise, 😨&nbsp; fear, 😡&nbsp; anger, 😒&nbsp; disgust, and 😐&nbsp; neutral.
        
    ## Steps:
    - **📤 &nbsp; Upload an Image**: The user selects an image from their device.
            
    - **👤 &nbsp; Face Detection**: The system detects all faces in the uploaded image.
            
    - **🔍 &nbsp; Emotion Prediction**: The CNN model analyzes each detected face and classifies the emotion from the seven categories.
            
    - **🖼️ &nbsp; Results Display**: The image is presented with bounding boxes around the faces, each labeled with the predicted emotion.
    
    This demo highlights the power of deep learning in analyzing and interpreting human emotions from visual inputs, demonstrating the effectiveness of CNNs in real-world applications.
'''

with gr.Blocks() as demo:
    image_input = gr.Image(
        label='Input Image'
    )
    image_box = gr.Interface(
        api.predict_on_image,
        image_input,
        gr.Image(label='Output Image'),
        title="🙂&nbsp; Facial Emotion Detection (FED) demo &nbsp;😐",
        description=description,
        allow_flagging='never')
    example_files = [os.path.join("demo-files", img_file) for img_file in os.listdir("demo-files") if img_file != '.DS_Store']
    print(example_files)
    examples = gr.Examples(example_files, image_input)
    gr.Markdown("""
        This demo does not aim to provide optimal results but rather to provide a quick look. See our [GitHub](https://github.com/1atharvad/facial-expression-detection) for more. 
        """)

app = gr.mount_gradio_app(app, demo, path='/')

if __name__ == '__main__':
    uvicorn.run(app, port=8080)

# uvicorn predict_expression.prediction_api:app --port 8080 --reload
