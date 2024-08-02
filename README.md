# Face Expression Prediction API

This Python program leverages **TensorFlow Keras** and **OpenCV** to recognize facial expressions in images, videos, or live webcam feeds.

## Overview

The core of this program is a multi-layered **Convolutional Neural Network (CNN)** architecture to predict facial expressions, categorizing them into seven distinct emotions. The model is first trained on both training and validation datasets. Upon completion of training, the model's architecture and weights are saved as `model-detail.json` and `model.h5` within the `models` folder.

The trained model can predict facial expressions from images, videos, and live webcam feeds.

Additionally, users can upload images through a web browser via the FastAPI-based API for facial expression predictions.

## Usage

### Training the Model

To train the model, run the following command:
```shell
python train_model 
```

### Predicting Emotions

Images

To predict the emotions in the images, use:
```shell
python predict_expression img image_name.png
```

Videos

To predict the emotions in videos, use:
```shell
python predict_expression video video_name.mp4
```

Live Webcam

To predict emotions using live webcam feed, use:
```shell
python predict_expression webcam
```

### Running the API locally

To start the server for the emotion prediction API, execute:
```shell
uvicorn predict_expression.prediction_api:app --port 8080 --reload
```

To interact with the API, send a POST request to the following URL:
```bash
http://localhost:8080/api/predict-emotion/
```

The request should include:

* `img_file`: The image file to be analyzed.
* `img_height` and `img_width`: The dimensions of the uploaded image for accurate detection.

The API will respond with a list containing the coordinates of the face(s) detected in the image and the corresponding predicted emotion.

## How it Works

1. **Training**: The model is trained using a dataset of labeled images representing various facial expressions. During training, the model's parameters are optimized to accurately classify the emotions.
2. **Prediction**: Post-training, the model can analyze input from images, videos, or live webcam feeds, processing each frame to detect and predict facial expressions.
3. **Prediction API**: The trained model is accessible via a locally hosted API. You can send a POST request to the server on port 8080, including details such as `img_file` (the image file for analysis), `img_height`, and `img_width` (to define the dimensions for accurate detection). The API will respond with a list containing the coordinates of the face(s) in the image and the corresponding predicted emotion.

## Requirements

To install the necessary dependencies, run:
```shell
pip install -r requirements.txt
```

Ensure these dependencies are listed in your `requirements.txt` file and installed to run all the program successfully.

* Python
* TensorFlow
* Keras
* OpenCV
* FastAPI
* Uvicorn

## Contributing

Contributions are welcome! If you have suggestions for improvements, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

* TensorFlow and Keras for the deep learning framework.
* OpenCV for real-time computer vision.
* FastAPI for building the API.
* Uvicorn for running the ASGI server.
* **FER-2023 Dataset** for providing the comprehensive dataset used in training and evaluation.

If you like this project, consider starring the repository to show your support!