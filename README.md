# Face Expression Prediction API

This Python program leverages **TensorFlow Keras** and **OpenCV** to recognize facial expressions in images, videos, or live webcam feeds.

## Overview

Using a multi-layered **CNN** model architecture, this program predicts facial expressions by categorizing them into seven different emotions. The model is initially trained using training and validation datasets. Upon training completion, the model details and weights are stored in `model.json` and `model_weights.keras`, respectively.

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

## How it Works

1. **Training**: The model is trained using a dataset consisting of images categorized into different facial expressions. This training process involves optimizing the model's parameters to minimize prediction errors.
2. **Prediction**: Once trained, the model can be used to predict emotions in various scenarios. It accepts input from images, videos, or live webcam feeds, and processes each frame to detect and predict facial expressions.

## Requirements

* Python
* TensorFlow
* Keras
* OpenCV

Ensure these dependencies are installed to run the program successfully.