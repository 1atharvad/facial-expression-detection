import sys
from prediction_on import PredictionOn

prediction_model = PredictionOn()

if len(sys.argv) > 2 and sys.argv[1] == 'img':
    prediction_model.predict_on_img(sys.argv[2])
elif len(sys.argv) > 2 and sys.argv[1] == 'video':
    prediction_model.predict_on_video(sys.argv[2])
elif len(sys.argv) > 1 and sys.argv[1] == 'webcam':
    prediction_model.predict_on_webcam()
else:
    print(f'Wrong input. Try again')