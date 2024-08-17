from flask import Flask, render_template, Response
import cv2
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
import io
import time
from mtcnn import MTCNN
from keras.models import load_model

app = Flask(__name__)
VGG16_model = load_model('hocusfocusplease.h5')

SIZE = 224
video_stream = cv2.VideoCapture(0)  # Assuming the camera is connected to the computer
label_html = 'Capturing...'
bbox = ''
count = 0
filenames = ['/static/image_1.jpg', '/static/image_2.jpg', '/static/image_3.jpg', '/static/image_4.jpg', '/static/image_5.jpg']
predictions = []
i = 0


def crop_face_and_return(image):
    cropped_face = None
    detector = MTCNN()
    faces = detector.detect_faces(image)
    if faces:
        x, y, width, height = faces[0]['box']
        cropped_face = image[y:y + height, x:x + width]
    return cropped_face


def gen_frames():
    global bbox, count, predictions, i
    while True:
        success, frame = video_stream.read()
        if not success:
            break

        data = {'create': 0, 'show': 0, 'capture': 0, 'img': ''}
        if count < 5:
            # Process every 5th frame
            if count % 5 == 0:
                # Save image
                cv2.imwrite(filenames[i], frame)

                # Process saved image
                image = cv2.imread(filenames[i])
                cropped_face = crop_face_and_return(image)
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

                if cropped_face is not None and cropped_face.size != 0:
                    pil_image = Image.fromarray(cropped_face, 'RGB')
                    pil_image = pil_image.resize((SIZE, SIZE))
                    cropped_face = np.array(pil_image)
                    image = tf.reshape(cropped_face, (1, SIZE, SIZE, 3))
                    predictions.append(VGG16_model.predict(image))

                i += 1
            count += 1
        else:
            # Process predictions
            predictions = [np.argmax(pr) for pr in predictions]
            count = 0
            i = 0

        # Convert the frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index1.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
