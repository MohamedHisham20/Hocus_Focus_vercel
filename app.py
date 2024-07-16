import time
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
from keras.models import load_model
from mtcnn import MTCNN

#size of the image to be given to the model

SIZE = 224
# load the model
VGG16_model = load_model('hocusfocusplease.h5')

# create the app
app = Flask(__name__)
#to capture the video from the camera
camera = cv2.VideoCapture(0)


def crop_face_and_return(image):
    cropped_face = None
    #create instance of haarcascade
    detector = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
    #capture the face using the model
    faces = detector.detectMultiScale(image, 1.1, 7)
    #crop the face exactly
    for (x, y, w, h) in faces:
        cropped_face = image[y:y + h, x:x + w]
    return cropped_face

# def crop_face_and_return(image):
#    cropped_face = None
#    detector = MTCNN()
#    faces = detector.detect_faces(image)
#    if faces:
#         x, y, width, height = faces[0]['box']
#         cropped_face = image[y:y + height, x:x + width]
#    return cropped_face


# Function to check if eyes are closed based on aspect ratio
#takes array of eyes that are detected
# def are_eyes_closed(eyes):
#     awake = 0
#     for eye in eyes:
#         #get the exact ratio of the eye
#         (x, y, w, h) = eye
#         aspect_ratio = float(w) / h  # the greater the value the more sleepy
#         # Set a threshold for the aspect ratio to determine closed eyes
#         closed_threshold = 5.0  # may be modified
#         if aspect_ratio < closed_threshold:
#             awake += 1 #an eye is detected as open
#     if awake > 0:
#         return False
#     else:
#         return True


prediction = [] #prediction array used to calculate the average

#main function of the video and prediction
def generate_frames():
    timey = 0 #to use time fn instead of delay
    last_pred = 0 #used for the sleep (to eliminate wrong frames of sleep)
    while True:
        ## read the camera frame
        success, frame = camera.read()
        frame = cv2.flip(frame,1)
        if time.time() - timey > 3: # enter each 5 seconds
            timey = time.time() #update the time
            if not success: #couldn't get the camera
                break
            else:
                #create instance of face detection
                detector = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
                #instance of eye detection
                eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
                #get the face
                faces = detector.detectMultiScale(frame, 1.1, 7)
                #convert to gray scale to enhance the detection of face and eye
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # frameBGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                x = y = w = h = 0
                # Draw the rectangle around each face
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                #range of interest (to faster the calculations)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                #detect the eyes
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
                #draw rectangle around the eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                # select the frame from the video (gray scale)
                cropped_face = crop_face_and_return(gray)
                if cropped_face is not None: # there's a face detected
                    #get it back to color image
                    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_GRAY2BGR)
                    # Convert the NumPy array 'cropped_face' into a PIL Image
                    pil_image = Image.fromarray(cropped_face, 'RGB')
                    #resize it to go into the model
                    pil_image = pil_image.resize((SIZE, SIZE))
                    #get it back to array
                    cropped_face = np.array(pil_image)
                    #reshape the array using tensor flow
                    image = tf.reshape(cropped_face, (1, SIZE, SIZE, 3))

                    # get the prediction
                    pred = VGG16_model.predict(image)
                    pred = np.argmax(pred)


                else:  # no face detected
                    if len(eyes) == 0:  # no eyes detected (absent)
                        pred = -1
                    else:  # there's eyes (active)
                        pred = 0
                #see the output on the terminal
                print("last pred", last_pred)
                print(pred)

                if pred == 1 or pred == -1 or pred == 2: #sleep or absent (to eliminate prediction errors)
                    if last_pred == pred: #check that  it repeated twice in a row
                        prediction.append(pred)
                else: #any thing other than sleep or absent
                    prediction.append(pred)
                #update the last prediction
                last_pred = pred
                print(prediction)

            # display the video
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#main application
@app.route('/')
def index():
    return render_template('index.html')

#to calculate the average
summ = 0
timeyy = 0
#to display the output average
@app.route('/_stuff', methods=['GET'])
def stuff():
    global summ
    global timeyy
    message = ''
    if len(prediction): #avoid first empty prediction
        while time.time() - timeyy > 1: #enter each 4 seconds
            timeyy = time.time()
            if len(prediction) % 5 == 0:  # each 5 readings of the prediction
                if summ > 5: summ = 5
                avg = (summ / 5) * 100
                message = 'avg=' + str(round(avg, 2)) + '%'
                summ = 0
            else:
                l_pred = prediction[-1] #get last prediction to display it
                if l_pred == 0:
                    message = 'Engaged'
                    summ += 1
                elif l_pred == -1:
                    message = "Absent"
                elif l_pred == 1:
                    message = "Sleeping"
                else:
                    message = "Yawningggg"
    return jsonify(result=message)

#app to display the frames
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Ensure that static files are served
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


if __name__ == "__main__":
    app.run(debug=True)
