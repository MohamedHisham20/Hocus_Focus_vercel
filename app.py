from flask import Flask, Response, jsonify, send_from_directory, request
import cv2
import numpy as np

# create the app
app = Flask(__name__)


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


#main application
@app.route('/')
def index():
    return jsonify({"message": "Welcome to the face detection app"})


@app.route('/video', methods=['POST'])
def generate_frames():
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({"error": "No image found"}), 400
    else:
        # Read the image file and convert it to a numpy array
        image_data = np.frombuffer(image_file.read(), np.uint8)
        # Decode the numpy array to an image
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Create instance of face detection
        detector = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = detector.detectMultiScale(gray_image, 1.1, 7)
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Return the image with rectangles around faces
        # return Response(cv2.imencode('.jpg', image)[1].tobytes(), mimetype='image/jpeg')
        # return the coordinates of the faces
        return jsonify({"faces": faces.tolist()})

# Ensure that static files are served
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


if __name__ == "__main__":
    app.run(debug=True)