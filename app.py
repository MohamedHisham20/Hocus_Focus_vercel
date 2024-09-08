#create flask app to receive an image from a mobile app and returns a signal that it received the image

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return '<h1> hello there <h1>'

#get the image from the mobile app and return a signal that the image is received
@app.route('/images',methods=['POST'])
def get_image():
    #get the image from the mobile app
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({"error": "No image found"}), 400
    else:
        #return a signal that the image is received
        #return the image to show on website
        return jsonify({"message": "Image received"}), 200