import base64
import time
from io import BytesIO
from flask import Flask, render_template, jsonify, send_from_directory, request
import cv2
import torch
from torchvision import transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define the number of classes
num_classes = 2

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # expected 224,224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # expected mean and std
])


# Load and transform the image
def load_image(image, transform):
    pil_image = Image.fromarray(image)
    pil_image = pil_image.convert("RGB")
    pil_image = transform(pil_image)
    return pil_image.unsqueeze(0)  # add batch dimension


# Define the models (LeNet, STN, Drowness)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
        )

    def forward(self, x):
        a1 = self.feature_extractor(x)
        return a1


class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(128, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(90, 64),  # Ensure this matches the flattened size of localization output
            nn.ReLU(True),
            nn.Linear(64, 6)  # 6 parameters for the affine transformation matrix
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x_encoder, x):
        xs = self.localization(x_encoder)
        xs = xs.view(xs.size(0), -1)  # Flatten
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)  # Affine transformation matrix
        grid = F.affine_grid(theta, x_encoder.size(), align_corners=True)
        x_transformed = F.grid_sample(x_encoder, grid, align_corners=True)
        return x_transformed


class Drowness(nn.Module):
    def __init__(self, num_classes):
        super(Drowness, self).__init__()
        self.model = LeNet()
        self.stn = STN()
        self.encoder = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-5])
        self.channel_reshape = nn.Conv2d(64, 3, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Linear(5880, 2)
        )

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_encoded = self.channel_reshape(x_encoded)
        x1 = self.stn(x_encoded, x)
        features = self.model(x1)
        features = torch.flatten(features, 1)
        pred = self.fc(features)
        return pred, x1


# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eye_model = Drowness(num_classes).to(device)
mouth_model = Drowness(num_classes).to(device)

eye_model.load_state_dict(torch.load("Model2_stn.pth", map_location=device))
mouth_model.load_state_dict(torch.load("Model_mouth_stn2.pth", map_location=device))

eye_model.eval()
mouth_model.eval()


# Predict function
def predict(passed_model, image):  # image is the frame from the camera
    image = load_image(image, transform).to(device)
    pred_state, stn = passed_model(image)
    pred_probs = torch.nn.functional.softmax(pred_state, dim=1).cpu().detach().numpy()
    predicted_state = np.argmax(pred_probs, axis=1)
    predicted_state = int(predicted_state[0])

    return predicted_state


# Create Flask app
app = Flask(__name__)


# Crop the face from the frame
def crop_face_and_return(image):
    cropped_face = None

    detector = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
    faces = detector.detectMultiScale(image, 1.1, 7)
    for (x, y, w, h) in faces:
        cropped_face = image[y:y + h, x:x + w]
    return cropped_face


prediction = []
map_prediction = {0: 'Active', 1: 'Sleep', 2: 'Yawn', -1: 'Absent'}
last_pred = -1
pred = -1


# Main application
@app.route('/')
def index():
    return render_template('index.html')


# To calculate the average
summ = 0
timeyy = 0


# Display the output average
@app.route('/_stuff', methods=['GET'])
def stuff():
    global summ
    global timeyy
    message = ''
    if len(prediction):
        while time.time() - timeyy > 2:  # Enter every 1 second
            timeyy = time.time()
            if len(prediction) % 5 == 0:  # Each 5 readings of the prediction
                if summ > 5: summ = 5
                avg = (summ / 5) * 100
                message = 'avg=' + str(round(avg, 2)) + '%'
                summ = 0
            else:
                l_pred = prediction[-1]  # Get last prediction to display it
                if l_pred == 0:
                    message = 'Engaged'
                    summ += 1
                elif l_pred == -1:
                    message = "Absent"
                else:
                    message = "Disengaged"
    return jsonify(result=message)


# To display the frames
# Change this route
@app.route('/process_frame', methods=['POST'])
def process_frame():
    global pred, last_pred, prediction  # Declare variables globally for persistence across requests

    if 'frame' not in request.form:
        return jsonify({"error": "No frame key found in request"}), 400

    frame_data = request.form['frame']
    frame_data = frame_data.split(',')[1]
    frame = Image.open(BytesIO(base64.b64decode(frame_data)))

    if not frame:
        return jsonify({"error": "No image found"}), 400
    else:
        numpy_image = np.array(frame)
        # Read the image file and convert it to a numpy array
        if numpy_image.shape[2] == 4:
            cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2RGB)
        else:
            cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        # if image is None:
        #     return jsonify({"error": "Failed to decode image"}), 400


        # select the frame from the video (gray scale)
        cropped_face = crop_face_and_return(cv2_image)  # gray

        if cropped_face is not None:  # there's a face detected
            eye_state = predict(eye_model, cropped_face)
            # eye_state = eye_state['state']
            mouth_state = predict(mouth_model, cropped_face)
            # mouth_state = mouth_state['state']
            print(f'eye_state:', eye_state)
            print(f'mouth_state:', mouth_state)

            if eye_state == 0:  # closed eyes
                if mouth_state == 0:  # open mouth
                    pred = 2  # yawn closed eyes
                else:  # closed mouth
                    pred = 1  # sleep
            elif eye_state == 1:  # open eyes
                # if mouth_state == 0:  #open mouth
                #     pred = 2  #yawn open eyes
                # else:  #closed mouth
                pred = 0  # active

        ##############################################################################################################
        ############################################## 0 = active, 1 = sleep, 2 = yawn, -1 = absent ########################################
        ##################################################################################################################

        else:  # no face detected
            pred = -1  # absent

        # Update prediction history only if necessary
        if pred == 1 or pred == -1 or pred == 2:  # Update only for sleep/absent states
            if pred == last_pred:  # Check if the same state is repeated twice
                prediction.append(pred)
        else:
            prediction.append(pred)

        print(f'prediction:', prediction)
        print(f'pred', pred)
        print(f'last_pred', last_pred)
        last_pred = pred  # Update last_pred
    return jsonify({"state": map_prediction[pred]})


# Ensure that static files are served
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


if __name__ == "__main__":
    app.run(debug=True)
