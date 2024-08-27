from flask import Flask, jsonify, send_from_directory, request
import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

num_classes = 2

# define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # expected 224,224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # expected mean and std
])


# load image
def load_image(image, transform):
    pil_image = Image.fromarray(image)
    pil_image = pil_image.convert("RGB")
    pil_image = transform(pil_image)

    return pil_image.unsqueeze(0)  # this for the batch dimension model expected the batch dimension


#define the model
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
        # Initialize with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x_encoder, x):
        # Apply STN
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

eye_model = Drowness(num_classes)  # num of classes is the num of labels
mouth_model = Drowness(num_classes)

eye_model.load_state_dict(
    torch.load("Model2_stn.pth", map_location=torch.device('cpu')))  # PATH is the path of the model
mouth_model.load_state_dict(
    torch.load("Model_mouth_stn2.pth", map_location=torch.device('cpu')))  # PATH is the path of the model

eye_model.to(device)
mouth_model.to(device)

eye_model.eval()
mouth_model.eval()


def predict(passed_model, image_path):  # image path is the path of the image
    image = load_image(image_path, transform).to(device)
    pred_state, stn = passed_model(image)
    pred_probs = torch.nn.functional.softmax(pred_state, dim=1).cpu().detach().numpy()
    predicted_state = np.argmax(pred_probs, axis=1)
    #convert predicted state from np array to int
    predicted_state = int(predicted_state[0])

    return {'state': predicted_state}


# create the app with port 3000
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

prediction = []  #prediction array used to calculate the average
last_pred = -1
pred = -1

#main application
@app.route('/')
def index():
    return "Hello, World!"

map_prediction = {0: 'Active', 1: 'disengaged', 2: 'disengaged', -1: 'disengaged'}

#main function of the video and prediction
@app.route('/video', methods=['POST'])
def generate_frames():
    global pred, last_pred, prediction  # Declare variables globally for persistence across requests

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

        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # select the frame from the video (gray scale)
        cropped_face = crop_face_and_return(frame)  #gray
        if cropped_face is not None:  # there's a face detected

            eye_state = predict(eye_model, cropped_face)
            eye_state = eye_state['state']
            mouth_state = predict(mouth_model, cropped_face)
            mouth_state = mouth_state['state']

            if eye_state == 0:  #closed eyes
                if mouth_state == 0: #open mouth
                    pred = 2  #yawn closed eyes
                else:  #closed mouth
                    pred = 1  #sleep
            elif eye_state == 1:  #open eyes
                # if mouth_state == 0:  #open mouth
                #     pred = 2  #yawn open eyes
                # else:  #closed mouth
                    pred = 0  #active

##############################################################################################################
############################################## 0 = active, 1 = sleep, 2 = yawn, -1 = absent ########################################
##################################################################################################################

        else:  # no face detected
            pred = -1  #absent

        # Update prediction history only if necessary
        if pred == 1 or pred == -1 or pred == 2:  # Update only for sleep/absent states
            if pred == last_pred:  # Check if the same state is repeated twice
                prediction.append(pred)
        else:
            prediction.append(pred)

        last_pred = pred  # Update last_pred
        print(f'prediction:', prediction)
    return jsonify({"state": map_prediction[pred]})


#to calculate the average
summ = 0
timeyy = 0


# Ensure that static files are served
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
