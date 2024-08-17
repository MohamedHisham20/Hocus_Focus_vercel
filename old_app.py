import time
from flask import Flask, render_template, Response, jsonify, send_from_directory, request
import cv2
import torch
from torchvision import transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

num_classes = 2

# define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # expected 224,224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # expected mean and std
])

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
            # nn.AvgPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(in_channels=16, out_channels=3, kernel_size=5, stride=1, padding=0),
            #   nn.ReLU(),
            # nn.AvgPool2d(kernel_size=2, stride=2),

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
    torch.load("Model_mouth_stn.pth", map_location=torch.device('cpu')))  # PATH is the path of the model

eye_model.to(device)
mouth_model.to(device)

eye_model.eval()
mouth_model.eval()

# load image
def load_image(image, transform):
    pil_image = Image.fromarray(image)
    pil_image = pil_image.convert("RGB")
    pil_image = transform(pil_image)
    return pil_image.unsqueeze(0)  # this for the batch dimension model expected the batch dimension

def predict(passed_model, image_path):  # image path is the path of the image
    image = load_image(image_path, transform).to(device)
    pred_state, stn = passed_model(image)
    pred_probs = torch.nn.functional.softmax(pred_state, dim=1).cpu().detach().numpy()
    predicted_state = np.argmax(pred_probs, axis=1)
    #convert predicted state from np array to int
    predicted_state = int(predicted_state[0])
    image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    stn = stn[0].permute(1, 2, 0).cpu().detach().numpy()  # Move to CPU before converting to NumPy array

    stn = np.clip(stn, 0, 1)

    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.imshow(image)
    plt.title('original')

    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.imshow(stn)
    plt.title('transformed')

    plt.show()

    return {'state': predicted_state}



# create the app
app = Flask(__name__)

# get camera
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


#main application


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

                # frameBGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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


@app.route('/')
def index():
    return render_template('index.html')

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
