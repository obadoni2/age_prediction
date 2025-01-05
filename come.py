 
import cv2
import numpy as np
import math 
import argparse
from flask import Flask, render_template, Response, request
from PIL import Image


import io

UPLOAD_FOLDER = './UPLOAD_FOLDER'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()

    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes


# gives input img to the prg for detection
# using argparse library which was imported

parser = argparse.ArgumentParser()

# if the input argument is given it will skip this and  open webcam for detection
parser.add_argument('--image')

args = parser.parse_args()


''' 
Each model comes with two files: weight file and model file
weight file store the data of the deployment of the model
model file store actual predication done by the model
we are using pre trained models

The .prototxt file(s) which define the model architecture(i.e., the  layers themselves)
the .caffemodel file which contains the weights for the  actual layers
Both file are required when using models trained using  Caffe for deep learning.



'''


def gen_frames():
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"  # Added missing model file
    
    MODEL_MEAN_VALUES = (78.4263377603, 87.768914744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']  # Corrected 'female' to 'Female' for consistency
    
    # LOAD NETWORK
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    video = cv2.VideoCapture(0)
    padding = 20
    while True:  # Changed from cv2.waitKey(1) < 0 for clarity, but you might want to keep it as is
        hasFrame, frame = video.read()
        if not hasFrame:
            break  # Exit loop if no more frames
        
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:
            print("No face detected")
        else:
            for faceBox in faceBoxes:
                face = frame[max(0, faceBox[1]-padding):min(faceBox[3]+padding, frame.shape[0]-1), 
                             max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                print(f'Gender: {gender}')
                
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                print(f'Age: {age[1:-1]} years')
                
                cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        if resultImg is None:
            continue

        ret, encodedImg = cv2.imencode('.jpg', resultImg)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')
                            
                            
    def gen_frames_photo(img_file):
        # Correct model names
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_deploy.prototxt"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"
    
    # ... (rest of the function remains similar)

    while True:  # Changed from while cv2.waitKey(1) < 0:
        resultImg, faceBoxes = highlightFace(faceNet, img_file)
        if not faceBoxes:
            print("No face detected")
            
        for faceBox in faceBoxes:
            face = img_file[max(0, faceBox[1]-padding):min(faceBox[3]+padding, img_file.shape[0]-1), max(0, faceBox[0]-padding):min(faceBox[2]+padding, img_file.shape[1]-1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')
            
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')
            
            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            
            if resultImg is None:
                continue
            
            ret, encodedImg = cv2.imencode('.jpg', resultImg)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')                      
                            
             
@app.route('/')
def index():
    """"Video streaming home page."""
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    # video streaming rounte. put this in the src attribute of an img tag 
    return Response(gen_frames(), mimetype='multipart/x-mixed- replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'fileToUpload' not in request.files:
            return "No file uploaded", 400

        f = request.files['fileToUpload'].read()
        img = Image.open(io.BytesIO(f))
        img_ip = np.asarray(img, dtype="uint8")
        return Response(gen_frames_photo(img_ip), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
    
        