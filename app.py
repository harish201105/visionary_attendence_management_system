# app.py

import cv2
from flask import Flask, render_template, request
from face_detection import detect_faces
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from keras.models import load_model
import numpy as np
import progressbar
import os

app = Flask(__name__)

# Load the pre-trained face recognition model
model_path = os.path.join('models', 'face_recognition_model.h5')
face_model = load_model(model_path)

# Load the student data (register numbers and corresponding faces)
data_path = os.path.join('student_data.csv')
student_data = pd.read_csv(data_path)

# Encode register numbers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(student_data['RegisterNumber'])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        # Save the uploaded file
        file_path = os.path.join('static/images', file.filename)
        file.save(file_path)

        # Detect faces in the uploaded image
        faces = detect_faces(file_path)

        # Read the image for displaying purposes
        img = Image.open(file_path)
        
        # Recognize faces using the pre-trained model
        recognized_labels = []

        # Use a progress bar to visualize face recognition progress
        with progressbar.ProgressBar(max_value=len(faces)) as bar:
            for i, face in enumerate(faces):
                x, y, width, height = face['box']
                face_img = np.array(img)[y:y+height, x:x+width]
                face_img = cv2.resize(face_img, (160, 160))
                face_img = np.expand_dims(face_img, axis=0)
                face_img = face_img / 255.0  # Normalize

                # Perform face recognition
                prediction = face_model.predict(face_img)

                # Get the predicted label
                predicted_label = np.argmax(prediction)
                recognized_labels.append(predicted_label)

                bar.update(i)

        # Map recognized labels to register numbers
        recognized_register_numbers = label_encoder.inverse_transform(recognized_labels)

        # Check attendance and update the database
        # (Add your logic for attendance tracking here)

        return render_template('index.html', image_path=file_path, recognized_faces=recognized_register_numbers)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
