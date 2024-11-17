from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import cv2
import pickle
import os
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

upload_folder = './uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Load the trained model
with open('model3.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

IMG_SIZE = 50  # Image size used during training

def preprocess_image(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            image_path = './uploads/' + file.filename
            file.save(image_path)
            image_array = preprocess_image(image_path)
            
            # Predict
            prediction = model.predict(image_array)
            result = np.argmax(prediction, axis=1)
            label = 'Masked' if result[0] == 0 else 'Unmasked'

            return render_template('result.html', label=label, filename=file.filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)
