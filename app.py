from flask import Flask, render_template, request
import pickle
import os
import numpy as np

from models import ResNet, image_size
from PIL import Image
from utils import resizeImage
from datasets import categories, num_classes

app = Flask(__name__)

def predict(img):
    model = ResNet(num_classes=num_classes)
    model.load_weights('weights/category.hdf5')
    img = np.expand_dims(img, axis=0)
    y = model.predict(img)
    return np.argmax(y[0])

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/result', methods=['POST'])
def result():
    if 'img' not in request.files:
        return "400"
    file = request.files['img']
    img = Image.open(file).convert('RGB')
    resized = resizeImage(img, target_size=image_size)
    y = predict(np.array(resized))
    return render_template('results.html',
                           prediction=categories[y]['name'])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
