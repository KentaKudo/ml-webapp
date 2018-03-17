from flask import Flask, render_template, request
import pickle
import os
import numpy as np

from models import InceptionV3
from PIL import Image
from utils import resizeImage

app = Flask(__name__)

def predict(img):
    model = InceptionV3()
    model.load_weights('weights/jeans.hdf5')
    return model.predict(img)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/result', methods=['POST'])
def result():
    if 'img' not in request.files:
        return "400"
    file = request.files['img']
    img = Image.open(file)
    resized = resizeImage(img, target_size=(225, 225))
    y = predict(np.array(resized))
    return render_template('results.html',
                           prediction=y)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
