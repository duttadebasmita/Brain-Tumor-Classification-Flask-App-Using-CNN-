import os
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('C:\\repo2\\my_BrainTumor10EpochsCategorical.keras')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"


def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        result = getResult(file_path)
        classNo = np.argmax(result)  # Get the class number with highest probability
        class_name = get_className(classNo)  # Get the class name corresponding to the class number
        return class_name

    return None


if __name__ == '__main__':
    app.run(debug=True)
