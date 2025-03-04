from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2 
import numpy as np

label_converter = {0 : 'Age related Macular Degeneration',
                   1: 'Cataract',
                   2: 'Diabetes',
                   3: 'Glaucoma',
                   4: 'Hypertension',
                   5: 'Pathological Myopia',
                   6: 'Normal',
                   7: 'Other diseases/abnormalities'}

new_model = load_model('/Users/becketthayes/Desktop/team_lambda/model/final_CNN.h5')

def prepare_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    return img.astype(np.float32)

app = Flask(__name__)

@app.route('/', methods=['GET'])

def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    img_info = prepare_image(image_path)
    prediction  = new_model.predict(np.expand_dims(img_info, axis=0))
    confidence = np.max(prediction)
    class_prediction = label_converter[np.argmax(prediction)]

    return render_template('index.html', prediction=class_prediction+f" with {confidence*100:.4f}% confidence.")

if __name__ == '__main__':
    app.run(port=3000, debug=True)