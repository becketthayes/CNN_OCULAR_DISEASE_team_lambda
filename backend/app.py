import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2 
import numpy as np

label_converter = {0 : 'Age related Macular Degeneration',
                   1: 'Cataract',
                   2: 'Diabetic Retinopathy',
                   3: 'Glaucoma',
                   4: 'Hypertension',
                   5: 'Pathological Myopia',
                   6: 'Normal',
                   7: 'Other diseases/abnormalities'}

new_model = load_model('model/final_CNN.h5')

def prepare_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary mask
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, return the original image
    if not contours:
        return img.astype(np.float32)

    # Find the largest contour (assuming it's the fundus)
    c = max(contours, key=cv2.contourArea)

    # Get bounding box coordinates
    x, y, w, h = cv2.boundingRect(c)

    # Crop the image
    cropped_img = img[y:y+h, x:x+w]
    cropped_img = cv2.resize(cropped_img, (256, 256))
    cropped_img = cropped_img / 255.0

    cropped_image_path = os.path.splitext(image_path)[0] + "_cropped.jpg"
    cv2.imwrite(cropped_image_path, (cropped_img * 255).astype(np.uint8))

    return cropped_img.astype(np.float32), cropped_image_path

app = Flask(__name__)


@app.route('/', methods=['GET'])

def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "static/images/" + imagefile.filename
    imagefile.save(image_path)

    image_url = f"/static/images/{imagefile.filename}"

    img_info, path = prepare_image(image_path)
    prediction  = new_model.predict(np.expand_dims(img_info, axis=0))
    confidence = np.max(prediction)
    class_prediction = label_converter[np.argmax(prediction)]

    return render_template('index.html', prediction=class_prediction+f" with {confidence*100:.4f}% confidence.",
                           image_url=path)

if __name__ == '__main__':
    app.run(port=3000, debug=True)