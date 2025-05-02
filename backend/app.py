import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2 
from flask_cors import CORS
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

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


@app.route('/predict', methods=['POST'])
def predict():
    imagefile = request.files['image']
    image_path = "static/images/" + imagefile.filename
    imagefile.save(image_path)

    image_url = f"/static/images/{imagefile.filename}"

    img_info, path = prepare_image(image_path)
    prediction  = new_model.predict(np.expand_dims(img_info, axis=0))
    confidence = np.max(prediction)
    class_prediction = label_converter[np.argmax(prediction)]

    return jsonify({'prediction': class_prediction})

@app.route('/api', methods=['POST']) 
def api():
    prompt = (
    "Please answer the following question if it is related to eye diseases, eye clinics, or anything related to vision or eye health. "
    "If the question is not related to eyes, respond instead with an interesting or fun fact about the human eye (do not just make the fun fact always about how the human eye can see 10 million different colors). Try to keep the response within 5 sentences long: "
    )

    user_input = request.json.get("message")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    
    response = model.generate_content(prompt+user_input)
    return jsonify({"response": response.text})

if __name__ == '__main__':
    app.run(port=5001, debug=True)