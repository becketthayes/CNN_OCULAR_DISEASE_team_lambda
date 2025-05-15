import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2 
from flask_cors import CORS
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "https://team-lambda-nine.vercel.app"])

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize models as None for lazy loading
_cnn_model = None

# Lazy load the models when needed
def get_cnn_model():
    global _cnn_model
    if _cnn_model is None:
        _cnn_model = load_model('model/final_CNN.h5')
    return _cnn_model

# Load Gemini model once
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

label_converter = {0 : 'Age related Macular Degeneration',
                   1: 'Cataract',
                   2: 'Diabetic Retinopathy',
                   3: 'Glaucoma',
                   4: 'Hypertension',
                   5: 'Pathological Myopia',
                   6: 'Normal',
                   7: 'Other diseases/abnormalities'}

def process_image_from_bytes(image_bytes):
    """Process image directly from bytes without saving to disk"""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to create a binary mask
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, use the original image
    if not contours:
        img_resized = cv2.resize(img, (256, 256))
        return img_resized.astype(np.float32) / 255.0
    
    # Find the largest contour (assuming it's the fundus)
    c = max(contours, key=cv2.contourArea)
    
    # Get bounding box coordinates
    x, y, w, h = cv2.boundingRect(c)
    
    # Crop the image
    cropped_img = img[y:y+h, x:x+w]
    cropped_img = cv2.resize(cropped_img, (256, 256))
    
    # Normalize pixel values
    normalized_img = cropped_img / 255.0
    
    return normalized_img.astype(np.float32)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return jsonify({'status': 'success'})
    
    try:
        # Check if image was provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        imagefile = request.files['image']
        if not imagefile.filename:
            return jsonify({'error': 'Empty image file'}), 400
        
        # Process image directly from the uploaded file
        image_bytes = imagefile.read()
        
        # Process the image in memory
        processed_img = process_image_from_bytes(image_bytes)
        
        # Get CNN model (lazy loaded)
        cnn_model = get_cnn_model()
        
        # Make prediction
        prediction = cnn_model.predict(np.expand_dims(processed_img, axis=0))
        
        # Get results
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))
        class_prediction = label_converter[predicted_class]
        
        return jsonify({
            'prediction': class_prediction, 
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api', methods=['POST', 'OPTIONS']) 
def api():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return jsonify({'status': 'success'})
    
    try:
        prompt = (
        "Please answer the following question if it is related to eye diseases, eye clinics, or anything related to vision or eye health. "
        "If the question is not related to eyes, respond instead with an interesting or fun fact about the human eye (do not just make the fun fact always about how the human eye can see 10 million different colors). Try to keep the response within 5 sentences long: "
        )

        user_input = request.json.get("message")

        if not user_input:
            return jsonify({"error": "No message provided"}), 400
        
        response = model.generate_content(prompt+user_input)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(port=5001, debug=True)