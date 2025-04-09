from flask import Flask, render_template, request, jsonify ,session, redirect, url_for
import numpy as np
import cv2
import tensorflow as tf
import os
from flask_cors import CORS  # Add this import

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained model (update path if needed)
try:
    model = tf.keras.models.load_model("dysgraphia_model.keras")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Read image
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (224, 224))  # Resize to 224x224
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Reshape for model input
    return img

# Prediction function
def predict_dysgraphia(image_path):
    processed_image = preprocess_image(image_path)
    dummy_features = np.zeros((1, 10))  # Dummy feature input

    prediction = model.predict({"image_input": processed_image, "feature_input": dummy_features})
    raw_value = prediction[0][0]  # Extract raw sigmoid output

    print(f"Raw Prediction Output: {raw_value}")  # Debugging output

    # Define threshold (flipped if necessary)
    label = "Dysgraphic" if raw_value < 0.5 else "Non-dysgraphic"
    confidence = 1 - raw_value if label == "Dysgraphic" else raw_value

    return label, float(confidence)

# 📌 Routes for HTML pages
@app.route('/')
def index():
    """Render the login page"""
    try:
        return render_template('login.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"

@app.route('/parent-login')
def parent_login():
    """Render the parent login page"""
    return render_template('parentlogin.html')

@app.route('/teacher-login')
def teacher_login():
    """Render the teacher login page"""
    return render_template('teacher.html')

@app.route('/counselor-login')
def counselor_login():
    """Render the counselor login page"""
    return render_template('counsellorlogin.html')

@app.route('/dyscalculia-test')
def dyscalculia_test():
    """Render the dyscalculia test page"""
    try:
        return render_template('dyscaculiatest.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"

@app.route('/dyslexia-test')
def dyslexia_test():
    """Render the dyslexia test page"""
    try:
        return render_template('dyslexiatest.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"
    
@app.route('/memorygame')
def memory_game():
    return render_template('memorygame.html')  # Make sure this file is in /templates folder

@app.route('/attentiongame')
def attention_game():
    return render_template('attentiongame.html')  # Make sure this file is in /templates folder

@app.route('/languagegame')
def language_game():
    return render_template('languagegame.html')  # Make sure this file is in /templates folder
    
@app.route('/processinggame')
def processing_game():
    return render_template('processinggame.html')

@app.route('/memorymaze')
def memory_maze():
    return render_template('memorymaze.html')

# 📌 Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        print("No file in request")
        return jsonify({"error": "No file uploaded"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        print("Empty filename")
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the uploaded file
        image_path = "temp_image.jpg"
        image_file.save(image_path)
        print(f"Image saved to {image_path}")
        
        # Call prediction function with corrected parameters
        label, confidence = predict_dysgraphia(image_path)
        
        # Return in the format expected by your frontend
        return jsonify({
            "prediction": label,
            "confidence": confidence
        })
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)