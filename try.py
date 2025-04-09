import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from scipy import stats
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import pickle
import logging
from flask_cors import CORS



app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and features dictionary
MODEL_PATH = 'dysgraphia_model.keras'
FEATURES_PATH = 'features_dict.pkl'
model = None
features_dict = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
def check_files_exist():
    model_exists = os.path.exists(MODEL_PATH)
    features_exist = os.path.exists(FEATURES_PATH)
    
    if not model_exists:
        logger.error(f"Model file not found at: {MODEL_PATH}")
    if not features_exist:
        logger.error(f"Features dictionary not found at: {FEATURES_PATH}")
        
    return model_exists and features_exist

def load_model_and_features():
    global model, features_dict
    model_success = False
    features_success = False
    
    try:
        model = keras.models.load_model(MODEL_PATH)
        model_success = True
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        
    try:
        with open(FEATURES_PATH, 'rb') as f:
            features_dict = pickle.load(f)
        features_success = True
        logger.info(f"Features dictionary loaded with {len(features_dict)} entries")
    except Exception as e:
        logger.error(f"Error loading features dictionary: {e}")
    
    if not model_success or not features_success:
        logger.warning("Some components failed to load. Application will not function properly.")

# Feature extraction functions from your code
def extract_stroke_consistency(img):
    """Measure stroke consistency based on intensity variations"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Calculate standard deviation of edge intensity as a measure of consistency
    if np.sum(edges) > 0:  # Check if there are any edges detected
        return np.std(edges[edges > 0]) / 255.0  # Normalize
    return 0.0

def extract_letter_spacing(img):
    """Measure letter spacing using horizontal projection"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img

    # Threshold to binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection (sum of white pixels in each column)
    h_proj = np.sum(binary, axis=0) / 255.0

    # Calculate spacing metric (standard deviation of distances between peaks)
    peaks = []
    for i in range(1, len(h_proj) - 1):
        if h_proj[i] > h_proj[i-1] and h_proj[i] > h_proj[i+1] and h_proj[i] > 5:
            peaks.append(i)

    if len(peaks) > 1:
        distances = [peaks[i+1] - peaks[i] for i in range(len(peaks) - 1)]
        return np.std(distances) / img.shape[1]  # Normalize by image width
    return 0.0

def extract_alignment(img):
    """Measure alignment of text lines"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img

    # Threshold to binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection (sum of white pixels in each row)
    v_proj = np.sum(binary, axis=1) / 255.0

    # Find rows with text (where projection > threshold)
    text_rows = [i for i, val in enumerate(v_proj) if val > binary.shape[1] * 0.1]

    if text_rows:
        # Split into continuous line segments
        lines = []
        current_line = [text_rows[0]]

        for i in range(1, len(text_rows)):
            if text_rows[i] - text_rows[i-1] <= 2:  # If rows are adjacent or close
                current_line.append(text_rows[i])
            else:
                if len(current_line) > 5:  # Minimum line length
                    lines.append(current_line)
                current_line = [text_rows[i]]

        if len(current_line) > 5:
            lines.append(current_line)

        # Calculate left margin variation
        margins = []
        for line in lines:
            line_img = binary[min(line):max(line), :]
            for col in range(line_img.shape[1]):
                if np.sum(line_img[:, col]) > 0:
                    margins.append(col)
                    break

        if len(margins) > 1:
            return np.std(margins) / img.shape[1]  # Normalize by image width

    return 0.0

def extract_features(image):
    """Extract multiple handwriting features from an image"""
    features = {
        'stroke_consistency': extract_stroke_consistency(image),
        'letter_spacing': extract_letter_spacing(image),
        'alignment': extract_alignment(image)
    }

    # Create a fixed-length feature vector (same as in your model)
    feature_vector = [
        features['stroke_consistency'],
        features['letter_spacing'],
        features['alignment'],
        # Add more features as needed to reach desired length (padding with zeros)
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]

    return feature_vector[:10]  # Return first 10 features

def preprocess_image(image, img_size=(224, 224)):
    """Preprocess an image for model input"""
    # Resize image
    resized_img = cv2.resize(image, img_size)
    
    # Normalize
    normalized_img = resized_img / 255.0
    
    # Expand dimensions to match model input shape
    expanded_img = np.expand_dims(normalized_img, axis=0)
    
    return expanded_img

def predict_dysgraphia(image_path, img_size=(224, 224)):
    """Predict dysgraphia from an image file"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Failed to read image"}, 400

        # Extract handwriting features
        features = extract_features(img)
        features_array = np.array(features).reshape(1, -1)
        
        # Normalize features
        try:
            features_array = stats.zscore(features_array, axis=0, nan_policy='omit')
            features_array = np.nan_to_num(features_array)  # Replace NaNs with zeros
        except Exception as e:
            logger.warning(f"Error normalizing features: {e}")
            features_array = np.zeros((1, 10))

        # Preprocess image for CNN
        img_input = preprocess_image(img, img_size)
        
        # If model is not loaded, return dummy prediction
        if model is None:
            return {
                "status": "success",
                "prediction": "Non-dysgraphic",
                "confidence": 0.7,
                "warning": "Using dummy prediction (model not loaded)"
            }
        
        # Make prediction
        prediction = model.predict([img_input, features_array])  # Try passing inputs as a list


        raw_value = float(prediction[0][0])
        
        # Interpret results (adjust threshold based on your model)
        # Note: Based on your original code, it seems like lower values indicate dysgraphia
        prediction_label = "Dysgraphic" if raw_value < 0.5 else "Non-dysgraphic"
        confidence = 1 - raw_value if prediction_label == "Dysgraphic" else raw_value
        
        # Return results
        return {
            "status": "success",
            "prediction": prediction_label,
            "confidence": float(confidence),
            "raw_score": float(raw_value),
            "features": features
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return {"error": str(e)}, 500

@app.route('/')
def index():
    """Render the login page"""
    return render_template('login.html')

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

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    features_count = len(features_dict) if features_dict is not None else 0
    model_loaded = model is not None
    features_loaded = features_dict is not None
    status = "healthy" if (model_loaded and features_loaded) else "unhealthy"
    
    return jsonify({
        "features_count": features_count,
        "features_loaded": features_loaded,
        "model_loaded": model_loaded,
        "status": status
    })




if __name__ == '__main__':
    # Load model and features at startup
    load_model_and_features()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)