# Fixed Flask App - app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import cv2
import tensorflow as tf
import os
from flask_cors import CORS
import traceback
from werkzeug.utils import secure_filename
from mindtrack import extract_features, normalize_features
from scipy import stats

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = 'your-secret-key-here'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/images', exist_ok=True)
os.makedirs('static/videos', exist_ok=True)
os.makedirs('static/css', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the trained model with error handling
model = None
feature_scaler = None

try:
    # Try to load model and scaler
    possible_paths = [
        "dysgraphia_model.keras",
        "./dysgraphia_model.keras", 
        "dysgraphia_model.h5",
        "./dysgraphia_model.h5"
    ]
    
    model_loaded = False
    for model_path in possible_paths:
        if os.path.exists(model_path):
            print(f"Found model at: {model_path}")
            try:
                model = tf.keras.models.load_model(model_path)
                print(f"Model loaded successfully from {model_path}")
                
                # Load feature scaler if available
                import pickle
                scaler_path = model_path.replace('.keras', '_scaler.pkl').replace('.h5', '_scaler.pkl')
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        feature_scaler = pickle.load(f)
                    print("Feature scaler loaded successfully")
                
                # Test model with dummy data
                dummy_image = np.random.random((1, 224, 224, 3))
                dummy_features = np.zeros((1, 15))  # Updated to 15 features
                test_pred = model.predict({"image_input": dummy_image, "feature_input": dummy_features})
                print(f"Model test successful. Output shape: {test_pred.shape}")
                model_loaded = True
                break
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                continue
    
    if not model_loaded:
        print("Warning: No dysgraphia model found.")
        print("Please train the model first using the fixed training script.")
        
except Exception as e:
    print(f"Error during model loading: {e}")
    print("Continuing without model - predictions will use dummy data")
    model = None

def preprocess_image(image_path):
    """Preprocess image for CNN input"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise

def predict_dysgraphia(image_path):
    """Fixed prediction function"""
    try:
        if model is None:
            print("Model not available, returning dummy prediction")
            return "Non-dysgraphic", 0.75

        # Preprocess image for CNN input
        processed_image = preprocess_image(image_path)

        # Load original image for feature extraction
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image for feature extraction")
        
        # Extract handwriting features
        features = extract_features(img)
        print(f"Extracted features: {features}")
        
        # Normalize features
        if feature_scaler is not None:
            features_array = np.array(features).reshape(1, -1)
            features_normalized = feature_scaler.transform(features_array)
        else:
            # Manual normalization if scaler not available
            features_normalized = normalize_features(np.array(features).reshape(1, -1))
        
        print(f"Normalized features shape: {features_normalized.shape}")
        print(f"Normalized features: {features_normalized}")

        # Make prediction
        prediction = model.predict({
            "image_input": processed_image,
            "feature_input": features_normalized
        }, verbose=0)

        raw_value = float(prediction[0][0])
        print(f"Raw prediction value: {raw_value}")

        # FIXED INTERPRETATION:
        # Model is trained with labels: dysgraphic=0, non-dysgraphic=1
        # So raw_value > 0.5 means non-dysgraphic, raw_value < 0.5 means dysgraphic
        
        if raw_value > 0.5:
            label = "Non-dysgraphic"
            confidence = raw_value
        else:
            label = "Dysgraphic" 
            confidence = 1.0 - raw_value

        print(f"Final prediction: {label} with confidence: {confidence}")
        
        return label, confidence

    except Exception as e:
        print(f"Error in predict_dysgraphia: {e}")
        traceback.print_exc()
        # Return a default prediction in case of error
        return "Error in prediction", 0.0

# Create missing static files
def create_missing_files():
    css_content = """
/* Basic styles for the application */
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f5f5f5;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

button {
    background-color: #007bff;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 4px;
    cursor: pointer;
}

button:hover {
    background-color: #0056b3;
}

.error {
    color: red;
    margin: 10px 0;
}

.success {
    color: green;
    margin: 10px 0;
}

.prediction-result {
    margin-top: 15px;
    padding: 10px;
    border-radius: 5px;
    font-weight: bold;
}

.dysgraphic {
    background-color: #ffebee;
    color: #c62828;
    border-left: 4px solid #c62828;
}

.non-dysgraphic {
    background-color: #e8f5e8;
    color: #2e7d32;
    border-left: 4px solid #2e7d32;
}
"""
    
    try:
        with open('static/css/styles.css', 'w') as f:
            f.write(css_content)
        print("Created basic styles.css file")
    except Exception as e:
        print(f"Could not create styles.css: {e}")

create_missing_files()

# Routes (keeping existing routes unchanged)
@app.route('/')
def index():
    try:
        return render_template('login.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"

@app.route('/parent-login')
def parent_login():
    try:
        return render_template('parentlogin.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"

@app.route('/teacher-login')
def teacher_login():
    try:
        return render_template('teacher.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"

@app.route('/counselor-login')
def counselor_login():
    try:
        return render_template('counsellorlogin.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"

@app.route('/dyscalculia-test')
def dyscalculia_test():
    try:
        return render_template('dyscaculiatest.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"

@app.route('/dyslexia-test')
def dyslexia_test():
    try:
        return render_template('dyslexiatest.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"

@app.route('/dysgraphia-test')
def dysgraphia_test():
    try:
        return render_template('dysgraphia_test.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"
    
@app.route('/memorygame')
def memory_game():
    try:
        return render_template('memorygame.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"

@app.route('/attentiongame')
def attention_game():
    try:
        return render_template('attentiongame.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"

@app.route('/languagegame')
def language_game():
    try:
        return render_template('languagegame.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"
    
@app.route('/processinggame')
def processing_game():
    try:
        return render_template('processinggame.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"

@app.route('/memorymaze')
def memory_maze():
    try:
        return render_template('memorymaze.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"

# FIXED PREDICTION ENDPOINT
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("=== Prediction Request Started ===")
        
        if 'image' not in request.files:
            print("No file in request")
            return jsonify({"error": "No file uploaded"}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            print("Empty filename")
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(image_file.filename):
            return jsonify({"error": "Invalid file type. Please upload an image."}), 400

        # Save the uploaded file
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(image_path)
        print(f"Image saved to {image_path}")
        
        # Make prediction
        label, confidence = predict_dysgraphia(image_path)
        
        # Clean up uploaded file
        try:
            os.remove(image_path)
        except:
            pass
        
        # Return improved result
        result = {
            "prediction": label,
            "confidence": round(float(confidence), 4),
            "interpretation": get_interpretation(label, confidence)
        }
        
        print(f"=== Final Result: {result} ===")
        return jsonify(result)
    
    except Exception as e:
        error_msg = f"Error in prediction: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

def get_interpretation(label, confidence):
    """Provide interpretation of the prediction"""
    if label == "Dysgraphic":
        if confidence > 0.8:
            return "Strong indication of dysgraphia. Consider professional assessment."
        elif confidence > 0.6:
            return "Moderate indication of dysgraphia. Monitoring recommended."
        else:
            return "Mild indication of dysgraphia. Further evaluation may be helpful."
    else:  # Non-dysgraphic
        if confidence > 0.8:
            return "Writing appears typical for age group."
        elif confidence > 0.6:
            return "Generally typical writing with minor concerns."
        else:
            return "Writing shows some areas for improvement but appears within normal range."

@app.route('/styles.css')
def serve_styles():
    try:
        with open('static/css/styles.css', 'r') as f:
            css_content = f.read()
        response = app.response_class(css_content, mimetype='text/css')
        return response
    except Exception as e:
        basic_css = """
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; }
        """
        return app.response_class(basic_css, mimetype='text/css')

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": feature_scaler is not None
    })

if __name__ == '__main__':
    print("Starting Flask application...")
    print(f"Model status: {'Loaded' if model else 'Not loaded'}")
    print(f"Feature scaler status: {'Loaded' if feature_scaler else 'Not loaded'}")
    app.run(debug=True, host='0.0.0.0', port=5000)