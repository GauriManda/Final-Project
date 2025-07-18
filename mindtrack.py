# -*- coding: utf-8 -*-
"""MINDTRACK - Fixed for Local Dataset

Fixed version to work with local dataset structure
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from scipy import stats
import pickle
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Feature extraction functions
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

    # More feature extraction can be added here

    # Create a fixed-length feature vector (15 features to match the model)
    feature_vector = [
        features['stroke_consistency'],
        features['letter_spacing'],
        features['alignment'],
        # Add more features as needed to reach desired length (padding with zeros)
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]

    return feature_vector[:15]  # Return first 15 features to match model expectation

def normalize_features(features):
    """Normalize features using z-score normalization"""
    try:
        # Convert to numpy array if not already
        features_array = np.array(features)
        
        # If it's a single sample (1D), make it 2D
        if features_array.ndim == 1:
            features_array = features_array.reshape(1, -1)
        
        # Apply z-score normalization
        normalized = stats.zscore(features_array, axis=0, nan_policy='omit')
        
        # Replace any NaN values with 0
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        return normalized
    except Exception as e:
        print(f"Error normalizing features: {e}")
        # Return original features if normalization fails
        return np.array(features).reshape(1, -1) if np.array(features).ndim == 1 else np.array(features)

def preprocess_images(data_dir, img_size):
    """Process all images and extract features"""
    processed_dir = os.path.join(data_dir, 'processed')
    features_file = os.path.join(data_dir, 'handwriting_features.pkl')

    # Create processed directory if it doesn't exist
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

        # Process each image in the dataset
        features_dict = {}

        # For each class directory
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)

            # Skip if not a directory or is the processed directory
            if not os.path.isdir(class_dir) or class_name == 'processed':
                continue

            # Create class directory in processed folder
            processed_class_dir = os.path.join(processed_dir, class_name)
            if not os.path.exists(processed_class_dir):
                os.makedirs(processed_class_dir)

            # Process each image in the class directory
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)

                    # Read and resize image
                    img = cv2.imread(img_path)
                    if img is not None:
                        resized_img = cv2.resize(img, img_size)

                        # Save processed image
                        processed_img_path = os.path.join(processed_class_dir, img_name)
                        cv2.imwrite(processed_img_path, resized_img)

                        # Extract features
                        features = extract_features(resized_img)

                        # Store features with relative path as key
                        feature_key = os.path.join(class_name, img_name)
                        features_dict[feature_key] = features

        # Save features dictionary
        with open(features_file, 'wb') as f:
            pickle.dump(features_dict, f)

        print(f"Processed images and extracted features for {len(features_dict)} images")

    else:
        # Load existing features
        if os.path.exists(features_file):
            with open(features_file, 'rb') as f:
                features_dict = pickle.load(f)
            print(f"Loaded existing features for {len(features_dict)} images")
        else:
            features_dict = {}
            print("No existing features found")

    return processed_dir, features_file

# Create CustomDataGenerator for combined features
class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, directory=None, batch_size=32, img_size=(224, 224),
                 features_dict=None, class_indices=None, class_mode='binary',
                 subset=None, image_paths=None, labels=None, **kwargs):
        # Call parent class init
        super().__init__(**kwargs)

        self.directory = directory
        self.batch_size = batch_size
        self.img_size = img_size
        self.features_dict = features_dict or {}
        self.class_indices = class_indices
        self.class_mode = class_mode
        self.image_paths = []
        self.labels = []

        # If image_paths and labels are provided directly, use them
        if image_paths is not None and labels is not None:
            self.image_paths = np.array(image_paths)
            self.labels = np.array(labels)
        # Otherwise, scan directory for images
        elif directory and os.path.exists(directory):
            for class_name in os.listdir(directory):
                class_path = os.path.join(directory, class_name)
                if os.path.isdir(class_path):
                    class_label = self.class_indices[class_name]
                    for img_name in os.listdir(class_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_path, img_name)
                            feature_key = os.path.join(class_name, img_name)

                            # Only add images with extracted features if features_dict provided
                            if not features_dict or feature_key in features_dict:
                                self.image_paths.append(img_path)
                                self.labels.append(class_label)

            # Convert to numpy arrays
            self.image_paths = np.array(self.image_paths)
            self.labels = np.array(self.labels)

        # Print debug info
        print(f"Created generator with {len(self.image_paths)} images")

        # Shuffle data
        if len(self.image_paths) > 0:
            indices = np.arange(len(self.image_paths))
            np.random.shuffle(indices)
            self.image_paths = self.image_paths[indices]
            self.labels = self.labels[indices]

        # Split into training and validation if needed
        if subset and len(self.image_paths) > 0:
            split_idx = int(len(self.image_paths) * 0.8)  # 80% for training
            if subset == 'training':
                self.image_paths = self.image_paths[:split_idx]
                self.labels = self.labels[:split_idx]
            elif subset == 'validation':
                self.image_paths = self.image_paths[split_idx:]
                self.labels = self.labels[split_idx:]

    def __len__(self):
        """Return the number of batches per epoch"""
        return max(1, int(np.ceil(len(self.image_paths) / self.batch_size)))

    @property
    def num_batches(self):
        """Required property for Keras training"""
        return self.__len__()

    def __getitem__(self, idx):
        # Safety check for empty batches
        if len(self.image_paths) == 0:
            # Return empty batch with correct shapes
            empty_images = np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.float32)
            empty_features = np.zeros((1, 15), dtype=np.float32)  # Changed to 15 features
            empty_labels = np.zeros(1, dtype=np.int32)
            return {"image_input": empty_images, "feature_input": empty_features}, empty_labels

        # Get batch indices with bounds checking
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.image_paths))
        batch_paths = self.image_paths[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]

        # Initialize batch arrays
        batch_size = len(batch_paths)
        batch_images = np.zeros((batch_size, self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        batch_features = np.zeros((batch_size, 15), dtype=np.float32)  # Changed to 15 features

        for i, img_path in enumerate(batch_paths):
            try:
                # Load and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not load image {img_path}")
                    img = np.zeros((self.img_size[0], self.img_size[1], 3))
                img = cv2.resize(img, self.img_size)
                img = img / 255.0  # Normalize
                batch_images[i] = img

                # Get features if available
                if self.features_dict:
                    rel_path = os.path.relpath(img_path, self.directory) if self.directory else os.path.basename(img_path)
                    feature_key = os.path.join(os.path.dirname(rel_path), os.path.basename(rel_path))
                    batch_features[i] = np.array(self.features_dict.get(feature_key, np.zeros(15)))
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                # Use zeros for this sample
                batch_images[i] = np.zeros((self.img_size[0], self.img_size[1], 3))
                batch_features[i] = np.zeros(15)

        # Normalize features if we have any
        if batch_size > 0 and np.any(batch_features):
            try:
                batch_features = normalize_features(batch_features)
            except Exception as e:
                print(f"Error normalizing features: {e}")

        # Return in the format TensorFlow expects
        return {"image_input": batch_images, "feature_input": batch_features}, batch_labels

    @property
    def classes(self):
        return self.labels

    def on_epoch_end(self):
        """Method called at the end of every epoch"""
        indices = np.arange(len(self.image_paths))
        np.random.shuffle(indices)
        self.image_paths = self.image_paths[indices]
        self.labels = self.labels[indices]

def build_model(img_size=(224, 224), num_features=15):  # Changed to 15 features
    """Build a model that combines CNN features with handwriting features"""
    # Input for images
    img_input = Input(shape=(img_size[0], img_size[1], 3), name='image_input')

    # Use MobileNetV2 for image feature extraction
    base_model = MobileNetV2(input_tensor=img_input, include_top=False, weights='imagenet')

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Get CNN features
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Input for handwriting features
    feature_input = Input(shape=(num_features,), name='feature_input')

    # Process handwriting features
    y = Dense(32, activation='relu')(feature_input)
    y = Dropout(0.3)(y)

    # Combine both feature sets
    combined = Concatenate()([x, y])

    # Output layer
    output = Dense(1, activation='sigmoid')(combined)

    # Create and compile model
    model = Model(inputs=[img_input, feature_input], outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(model, data_dir, epochs=10, batch_size=32, img_size=(224, 224)):
    """Train the model using data generator"""
    # Process data directory to get features
    processed_dir, features_file = preprocess_images(data_dir, img_size)

    # Load features dictionary
    features_dict = {}
    if os.path.exists(features_file):
        with open(features_file, 'rb') as f:
            features_dict = pickle.load(f)
        print(f"Loaded features for {len(features_dict)} images")

    # Get class names
    class_names = sorted(os.listdir(processed_dir))
    class_indices = {name: idx for idx, name in enumerate(class_names)}

    # Create training and validation generators
    train_gen = CustomDataGenerator(
        directory=processed_dir,
        batch_size=batch_size,
        img_size=img_size,
        features_dict=features_dict,
        class_indices=class_indices,
        subset='training'
    )

    val_gen = CustomDataGenerator(
        directory=processed_dir,
        batch_size=batch_size,
        img_size=img_size,
        features_dict=features_dict,
        class_indices=class_indices,
        subset='validation'
    )

    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]

    # Train the model
    print(f"Starting training with {len(train_gen)} training batches and {len(val_gen)} validation batches")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks
    )

    return model, history

def predict_single_image(model, image_path, img_size=(224, 224)):
    """Predict dysgraphia for a single image"""
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
        
    img = cv2.resize(img, img_size)
    img_normalized = img / 255.0

    # Extract features
    features = extract_features(img)

    # Normalize features
    features_normalized = normalize_features(np.array(features).reshape(1, -1))

    # Prepare input for model
    img_input = np.expand_dims(img_normalized, axis=0)

    # Make prediction
    prediction = model.predict({"image_input": img_input, "feature_input": features_normalized})

    # Get binary prediction
    has_dysgraphia = bool(prediction[0][0] > 0.5)

    # Return result
    result = {
        "has_dysgraphia": has_dysgraphia,
        "confidence": float(prediction[0][0]),
        "features": features_normalized.flatten().tolist()
    }

    return result

# Main execution - UPDATED FOR LOCAL DATASET
# Replace the entire "if __name__ == "__main__":" section in mindtrack.py with this:

if __name__ == "__main__":
    import os
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    
    print(f"Looking for data at: {data_dir}")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        print("Please ensure your data folder structure is:")
        print("./data/")
        print("  ├── dysgraphic/")
        print("  │   ├── image1.jpg")
        print("  │   └── image2.jpg")
        print("  └── non-dysgraphic/")
        print("      ├── image3.jpg")
        print("      └── image4.jpg")
        exit(1)
    
    # Verify class folders exist
    expected_classes = ['dysgraphic', 'non-dysgraphic']
    for class_name in expected_classes:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            print(f"Error: Class folder '{class_path}' not found!")
            exit(1)
        
        # Count images in each class
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(images)} images in {class_name} folder")
        
        if len(images) == 0:
            print(f"Warning: No images found in {class_path}")
    
    img_size = (224, 224)
    batch_size = 16  # Smaller batch size if dataset is small
    epochs = 20

    # Build enhanced model
    model = build_model(img_size=img_size, num_features=15)
    print("Enhanced model built successfully")

    # Train model
    model, history = train_model(model, data_dir, epochs=epochs, batch_size=batch_size, img_size=img_size)
    print("Training complete")

    # Save model
    model.save('dysgraphia_model.keras')
    print("Model saved as 'dysgraphia_model.keras'")

# Test prediction function
def test_prediction(model_path='dysgraphia_model.keras', test_image_path=None):
    """Test the model with a single image"""
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found!")
        return
    
    model = keras.models.load_model(model_path)
    
    if test_image_path is None:
        # Try to find a test image
        test_dirs = ['./data/dysgraphic', './data/non-dysgraphic']
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if images:
                    test_image_path = os.path.join(test_dir, images[0])
                    break
    
    if test_image_path and os.path.exists(test_image_path):
        print(f"Testing with image: {test_image_path}")
        result = predict_single_image(model, test_image_path)
        if result:
            print(f"Prediction: {'Dysgraphic' if result['has_dysgraphia'] else 'Non-dysgraphic'}")
            print(f"Confidence: {result['confidence']:.4f}")
    else:
        print("No test image found or specified")