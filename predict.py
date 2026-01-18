"""
Traffic Sign Classification - Prediction API
============================================

Flask API for serving traffic sign classification predictions.

Usage:
    python predict.py

API Endpoints:
    GET  /              - Health check
    POST /predict       - Predict traffic sign from uploaded image
    GET  /classes       - Get list of all traffic sign classes
"""

import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
MODEL_PATH = 'model/traffic_sign_classifier.h5'
IMG_SIZE = 48

# Class names mapping (GTSRB dataset)
CLASS_NAMES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# Global model variable
model = None


def load_model():
    """Load the trained model"""
    global model
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
    
    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully!")
    print(f"   Model input shape: {model.input_shape}")
    print(f"   Model output shape: {model.output_shape}")
    
    return model


def preprocess_image(image):
    """
    Preprocess image for prediction
    
    Args:
        image: PIL Image object
    
    Returns:
        Preprocessed numpy array
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to numpy array and normalize
    img_array = np.array(image)
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'Traffic Sign Classification API',
        'model_loaded': model is not None,
        'version': '1.0.0'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict traffic sign from uploaded image
    
    Expected: multipart/form-data with 'file' field containing image
    
    Returns:
        JSON with prediction results
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 500
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'message': 'Please upload an image file using the "file" field'
            }), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({
                'error': 'Empty filename',
                'message': 'Please select a file'
            }), 400
        
        # Read and preprocess image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get top 5 predictions
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        
        top_5_predictions = [
            {
                'class_id': int(idx),
                'class_name': CLASS_NAMES[int(idx)],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_5_indices
        ]
        
        # Get best prediction
        best_class_id = int(np.argmax(predictions[0]))
        best_confidence = float(np.max(predictions[0]))
        
        return jsonify({
            'success': True,
            'prediction': {
                'class_id': best_class_id,
                'class_name': CLASS_NAMES[best_class_id],
                'confidence': best_confidence
            },
            'top_5_predictions': top_5_predictions
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of all traffic sign classes"""
    return jsonify({
        'num_classes': len(CLASS_NAMES),
        'classes': CLASS_NAMES
    })


@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'num_classes': len(CLASS_NAMES),
        'input_size': f'{IMG_SIZE}x{IMG_SIZE}'
    })


if __name__ == '__main__':
    # Load model at startup
    try:
        load_model()
    except Exception as e:
        print(f"⚠️  Warning: Could not load model: {e}")
        print(f"   The API will start but predictions will fail.")
        print(f"   Please train the model first using: python train.py")
    
    # Run Flask app
    print(f"\n{'='*50}")
    print(f"🚀 Starting Traffic Sign Classification API")
    print(f"{'='*50}")
    print(f"📍 Server running on: http://localhost:5000")
    print(f"📝 Endpoints:")
    print(f"   GET  /         - Health check")
    print(f"   POST /predict  - Predict traffic sign")
    print(f"   GET  /classes  - List all classes")
    print(f"   GET  /health   - Detailed health check")
    print(f"{'='*50}\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)