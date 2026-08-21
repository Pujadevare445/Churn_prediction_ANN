import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Path to your pickled/saved model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ANN.pkl')

# Load the model
try:
    # tf.keras safely loads Keras 3 / h5 serialized models stored inside pkl
    model = tf.keras.models.load_model(MODEL_PATH)
    print("ANN Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Feature definitions for the UI
FEATURE_NAMES = [f"Feature {i+1}" for i in range(10)]

@app.route('/')
def home():
    return render_template('index.html', features=FEATURE_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model failed to load on server.'}), 500

    try:
        # Parse inputs from the form request
        data = request.form
        input_data = []
        
        for name in FEATURE_NAMES:
            val = float(data.get(name, 0.0))
            input_data.append(val)
        
        # Reshape to (1, 10) for model prediction
        input_array = np.array([input_data], dtype=np.float32)
        
        # Run prediction
        raw_prediction = model.predict(input_array)[0][0]
        probability = float(raw_prediction)
        predicted_class = 1 if probability >= 0.5 else 0
        confidence = probability if predicted_class == 1 else (1.0 - probability)

        return render_template(
            'index.html',
            features=FEATURE_NAMES,
            prediction_text=f"Class {predicted_class}",
            probability=f"{probability * 100:.2f}%",
            confidence=f"{confidence * 100:.2f}%",
            inputs=data
        )

    except Exception as e:
        return render_template('index.html', features=FEATURE_NAMES, error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
