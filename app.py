import os
import io
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ANN.pkl')
FEATURE_NAMES = [f"Feature {i+1}" for i in range(10)]

session = None

def load_onnx_model():
    """Converts Keras model to ONNX runtime session to minimize memory footprint."""
    global session
    import tf2onnx
    import tensorflow as tf
    import onnxruntime as ort

    try:
        # Load Keras model from pkl file
        keras_model = tf.keras.models.load_model(MODEL_PATH)
        
        # Define input signature matching model config (10 features)
        input_signature = [tf.TensorSpec([None, 10], tf.float32, name='input')]
        
        # Convert to ONNX format in-memory
        onnx_model, _ = tf2onnx.convert.from_keras(
            keras_model, 
            input_signature=input_signature
        )
        
        # Create lightweight ONNX Runtime session
        session = ort.InferenceSession(onnx_model.SerializeToString())
        print("Model converted to ONNX and loaded successfully.")
    except Exception as e:
        print(f"Error initializing model: {e}")
        session = None

# Initialize model
load_onnx_model()

@app.route('/')
def home():
    return render_template('index.html', features=FEATURE_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    if session is None:
        return jsonify({'error': 'Model failed to load on server.'}), 500

    try:
        data = request.form
        input_data = [float(data.get(name, 0.0)) for name in FEATURE_NAMES]
        
        # Prepare input array (1, 10) float32
        input_array = np.array([input_data], dtype=np.float32)
        
        # Perform inference via ONNX Runtime
        input_name = session.get_inputs()[0].name
        raw_prediction = session.run(None, {input_name: input_array})[0][0][0]
        
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
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
