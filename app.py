import os
import io
import numpy as np
import h5py
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ANN.pkl')
FEATURE_NAMES = [f"Feature {i+1}" for i in range(10)]

class LightweightANN:
    """Pure NumPy inference engine extracting weights directly from HDF5 binary inside PKL."""
    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = f.read()

        # Find starting offset of HDF5 binary signature (\x89HDF)
        h5_offset = data.find(b'\x89HDF\r\n\x1a\n')
        if h5_offset == -1:
            raise ValueError("HDF5 binary payload missing from PKL file.")

        # Read layer weights into NumPy arrays
        h5_bytes = io.BytesIO(data[h5_offset:])
        with h5py.File(h5_bytes, 'r') as h5f:
            vars_grp = h5f['vars']
            
            # Dense Layer 1: 10 inputs -> 8 hidden nodes (ReLU)
            self.w0 = np.array(vars_grp['0']['0'])
            self.b0 = np.array(vars_grp['0']['1'])
            
            # Dense Layer 2: 8 hidden nodes -> 7 hidden nodes (ReLU)
            self.w1 = np.array(vars_grp['1']['0'])
            self.b1 = np.array(vars_grp['1']['1'])
            
            # Dense Layer 3: 7 hidden nodes -> 1 output node (Sigmoid)
            self.w2 = np.array(vars_grp['2']['0'])
            self.b2 = np.array(vars_grp['2']['1'])

    def predict(self, x):
        """Forward pass matching original Keras architecture."""
        # Layer 1: Dense + ReLU activation
        h1 = np.maximum(0, np.dot(x, self.w0) + self.b0)
        # Layer 2: Dense + ReLU activation
        h2 = np.maximum(0, np.dot(h1, self.w1) + self.b1)
        # Layer 3: Dense + Sigmoid activation
        z3 = np.dot(h2, self.w2) + self.b2
        out = 1.0 / (1.0 + np.exp(-z3))
        return float(out[0][0])

# Initialize model on server start
try:
    model = LightweightANN(MODEL_PATH)
    print("Lightweight ANN loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html', features=FEATURE_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model failed to load on server.'}), 500

    try:
        data = request.form
        input_data = [float(data.get(name, 0.0)) for name in FEATURE_NAMES]
        
        # Reshape to (1, 10) array
        input_array = np.array([input_data], dtype=np.float32)
        
        # Forward pass inference
        probability = model.predict(input_array)
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
