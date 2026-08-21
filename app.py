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
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Model file not found at {pkl_path}")

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
            self.w0 = np.array(vars_grp['0']['0'], dtype=np.float32)
            self.b0 = np.array(vars_grp['0']['1'], dtype=np.float32)
            
            # Dense Layer 2: 8 hidden nodes -> 7 hidden nodes (ReLU)
            self.w1 = np.array(vars_grp['1']['0'], dtype=np.float32)
            self.b1 = np.array(vars_grp['1']['1'], dtype=np.float32)
            
            # Dense Layer 3: 7 hidden nodes -> 1 output node (Sigmoid)
            self.w2 = np.array(vars_grp['2']['0'], dtype=np.float32)
            self.b2 = np.array(vars_grp['2']['1'], dtype=np.float32)

    def predict(self, x):
        """Forward pass with numerically stable activation functions."""
        # Layer 1: Dense + ReLU activation
        h1 = np.maximum(0, np.dot(x, self.w0) + self.b0)
        # Layer 2: Dense + ReLU activation
        h2 = np.maximum(0, np.dot(h1, self.w1) + self.b1)
        # Layer 3: Dense + Numerically Stable Sigmoid activation
        z3 = np.dot(h2, self.w2) + self.b2
        
        # Clip z3 values to prevent floating point overflow in exp()
        z3_clipped = np.clip(z3, -500, 500)
        out = 1.0 / (1.0 + np.exp(-z3_clipped))
        return float(out[0][0])

# Initialize model on startup
model = None
model_error_msg = None

try:
    model = LightweightANN(MODEL_PATH)
    print("Lightweight ANN loaded successfully.")
except Exception as e:
    model_error_msg = str(e)
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html', features=FEATURE_NAMES, model_error=model_error_msg)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template(
            'index.html', 
            features=FEATURE_NAMES, 
            error=f"Model is not initialized: {model_error_msg}"
        ), 500

    try:
        data = request.form
        input_data = []

        # Parse inputs with fallback values to prevent KeyError
        for name in FEATURE_NAMES:
            raw_val = data.get(name, "0.0")
            try:
                val = float(raw_val) if raw_val.strip() != "" else 0.0
            except ValueError:
                val = 0.0
            input_data.append(val)
        
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
        # Catch internal processing errors gracefully without crashing to 500 page
        return render_template(
            'index.html', 
            features=FEATURE_NAMES, 
            error=f"Prediction error: {str(e)}"
        )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
