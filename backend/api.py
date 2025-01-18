from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model('gesture_model.keras')

# Load the scaler used during training
scaler = StandardScaler()
scaler.mean_ = np.load('scaler_mean.npy')
scaler.scale_ = np.load('scaler_scale.npy')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['vector']
        print(f"Received vector: {data}")

        vector = np.array(data).reshape(1, -1)

        # Normalize the vector using the scaler
        vector = scaler.transform(vector)

        # Predict gesture
        prediction = model.predict(vector)
        gesture_class = np.argmax(prediction)  # Get the class with highest probability

        if gesture_class == 0:
            gesture = "No Gesture Detected"
        elif gesture_class == 1:
            gesture = "One Finger Gesture"
        elif gesture_class == 2:
            gesture = "Two Fingers Gesture"

        return jsonify({'gesture': gesture})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)










