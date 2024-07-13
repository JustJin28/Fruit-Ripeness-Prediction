from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)
CORS(app)

# Load models
models_dir = os.path.join(os.getcwd(), 'models')
cnn_model = load_model(os.path.join(models_dir, 'cnn_model.keras'))
cnn_model_v2 = load_model(os.path.join(models_dir, 'cnn_model_v2.keras'))
mlp_model = load_model(os.path.join(models_dir, 'mlp_model.keras'))
mlp_model_v2 = load_model(os.path.join(models_dir, 'mlp_model_v2.keras'))
rf_model = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
meta_model = joblib.load(os.path.join(models_dir, 'meta_model.pkl'))

def preprocess_image(image):
    img_resized = cv2.resize(image, (224, 224))
    img_flattened = img_resized.flatten().reshape(1, -1)
    return img_resized, img_flattened

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file.save('temp.jpg')
    img = cv2.imread('temp.jpg')
    img_resized, img_flattened = preprocess_image(img)

    # Generate predictions from each model
    predictions_stack = np.zeros((1, 5))
    predictions_stack[0, 0] = cnn_model.predict(np.array([img_resized])).flatten()[0]
    predictions_stack[0, 1] = cnn_model_v2.predict(np.array([img_resized])).flatten()[0]
    predictions_stack[0, 2] = rf_model.predict(img_flattened).flatten()[0]
    predictions_stack[0, 3] = mlp_model.predict(img_flattened).flatten()[0]
    predictions_stack[0, 4] = mlp_model_v2.predict(img_flattened).flatten()[0]

    # Meta-model prediction
    final_prediction = meta_model.predict(predictions_stack)

    return jsonify({'prediction': int(final_prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)