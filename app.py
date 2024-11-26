from flask import Flask, request, jsonify, render_template, g
import torch
from model import CNN  # Import your model from model.py
from PIL import Image
import numpy as np
import os
import sqlite3
from database import save_image_prediction, predict_new_image, load_old_predictions, get_db  # Import database functions
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Ensure the Images directory exists
if not os.path.exists('Images'):
    os.makedirs('Images')

# Initialize Flask app
app = Flask(__name__)

# Load model
model = CNN()
model.load_state_dict(torch.load('digit_model.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

DATABASE = 'predictions.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # You can create a basic HTML page in the templates folder

# Route for favicon
@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content

# Predict route that handles POST requests
@app.route('/predict', methods=['POST'])
def predict():
    logging.debug("Received a prediction request")
    # Check if image is sent in the request
    if 'image' not in request.files:
        logging.error("No image provided in the request")
        return jsonify({'error': 'No image provided!'}), 400

    image_file = request.files['image']
    try:
        image = Image.open(image_file.stream).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))
        image = np.array(image).reshape(1, 1, 28, 28) / 255.0
        image = torch.tensor(image, dtype=torch.float32)
        image = (image - 0.5) / 0.5  # Normalize for the model

        # Predict using the loaded model
        with torch.no_grad():
            outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        confidence = torch.softmax(outputs, dim=1) * 100  # Get confidence as a percentage

        confidences = confidence[0].tolist()
        confidence_percentages = {str(i): confidences[i] for i in range(10)}

        # Save image and prediction to database
        image_file.seek(0)  # Reset file pointer to the beginning
        save_image_prediction(image_file, predicted.item(), get_db())

        logging.debug(f"Prediction: {predicted.item()}, Confidence: {confidence_percentages}")
        return jsonify({'digit': predicted.item(), 'confidence': confidence_percentages})
    except Exception as e:
        logging.error(f"Error processing the image: {str(e)}")
        return jsonify({'error': f"Error processing the image: {str(e)}"}), 500

# Route to load old predictions
@app.route('/predictions', methods=['GET'])
def predictions():
    try:
        db = get_db()
        old_predictions = load_old_predictions(db)
        predictions_data = [{'image_path': image.filename, 'prediction': prediction} for image, prediction in old_predictions]
        return jsonify(predictions_data)
    except Exception as e:
        logging.error(f"Error loading predictions: {str(e)}")
        return jsonify({'error': f"Error loading predictions: {str(e)}"}), 500

# Run the app
if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        logging.error(f"Failed to start the Flask application: {str(e)}")