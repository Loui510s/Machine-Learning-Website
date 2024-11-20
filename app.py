from flask import Flask, request, jsonify, render_template
import torch
from model import CNN  # Import your CNN model class (make sure model.py is in the same directory)
from PIL import Image
import numpy as np
import io

# Initialize Flask app
app = Flask(__name__)

# Load model
model = CNN()
model.load_state_dict(torch.load('digit_model.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

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
    # Check if image is sent in the request
    if 'image' not in request.files:
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

        return jsonify({'digit': predicted.item(), 'confidence': confidence[0][predicted].item()})
    
    except Exception as e:
        return jsonify({'error': f"Error processing the image: {str(e)}"}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
