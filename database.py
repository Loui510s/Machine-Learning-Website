import sqlite3
import os
from PIL import Image
import numpy as np
import torch
from model import CNN  # Import your model from model.py
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Ensure the Images directory exists
if not os.path.exists('Images'):
    os.makedirs('Images')

def get_db():
    db = sqlite3.connect('predictions.db')
    return db

def create_table():
    db = get_db()
    c = db.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY, image_path TEXT, prediction TEXT)''')
    db.commit()
    db.close()

def save_image_prediction(image_file, prediction, db):
    logging.debug(f"Saving image {image_file.filename} with prediction {prediction}")
    # Save image to the Images folder
    image_path = os.path.join('Images', image_file.filename)
    image_file.save(image_path)
    
    # Insert image path and prediction into database
    c = db.cursor()
    c.execute("INSERT INTO predictions (image_path, prediction) VALUES (?, ?)", (image_path, prediction))
    db.commit()
    logging.debug(f"Saved image {image_file.filename} with prediction {prediction} to database")

def predict_new_image(image_file, model, db):
    logging.debug(f"Predicting new image {image_file.filename}")
    # Save image to the Images folder
    image_path = os.path.join('Images', image_file.filename)
    image_file.save(image_path)
    
    # Load image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))
    image = np.array(image).reshape(1, 1, 28, 28) / 255.0
    image = torch.tensor(image, dtype=torch.float32)
    image = (image - 0.5) / 0.5  # Normalize for the model

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    
    # Save image and prediction to database
    save_image_prediction(image_file, predicted.item(), db)
    
    logging.debug(f"Prediction for image {image_file.filename}: {predicted.item()}")
    return predicted.item()

def load_old_predictions(db):
    logging.debug("Loading old predictions from database")
    # Retrieve all predictions from database
    c = db.cursor()
    c.execute("SELECT * FROM predictions")
    rows = c.fetchall()
    
    predictions = []
    for row in rows:
        image_path = row[1]
        prediction = row[2]
        
        # Load image from file path
        image = Image.open(image_path)
        
        predictions.append((image, prediction))
    
    logging.debug(f"Loaded {len(predictions)} predictions from database")
    return predictions

# Create the table if it doesn't exist
create_table()