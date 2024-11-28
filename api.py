from fastapi import FastAPI, UploadFile
import torch
from torchvision import transforms
from PIL import Image
from model import YourModelClass  # Import your model class

# Initialize FastAPI
app = FastAPI()

# Load the trained model
model = YourModelClass()
model.load_state_dict(torch.load("digit_model.pth", map_location=torch.device('cpu')))
model.eval()

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.post("/predict/")
async def predict_digit(file: UploadFile):
    """
    Predict the digit from the uploaded image.
    """
    # Load the image
    image = Image.open(file.file)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1).item()

    return {"prediction": prediction}
