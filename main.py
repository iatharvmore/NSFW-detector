import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load the trained model
model_path = "nsfw_classifier.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# Use `weights_only=True` for secure loading
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()

# Transform for preprocessing the uploaded image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet normalization
])

# Function to predict NSFW status
def classify_image(image_path):
    try:
        # Open and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            outputs = model(image.to(device))
            _, predicted = outputs.max(1)

        # Decode prediction
        classes = ["Safe for Work", "Not Safe for Work (NSFW)"]
        return classes[predicted.item()]
    except Exception as e:
        return f"Error processing image: {e}"

# Gradio interface
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath"),  # Use `filepath` to pass image path to the function
    outputs="text",
    title="NSFW Image Classifier",
    description="Upload an image to classify whether it is Safe for Work (SFW) or Not Safe for Work (NSFW).",
    examples=[
        ["example1.jpg"],
        ["example2.jpg"]
    ]
)

# Launch the app
interface.launch()
