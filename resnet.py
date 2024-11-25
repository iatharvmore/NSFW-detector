import os
import requests
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import validators
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, filename='image_processing.log', filemode='w')
logger = logging.getLogger()

# Paths
data_csv_path = 'Database.csv'
image_dir = "reddit_images"
os.makedirs(image_dir, exist_ok=True)

# Load the data
data = pd.read_csv(data_csv_path)
data = data.head(1299)  # Adjust if needed

# Function to download images with retries
def download_with_retry(url, target_path, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                with open(target_path, 'wb') as f:
                    f.write(response.content)
                # Verify the image
                with Image.open(target_path) as img:
                    img.verify()
                return True
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}: Failed to download {url}. Error: {e}")
    logger.error(f"Failed to download after {retries} attempts: {url}")
    return False

# Download images
def download_images(df, target_dir):
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        image_url = row['Imagelink']
        image_path = os.path.join(target_dir, f"{idx}.jpg")
        if os.path.exists(image_path):
            continue  # Skip already downloaded images
        if not validators.url(image_url):
            logger.warning(f"Invalid URL: {image_url}")
            continue
        if not download_with_retry(image_url, image_path):
            logger.error(f"Failed to download: {image_url}")

# Validate dataset
def validate_dataset(df, image_dir):
    valid_indices = []
    for idx, row in df.iterrows():
        image_path = os.path.join(image_dir, f"{idx}.jpg")
        if os.path.exists(image_path):
            try:
                with Image.open(image_path) as img:
                    img.verify()
                valid_indices.append(idx)
            except Exception as e:
                logger.warning(f"Corrupted image {image_path}: {e}")
                os.remove(image_path)  # Remove corrupted images
        else:
            logger.warning(f"Missing image {image_path}")
    return df.loc[valid_indices].reset_index(drop=True)

# Download and validate images
download_images(data, image_dir)
data = validate_dataset(data, image_dir)

# Dataset class
class RedditMemesDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_dir, f"{row.name}.jpg")
        label = int(row['Nsfw'])

        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            image = torch.zeros(3, 224, 224)  # Dummy image
        return image, label

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet normalization
])

# Split data
train_df, val_df = train_test_split(data, test_size=0.2)
train_dataset = RedditMemesDataset(train_df, image_dir, transform)
val_dataset = RedditMemesDataset(val_df, image_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)

# Training parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=5):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Train Accuracy: {100. * correct / total:.2f}%, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, "
              f"Val Accuracy: {100. * val_correct / val_total:.2f}%")

# Train the model
train_model(model, criterion, optimizer, train_loader, val_loader, epochs=5)

# Save the model
model_path = "nsfw_classifier.pkl"
torch.save(model.state_dict(), model_path)
print(f"Model saved at {model_path}")
