# DETR (DEtection TRansformer) for Object Detection

## Model Description

This model is a pre-trained DETR model for object detection. It uses a Transformer architecture to predict bounding boxes and class labels for each object in an image. It was trained on the COCO dataset and is capable of detecting a wide variety of objects in real-world images.

## Model Details

- Model: `facebook/detr-resnet-50`
- Framework: PyTorch
- Task: Object Detection
- Input: Image (H, W, C)
- Output: Bounding boxes and class labels for detected objects
- License: MIT

## How to Use

```python
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
import torch

# Load the processor and model
processor = DetrImageProcessor.from_pretrained("your-username/detr-object-detection")
model = DetrForObjectDetection.from_pretrained("your-username/detr-object-detection")

# Prepare the image
image = Image.open("path_to_image.jpg")

# Process the image
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Post-process and display the results
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# Print and visualize detected objects
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box}")
