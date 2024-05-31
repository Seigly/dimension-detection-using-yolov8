import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
from torchvision.ops import nms
import numpy as np

# Load the custom YOLOv8 model
model = YOLO('D:/bestn.pt')
def find(image_path, model):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image {image_path}")
        return

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(image)

    # Extract bounding boxes,confidence scores,and class labels
    boxes = torch.tensor(results[0].boxes.xyxy).cpu()  # Bounding boxes
    scores = torch.tensor(results[0].boxes.conf).cpu()  # Confidence scores
    class_ids = torch.tensor(results[0].boxes.cls).cpu()  # Class IDs

    # Apply Non-Maximum Suppression (NMS)
    iou_threshold = 0.5  # Intersection over Union threshold for NMS
    nms_indices = nms(boxes, scores, iou_threshold)

    # Filter boxes, scores, and class_ids using NMS indices
    boxes = boxes[nms_indices].numpy()
    scores = scores[nms_indices].numpy()
    class_ids = class_ids[nms_indices].numpy()


    # Loop through the detections, draw them on the image, and print coordinates
    for box, score, class_id in zip(boxes, scores, class_ids):
        xmin, ymin, xmax, ymax = map(int, box)
        label = f'{model.names[int(class_id)]}: {score:.2f}'
        # Draw bounding box and label on the image
        cv2.rectangle(image_rgb, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(image_rgb, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        width = xmax - xmin
        height = ymax - ymin
        print(f"Object {model.names[int(class_id)]} detected with width: {width} and height: {height}")

    # Display the image with detections using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axes
    plt.show()

# Path to your image
image_path = 'img_3.png'

# Perform detection and visualize results
find(image_path, model)