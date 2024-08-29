import cv2
from ultralytics import YOLO
import torch
from torchvision.ops import nms
import numpy as np

# Load the custom YOLOv8 model
model = YOLO('D:/bestn.pt')

def dim(model):
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break


        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        results = model(image_rgb)

        boxes = results[0].boxes.xyxy.clone().detach()
        scores = results[0].boxes.conf.clone().detach()
        class_ids = results[0].boxes.cls.clone().detach()


        iou_threshold = 0.5
        nms_indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold)


        boxes = boxes[nms_indices].numpy()
        scores = scores[nms_indices].numpy()
        class_ids = class_ids[nms_indices].numpy()


        for box, score, class_id in zip(boxes, scores, class_ids):
            xmin, ymin, xmax, ymax = map(int, box)
            label = f'{model.names[int(class_id)]}: {score:.2f}'

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            width = xmax - xmin
            height = ymax - ymin
            print(f"Object {model.names[int(class_id)]} detected with width: {width} and height: {height}")


        cv2.imshow('Webcam', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


dim(model)