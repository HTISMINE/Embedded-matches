from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("/best.pt")

# Run inference on 'bus.jpg' with arguments

model.predict("5.jpg",save=True)
