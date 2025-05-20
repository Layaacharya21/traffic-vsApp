from ultralytics import YOLO

# Load the uploaded model
model = YOLO("models/best.pt")

# Print class labels
print(model.names)
