import torch
import cv2
import numpy as np
from PIL import Image

# Load the YOLOv5 model (update the path to your trained model)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True)

def detect_image(image_np):
    """
    Detect objects in an image using YOLOv5.

    Args:
        image_np (numpy.ndarray): Image as NumPy array (BGR or RGB)

    Returns:
        numpy.ndarray: Annotated image with detections
    """
    # Ensure RGB format
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    results = model(image_np)
    detected_img = np.squeeze(results.render())  # Rendered image (with bounding boxes)
    return detected_img

def detect_video(video_path, output_path='output.mp4'):
    """
    Process a video and detect helmet/seatbelt violations.

    Args:
        video_path (str): Input video path
        output_path (str): Output video file with detections

    Returns:
        str: Path to the output video
    """
    cap = cv2.VideoCapture(video_path)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result_frame = detect_image(frame)
        out.write(result_frame)

    cap.release()
    out.release()
    return output_path
