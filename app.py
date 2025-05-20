import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# Load YOLOv8 model
model = YOLO("models/best.pt")  # Replace with your custom trained model

st.title("🚦 Traffic Violation Detection System")
st.markdown("Real-time helmet and seatbelt monitoring using YOLOv8 and OpenCV")

# File uploader
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

def detect_image(image_np):
    # Run detection with lower confidence threshold
    results = model(image_np, conf=0.20)
    
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return None, results
    else:
        return results[0].plot(), results

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type in ['jpg', 'jpeg', 'png']:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        output_img, results = detect_image(image_np)

        if output_img is not None:
            st.success(f"✅ {len(results[0].boxes)} object(s) detected.")
            st.image(output_img, caption='Detection Result', use_column_width=True)
        else:
            st.warning("⚠️ No helmet or seatbelt detected in the image.")

    elif file_type == 'mp4':
        # Save the uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img, results = detect_image(frame_rgb)

            if output_img is not None:
                stframe.image(output_img, channels="RGB", use_column_width=True)
            else:
                stframe.image(frame_rgb, caption="No detection", channels="RGB", use_column_width=True)

        cap.release()
        os.remove(video_path)  # Clean up after use
