import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# ----------------------
# Load YOLOv8 model
# ----------------------
# 使用官方 YOLOv8n 模型，可以替换成你训练的 custom model e.g. "best.pt"
model = YOLO("yolov8n.pt")

st.title("Vehicle Detection App")
st.write("Upload an image and the app will detect if it contains a vehicle (car, truck, bus, motorcycle).")

# ----------------------
# File uploader
# ----------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # ----------------------
    # YOLOv8 inference
    # ----------------------
    results = model.predict(source=image_cv, imgsz=640)

    # Annotate image with bounding boxes
    annotated_image = results[0].plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    st.image(annotated_image, caption="Detection Result", use_column_width=True)

    # ----------------------
    # Check if any vehicle is detected
    # ----------------------
    # YOLOv8 COCO class names that are vehicles
    vehicle_classes = ["car", "truck", "bus", "motorcycle"]

    detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]  # list of detected class names

    if any(cls in vehicle_classes for cls in detected_classes):
        st.success("Vehicle detected in the image!")
    else:
        st.warning("No vehicle detected in the image.")
