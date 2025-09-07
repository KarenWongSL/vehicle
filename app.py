import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# ----------------------
# Load YOLOv8 model
# ----------------------
# 可替换为你自己的训练模型，如 "best.pt"
model = YOLO("yolov8n.pt")

st.title("Vehicle Detection App")
st.write("Upload an image and the app will detect if it contains a vehicle (car, truck, bus, motorcycle).")

# ----------------------
# File uploader
# ----------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to numpy array
    image_np = np.array(image)

    # ----------------------
    # YOLOv8 inference
    # ----------------------
    results = model.predict(source=image_np, imgsz=640, device='cpu')

    # Annotated image (YOLOv8 returns numpy array)
    annotated_image = results[0].plot()
    annotated_image_pil = Image.fromarray(annotated_image)
    st.image(annotated_image_pil, caption="Detection Result", use_column_width=True)

    # ----------------------
    # Check for vehicles
    # ----------------------
    vehicle_classes = ["car", "truck", "bus", "motorcycle"]
    detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]

    if any(cls in vehicle_classes for cls in detected_classes):
        st.success("Vehicle detected in the image!")
    else:
        st.warning("No vehicle detected in the image.")
