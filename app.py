import streamlit as st
from PIL import Image, ImageDraw
import torch
from torchvision.transforms import functional as F
from pathlib import Path
from torchvision import transforms
import numpy as np

# Add yolov5 directory to the system path
import sys
sys.path.append("yolov5")

from models.experimental import attempt_load
from utils.general import non_max_suppression

# Load YOLOv5 model
weights_path = "traffic_sign_detection_yolo.pth"  # Path to your trained weights
model = attempt_load(weights_path, map_location=torch.device('cuda'))  # Change map_location if you're using GPU


transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert numpy array to PIL Image
    transforms.Resize((40, 40)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

# Streamlit app
st.title("YOLOv5 Object Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    img_tensor = transform(np.array(image))

    # Perform inference
    with torch.no_grad():
        results = model(img_tensor)

    # Post-process the detections
    detections = non_max_suppression(results.pred[0], conf_thres=0.5, iou_thres=0.5)

    # Draw bounding boxes and show class names
    if detections[0] is not None:
        for det in detections[0]:
            bbox = det[:4].tolist()
            conf = det[4].item()
            label = int(det[5].item())  # Class label
            class_name = model.names[label]
            st.write(f"Detected Object: {class_name} - Confidence: {conf:.2f}")

            # Draw bounding box
            draw = ImageDraw.Draw(image)
            draw.rectangle(bbox, outline="red", width=3)

    # Display the annotated image
    st.image(image, caption="Annotated Image.", use_column_width=True)