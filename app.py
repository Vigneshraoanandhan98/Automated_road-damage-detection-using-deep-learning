import streamlit as st
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

# Page settings
st.set_page_config(page_title="Road Damage Detection", layout="centered")
st.title("🚧 Road Damage Detection ")
st.write("Upload a road image to detect damage types (Pothole, Crack, Manhole).")

# Model path
MODEL_PATH = r"C:\Users\VICKY\Desktop\Guvi\projects\Final Project\runs\detect\road_damage_output\pothole_detection_fast8\weights\best.pt"

# Load trained model
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Cannot find model file: {MODEL_PATH}")
        return None
    model = YOLO(MODEL_PATH)
    return model

model = load_my_model()

if model is None:
    st.stop()
    
# Class names & Recommendations
# YOLO model classes: 0: pothole, 1: crack, 2: manhole
classes = ["Pothole", "Crack", "Manhole"]
recommendations = {
    "Pothole": "⚠️ Recommended Action: Schedule immediate repair.",
    "Crack": "🔍 Recommended Action: Monitor and seal the crack soon.",
    "Manhole": "✅ Recommended Action: Inspect manhole cover condition and alignment."
}

# File uploader & Detection
uploaded_file = st.file_uploader("Upload an image (Pothole, Crack, Manhole)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    with st.spinner('Detecting...'):
        # Predict using YOLOv8 with a minimum confidence threshold to filter bad guesses
        results = model.predict(image, conf=0.45)
        
    if len(results) > 0:
        result = results[0]
        
        # Plot the detections on the image
        res_plotted = result.plot()  # Returns a BGR numpy array
        res_plotted_rgb = res_plotted[:, :, ::-1]  # Convert BGR to RGB
        
        st.subheader("Detection Result")
        st.image(res_plotted_rgb, use_container_width=True)
        
        boxes = result.boxes
        if len(boxes) > 0:
            st.subheader("Detections")
            
            detected_classes = set()
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = classes[cls_id]
                detected_classes.add(class_name)
                
                st.write(f"**Detected:** {class_name} with {conf * 100:.2f}% confidence")
            
            st.subheader("Recommendations")
            for class_name in detected_classes:
                st.info(recommendations.get(class_name, ""))
        else:
            st.info("No damage detected in the image.")

st.markdown("---")
st.write("Built with Streamlit and YOLOv8")