import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Initialize the Mesh outside to check for errors
try:
    FaceMesh = mp.solutions.face_mesh.FaceMesh
except AttributeError:
    st.error("Mediapipe initialization failed. Please check your requirements.txt")

st.set_page_config(page_title="AI Beauty Consultant", layout="wide")
st.title("💄 AI Virtual Makeup Consultant")

# Sidebar
st.sidebar.header("Makeup Palette")
lp_color = st.sidebar.color_picker("Lipstick Color", "#E91E63")
lp_intensity = st.sidebar.slider("Lipstick Intensity", 0.0, 1.0, 0.4)
blush_color = st.sidebar.color_picker("Blush Color", "#FFB6C1")
blush_intensity = st.sidebar.slider("Blush Intensity", 0.0, 1.0, 0.2)

def hex_to_bgr(hex_str):
    hex_str = hex_str.lstrip('#')
    rgb = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])

def apply_makeup(img, points, color_bgr, intensity, blur_size=7):
    mask = np.zeros_like(img)
    points = np.array(points, np.int32)
    cv2.fillPoly(mask, [points], color_bgr)
    # Syllabus Topic: Gaussian Filtering for edge smoothing
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 11)
    # Syllabus Topic: Alpha Blending for realism
    return cv2.addWeighted(img, 1.0, mask, intensity, 0)

uploaded_file = st.file_uploader("Upload Portrait", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Handle RGBA images (convert to RGB)
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape

    # Deep Learning Inference
    with FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(img_array)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Key Indices for Lips and Cheeks
            LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 273, 287, 410, 322, 391, 308, 307, 375, 0, 267, 269, 270, 409, 291, 306, 292, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
            L_CHEEK = [205, 203, 98, 97, 2]
            R_CHEEK = [425, 423, 327, 326, 2]

            def get_pts(indices):
                return [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in indices]
            
            # Apply Image Processing Pipeline
            out = apply_makeup(img_bgr, get_pts(LIPS), hex_to_bgr(lp_color), lp_intensity)
            out = apply_makeup(out, get_pts(L_CHEEK), hex_to_bgr(blush_color), blush_intensity, 51)
            out = apply_makeup(out, get_pts(R_CHEEK), hex_to_bgr(blush_color), blush_intensity, 51)

            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.warning("Face not detected. Please use a clear, front-facing photo.")
