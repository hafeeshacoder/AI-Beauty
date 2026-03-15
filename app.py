import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Setup Mediapipe with a safe check
mp_face_mesh = mp.solutions.face_mesh

st.set_page_config(page_title="AI Beauty Consultant", layout="wide")
st.title("💄 AI Virtual Makeup Consultant")

# Sidebar for Palette
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
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 11)
    return cv2.addWeighted(img, 1.0, mask, intensity, 0)

# File Uploader
uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    # Ensure RGB for Mediapipe and BGR for OpenCV
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB) if img_array.shape[-1] == 4 else img_array
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Landmark Indices
            LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 273, 287, 410, 322, 391, 308, 307, 375, 0, 267, 269, 270, 409, 291, 306, 292, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
            L_CHEEK = [205, 203, 98, 97, 2]
            R_CHEEK = [425, 423, 327, 326, 2]

            # Get Coordinates
            get_coords = lambda indices: [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in indices]
            
            # Processing
            out = apply_makeup(img_bgr, get_coords(LIPS), hex_to_bgr(lp_color), lp_intensity)
            out = apply_makeup(out, get_coords(L_CHEEK), hex_to_bgr(blush_color), blush_intensity, 51)
            out = apply_makeup(out, get_coords(R_CHEEK), hex_to_bgr(blush_color), blush_intensity, 51)

            # Display
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)
        else:
            st.error("No face detected. Please try another photo.")
