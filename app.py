import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Direct access to solutions to prevent AttributeError
mp_face_mesh = mp.solutions.face_mesh

# 1. PAGE SETUP
st.set_page_config(page_title="AI Beauty Consultant", layout="wide")
st.title("💄 AI Virtual Makeup Consultant")
st.markdown("---")

# 2. SIDEBAR CONTROLS
st.sidebar.header("💄 Makeup Palette")
lp_color = st.sidebar.color_picker("Lipstick Color", "#E91E63")
lp_intensity = st.sidebar.slider("Lipstick Intensity", 0.0, 1.0, 0.4)

st.sidebar.divider()
blush_color = st.sidebar.color_picker("Blush Color", "#FFB6C1")
blush_intensity = st.sidebar.slider("Blush Intensity", 0.0, 1.0, 0.2)

def hex_to_bgr(hex_str):
    hex_str = hex_str.lstrip('#')
    rgb = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0]) # BGR Format

# 3. CORE IMAGE PROCESSING FUNCTIONS
def apply_makeup(img, points, color_bgr, intensity, blur_size=7):
    mask = np.zeros_like(img)
    points = np.array(points, np.int32)
    cv2.fillPoly(mask, [points], color_bgr)
    
    # Filtering: Softens the edges (Syllabus Topic: Gaussian Filtering)
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 11)
    
    # Alpha Blending: Mix original image with the color mask
    return cv2.addWeighted(img, 1.0, mask, intensity, 0)

# 4. MAIN APPLICATION LOGIC
uploaded_file = st.file_uploader("Upload a Face Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    h, w, _ = image_bgr.shape

    # Initialize Face Mesh inside the processing block for stability
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:
        
        results = face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Landmark Indices (Standard 468 Face Mesh mapping)
            LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 273, 287, 410, 322, 391, 308, 307, 375, 0, 267, 269, 270, 409, 291, 306, 292, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
            LEFT_CHEEK = [205, 203, 98, 97, 2]
            RIGHT_CHEEK = [425, 423, 327, 326, 2]

            # Coordinate Extraction
            lip_pts = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LIPS]
            l_cheek_pts = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_CHEEK]
            r_cheek_pts = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_CHEEK]

            # Apply Makeup Processing
            output = apply_makeup(image_bgr, lip_pts, hex_to_bgr(lp_color), lp_intensity)
            output = apply_makeup(output, l_cheek_pts, hex_to_bgr(blush_color), blush_intensity, blur_size=51)
            output = apply_makeup(output, r_cheek_pts, hex_to_bgr(blush_color), blush_intensity, blur_size=51)

            # 5. UI DISPLAY
            col1, col2 = st.columns(2)
            col1.image(image_np, caption="Original Portrait", use_container_width=True)
            col2.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="AI Virtual Makeup Applied", use_container_width=True)
            
            st.sidebar.success("Analysis Successful!")
        else:
            st.error("Face not detected. Please use a clear, front-facing portrait.")
