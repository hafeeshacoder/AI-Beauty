import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# 1. PAGE SETUP
st.set_page_config(page_title="AI Beauty Consultant", layout="wide")
st.title("💄 AI Virtual Makeup Consultant")
st.write("Upload a portrait photo to apply virtual lipstick and blush.")

# 2. INITIALIZE MEDIAPIPE FACE MESH
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Lip Landmark Indices (standard indices for MediaPipe Face Mesh)
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 273, 287, 410, 322, 391, 308, 307, 375, 0, 267, 269, 270, 409, 291, 306, 292, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

def apply_lipstick(img, points, color_bgr, intensity):
    """Applies lipstick using Alpha Blending and Gaussian Blur."""
    mask = np.zeros_like(img)
    # Create the lip shape from landmarks
    points = np.array(points, np.int32)
    cv2.fillPoly(mask, [points], color_bgr)
    
    # Soften the edges for realism (Filtering)
    mask = cv2.GaussianBlur(mask, (7, 7), 11)
    
    # Blend the mask with original image (Alpha Blending)
    lipstick_img = cv2.addWeighted(img, 1.0, mask, intensity, 0)
    return lipstick_img

# 3. SIDEBAR CONTROLS
st.sidebar.header("Makeup Palette")
color = st.sidebar.color_picker("Pick a Lipstick Color", "#E91E63")
intensity = st.sidebar.slider("Lipstick Intensity", 0.0, 1.0, 0.4)

# Convert Hex to BGR
hex_color = color.lstrip('#')
rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])

# 4. FILE UPLOADER
uploaded_file = st.file_uploader("Upload a Face Photo (PNG, JPG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    h, w, _ = image_bgr.shape

    # Detect Facial Landmarks
    results = face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lip_coords = []
            for idx in LIPS:
                lm = face_landmarks.landmark[idx]
                lip_coords.append((int(lm.x * w), int(lm.y * h)))
            
            # Apply Makeup
            output_img = apply_lipstick(image_bgr, lip_coords, bgr_color, intensity)
            
            # Display Side-by-Side Result
            col1, col2 = st.columns(2)
            col1.image(image_np, caption="Original Image")
            col2.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), caption="Virtual Makeup Applied")
            
            # 5. DOWNLOAD FEATURE
            final_pil = Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
            st.sidebar.download_button("Save the Look", data=uploaded_file, file_name="ai_makeup.png", mime="image/png")
    else:
        st.error("No face detected. Please use a clearer portrait photo.")
else:
    st.info("Please upload a portrait photo to begin the virtual makeover.")
