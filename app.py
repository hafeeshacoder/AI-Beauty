import streamlit as st
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import io

st.set_page_config(page_title="AI Cartoon Studio", page_icon="🎨")

st.title("🎨 AI Cartoon Studio")
st.write("Upload an image to generate a cartoon version or detect a mask.")

# Sidebar menu
option = st.sidebar.selectbox(
    "Select Feature",
    ("Cartoon Generator", "Mask Detection")
)

# -------- CARTOON GENERATOR --------
if option == "Cartoon Generator":

    st.header("🎨 Cartoon Image Generator")

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        image = Image.open(file).convert("RGB")

        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        # Smooth image
        smooth = image.filter(ImageFilter.SMOOTH_MORE)

        # Reduce color palette
        poster = ImageOps.posterize(smooth, 3)

        # Edge detection
        edges = image.filter(ImageFilter.FIND_EDGES)

        # Blend edges and colors
        cartoon = Image.blend(poster, edges, 0.25)

        # Enhance colors
        cartoon = ImageEnhance.Color(cartoon).enhance(1.5)

        st.subheader("Cartoon Result")
        st.image(cartoon, use_column_width=True)

        # Download button
        buf = io.BytesIO()
        cartoon.save(buf, format="PNG")

        st.download_button(
            "Download Cartoon Image",
            buf.getvalue(),
            "cartoon.png",
            "image/png"
        )

# -------- MASK DETECTION --------
elif option == "Mask Detection":

    st.header("😷 Face Mask Detection")

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        image = Image.open(file).convert("RGB")

        st.image(image, caption="Uploaded Image", use_column_width=True)

        pixels = list(image.getdata())
        blue_pixels = 0

        for p in pixels:
            r, g, b = p
            if b > r and b > g:
                blue_pixels += 1

        ratio = blue_pixels / len(pixels)

        if ratio > 0.20:
            st.success("Mask Detected ✔")
        else:
            st.error("No Mask Detected ❌")
