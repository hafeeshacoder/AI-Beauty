import streamlit as st
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import io

st.set_page_config(page_title="AI Cartoon Studio", page_icon="🎨")

st.title("🎨 AI Cartoon & Mask Detection Studio")
st.write("Upload an image and convert it into cartoon styles or detect face mask.")

# Sidebar menu
option = st.sidebar.selectbox(
    "Select Feature",
    ("Classic Cartoon", "Ghibli Style Cartoon", "Mask Detection")
)

# -------- CLASSIC CARTOON --------
if option == "Classic Cartoon":

    st.header("🎨 Classic Cartoon Generator")

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        image = Image.open(file).convert("RGB")

        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        # Edge enhancement
        edges = image.filter(ImageFilter.FIND_EDGES)

        # Smooth colors
        smooth = image.filter(ImageFilter.SMOOTH_MORE)

        # Reduce colors
        poster = ImageOps.posterize(smooth, 3)

        # Blend edges with poster
        cartoon = Image.blend(poster, edges, 0.2)

        st.subheader("Cartoon Result")
        st.image(cartoon, use_column_width=True)

        # Download
        buf = io.BytesIO()
        cartoon.save(buf, format="PNG")

        st.download_button(
            "Download Cartoon Image",
            buf.getvalue(),
            "cartoon.png"
        )

# -------- GHIBLI STYLE --------
elif option == "Ghibli Style Cartoon":

    st.header("🌸 Ghibli Style Cartoon")

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        image = Image.open(file).convert("RGB")

        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        # Soft painting effect
        smooth = image.filter(ImageFilter.GaussianBlur(1.5))

        # Reduce color palette
        poster = ImageOps.posterize(smooth, 4)

        # Enhance colors
        color = ImageEnhance.Color(poster).enhance(1.6)

        # Slight brightness boost
        ghibli = ImageEnhance.Brightness(color).enhance(1.1)

        st.subheader("Ghibli Style Result")
        st.image(ghibli, use_column_width=True)

        buf = io.BytesIO()
        ghibli.save(buf, format="PNG")

        st.download_button(
            "Download Ghibli Image",
            buf.getvalue(),
            "ghibli_style.png"
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
