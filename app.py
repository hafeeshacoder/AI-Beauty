import streamlit as st
from PIL import Image, ImageFilter, ImageOps, ImageStat
import io

st.set_page_config(page_title="AI Face Studio", page_icon="🎭")

st.title("🎭 AI Face Studio")
st.write("Choose a feature and upload an image.")

# -------- CATEGORY SELECT --------
option = st.sidebar.selectbox(
    "Select Feature",
    ("Cartoon Generator", "Mask Detection", "Beauty Score")
)

# -------- CARTOON GENERATOR --------
if option == "Cartoon Generator":

    st.header("🎨 Cartoon Image Generator")

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        image = Image.open(file).convert("RGB")

        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        cartoon = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        cartoon = ImageOps.posterize(cartoon, 3)

        st.subheader("Cartoon Result")
        st.image(cartoon, use_column_width=True)

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

# -------- BEAUTY SCORE --------
elif option == "Beauty Score":

    st.header("⭐ Beauty Score Analyzer")

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        image = Image.open(file).convert("RGB")

        st.image(image, caption="Uploaded Image", use_column_width=True)

        stat = ImageStat.Stat(image)
        brightness = sum(stat.mean) / 3

        if brightness > 170:
            score = 9
        elif brightness > 140:
            score = 8
        elif brightness > 110:
            score = 7
        elif brightness > 80:
            score = 6
        else:
            score = 5

        st.subheader(f"Beauty Score: {score}/10")

        st.progress(score/10)

        if score >= 8:
            st.success("Great Photo Quality!")
        elif score >= 6:
            st.info("Good Photo")
        else:
            st.warning("Try better lighting or angle")
