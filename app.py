#############################################
#   COFFEE ROAST RECOGNITION — FINAL APP
#   KERAS VERSION (NO TFLITE)
#############################################

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader


# ======================
# PAGE SETTINGS
# ======================
st.set_page_config(
    page_title="Coffee Roast Recognition",
    page_icon="☕",
    layout="wide"
)


# ======================
# HEADER
# ======================
st.markdown(
    """
    <h1 style='text-align:center; color:#3e2723;'>
        ☕ Coffee Roast Recognition
    </h1>
    <p style='text-align:center; color:#5d4037; font-size:18px;'>
        AI-powered tool to classify coffee bean roast levels from image or camera.
    </p>
    """,
    unsafe_allow_html=True
)


# ======================
# LABELS + DESCRIPTIONS
# ======================
labels = ["Dark", "Green", "Light", "Medium"]

descriptions = {
    "Dark": "Dark roast beans are bold, smoky, and have the lowest acidity.",
    "Green": "Green beans are raw and unroasted — earthy, pale, and dense.",
    "Light": "Light roast highlights the bean's origin, acidity, and floral notes.",
    "Medium": "Medium roast is balanced, smooth, and widely preferred globally."
}


# ======================
# LOAD KERAS MODEL
# ======================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/efficientnet_B1.keras")

model = load_model()
IMG_SIZE = 240


# ======================
# PREPROCESS
# ======================
def preprocess(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img).astype("float32")
    x = x / 255.0
    return np.expand_dims(x, axis=0)


# ======================
# FEATURE VISUALIZATION
# ======================
def extract_edge_map(pil_img):
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    return edges


def plot_color_histogram(pil_img):
    img = np.array(pil_img)
    colors = ('r', 'g', 'b')

    fig, ax = plt.subplots()

    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax.plot(hist, color=col)
        ax.set_xlim([0, 256])

    ax.set_title("RGB Color Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")

    return fig


# ======================
# PDF REPORT
# ======================
def create_pdf_report(pred_label, confidence, probs, pil_img, hist_fig, edge_img):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # PAGE 1
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(40, height - 50, "Coffee Roast Prediction Report")

    pdf.setFont("Helvetica", 13)
    pdf.drawString(40, height - 90, f"Roast Level: {pred_label}")
    pdf.drawString(40, height - 110, f"Confidence: {confidence:.2f}")

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(40, height - 150, "Probability per Class:")

    pdf.setFont("Helvetica", 11)
    y = height - 170
    for label, p in zip(labels, probs):
        pdf.drawString(55, y, f"{label:<10} : {p:.2f}")
        y -= 18

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer


# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.markdown("## 📌 Quick Info")
    st.markdown("""
    **Roast Classes:**
    - 🟫 Dark  
    - 🟩 Green  
    - 🟧 Light  
    - 🟤 Medium  

    **Model:** EfficientNet-B1  
    **Dataset:** 1600 images  
    """)


# ======================
# INPUT
# ======================
st.markdown("### 📤 Choose Input Method")

input_method = st.radio(
    "Select how you want to provide the image:",
    ("Upload Image", "Use Camera"),
    horizontal=True
)

uploaded_file = None
camera_photo = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload Coffee Bean Image", type=["jpg", "jpeg", "png"])
else:
    camera_photo = st.camera_input("Take a coffee bean photo")


# ======================
# MAIN PREDICTION
# ======================
if uploaded_file or camera_photo:

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
    else:
        img = Image.open(camera_photo).convert("RGB")

    st.markdown("---")

    left, right = st.columns([1, 1.3])

    with left:
        st.subheader("📸 Uploaded Image")
        st.image(img, use_container_width=True)

    with right:
        input_tensor = preprocess(img)
        probs = model.predict(input_tensor)[0]

        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx]
        confidence = probs[pred_idx]

        st.subheader("🔍 Prediction Result")
        st.success(f"**Roast Level:** {pred_label}")
        st.info(f"**Confidence:** {confidence:.2f}")


    st.markdown("---")
    st.markdown("### 📊 Probability Chart")

    fig, ax = plt.subplots()
    bars = ax.bar(labels, probs)
    ax.set_ylim(0, 1)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.02, f"{p:.2f}", ha="center")
    st.pyplot(fig)


# ======================
# FOOTER
# ======================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#8d6e63;'>"
    "☕ Built with EfficientNet • Streamlit"
    "</p>",
    unsafe_allow_html=True
)
