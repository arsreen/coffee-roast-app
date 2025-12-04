#############################################
#   COFFEE ROAST RECOGNITION — FINAL APP
#   TFLite Version (No TensorFlow)
#############################################

import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import tflite_runtime.interpreter as tflite


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
# LOAD TFLITE MODEL
# ======================
@st.cache_resource
def load_interpreter():
    interpreter = tflite.Interpreter(model_path="model/efficientnet_B1.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_interpreter()
IMG_SIZE = 240


# ======================
# PREPROCESS
# (EfficientNet style: scale to [-1, 1])
# ======================
def preprocess(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img).astype("float32")
    x = x / 255.0
    x = (x - 0.5) * 2.0  # [-1, 1], mirip preprocess_input EfficientNet
    x = np.expand_dims(x, axis=0)
    return x



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
# PDF REPORT — AESTHETIC + RAPIH
# ======================
def create_pdf_report(pred_label, confidence, probs, pil_img, hist_fig, edge_img):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # ---------- PAGE 1 ----------
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(40, height - 50, "Coffee Roast Prediction Report")

    pdf.setFont("Helvetica", 13)
    pdf.drawString(40, height - 90, f"Roast Level: {pred_label}")
    pdf.drawString(40, height - 110, f"Confidence: {confidence:.2f}")

    # Probabilities
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(40, height - 150, "Probability per Class:")

    pdf.setFont("Helvetica", 11)
    y = height - 170
    for label, p in zip(labels, probs):
        pdf.drawString(55, y, f"{label:<10} : {p:.2f}")
        y -= 18

    # Uploaded image
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(320, height - 150, "Uploaded Image:")

    img_reader = ImageReader(pil_img)
    pdf.drawImage(
        img_reader,
        310,
        height - 370,
        width=200,
        height=200,
        preserveAspectRatio=True
    )

    # Histogram
    pdf.drawString(40, height - 370, "Color Histogram:")
    hist_buffer = io.BytesIO()
    hist_fig.savefig(hist_buffer, format="png", bbox_inches="tight")
    hist_buffer.seek(0)
    pdf.drawImage(
        ImageReader(hist_buffer),
        40,
        height - 650,
        width=260,
        height=260,
        preserveAspectRatio=True
    )

    pdf.showPage()

    # ---------- PAGE 2 ----------
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(40, height - 50, "Image Feature Visualizations")

    # Edge map
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(40, height - 90, "Edge Map:")

    edge_rgb = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2RGB)
    edge_buffer = io.BytesIO()
    Image.fromarray(edge_rgb).save(edge_buffer, format="PNG")
    edge_buffer.seek(0)

    pdf.drawImage(
        ImageReader(edge_buffer),
        40,
        height - 650,
        width=500,
        height=500,
        preserveAspectRatio=True
    )

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

    **Model:** EfficientNet-B1 (TFLite)  
    **Dataset:** 1600 images  
    """)

    st.markdown("---")
    st.markdown("### 🔧 Tips:")
    st.markdown("""
    - Use good lighting  
    - Avoid blur  
    - Capture only beans  
    """)



# ======================
# INPUT SELECTOR
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

    # Select image source
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
    else:
        img = Image.open(camera_photo).convert("RGB")

    st.markdown("---")

    # ---------- IMAGE LEFT / PREDICTION RIGHT ----------
    left, right = st.columns([1, 1.3])

    with left:
        st.subheader("📸 Uploaded Image")
        st.image(img, use_container_width=True)

    with right:
        input_tensor = preprocess(img)

        # TFLite inference
        interpreter.set_tensor(input_details[0]['index'], input_tensor.astype("float32"))
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details[0]['index'])[0]

        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx]
        confidence = probs[pred_idx]

        st.subheader("🔍 Prediction Result")
        st.success(f"**Roast Level:** {pred_label}")
        st.info(f"**Confidence:** {confidence:.2f}")

        st.markdown(
            f"""
            <div style='background:#f7f2ee; padding:18px; border-radius:12px;
            border:1px solid #d7ccc8; margin-top:10px;'>
                <h4 style='color:#4e342e;'>About this roast:</h4>
                <p style='color:#4e342e; font-size:15px;'>{descriptions[pred_label]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ---------- PROBABILITY CHART ----------
    st.markdown("---")
    st.markdown("### 📊 Probability Chart")

    center_cols = st.columns([0.15, 0.7, 0.15])
    with center_cols[1]:
        fig, ax = plt.subplots()
        bars = ax.bar(labels, probs, color=["#4e342e", "#7cb342", "#ffb74d", "#5d4037"])
        ax.set_ylim(0, 1)
        for bar, p in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width() / 2, p + 0.02, f"{p:.2f}", ha="center")
        st.pyplot(fig)


    # ---------- FEATURE VISUALIZATION ----------
    st.markdown("---")
    st.subheader("🧪 Feature Visualization")

    edges = extract_edge_map(img)
    hist_fig = plot_color_histogram(img)

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Edge Map")
        st.image(edges, use_container_width=True)

    with colB:
        st.markdown("### Color Histogram")
        st.pyplot(hist_fig)



    # ---------- PDF REPORT ----------
    st.markdown("---")
    st.subheader("📄 Download Prediction Report")

    pdf_buffer = create_pdf_report(
        pred_label, confidence, probs, img, hist_fig, edges
    )

    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="coffee_prediction_report.pdf",
        mime="application/pdf"
    )



# ======================
# FOOTER
# ======================
st.markdown("---")
st.markdown("""
<p style='text-align:center; color:#8d6e63;'>
    ☕ Built with EfficientNet (TFLite) • Streamlit • Machine Learning
</p>
""", unsafe_allow_html=True)