import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# MUST be first
st.set_page_config(page_title="Steel Defect Detector", layout="centered")

# Constants
MODEL_DIR = "converted_savedmodel/model.savedmodel"
LABELS_PATH = "converted_savedmodel/labels.txt"
PASS_LABELS = ["no_defect", "pass", "ok", "normal"]  # update if needed

# Load class names
with open(LABELS_PATH, "r") as f:
    class_names = [line.strip().lower() for line in f.readlines()]

# Load model
@st.cache_resource
def load_model():
    model = tf.saved_model.load(MODEL_DIR)
    return model.signatures["serving_default"]

model = load_model()

# Page title
st.title("🧠 Hot-Rolled Steel Surface Defect Detection")
st.markdown("Upload an image or use your webcam to detect surface defects.")

# UI Tabs
tab1, tab2 = st.tabs(["📤 Upload Image", "📸 Use Webcam"])

# Prediction logic
def predict_image(image: Image.Image):
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    input_tensor = tf.convert_to_tensor(img_array)

    result = model(input_tensor)
    predictions = list(result.values())[0].numpy()[0]
    predicted_index = int(np.argmax(predictions))
    predicted_label = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    return predicted_label, confidence, predictions

# ----------- Tab 1: Upload -------------
with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        predicted_label, confidence, predictions = predict_image(image)

        st.divider()
        st.subheader("🔎 Prediction Result")
        if predicted_label in PASS_LABELS or confidence < 50:
            st.success(f"✅ PASS — No Defect Detected ({confidence:.2f}% confident)")
        else:
            st.error(f"❌ DEFECT: **{predicted_label.upper()}** ({confidence:.2f}% confident)")

        if st.checkbox("Show all class probabilities"):
            st.markdown("### 📊 Class-wise Probabilities")
            for i, prob in enumerate(predictions):
                st.write(f"{class_names[i].capitalize()}: {prob * 100:.2f}%")

# ----------- Tab 2: Webcam -------------
with tab2:
    st.info("Capture a photo using your webcam")
    camera_img = st.camera_input("📸 Take a Picture")

    if camera_img is not None:
        image = Image.open(camera_img).convert("RGB")
        st.image(image, caption="Captured Image", use_column_width=True)

        predicted_label, confidence, predictions = predict_image(image)

        st.divider()
        st.subheader("🔎 Prediction Result")
        if predicted_label in PASS_LABELS or confidence < 50:
            st.success(f"✅ PASS — No Defect Detected ({confidence:.2f}% confident)")
        else:
            st.error(f"❌ DEFECT: **{predicted_label.upper()}** ({confidence:.2f}% confident)")

        if st.checkbox("Show all class probabilities (Camera)"):
            st.markdown("### 📊 Class-wise Probabilities")
            for i, prob in enumerate(predictions):
                st.write(f"{class_names[i].capitalize()}: {prob * 100:.2f}%")
