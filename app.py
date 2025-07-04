import streamlit as st

# MUST be first Streamlit command
st.set_page_config(page_title="Steel Defect Detector", layout="centered")

import numpy as np
from PIL import Image
import tensorflow as tf

# Constants
MODEL_DIR = "converted_savedmodel/model.savedmodel"
LABELS_PATH = "converted_savedmodel/labels.txt"
PASS_LABELS = ["no_defect", "pass", "ok", "normal"]  # Customize as per your dataset

# Load labels
with open(LABELS_PATH, "r") as f:
    class_names = [line.strip().lower() for line in f.readlines()]

# Load model
@st.cache_resource
def load_model():
    model = tf.saved_model.load(MODEL_DIR)
    return model.signatures["serving_default"]

model = load_model()

# Streamlit UI
st.title("Hot-Rolled Steel Surface Defect Detection")
st.markdown("Upload an image")

uploaded_file = st.file_uploader("📤 Upload a steel surface image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    input_tensor = tf.convert_to_tensor(img_array)

    # Inference
    result = model(input_tensor)
    predictions = list(result.values())[0].numpy()[0]
    predicted_index = int(np.argmax(predictions))
    predicted_label = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    # Show result
    st.divider()
    st.subheader("🔎 Prediction Result")
    if predicted_label in PASS_LABELS or confidence < 50:
        st.success(f"✅ PASS — No Defect Detected ({confidence:.2f}% confident)")
    else:
        st.error(f"❌ DEFECT: **{predicted_label.upper()}** ({confidence:.2f}% confident)")

    # Optional toggle: Show full class probabilities
    if st.checkbox("Show all class probabilities"):
        st.markdown("### Class-wise Probabilities")
        for i, prob in enumerate(predictions):
            st.write(f"{class_names[i].capitalize()}: {prob * 100:.2f}%")
