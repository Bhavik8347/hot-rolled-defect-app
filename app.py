import streamlit as st

# MUST be first Streamlit command
st.set_page_config(page_title="Steel Defect Detector", layout="centered")

import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

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

# Sidebar option for mode
st.sidebar.title("🛠 Mode Selection")
mode = st.sidebar.radio("Choose Input Mode:", ("Image Upload", "Real-time Camera"))

# Streamlit UI
st.title("Hot-Rolled Steel Surface Defect Detection")

# ---------- Mode 1: Image Upload ----------
if mode == "Image Upload":
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

        if st.checkbox("Show all class probabilities"):
            st.markdown("### Class-wise Probabilities")
            for i, prob in enumerate(predictions):
                st.write(f"{class_names[i].capitalize()}: {prob * 100:.2f}%")

# ---------- Mode 2: Real-time Camera ----------
elif mode == "Real-time Camera":
    st.markdown("🎥 Start real-time defect detection using your webcam")

    run_camera = st.button("📷 Start Camera")

    if run_camera:
        cap = cv2.VideoCapture(0)
        FRAME_WINDOW = st.image([])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame.")
                break

            # Resize and preprocess
            img = cv2.resize(frame, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = img.astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            input_tensor = tf.convert_to_tensor(img_array)

            # Inference
            result = model(input_tensor)
            predictions = list(result.values())[0].numpy()[0]
            predicted_index = int(np.argmax(predictions))
            predicted_label = class_names[predicted_index]
            confidence = predictions[predicted_index] * 100

            # Annotate result on frame
            label_text = f"{'PASS ✅' if predicted_label in PASS_LABELS or confidence < 50 else f'DEFECT ❌: {predicted_label.upper()}'} ({confidence:.1f}%)"
            color = (0, 255, 0) if predicted_label in PASS_LABELS or confidence < 50 else (0, 0, 255)
            cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Display frame
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        cv2.destroyAllWindows()
