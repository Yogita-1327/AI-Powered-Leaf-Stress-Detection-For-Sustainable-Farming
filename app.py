import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="LeafGuard - Leaf Stress Detection", page_icon="🌿", layout="centered")
st.title("🌿 LeafGuard: Leaf Stress Detection (No Hardware)")
st.caption("Upload or capture a leaf image. The model runs locally using TensorFlow Lite.")

def load_labels(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return ["Healthy Leaf", "Stressed Leaf"]

LABELS = load_labels("labels.txt")

@st.cache_resource(show_spinner=False)
def load_interpreter(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]
    return interpreter, input_details, output_details, input_shape

def preprocess_pil(img, target_size=(224, 224)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(interpreter, input_details, output_details, img_arr):
    interpreter.set_tensor(input_details[0]['index'], img_arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    if np.any(np.isnan(preds)) or preds.sum() <= 0:
        preds = np.clip(preds, 0, None)
        s = preds.sum()
        preds = preds / s if s > 0 else np.ones_like(preds) / len(preds)
    return preds

def advisory(label):
    if "Healthy" in label:
        return ("✅ Looks healthy.",
                ["Maintain consistent watering; avoid overwatering.",
                 "Inspect weekly for early pest signs.",
                 "Continue current fertilizer schedule."])
    else:
        return ("⚠️ Stress detected.",
                ["Check soil moisture; water if topsoil is dry 2–3 cm below surface.",
                 "Inspect underside of leaves for pests; remove affected leaves if necessary.",
                 "Look for yellowing/browning (possible nutrient deficiency); consider balanced fertilizer.",
                 "Recheck in 3–5 days after action."])

st.sidebar.header("Setup")
model_file = st.sidebar.text_input("Path to your .tflite model", value="model.tflite")
st.sidebar.write("Ensure labels.txt matches your training class order.")

uploaded = st.file_uploader("Upload a leaf photo", type=["jpg","jpeg","png"])
cam_photo = st.camera_input("Or use your webcam")

image_to_use = None
if uploaded:
    image_to_use = Image.open(uploaded)
elif cam_photo:
    image_to_use = Image.open(cam_photo)

if not os.path.exists(model_file):
    st.info("➡️ Export a **TensorFlow Lite** model from Teachable Machine and place it here as model.tflite.")
else:
    interpreter, input_details, output_details, input_shape = load_interpreter(model_file)
    st.caption(f"Model loaded. Expected input shape: {tuple(input_shape)}")

if image_to_use and os.path.exists(model_file):
    st.image(image_to_use, caption="Input image", use_container_width=True)
    arr = preprocess_pil(image_to_use, target_size=(input_shape[1], input_shape[2]))
    probs = predict_image(interpreter, input_details, output_details, arr)

    if len(LABELS) != len(probs):
        LABELS = [f"Class {i}" for i in range(len(probs))]

    top_idx = int(np.argmax(probs))
    top_label = LABELS[top_idx]
    top_conf = float(probs[top_idx]) * 100.0

    st.subheader(f"Prediction: **{top_label}**")
    st.write(f"Confidence: **{top_conf:.2f}%**")

    st.markdown("**Class probabilities:**")
    for lab, p in zip(LABELS, probs):
        st.write(f"- {lab}: {p*100:.2f}%")

    title, tips = advisory(top_label)
    st.markdown(f"### {title}")
    for t in tips:
        st.write(f"- {t}")
