"""MRI brain tumor classifier (Streamlit).

Run from anywhere:  streamlit run app.py
Requires model.h5 in the same folder as this file.
"""
import os

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.h5")
IMAGE_SIZE = (128, 128)  # the original model.h5 was trained on 128x128 images

# Verified against the original model.h5 using known sample images
# (glioma -> index 0, meningioma -> index 1; notumor/pituitary by test-set counts).
CLASS_LABELS = ["glioma", "meningioma", "notumor", "pituitary"]

st.set_page_config(page_title="MRI Tumor Detection", page_icon="🧠")


@st.cache_resource
def get_model():
    """Load the original model.h5 (no retraining needed).

    Newer TensorFlow/Keras versions often fail on load_model() for this old file,
    so if that happens we rebuild the same architecture and copy the weights in.
    """
    try:
        return load_model(MODEL_PATH, compile=False)
    except Exception:
        import h5py
        from tensorflow.keras.applications import VGG16
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input, Dense, Flatten, Dropout

        base = VGG16(input_shape=(128, 128, 3), include_top=False, weights=None)
        dense1 = Dense(128, activation="relu")
        dense2 = Dense(4, activation="softmax")
        model = Sequential([Input((128, 128, 3)), base, Flatten(), Dropout(0.3),
                            dense1, Dropout(0.2), dense2])
        with h5py.File(MODEL_PATH, "r") as f:
            w = f["model_weights"]
            for layer in base.layers:
                if layer.name in w["vgg16"]:
                    g = w["vgg16"][layer.name]
                    layer.set_weights([g["kernel"][:], g["bias"][:]])
            g = w["dense"]["sequential"]["dense"]
            dense1.set_weights([g["kernel"][:], g["bias"][:]])
            g = w["dense_1"]["sequential"]["dense_1"]
            dense2.set_weights([g["kernel"][:], g["bias"][:]])
        return model


def predict_tumor(image: Image.Image):
    """Return (label, confidence, all_probabilities) for a PIL image."""
    # Same preprocessing as training: Keras load_img (nearest resize) + divide by 255.
    img = image.convert("RGB").resize(IMAGE_SIZE, Image.NEAREST)
    arr = np.asarray(img, dtype="float32") / 255.0
    probs = get_model().predict(np.expand_dims(arr, axis=0), verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_LABELS[idx], float(probs[idx]), probs


st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] { background-color: #87ceeb; }
        [data-testid="stHeader"] { background-color: #add8e6; }
        .title-box {
            background-color: #4da6ff; padding: 20px 15px; border-radius: 12px;
            text-align: center; color: white; font-size: 36px; font-weight: bold;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.25); margin-bottom: 25px;
        }
        .instruction-text { color: darkblue; font-size: 20px; font-weight: 500; line-height: 1.6; }
        [data-testid="stFileUploader"] > section { background-color: steelblue; border-color: lightblue; }
        h3 { font-size: 26px; }
    </style>
    <div class="title-box">🧠 MRI Tumor Detection</div>
    <p class="instruction-text">
        Upload an MRI brain scan image. The model will analyze the image and predict
        if a tumor is detected. Recommended formats: JPG, JPEG, PNG.
    </p>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Educational project only. This is **not** a medical device and must not be "
    "used for diagnosis. Always consult a qualified medical professional."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "model.h5 was not found next to app.py. Place your trained model.h5 "
        "in this folder and restart the app."
    )
    st.stop()

uploaded_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)
    st.markdown("---")

    with st.spinner("Analyzing image..."):
        label, confidence, probs = predict_tumor(image)

    if label == "notumor":
        result, color = "No tumor detected", "green"
    else:
        result, color = f"Tumor: {label}", "red"

    st.markdown(
        f'<h3 style="color: {color};">Prediction: {result}</h3>'
        f"<p><strong>Confidence:</strong> {confidence * 100:.2f}%</p>",
        unsafe_allow_html=True,
    )

    with st.expander("Probability for each class"):
        for name, p in zip(CLASS_LABELS, probs):
            st.write(f"{name}: {p * 100:.2f}%")
            st.progress(float(p))

    st.markdown("---")
    st.info("Note: For best results, upload clear and focused MRI brain images.")