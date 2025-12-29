import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import re

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Handwritten Digit & Symbol Calculator",
    layout="centered"
)

st.title("✍️ Handwritten Digit & Symbol Recognition")
st.caption("CNN-based multi-symbol handwritten calculator")

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("handwritten_model.h5")

model = load_model()

# Class labels (same order used during training)
CLASS_NAMES = [
    '+', '-', '0', '1', '2', '3', '4', '5',
    '6', '7', '8', '9', '[', ']', 'd', 't'
]

# -------------------------------------------------
# Symbol recognition function
# -------------------------------------------------
def recognize_symbols(image):
    image = image.convert("L")
    img = np.array(image)

    # Binary threshold
    _, img = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        255 - img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return "", 0.0

    # Sort contours left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    symbols = []
    confidences = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Ignore noise
        if w < 10 or h < 10:
            continue

        margin = 4
        cropped = img[
            max(0, y - margin):y + h + margin,
            max(0, x - margin):x + w + margin
        ]

        resized = cv2.resize(cropped, (22, 22), interpolation=cv2.INTER_AREA)

        canvas = np.ones((28, 28), dtype=np.uint8) * 255
        canvas[3:25, 3:25] = resized

        canvas = canvas.astype("float32") / 255.0
        canvas = canvas.reshape(1, 28, 28, 1)

        pred = model.predict(canvas, verbose=0)
        idx = np.argmax(pred)

        symbols.append(CLASS_NAMES[idx])
        confidences.append(float(np.max(pred)))

    if not symbols:
        return "", 0.0

    return "".join(symbols), sum(confidences) / len(confidences)

# -------------------------------------------------
# UI
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a handwritten expression image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    prediction, confidence = recognize_symbols(image)

    if prediction:
        # Convert symbols to operators
        expression = prediction.replace("t", "*").replace("d", "/")

        st.subheader("🧠 Prediction")
        st.write(f"**Recognized Expression:** `{expression}`")
        st.write(f"**Average Confidence:** `{confidence:.2f}`")

        # Safety check
        if re.fullmatch(r"[0-9+\-*/().]+", expression):
            try:
                result = eval(expression)
                st.success(f"✅ Calculated Result: **{result}**")
            except:
                st.error("❌ Calculation failed")
        else:
            st.error("❌ Invalid mathematical expression")
    else:
        st.warning("No symbols detected")
