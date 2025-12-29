import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import cv2
import re

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("handwritten_model.h5")

CLASS_NAMES = ['+', '-', '0', '1', '2', '3', '4', '5',
               '6', '7', '8', '9', '[', ']', 'd', 't']


def recognize_symbols(image_base64):
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes)).convert("L")
    img = np.array(image)

    # Threshold
    _, img = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        255 - img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return "", 0.0

    # Sort left → right
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
        offset = 3
        canvas[offset:offset + 22, offset:offset + 22] = resized

        canvas = canvas.astype("float32") / 255.0
        canvas = canvas.reshape(1, 28, 28, 1)

        pred = model.predict(canvas, verbose=0)
        idx = np.argmax(pred)

        symbols.append(CLASS_NAMES[idx])
        confidences.append(float(np.max(pred)))

    if not symbols:
        return "", 0.0

    return "".join(symbols), sum(confidences) / len(confidences)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    prediction, confidence = recognize_symbols(data["image"])
    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 3)
    })


@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.get_json()
    expression, _ = recognize_symbols(data["image"])

    # Convert symbols to math operators
    expression = expression.replace("d", "/").replace("t", "*")

    # Safety check
    if not re.fullmatch(r"[0-9+\-*/().]+", expression):
        return jsonify({"error": "Invalid expression"})

    try:
        result = eval(expression)
        return jsonify({
            "expression": expression,
            "result": result
        })
    except:
        return jsonify({"error": "Calculation failed"})


if __name__ == "__main__":
    app.run(debug=True)
