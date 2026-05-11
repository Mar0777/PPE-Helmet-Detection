import cv2
import numpy as np
import tensorflow as tf

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(
    r"E:\Desktop\PPE_Project\saved_models\best_mobilenet.keras"
)

# =========================
# CLASS NAMES
# =========================
classes = ["helmet", "no_helmet"]

# =========================
# START WEBCAM
# =========================
cap = cv2.VideoCapture(0)

IMG_SIZE = 224
cv2.namedWindow("PPE Detection")
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize for model
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)

    confidence = prediction[0][0]

    if confidence > 0.5:
        label = "NO HELMET"
        color = (0, 0, 255)
        score = confidence
    else:
        label = "HELMET"
        color = (0, 255, 0)
        score = 1 - confidence

    text = f"{label}: {score:.2f}"

    # Draw label
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("PPE Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()