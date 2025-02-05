import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from flask import Flask, Response, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET"])
# ✅ Load the trained model
MODEL_PATH = "model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file '{MODEL_PATH}' not found!")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")

# ✅ Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ✅ Define sign labels (adjust according to training data)
sign_labels = [str(i) for i in range(10)]

# ✅ Initialize Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ ERROR: Webcam not found! Check your camera connection.")

# ✅ Real-time video streaming generator
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        predicted_sign = "No Hand Detected"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(frame.shape[1], x_max), min(frame.shape[0], y_max)
                hand_region = frame[y_min:y_max, x_min:x_max]

                if hand_region.size > 0:
                    gray_hand = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
                    resized_hand = cv2.resize(gray_hand, (28, 28))
                    model_input = resized_hand.reshape(1, 1, 784) / 255.0  

                    prediction = model.predict(model_input)
                    predicted_class = np.argmax(prediction)

                    if 0 <= predicted_class < len(sign_labels):
                        predicted_sign = sign_labels[predicted_class]

                cv2.putText(frame, f"Sign: {predicted_sign}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ✅ Flask API Routes

@app.route('/video_feed')
def video_feed():
    """Streams video frames to the Flutter app."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['GET'])
def predict():
    """Returns the latest prediction."""
    return jsonify({"sign": predicted_sign})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
 
