import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# Load model and labels
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Sentence builder variables
sentence = ""
last_sign = ""
sign_start_time = 0
HOLD_TIME = 2  # seconds to hold sign before adding to sentence

cap = cv2.VideoCapture(0)
print("🎥 Real-time detection started! Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    predicted_sign = ""
    confidence = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(landmarks)[0]
            prob = model.predict_proba(landmarks).max()
            predicted_sign = prediction
            confidence = prob * 100

        # Sentence building logic
        current_time = time.time()
        if predicted_sign == last_sign:
            hold_duration = current_time - sign_start_time
            # Progress bar
            progress = min(hold_duration / HOLD_TIME, 1.0)
            bar_width = int(300 * progress)
            cv2.rectangle(frame, (10, 100), (310, 120), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 100), (10 + bar_width, 120),
                         (0, 255, 0), -1)

            if hold_duration >= HOLD_TIME:
                if predicted_sign == 'iloveyou':
                    display = 'I Love You'
                else:
                    display = predicted_sign
                if len(sentence) == 0 or sentence.split()[-1] != display:
                    sentence += display + " "
                sign_start_time = current_time
        else:
            last_sign = predicted_sign
            sign_start_time = current_time

    # Display
    h, w = frame.shape[:2]

    # Sign prediction box
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"Sign: {predicted_sign.upper()}  {confidence:.1f}%",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Sentence box at bottom
    cv2.rectangle(frame, (0, h-80), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Sentence: {sentence[-40:]}",
                (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Controls
    cv2.putText(frame, "SPACE=clear  Q=quit",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("AI Sign Language Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        sentence = ""

cap.release()
cv2.destroyAllWindows()