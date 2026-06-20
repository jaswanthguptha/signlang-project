import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# All signs you want to collect
SIGNS = [
    'A', 'z'
]

SAMPLES_PER_SIGN = 100  # 100 samples per sign
DATA_DIR = 'data'

# Create folders
os.makedirs(DATA_DIR, exist_ok=True)
for sign in SIGNS:
    os.makedirs(os.path.join(DATA_DIR, sign), exist_ok=True)

def extract_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

cap = cv2.VideoCapture(0)

for sign in SIGNS:
    print(f"\n📌 Get ready for sign: {sign.upper()}")
    print("Press SPACE to start collecting...")

    # Wait for space key
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"NEXT SIGN: {sign.upper()}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frame, "Press SPACE to start", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    # Collect 100 samples
    count = 0
    while count < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks,
                                       mp_hands.HAND_CONNECTIONS)
                landmarks = extract_landmarks(hand_landmarks)
                
                # Save landmarks to file
                save_path = os.path.join(DATA_DIR, sign, f'{count}.npy')
                np.save(save_path, landmarks)
                count += 1

        cv2.putText(frame, f"Sign: {sign.upper()}  Samples: {count}/{SAMPLES_PER_SIGN}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(1)

    print(f"✅ Done collecting {sign.upper()}!")

cap.release()
cv2.destroyAllWindows()
print("\n🎉 All signs collected! Ready to train!")
