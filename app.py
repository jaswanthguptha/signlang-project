import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from gtts import gTTS
import os
import tempfile

# Page config
st.set_page_config(
    page_title="AI Sign Language Communicator",
    page_icon="🤟",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        background: linear-gradient(90deg, #00ff88, #00aaff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .sentence-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #00ff88;
        border-radius: 15px;
        padding: 20px;
        font-size: 1.8em;
        color: #00ff88;
        text-align: center;
        min-height: 80px;
        margin: 10px 0;
    }
    .sign-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #00aaff;
        border-radius: 15px;
        padding: 15px;
        font-size: 2em;
        color: #00aaff;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-size: 1.1em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    return model, labels

model, labels = load_model()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Title
st.markdown('<div class="title">🤟 AI Sign Language Communicator</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Show your hand signs to the camera — AI will read them!</div>',
            unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📷 Live Camera Feed")
    camera_placeholder = st.empty()

with col2:
    st.markdown("### 🔤 Current Sign")
    sign_placeholder = st.empty()
    confidence_placeholder = st.empty()

    st.markdown("### 📝 Sentence")
    sentence_placeholder = st.empty()

    st.markdown("### 🎮 Controls")
    speak_btn = st.button("🔊 Speak Sentence", use_container_width=True)
    clear_btn = st.button("🗑️ Clear Sentence", use_container_width=True)
    stop_btn = st.button("⏹️ Stop Camera", use_container_width=True)

    st.markdown("### 📊 Stats")
    stats_placeholder = st.empty()

# Session state
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""
if 'running' not in st.session_state:
    st.session_state.running = True
if 'word_count' not in st.session_state:
    st.session_state.word_count = 0

# Controls
if clear_btn:
    st.session_state.sentence = ""
if stop_btn:
    st.session_state.running = False

# Text to Speech function
def speak(text):
    if text.strip():
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix='.mp3') as fp:
            tts.save(fp.name)
            st.audio(fp.name,format='audio/mp3')

if speak_btn:
    speak(st.session_state.sentence)

# Camera loop
cap = cv2.VideoCapture(0)
last_sign = ""
sign_start_time = 0
HOLD_TIME = 2

while st.session_state.running:
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

        # Sentence building
        current_time = time.time()
        if predicted_sign == last_sign and predicted_sign != "":
            hold_duration = current_time - sign_start_time
            if hold_duration >= HOLD_TIME:
                display = predicted_sign
                words = st.session_state.sentence.strip().split()
                if not words or words[-1].lower() != display.lower():
                    st.session_state.sentence += display + " "
                    st.session_state.word_count += 1
                sign_start_time = current_time
        else:
            last_sign = predicted_sign
            sign_start_time = current_time

    # Show frame
    camera_placeholder.image(frame, channels="RGB", use_column_width=True)

    # Update UI
    if predicted_sign:
        sign_placeholder.markdown(
            f'<div class="sign-box">{predicted_sign.upper()}</div>',
            unsafe_allow_html=True)
        confidence_placeholder.progress(int(confidence))
    else:
        sign_placeholder.markdown(
            '<div class="sign-box">No Hand Detected</div>',
            unsafe_allow_html=True)

    sentence_placeholder.markdown(
        f'<div class="sentence-box">{st.session_state.sentence or "Start signing..."}</div>',
        unsafe_allow_html=True)

    stats_placeholder.markdown(f"""
    - 🔤 Words: **{st.session_state.word_count}**
    - 🎯 Confidence: **{confidence:.1f}%**
    - 📝 Signs Available: **40**
    """)

    time.sleep(0.03)

cap.release()
st.success("Camera stopped!")