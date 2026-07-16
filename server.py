import os
import sys
import base64
import pickle
import traceback
import datetime
import re

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS

def get_file_timestamp_str(filepath):
    if not os.path.exists(filepath):
        return "Not found"
    mtime = os.path.getmtime(filepath)
    dt = datetime.datetime.fromtimestamp(mtime)
    return dt.strftime("%Y-%m-%d %H:%M")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  APP INIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = Flask(__name__)
CORS(app)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

HAND_LANDMARKER_PATH  = os.path.join(MODELS_DIR, "hand_landmarker.task")
CUSTOM_MODEL_PATH     = os.path.join(MODELS_DIR, "model_custom.pkl")
CUSTOM_LABELS_PATH    = os.path.join(MODELS_DIR, "labels_custom.pkl")
ALPHA_MODEL_PATH      = os.path.join(MODELS_DIR, "model_alpha.pkl")
ALPHA_LABELS_PATH     = os.path.join(MODELS_DIR, "labels_alpha.pkl")
LSTM_MODEL_PATH       = os.path.join(MODELS_DIR, "lstm_words.keras")
REVERSE_MAP_PATH      = os.path.join(MODELS_DIR, "reverse_label_map.pkl")

NUM_HANDS       = 2
FEAT_PER_HAND   = 21 * 3       # 63
SEQUENCE_LENGTH = 30           # 30 frames for LSTM sequence

ALLOWED_WORDS = {
    'hello', 'help', 'yes', 'no', 'thanks', 'please', 'water', 'eat', 'stop', 'iloveyou',
    'good', 'bad', 'more', 'sorry', 'friend', 'sad', 'drink', 'year', 'day', 'phone',
    'home', 'happy', 'hungry', 'where', 'time'
}

# Global sequence buffer and state for single-hand word recognition
primary_sequence_buffer = []
alpha_prediction_buffer = []
last_normalized_vec = None
last_landmarks_list = []
hand_loss_counter = 0
GRACE_PERIOD_FRAMES = 10       # Increased for higher occlusion/dropout tolerance
SMOOTHING_FACTOR = 0.6         # Exponential Moving Average smoothing factor

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOAD MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 60)
print("  Loading models ...")
print("=" * 60)

hand_landmarker = None

def get_hand_landmarker():
    global hand_landmarker
    if hand_landmarker is None:
        print("Lazy loading hand_landmarker...")
        hand_landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
                num_hands=NUM_HANDS,
                min_hand_detection_confidence=0.35,  # Lowered to support wide angles/low-light
                min_hand_presence_confidence=0.35,   # Lowered to avoid dropouts
                min_tracking_confidence=0.35,        # Lowered to prevent fast-movement tracking loss
            )
        )
    return hand_landmarker

custom_model = None
custom_labels = None
alpha_model = None
alpha_labels = None
reverse_label_map = None
saved_label_map = None
verification_error = None
startup_error = None

# WLASL training vocabulary definition
WLASL_CLASSES = [
    'hello', 'help', 'yes', 'no', 'thanks', 'please', 'water', 'eat', 'stop', 'iloveyou', 'good', 'bad', 'more', 'sorry',
    'friend', 'study', 'angry', 'year', 'home', 'happy', 'sad', 'day', 'morning', 'month', 'phone', 'hungry', 'night',
    'drink', 'where', 'who', 'need', 'computer', 'doctor', 'time', 'like'
]

try:
    print("=" * 60)
    print("  Loading models...")
    print("=" * 60)

    with open(CUSTOM_MODEL_PATH, "rb") as f:
        custom_model = pickle.load(f)
    with open(CUSTOM_LABELS_PATH, "rb") as f:
        custom_labels = pickle.load(f)

    with open(ALPHA_MODEL_PATH, "rb") as f:
        alpha_model = pickle.load(f)
    with open(ALPHA_LABELS_PATH, "rb") as f:
        alpha_labels = pickle.load(f)

    with open(REVERSE_MAP_PATH, "rb") as f:
        reverse_label_map = pickle.load(f)

    # Load saved label_map.pkl
    LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.pkl")
    with open(LABEL_MAP_PATH, "rb") as f:
        saved_label_map = pickle.load(f)

    model_time = get_file_timestamp_str(LSTM_MODEL_PATH)
    label_time = get_file_timestamp_str(REVERSE_MAP_PATH)
    num_classes = len(reverse_label_map)

    # Print Phase 1 startup information (EXACT requested format)
    print("Loaded Model:\nmodels/lstm_words.keras")
    print(f"\nModified:\n{model_time}")
    print(f"\nLabel Map Modified:\n{label_time}")
    print(f"\nClasses:\n{num_classes}")
    print("=" * 60)

    # Print first and last 10 labels
    sorted_indices = sorted(list(reverse_label_map.keys()))
    first_10 = [f"{i}: {reverse_label_map[i]}" for i in sorted_indices[:10]]
    last_10 = [f"{i}: {reverse_label_map[i]}" for i in sorted_indices[-10:]]

    print("First 10 labels:")
    for item in first_10:
        print(f"  {item}")
    print("\nLast 10 labels:")
    for item in last_10:
        print(f"  {item}")
    print("=" * 60)

    # Phase 2: Assertions and Abort startup if mismatch
    mismatch = False
    print("Verifying server label map against WLASL training definition...")
    for idx, name in enumerate(WLASL_CLASSES):
        if idx not in reverse_label_map:
            print(f"[FATAL MISMATCH] Index {idx} is missing from reverse_label_map.pkl!")
            mismatch = True
        elif reverse_label_map[idx] != name:
            print(f"[FATAL MISMATCH] Index {idx}: Expected '{name}' but got '{reverse_label_map[idx]}'!")
            mismatch = True

    if len(reverse_label_map) != len(WLASL_CLASSES):
        print(f"[FATAL MISMATCH] Class count mismatch: WLASL training has {len(WLASL_CLASSES)} but server reverse_label_map has {len(reverse_label_map)}")
        mismatch = True

    if len(saved_label_map) != len(WLASL_CLASSES):
        print(f"[FATAL MISMATCH] Class count mismatch: WLASL training has {len(WLASL_CLASSES)} but saved_label_map has {len(saved_label_map)}")
        mismatch = True

    for idx, name in enumerate(WLASL_CLASSES):
        if name not in saved_label_map:
            print(f"[FATAL MISMATCH] Word '{name}' is missing from saved_label_map.pkl!")
            mismatch = True
        elif saved_label_map[name] != idx:
            print(f"[FATAL MISMATCH] Word '{name}': Expected index {idx} but got {saved_label_map[name]} in saved_label_map.pkl!")
            mismatch = True

    # Verify frontend translations
    print("Verifying frontend translations.js file...")
    trans_path = os.path.join(BASE_DIR, "frontend", "src", "translations.js")
    if os.path.exists(trans_path):
        with open(trans_path, "r", encoding="utf-8") as f_trans:
            trans_content = f_trans.read()
        
        # Extract keys from English translations block
        match_en = re.search(r"en:\s*\{([^}]+)\}", trans_content)
        if match_en:
            en_block = match_en.group(1)
            keys_found = re.findall(r"([a-zA-Z0-9_-]+)\s*:", en_block)
            keys_found += re.findall(r"['\"]([a-zA-Z0-9_-]+)['\"]\s*:", en_block)
            keys_set = set(keys_found)
            
            missing_keys = [w for w in WLASL_CLASSES if w not in keys_set]
            if missing_keys:
                print(f"[FATAL MISMATCH] Frontend translations missing keys for WLASL signs: {missing_keys}")
                mismatch = True
            else:
                print("  Frontend translation keys verification PASSED.")
        else:
            print("  [FATAL MISMATCH] Could not parse 'en' translations block from translations.js")
            mismatch = True
    else:
        print(f"  [FATAL MISMATCH] translations.js file not found at {trans_path}")
        mismatch = True

    if mismatch:
        verification_error = "Label map mismatch detected! Check server logs."
        print("\n[WARNING] Label map mismatch detected! Startup is proceeding in diagnostics mode.")
    else:
        print("Label map verification completed successfully. No mismatches found.")
    print("=" * 60)

    # Warmup: pre-JIT both models so first live request is fast
    print("  Warming up models...")
    _warm_rf = alpha_model.predict_proba(np.zeros((1, 63), dtype=np.float32))
    print("  Warmup complete.")
    print("=" * 60)
    print("  All models loaded and verified successfully.")
    print("=" * 60)

except Exception as e:
    startup_error = traceback.format_exc()
    print("=" * 60)
    print("  [CRITICAL ERROR DURING STARTUP] Server is continuing in DIAGNOSTICS mode.")
    print(startup_error)
    print("=" * 60)

# Configure ONNX session options
LSTM_ONNX_PATH = os.path.join(MODELS_DIR, "lstm_words.onnx")
lstm_session = None

def get_lstm_session():
    global lstm_session
    if lstm_session is None:
        print("Lazy loading lstm_session...")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        lstm_session = ort.InferenceSession(LSTM_ONNX_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
        # Warmup inside get_lstm_session
        print("  Warming up lstm_session...")
        input_name = lstm_session.get_inputs()[0].name
        lstm_session.run(None, {input_name: np.zeros((1, 30, 63), dtype=np.float32)})
        print("  lstm_session warmup complete.")
    return lstm_session

# ── Fast LSTM inference via ONNX Runtime ──
def fast_lstm_infer(x):
    session = get_lstm_session()
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: x})[0]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def decode_frame(frame_b64: str):
    try:
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]
        raw = base64.b64decode(frame_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.flip(img, 1)  # Flip horizontally to match mirrored training distribution
        return img
    except Exception:
        return None

def detect(bgr_frame):
    try:
        rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return get_hand_landmarker().detect(mp_img)
    except Exception:
        return None

def resample_sequence(seq, target_len=30):
    seq = np.array(seq, dtype=np.float32)
    n = len(seq)
    if n == target_len:
        return seq
    indices = np.linspace(0, n - 1, target_len)
    resampled = np.zeros((target_len, seq.shape[1]), dtype=seq.dtype)
    for i in range(seq.shape[1]):
        resampled[:, i] = np.interp(indices, np.arange(n), seq[:, i])
    return resampled

def normalize_landmarks(coords):
    coords = np.array(coords, dtype=np.float32).reshape(21, 3)
    wrist = coords[0]
    translated = coords - wrist
    dists = np.linalg.norm(translated, axis=1)
    max_dist = dists.max()
    if max_dist > 0:
        normalized = translated / max_dist
    else:
        normalized = translated
    return normalized.flatten()

def get_index_straightness(hand_landmarks):
    # landmarks 5, 6, 7, 8 are index MCP, PIP, DIP, Tip
    c5 = np.array([hand_landmarks[5].x, hand_landmarks[5].y, hand_landmarks[5].z])
    c6 = np.array([hand_landmarks[6].x, hand_landmarks[6].y, hand_landmarks[6].z])
    c7 = np.array([hand_landmarks[7].x, hand_landmarks[7].y, hand_landmarks[7].z])
    c8 = np.array([hand_landmarks[8].x, hand_landmarks[8].y, hand_landmarks[8].z])
    d58 = np.linalg.norm(c5 - c8)
    d56 = np.linalg.norm(c5 - c6)
    d67 = np.linalg.norm(c6 - c7)
    d78 = np.linalg.norm(c7 - c8)
    denom = d56 + d67 + d78
    return d58 / denom if denom > 0 else 1.0

def landmarks_to_vectors(result):
    """Return a list of 63-feature normalized landmark vectors for each detected hand."""
    if not result or not result.hand_landmarks:
        return []
    vectors = []
    for hand in result.hand_landmarks:
        vec = []
        for lm in hand:
            vec.extend([lm.x, lm.y, lm.z])
        vec = vec[:FEAT_PER_HAND]
        while len(vec) < FEAT_PER_HAND:
            vec.append(0.0)
        # Normalize landmarks to make them translation & scale-invariant
        normalized = normalize_landmarks(vec)
        vectors.append(normalized)
    return vectors

def landmarks_to_list(result):
    """Return list of lists of {x,y,z} dicts for all detected hands (for frontend skeletons)."""
    if not result or not result.hand_landmarks:
        return []
    res = []
    for hand in result.hand_landmarks:
        # Un-flip coordinates horizontally (1.0 - x) so they map correctly on CSS-mirrored preview
        res.append([{"x": 1.0 - lm.x, "y": lm.y, "z": lm.z} for lm in hand])
    return res

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ROUTES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok", "message": "SilentSpeak API is running"})

@app.route("/health", methods=["GET"])
def health():
    model_time = get_file_timestamp_str(LSTM_MODEL_PATH)
    label_time = get_file_timestamp_str(REVERSE_MAP_PATH)
    return jsonify({
        "status": "ok",
        "models_loaded": True,
        "num_hands": NUM_HANDS,
        "classes_count": len(custom_labels),
        "backend_version": "1.2.0",
        "model_version": f"GRU-2Layer-v2 ({model_time})",
        "label_map_version": f"WLASL-54-v2 ({label_time})"
    })

@app.route("/predict_alpha", methods=["POST"])
@app.route("/predict_alphabet", methods=["POST"])
def predict_alpha():
    global primary_sequence_buffer, last_normalized_vec, last_landmarks_list, hand_loss_counter
    try:
        print("[Server] Received predict_alpha request")
        data = request.get_json(silent=True)
        if not data:
            print("[Server] Request body is empty")
            return jsonify({"detected": False}), 400

        frame_data = data.get("image") or data.get("frame")
        if not frame_data:
            print("[Server] No frame data provided in request")
            return jsonify({"detected": False}), 400

        img = decode_frame(frame_data)
        if img is None:
            print("[Server] Failed to decode base64 frame")
            return jsonify({"detected": False, "message": "Could not decode image"})
        
        print(f"[Server] Frame decoded. Shape: {img.shape}")
        print("[Server] Running MediaPipe HandLandmarker...")
        result = detect(img)
        
        num_hands = len(result.hand_landmarks) if result and result.hand_landmarks else 0
        print(f"[Server] MediaPipe output: {num_hands} hand(s) detected")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CASE A: No hands detected (Grace Period or Clear Buffer)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if num_hands == 0:
            hand_loss_counter += 1
            print(f"[Server] No hand detected. hand_loss_counter={hand_loss_counter}/{GRACE_PERIOD_FRAMES}")
            
            if hand_loss_counter <= GRACE_PERIOD_FRAMES and last_normalized_vec is not None:
                # Grace period: forward-fill the sequence buffer with the last known frame
                primary_sequence_buffer.append(last_normalized_vec)
                if len(primary_sequence_buffer) > SEQUENCE_LENGTH:
                    primary_sequence_buffer.pop(0)
                
                print(f"[Server] Grace Period: Forward-filled buffer. Len: {len(primary_sequence_buffer)}")
                
                # Predict word using the filled buffer
                word_sign = None
                word_conf = 0.0
                word_top5 = []
                
                if len(primary_sequence_buffer) == SEQUENCE_LENGTH:
                    seq_arr = np.array(primary_sequence_buffer, dtype=np.float32)
                    X_lstm = seq_arr.reshape(1, SEQUENCE_LENGTH, FEAT_PER_HAND)
                    pred_lstm = fast_lstm_infer(X_lstm)[0]
                    best_idx_lstm = int(np.argmax(pred_lstm))
                    word_conf = float(pred_lstm[best_idx_lstm]) * 100.0
                    word_sign = reverse_label_map.get(best_idx_lstm, str(best_idx_lstm))
                    
                    # Extract top 5 predictions for word
                    top5_idxs_word = np.argsort(pred_lstm)[-5:][::-1]
                    for idx in top5_idxs_word:
                        lbl = reverse_label_map.get(idx, str(idx))
                        c_val = float(pred_lstm[idx]) * 100.0
                        word_top5.append({"label": lbl, "confidence": round(c_val, 2)})
                
                # Return prediction with skeleton persistence!
                print(f"[Server] Grace Period Word Prediction: {word_sign} ({word_conf:.2f}%)")
                return jsonify({
                    "detected":            True,
                    "alpha_sign":          None,
                    "alpha_confidence":    0.0,
                    "word_sign":           word_sign,
                    "word_confidence":     word_conf,
                    "landmarks":           last_landmarks_list,  # PERSISTENT SKELETON!
                    "top3":                word_top5[:3],
                    "raw_alpha_sign":      "—",
                    "raw_word_sign":       word_sign or "—",
                    "raw_word_confidence": word_conf,
                    "sequence_len":        len(primary_sequence_buffer),
                    "word_model_active":   True,
                    "message":             "grace_period",
                })
            else:
                # Exceeded grace period or no last frame: clear buffer
                primary_sequence_buffer = []
                last_normalized_vec = None
                last_landmarks_list = []
                print("[Server] Grace period exceeded or no last frame. Clearing sequence buffer.")
                return jsonify({
                    "detected":            False,
                    "alpha_sign":          None,
                    "alpha_confidence":    0.0,
                    "word_sign":           None,
                    "word_confidence":     0.0,
                    "landmarks":           [],
                    "top3":                [],
                    "raw_alpha_sign":      "—",
                    "raw_word_sign":       "—",
                    "raw_word_confidence": 0.0,
                    "sequence_len":        0,
                    "word_model_active":   False,
                    "message":             "No hand detected",
                })

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CASE B: Hand detected
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Reset hand loss counter
        hand_loss_counter = 0
        
        # Get primary hand landmarks (first detected hand)
        hand_lms = result.hand_landmarks[0]
        
        # Determine handedness label for logging
        if result.handedness and len(result.handedness) > 0:
            hand_label = result.handedness[0][0].category_name
        else:
            hand_label = "Right"
            
        # Get hand landmarks formatting for frontend skeleton rendering (both hands)
        lm_list = landmarks_to_list(result)
        last_landmarks_list = lm_list  # Save current landmarks for persistence
            
        # 1. Extract and normalize current frame vector (63 floats)
        vec = []
        for lm in hand_lms:
            vec.extend([lm.x, lm.y, lm.z])
        vec = vec[:FEAT_PER_HAND]
        while len(vec) < FEAT_PER_HAND:
            vec.append(0.0)
            
        # Align left hand (detected as "Left") to right-hand canonical space
        if hand_label.lower() == "left":
            for idx in range(0, 63, 3):
                vec[idx] = 1.0 - vec[idx]
                
        normalized_vec = normalize_landmarks(vec)
        
        # Apply Exponential Moving Average (EMA) landmark smoothing
        if last_normalized_vec is not None:
            smoothed_vec = SMOOTHING_FACTOR * normalized_vec + (1.0 - SMOOTHING_FACTOR) * last_normalized_vec
        else:
            smoothed_vec = normalized_vec
            
        # Update last normalized vec with the smoothed coordinates
        last_normalized_vec = smoothed_vec
        
        # 2. Add smoothed landmarks to the primary sequence buffer
        primary_sequence_buffer.append(smoothed_vec)
        if len(primary_sequence_buffer) > SEQUENCE_LENGTH:
            primary_sequence_buffer.pop(0)
            
        print(f"[Server Debug] First 6 smoothed coords: {[round(float(x), 6) for x in smoothed_vec[:6]]}")
        print(f"[Server] Hand detected. Primary buffer length: {len(primary_sequence_buffer)}")
        
        # 3. Predict alphabet ONLY using alpha_model
        X_alpha = smoothed_vec.reshape(1, -1)
        straightness = get_index_straightness(hand_lms)
        
        if hasattr(alpha_model, "predict_proba"):
            proba_alpha = alpha_model.predict_proba(X_alpha)[0].copy()
            
            # B vs F override
            try:
                idx_B = alpha_labels.index('B')
            except ValueError:
                idx_B = -1
            try:
                idx_F = alpha_labels.index('F')
            except ValueError:
                idx_F = -1
                
            if idx_B != -1 and idx_F != -1:
                if straightness > 0.90:
                    proba_alpha[idx_B] += proba_alpha[idx_F]
                    proba_alpha[idx_F] = 0.0
                elif straightness < 0.80:
                    proba_alpha[idx_B] = 0.0
                    
                p_sum = proba_alpha.sum()
                if p_sum > 0:
                    proba_alpha = proba_alpha / p_sum
            
            # Extract top 5 predictions for alphabet
            top5_idxs_alpha = np.argsort(proba_alpha)[-5:][::-1]
            hand_top5_alpha = []
            for idx in top5_idxs_alpha:
                label = str(alpha_labels[idx])
                conf = float(proba_alpha[idx]) * 100.0
                hand_top5_alpha.append({"label": label, "confidence": round(conf, 2)})
                
            alpha_sign = hand_top5_alpha[0]["label"]
            alpha_conf = hand_top5_alpha[0]["confidence"]
        else:
            alpha_sign = str(alpha_model.predict(X_alpha)[0])
            if alpha_sign == 'F' and straightness > 0.90:
                alpha_sign = 'B'
            elif alpha_sign == 'B' and straightness < 0.80:
                alpha_sign = 'F'
            alpha_conf = 100.0
            hand_top5_alpha = [{"label": alpha_sign, "confidence": 100.0}]

        # 4. Predict word using lstm_model if primary sequence buffer is full
        word_sign = None
        word_conf = 0.0
        word_top5 = []
        
        if len(primary_sequence_buffer) == SEQUENCE_LENGTH:
            seq_arr = np.array(primary_sequence_buffer, dtype=np.float32)  # shape (30, 63)
            X_lstm = seq_arr.reshape(1, SEQUENCE_LENGTH, FEAT_PER_HAND)
            
            pred_lstm = fast_lstm_infer(X_lstm)[0]
            best_idx_lstm = int(np.argmax(pred_lstm))
            word_conf = float(pred_lstm[best_idx_lstm]) * 100.0
            word_sign = reverse_label_map.get(best_idx_lstm, str(best_idx_lstm))
            
            # Extract top 5 predictions for word
            top5_idxs_word = np.argsort(pred_lstm)[-5:][::-1]
            for idx in top5_idxs_word:
                lbl = reverse_label_map.get(idx, str(idx))
                c_val = float(pred_lstm[idx]) * 100.0
                word_top5.append({"label": lbl, "confidence": round(c_val, 2)})
                
            print(f"[Server] Word predicted: {word_sign} ({word_conf:.2f}%)")

        # Confidence mediation:
        # Word wins if it has >=55% confidence OR it beats alphabet confidence at >=50%.
        # Alphabet wins otherwise (static postures without sequence context).
        is_word_confident = (word_sign and word_conf >= 55.0)
        if is_word_confident or (word_sign and word_conf > alpha_conf and word_conf >= 50.0):
            ret_word_sign = word_sign
            ret_word_conf = word_conf
            ret_alpha_sign = None
            ret_alpha_conf = 0.0
            display_top = word_top5
        else:
            ret_word_sign = None
            ret_word_conf = 0.0
            ret_alpha_sign = alpha_sign
            ret_alpha_conf = alpha_conf
            display_top = hand_top5_alpha

        print(f"[Server] Best Alpha: {alpha_sign} ({alpha_conf:.2f}%) | Best Word: {word_sign} ({word_conf:.2f}%) -> Winner: {ret_word_sign or ret_alpha_sign} ({ret_word_conf or ret_alpha_conf:.2f}%)")

        return jsonify({
            "detected":            True,
            "alpha_sign":          ret_alpha_sign,
            "alpha_confidence":    ret_alpha_conf,
            "word_sign":           ret_word_sign,
            "word_confidence":     ret_word_conf,
            "landmarks":           lm_list,
            "top3":                display_top,
            "raw_alpha_sign":      alpha_sign,
            "raw_word_sign":       word_sign or "—",
            "raw_word_confidence": word_conf,
            "sequence_len":        len(primary_sequence_buffer),
            "word_model_active":   True,
            "message":             "success",
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc), "detected": False}), 500


def get_index_straightness_from_raw(lm_raw):
    """lm_raw: list of 21 items, each [x, y, z]. Returns straightness ratio."""
    try:
        c5 = np.array(lm_raw[5], dtype=np.float32)
        c6 = np.array(lm_raw[6], dtype=np.float32)
        c7 = np.array(lm_raw[7], dtype=np.float32)
        c8 = np.array(lm_raw[8], dtype=np.float32)
        d58 = np.linalg.norm(c5 - c8)
        d56 = np.linalg.norm(c5 - c6)
        d67 = np.linalg.norm(c6 - c7)
        d78 = np.linalg.norm(c7 - c8)
        denom = d56 + d67 + d78
        return float(d58 / denom) if denom > 0 else 1.0
    except Exception:
        return 1.0


def get_finger_straightness_from_raw(lm_raw, base_idx, tip_idx):
    """Calculate straightness of a single finger using its base and tip index."""
    try:
        c_base = np.array(lm_raw[base_idx], dtype=np.float32)
        c_tip = np.array(lm_raw[tip_idx], dtype=np.float32)
        c1 = np.array(lm_raw[base_idx + 1], dtype=np.float32)
        c2 = np.array(lm_raw[base_idx + 2], dtype=np.float32)
        d_base_tip = np.linalg.norm(c_base - c_tip)
        d_joints = (np.linalg.norm(c_base - c1) + 
                    np.linalg.norm(c1 - c2) + 
                    np.linalg.norm(c2 - c_tip))
        return float(d_base_tip / d_joints) if d_joints > 0 else 1.0
    except Exception:
        return 0.0


def process_landmarks_prediction(data):
    """
    Core prediction logic that takes the JSON data dict and returns a tuple/dict of results.
    Specifically: returns (response_dict, raw_probabilities)
    where raw_probabilities is a dictionary of class -> float.
    """
    global primary_sequence_buffer, alpha_prediction_buffer, last_normalized_vec, last_landmarks_list, hand_loss_counter
    
    mode = data.get("mode", "alphabet")
    lm_raw = data.get("landmarks")
    if not lm_raw or len(lm_raw) != 21:
        if mode == "word" and data.get("sequence") is not None:
            lm_raw = [[0.0, 0.0, 0.0] for _ in range(21)]
        else:
            return {"error": "Need exactly 21 landmarks", "detected": False}, None

    # Mirror left hand (which is detected as "Left") to canonical right-hand
    handedness = data.get("handedness", "Right")
    if handedness.lower() == "right":
        # Copy lm_raw to avoid modifying in-place (since it might be reused)
        lm_raw = [[1.0 - lm[0], lm[1], lm[2]] for lm in lm_raw]
        
        client_sequence = data.get("sequence")
        if client_sequence is not None:
            new_seq = []
            for frame in client_sequence:
                f_copy = list(frame)
                for idx in range(0, 63, 3):
                    f_copy[idx] = 1.0 - f_copy[idx]
                new_seq.append(f_copy)
            data["sequence"] = new_seq
    else:
        # Just copy lm_raw to avoid shared side effects (Right hand is already canonical)
        lm_raw = [[lm[0], lm[1], lm[2]] for lm in lm_raw]

    # ── Build 63-float vector ────────────────────────────────────────────────
    vec = []
    for lm in lm_raw:
        vec.extend([float(lm[0]), float(lm[1]), float(lm[2])])
    vec = vec[:FEAT_PER_HAND]
    while len(vec) < FEAT_PER_HAND:
        vec.append(0.0)

    normalized_vec = normalize_landmarks(vec)

    # ── EMA smoothing ────────────────────────────────────────────────────────
    if last_normalized_vec is not None:
        smoothed_vec = SMOOTHING_FACTOR * normalized_vec + (1.0 - SMOOTHING_FACTOR) * last_normalized_vec
    else:
        smoothed_vec = normalized_vec
    last_normalized_vec = smoothed_vec
    hand_loss_counter = 0

    # ── Sequence buffer synchronization ──────────────────────────────────────
    primary_sequence_buffer.append(smoothed_vec)
    if len(primary_sequence_buffer) > SEQUENCE_LENGTH:
        primary_sequence_buffer.pop(0)

    client_sequence = data.get("sequence")
    if client_sequence is not None:
        primary_sequence_buffer_local = [normalize_landmarks(x) for x in client_sequence]
    else:
        primary_sequence_buffer_local = list(primary_sequence_buffer)

    last_landmarks_list = [{"x": lm[0], "y": lm[1], "z": lm[2]} for lm in lm_raw]

    mode = data.get("mode", "alphabet")
    client_seq_len = int(data.get("seq_len", 0)) if data else 0
    min_conf_alpha = float(data.get("min_conf_alpha", 15.0)) if data else 15.0
    min_conf_word = float(data.get("min_conf_word", 55.0)) if data else 55.0

    alpha_sign = None
    alpha_conf = 0.0
    top5_a = []
    word_sign = None
    word_conf = 0.0
    top5_w = []
    raw_probabilities = {}
    rejection_reason = ""

    if mode == "alphabet":
        # ── Alphabet prediction (Random Forest) - ALWAYS RUN (Frame-Based) ───────
        X_alpha = smoothed_vec.reshape(1, -1)
        straightness = get_index_straightness_from_raw(lm_raw)

        if hasattr(alpha_model, "predict_proba"):
            proba = alpha_model.predict_proba(X_alpha)[0].copy()
            # Overrides commented out for Alphabet Recovery Phase to restore raw model accuracy.
            # B vs F, R crossed, and N vs M vs A fist knuckling checks disabled.
            
            # B finger straightness override (strict, regression-free, includes R confusion fix)
            try:
                idx_B = alpha_labels.index('B')
                idx_F = alpha_labels.index('F')
                idx_M = alpha_labels.index('M')
                idx_N = alpha_labels.index('N')
                idx_C = alpha_labels.index('C')
                idx_R = alpha_labels.index('R')
                
                s_idx = get_finger_straightness_from_raw(lm_raw, 5, 8)
                s_mid = get_finger_straightness_from_raw(lm_raw, 9, 12)
                s_rng = get_finger_straightness_from_raw(lm_raw, 13, 16)
                s_pky = get_finger_straightness_from_raw(lm_raw, 17, 20)
                
                if s_idx > 0.90 and s_mid > 0.90 and s_rng > 0.90 and s_pky > 0.90 and proba[idx_B] > 0.20:
                    proba[idx_B] += proba[idx_F] + proba[idx_M] + proba[idx_N] + proba[idx_C] + proba[idx_R]
                    proba[idx_F] = 0.0
                    proba[idx_M] = 0.0
                    proba[idx_N] = 0.0
                    proba[idx_C] = 0.0
                    proba[idx_R] = 0.0
            except (ValueError, IndexError):
                pass

            top5_a = []
            for idx in np.argsort(proba)[-5:][::-1]:
                top5_a.append({"label": str(alpha_labels[idx]), "confidence": round(float(proba[idx]) * 100, 2)})
            alpha_sign = top5_a[0]["label"]
            alpha_conf = top5_a[0]["confidence"]
            raw_probabilities = {str(alpha_labels[idx]): float(proba[idx]) for idx in range(len(proba))}
        else:
            alpha_sign = str(alpha_model.predict(X_alpha)[0])
            alpha_conf = 100.0
            top5_a = [{"label": alpha_sign, "confidence": 100.0}]
            raw_probabilities = {alpha_sign: 1.0}

        # Gating / Majority Voting
        if alpha_conf >= min_conf_alpha:
            alpha_prediction_buffer.append(alpha_sign)
        else:
            alpha_prediction_buffer.append("No prediction")

        if len(alpha_prediction_buffer) > 5:
            alpha_prediction_buffer.pop(0)

        from collections import Counter
        counts = Counter(alpha_prediction_buffer)
        most_common_sign, count = counts.most_common(1)[0]
        
        smoothed_alpha_sign = None if most_common_sign == "No prediction" else most_common_sign
        ret_alpha_sign = smoothed_alpha_sign
        ret_alpha_conf = alpha_conf if ret_alpha_sign is not None else 0.0

        if ret_alpha_sign is None:
            rejection_reason = f"Confidence {alpha_conf:.1f}% below threshold {min_conf_alpha:.1f}%"

        res_dict = {
            "detected":           True,
            "alpha_sign":         ret_alpha_sign,
            "alpha_confidence":   ret_alpha_conf,
            "word_sign":          None,
            "word_confidence":    0.0,
            "sequence_len":       0,
            "top5":               top5_a,
            "top5_alpha":         top5_a,
            "top5_word":          [],
            "raw_alpha":          alpha_sign,
            "raw_alpha_conf":     alpha_conf,
            "raw_word":           "—",
            "raw_word_conf":      0.0,
            "word_model_active":  False,
            "active_model":       "Alphabet (RF)",
            "final_prediction":   ret_alpha_sign or "No prediction",
            "rejection_reason":   rejection_reason
        }
        return res_dict, raw_probabilities

    elif mode == "word":
        # ── Word prediction (LSTM) - RUN ONLY WHEN SEQUENCE IS FULL ──────────────
        is_seq_completed = (client_seq_len == SEQUENCE_LENGTH) or (len(primary_sequence_buffer_local) == SEQUENCE_LENGTH)
        if is_seq_completed and len(primary_sequence_buffer_local) > 0:
            while len(primary_sequence_buffer_local) < SEQUENCE_LENGTH:
                primary_sequence_buffer_local.insert(0, primary_sequence_buffer_local[0])
            seq_arr = np.array(primary_sequence_buffer_local, dtype=np.float32)
            X_lstm = seq_arr.reshape(1, SEQUENCE_LENGTH, FEAT_PER_HAND)
            pred = fast_lstm_infer(X_lstm)[0]
            best_idx = int(np.argmax(pred))
            word_conf = float(pred[best_idx]) * 100.0
            word_sign = reverse_label_map.get(best_idx, str(best_idx))
            for idx in np.argsort(pred)[-5:][::-1]:
                top5_w.append({"label": reverse_label_map.get(idx, str(idx)), "confidence": round(float(pred[idx]) * 100, 2)})
            raw_probabilities = {reverse_label_map[idx]: float(pred[idx]) for idx in range(len(pred))}
            
            # Print debug for target word logging
            TARGET_DEBUG_WORDS = {'hello', 'help', 'water', 'good', 'mother', 'family', 'school', 'go', 'come', 'what', 'where', 'need'}
            is_word_confident = (word_sign is not None and word_conf >= min_conf_word and word_sign in ALLOWED_WORDS)
            ret_word_sign = word_sign if is_word_confident else None
            ret_word_conf = word_conf if is_word_confident else 0.0
            
            if word_sign is not None and word_sign not in ALLOWED_WORDS:
                rejection_reason = f"Word '{word_sign}' is disabled/unstable"
            elif not is_word_confident:
                rejection_reason = f"Confidence {word_conf:.1f}% below threshold {min_conf_word:.1f}%"
            
            if word_sign in TARGET_DEBUG_WORDS:
                print(f"\n[DEBUG TARGET WORD] Word: {word_sign}")
                print(f"  Raw Prediction: {word_sign} | Confidence: {word_conf:.2f}% | Final: {ret_word_sign or 'No prediction'}")
                print(f"  Top 5: {top5_w}")
                print(f"  Rejection Reason: {rejection_reason or 'None (Passed)'}")
        else:
            rejection_reason = f"Sequence not completed (len: {len(primary_sequence_buffer_local)} < {SEQUENCE_LENGTH})"
            is_word_confident = False
            ret_word_sign = None
            ret_word_conf = 0.0

        res_dict = {
            "detected":           True,
            "alpha_sign":         None,
            "alpha_confidence":   0.0,
            "word_sign":          ret_word_sign,
            "word_confidence":    ret_word_conf,
            "sequence_len":       len(primary_sequence_buffer_local),
            "top5":               top5_w,
            "top5_alpha":         [],
            "top5_word":          top5_w,
            "raw_alpha":          "—",
            "raw_alpha_conf":     0.0,
            "raw_word":           word_sign or "—",
            "raw_word_conf":      word_conf,
            "word_model_active":  is_word_confident and is_seq_completed,
            "active_model":       "WLASL (LSTM)",
            "final_prediction":   ret_word_sign or "No prediction",
            "rejection_reason":   rejection_reason
        }
        return res_dict, raw_probabilities


@app.route("/predict_from_landmarks", methods=["POST"])
def predict_from_landmarks():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"detected": False, "message": "No JSON body"}), 400
            
        res_dict, _ = process_landmarks_prediction(data)
        if "error" in res_dict:
            return jsonify(res_dict), 400
            
        return jsonify(res_dict)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc), "detected": False}), 500


@app.route("/predict_debug", methods=["POST"])
def predict_debug():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON body"}), 400
            
        res_dict, raw_probs = process_landmarks_prediction(data)
        if "error" in res_dict:
            return jsonify(res_dict), 400
            
        # Format the debug response
        debug_response = {
            "status": "success",
            "raw_probabilities": raw_probs,
            "top_5": res_dict.get("top5", []),
            "confidence": res_dict.get("raw_word_conf" if data.get("mode") == "word" else "raw_alpha_conf", 0.0),
            "final_prediction": res_dict.get("final_prediction", "No prediction"),
            "rejection_reason": res_dict.get("rejection_reason", ""),
            "mode": data.get("mode", "alphabet")
        }
        return jsonify(debug_response)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/save_live_test", methods=["POST"])
def save_live_test():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload"}), 400
            
        debug_logs_dir = os.path.join(BASE_DIR, "debug_logs")
        os.makedirs(debug_logs_dir, exist_ok=True)
        
        save_path = os.path.join(debug_logs_dir, "live_test.json")
        with open(save_path, "w", encoding="utf-8") as f:
            import json
            json.dump(data, f, indent=2)
            
        print(f"[Server] Saved live test recording (length: {len(data.get('frames', []))} frames) to {save_path}")
        return jsonify({"status": "success", "message": f"Saved to {save_path}"})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/predict_word", methods=["POST"])

def predict_word_endpoint():
    global primary_sequence_buffer, last_normalized_vec, hand_loss_counter, last_landmarks_list
    try:
        print("\n[predict_word] --- Received word prediction request ---")
        data = request.get_json(silent=True)
        if not data:
            print("[predict_word] No JSON payload received")
            return jsonify({"error": "No JSON payload", "prediction": None, "confidence": 0.0}), 400

        # Case 1: Sequence of landmarks directly provided (shape [30, 63])
        if "sequence" in data:
            seq = data["sequence"]
            if not seq or len(seq) < 3:
                return jsonify({"error": "Sequence too short", "prediction": None, "confidence": 0.0}), 400
                
            # Align coordinates to right-hand canonical space
            handedness = data.get("handedness", "Right")
            aligned_seq = []
            for frame in seq:
                f_copy = list(frame)
                if handedness.lower() == "right":
                    # Flip x coordinates to map left hand to right hand space
                    for idx in range(0, 63, 3):
                        f_copy[idx] = 1.0 - f_copy[idx]
                aligned_seq.append(f_copy)
                
            # Resample to exactly SEQUENCE_LENGTH (30)
            resampled = resample_sequence(aligned_seq, SEQUENCE_LENGTH)
            
            # Normalize each frame in the sequence
            normalized_seq = [normalize_landmarks(frame) for frame in resampled]
            seq_arr = np.array(normalized_seq, dtype=np.float32)
            print(f"[predict_word] Direct sequence input resampled from {len(seq)} to {seq_arr.shape}")
            
            X_lstm = seq_arr.reshape(1, SEQUENCE_LENGTH, FEAT_PER_HAND)
            pred_lstm = fast_lstm_infer(X_lstm)[0]
            best_idx = int(np.argmax(pred_lstm))
            word_label = reverse_label_map.get(best_idx, str(best_idx))
            word_conf = float(pred_lstm[best_idx]) * 100.0
            
            if word_label not in ALLOWED_WORDS:
                print(f"[predict_word] BLOCKED disabled/unstable word prediction: '{word_label}'")
                return jsonify({
                    "prediction": None,
                    "confidence": 0.0,
                    "sequence_len": len(seq),
                    "required_len": SEQUENCE_LENGTH,
                    "status": "disabled"
                })
                
            print(f"[predict_word] SUCCESS: Predicted word: '{word_label}' | Confidence: {word_conf:.2f}% | Sequence length: {len(seq)}/{SEQUENCE_LENGTH}")
            return jsonify({
                "prediction": word_label,
                "confidence": word_conf,
                "sequence_len": len(seq),
                "required_len": SEQUENCE_LENGTH,
                "status": "success"
            })

        # Case 2: Image frame provided (accumulate in sliding window)
        frame_data = data.get("image") or data.get("frame")
        if not frame_data:
            print("[predict_word] No image or sequence provided in request")
            return jsonify({"error": "No image or sequence provided", "prediction": None, "confidence": 0.0}), 400

        img = decode_frame(frame_data)
        if img is None:
            print("[predict_word] Failed to decode base64 frame")
            return jsonify({"error": "Could not decode image", "prediction": None, "confidence": 0.0}), 400

        result = detect(img)
        num_hands = len(result.hand_landmarks) if result and result.hand_landmarks else 0
        
        if num_hands == 0:
            hand_loss_counter += 1
            print(f"[predict_word] No hand detected. hand_loss_counter={hand_loss_counter}/{GRACE_PERIOD_FRAMES}")
            
            if hand_loss_counter <= GRACE_PERIOD_FRAMES and last_normalized_vec is not None:
                primary_sequence_buffer.append(last_normalized_vec)
                if len(primary_sequence_buffer) > SEQUENCE_LENGTH:
                    primary_sequence_buffer.pop(0)
            else:
                primary_sequence_buffer = []
                last_normalized_vec = None
                last_landmarks_list = []
        else:
            hand_loss_counter = 0
            hand_lms = result.hand_landmarks[0]
            
            # Save landmarks list for persistence
            lm_list = landmarks_to_list(result)
            last_landmarks_list = lm_list
            
            vec = []
            for lm in hand_lms:
                vec.extend([lm.x, lm.y, lm.z])
            vec = vec[:FEAT_PER_HAND]
            while len(vec) < FEAT_PER_HAND:
                vec.append(0.0)
                
            # Align coordinates to right-hand canonical space
            if result.handedness and len(result.handedness) > 0:
                hand_label = result.handedness[0][0].category_name
            else:
                hand_label = "Right"
                
            if hand_label.lower() == "left":
                for idx in range(0, 63, 3):
                    vec[idx] = 1.0 - vec[idx]
                    
            normalized_vec = normalize_landmarks(vec)
            
            # Apply EMA smoothing
            if last_normalized_vec is not None:
                smoothed_vec = SMOOTHING_FACTOR * normalized_vec + (1.0 - SMOOTHING_FACTOR) * last_normalized_vec
            else:
                smoothed_vec = normalized_vec
                
            last_normalized_vec = smoothed_vec
            primary_sequence_buffer.append(smoothed_vec)
            if len(primary_sequence_buffer) > SEQUENCE_LENGTH:
                primary_sequence_buffer.pop(0)

        seq_len = len(primary_sequence_buffer)
        print(f"[predict_word] Request processed. Sequence buffer: {seq_len}/{SEQUENCE_LENGTH}")

        if seq_len == SEQUENCE_LENGTH:
            seq_arr = np.array(primary_sequence_buffer, dtype=np.float32)
            X_lstm = seq_arr.reshape(1, SEQUENCE_LENGTH, FEAT_PER_HAND)
            pred_lstm = fast_lstm_infer(X_lstm)[0]
            best_idx = int(np.argmax(pred_lstm))
            word_conf = float(pred_lstm[best_idx]) * 100.0
            word_label = reverse_label_map.get(best_idx, str(best_idx))

            print(f"[predict_word] SUCCESS: Predicted word: '{word_label}' | Confidence: {word_conf:.2f}% | Sequence length: {seq_len}/{SEQUENCE_LENGTH}")
            return jsonify({
                "prediction": word_label,
                "confidence": word_conf,
                "sequence_len": seq_len,
                "required_len": SEQUENCE_LENGTH,
                "status": "success"
            })
        else:
            print(f"[predict_word] Accumulating frames: {seq_len}/{SEQUENCE_LENGTH}")
            return jsonify({
                "prediction": None,
                "confidence": 0.0,
                "sequence_len": seq_len,
                "required_len": SEQUENCE_LENGTH,
                "status": "accumulating"
            })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "prediction": None, "confidence": 0.0}), 500


STARTUP_TIME = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@app.route("/version", methods=["GET"])
def version():
    mtime = os.path.getmtime(__file__)
    modified_time = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({
        "startup_timestamp": STARTUP_TIME,
        "pid": os.getpid(),
        "server_modified_timestamp": modified_time
    })


@app.route("/diagnostics", methods=["GET"])
def diagnostics():
    results = {}
    
    # Test cv2
    try:
        import cv2
        results["cv2"] = "SUCCESS"
    except Exception as e:
        results["cv2"] = f"FAILURE: {str(e)}"
        
    # Test MediaPipe
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        # Trigger hand_landmarker lazy load
        get_hand_landmarker()
        results["mediapipe"] = "SUCCESS"
    except Exception as e:
        results["mediapipe"] = f"FAILURE: {str(e)}"
        
    # Test ONNX Runtime
    try:
        import onnxruntime as ort
        # Trigger lstm_session lazy load
        get_lstm_session()
        results["onnxruntime"] = "SUCCESS"
    except Exception as e:
        results["onnxruntime"] = f"FAILURE: {str(e)}"
        
    # Test loaded models
    try:
        results["model_custom"] = "SUCCESS" if custom_model is not None else "LOADED_NONE"
        results["model_alpha"] = "SUCCESS" if alpha_model is not None else "LOADED_NONE"
        results["verification_error"] = verification_error
    except Exception as e:
        results["model_loading"] = f"FAILURE: {str(e)}"
        
    return jsonify(results), 200


@app.route("/reset_sequence", methods=["POST"])
def reset_sequence():
    global primary_sequence_buffer, last_normalized_vec, last_landmarks_list, hand_loss_counter, alpha_prediction_buffer
    primary_sequence_buffer = []
    alpha_prediction_buffer = []
    last_normalized_vec = None
    last_landmarks_list = []
    hand_loss_counter = 0
    print("[Server] Global primary sequence buffer reset successfully")
    return jsonify({"message": "Sequence buffer reset", "status": "success"})



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Server starting on http://0.0.0.0:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)