import os
import sys
import pickle
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Force UTF-8 encoding for standard output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = 'data'
MODELS_DIR = 'models'

# The 35 production ASL classes
WLASL_CLASSES = [
    'hello', 'help', 'yes', 'no', 'thanks', 'please', 'water', 'eat', 'stop', 'iloveyou', 'good', 'bad', 'more', 'sorry',
    'friend', 'study', 'angry', 'year', 'home', 'happy', 'sad', 'day', 'morning', 'month', 'phone', 'hungry', 'night',
    'drink', 'where', 'who', 'need', 'computer', 'doctor', 'time', 'like'
]

SEQUENCE_LENGTH = 30
FEATURES = 63  # 21 landmarks * 3 coordinates

def resample_sequence(seq, target_len=30):
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

def mirror_sequence(seq):
    mirrored = seq.copy()
    mirrored[:, 0::3] = -mirrored[:, 0::3]
    return mirrored

def rotate_sequence_z(seq, theta):
    coords = seq.copy().reshape(-1, 21, 3)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x = coords[:, :, 0]
    y = coords[:, :, 1]
    coords[:, :, 0] = x * cos_t - y * sin_t
    coords[:, :, 1] = x * sin_t + y * cos_t
    return coords.reshape(-1, 63)

def add_noise(seq, std=0.005):
    return seq + np.random.normal(0, std, seq.shape)

def prepare_data():
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    
    label_map = {word: idx for idx, word in enumerate(WLASL_CLASSES)}
    np.random.seed(42)
    random.seed(42)
    
    for word in WLASL_CLASSES:
        folder_path = os.path.join(DATA_DIR, word)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder for {word} not found")
            continue
            
        files = os.listdir(folder_path)
        all_seq_files = [f for f in files if f.startswith('seq_') and f.endswith('.npy')]
        original_files = [f for f in all_seq_files if 'aug' not in f]
        
        if len(all_seq_files) == 0:
            print(f"Warning: No valid sequences found for '{word}'")
            continue
            
        word_idx = label_map[word]
        
        if len(original_files) > 1:
            # Kaggle-based class
            train_files, val_files = train_test_split(original_files, test_size=0.2, random_state=42)
            is_local_only = False
        else:
            # Local-only class: clean original as validation, augmented as training
            train_files = all_seq_files
            val_files = original_files
            is_local_only = True
            
        # Process Training Sequences
        class_train_windows = []
        for f in train_files:
            raw_seq = np.load(os.path.join(folder_path, f))
            resampled = resample_sequence(raw_seq, SEQUENCE_LENGTH)
            norm_seq = np.zeros_like(resampled)
            for f_idx in range(len(resampled)):
                norm_seq[f_idx] = normalize_landmarks(resampled[f_idx])
                
            class_train_windows.append(norm_seq)
            
            # Apply augmentations (mirroring, rotating, rotating + mirroring)
            class_train_windows.append(mirror_sequence(norm_seq))
            
            theta = np.random.uniform(-0.12, 0.12)
            class_train_windows.append(rotate_sequence_z(norm_seq, theta))
            class_train_windows.append(mirror_sequence(rotate_sequence_z(norm_seq, theta)))
            
        # Balance to exactly 1000 windows per class
        TARGET_WINDOWS = 200
        n_windows = len(class_train_windows)
        if n_windows > 0:
            indices = np.random.choice(n_windows, TARGET_WINDOWS, replace=True)
            balanced = [class_train_windows[i] for i in indices]
            for win in balanced:
                # Add tiny random noise for regularization
                noise_std = np.random.uniform(0.001, 0.003)
                noisy_win = add_noise(win, noise_std)
                X_train.append(noisy_win)
                y_train.append(word_idx)
                
        # Process Validation Sequences (clean, resampled to 30 frames, NO augmentations)
        val_count = 0
        for f in val_files:
            raw_seq = np.load(os.path.join(folder_path, f))
            resampled = resample_sequence(raw_seq, SEQUENCE_LENGTH)
            norm_seq = np.zeros_like(resampled)
            for f_idx in range(len(resampled)):
                norm_seq[f_idx] = normalize_landmarks(resampled[f_idx])
            X_val.append(norm_seq)
            y_val.append(word_idx)
            val_count += 1
            
        type_str = "Local-Only" if is_local_only else "Kaggle"
        print(f"Class '{word:10s}' ({type_str:10s}): Balanced training to {TARGET_WINDOWS}. Validation samples: {val_count}")
        
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val)
    
    return X_train, y_train, X_val, y_val, label_map

def build_model(num_classes):
    # Winning 2-layer GRU with 128 units each for capacity on WLASL dataset
    model = keras.Sequential([
        layers.GRU(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURES)),
        layers.Dropout(0.4),
        layers.GRU(128, return_sequences=False),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model():
    print("Preparing full-resampled dataset...")
    X_train, y_train, X_val, y_val, label_map = prepare_data()
    
    print(f"\nDataset size:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val  : {X_val.shape}, y_val  : {y_val.shape}")
    
    y_train_cat = to_categorical(y_train, len(WLASL_CLASSES))
    y_val_cat = to_categorical(y_val, len(WLASL_CLASSES))
    
    model = build_model(len(WLASL_CLASSES))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODELS_DIR, "best_lstm_words.keras")
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=8, restore_best_weights=True
    )
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, min_lr=0.0001, verbose=1
    )
    
    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=35,
        batch_size=128,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )
    
    print(f"\nLoading best checkpoint from: {checkpoint_path}")
    best_model = keras.models.load_model(checkpoint_path)
    
    val_loss, val_acc = best_model.evaluate(X_val, y_val_cat, verbose=0)
    print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")
    
    # Save best model to final destination
    model_path = os.path.join(MODELS_DIR, "lstm_words.keras")
    best_model.save(model_path)
    print(f"Saved optimized model to: {model_path}")
    
    # Save label mappings
    label_map_path = os.path.join(MODELS_DIR, "label_map.pkl")
    reverse_map_path = os.path.join(MODELS_DIR, "reverse_label_map.pkl")
    
    with open(label_map_path, 'wb') as f:
        pickle.dump(label_map, f)
    print(f"Saved label_map to: {label_map_path}")
    
    reverse_map = {idx: word for word, idx in label_map.items()}
    with open(reverse_map_path, 'wb') as f:
        pickle.dump(reverse_map, f)
    print(f"Saved reverse_label_map to: {reverse_map_path}")
    print("Training process finished successfully.")

if __name__ == '__main__':
    train_model()
