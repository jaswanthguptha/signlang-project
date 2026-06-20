import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = 'data'
MODELS_DIR = 'models'

# Only the 26 ASL alphabet signs
SIGNS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

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

def add_noise(coords, std_dev=0.003):
    # Add minor Gaussian noise to coordinates
    noise = np.random.normal(0, std_dev, coords.shape)
    # Ensure wrist remains at 0, 0, 0
    augmented = coords + noise
    # Re-normalize just in case
    return normalize_landmarks(augmented)

def train_alpha_model():
    print("📂 Loading alphabet data...")
    X = []
    y = []
    
    for sign in SIGNS:
        sign_dir = os.path.join(DATA_DIR, sign)
        if not os.path.exists(sign_dir):
            print(f"⚠️ Warning: Folder for {sign} not found")
            continue
            
        files = [f for f in os.listdir(sign_dir) if f.endswith('.npy')]
        count = 0
        for file in files:
            landmarks = np.load(os.path.join(sign_dir, file))
            # Keep original raw coords (first 63 values)
            raw = landmarks[:63]
            norm = normalize_landmarks(raw)
            
            # 1. Original normalized sample
            X.append(norm)
            y.append(sign)
            count += 1
            
            # 2. Augmentation: Jitter level 1
            X.append(add_noise(raw, std_dev=0.003))
            y.append(sign)
            
            # 3. Augmentation: Jitter level 2
            X.append(add_noise(raw, std_dev=0.005))
            y.append(sign)
            
        print(f"Loaded {len(files)} raw samples for {sign} (augmented to {count * 3})")
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    print(f"✅ Total training set size (with augmentation): {len(X)}")
    
    # Stratified split to ensure balanced test representation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("🤖 Training Random Forest alphabet classifier...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Model Accuracy on Test Set: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    # Save the model and labels in exact sorted order of model.classes_
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "model_alpha.pkl")
    labels_path = os.path.join(MODELS_DIR, "labels_alpha.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"💾 Saved alphabet model to: {model_path}")
    
    # Crucial fix: serialize list(model.classes_) to avoid label mapping misalignment
    with open(labels_path, 'wb') as f:
        pickle.dump(list(model.classes_), f)
    print(f"💾 Saved alphabet labels to: {labels_path}")
    print(f"Classes list: {list(model.classes_)}")

if __name__ == '__main__':
    train_alpha_model()
