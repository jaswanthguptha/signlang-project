import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = 'data'
MODELS_DIR = 'models'
SIGNS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'hello', 'thanks', 'yes', 'no', 'help',
    'please', 'sorry', 'good', 'bad', 'eat',
    'water', 'more', 'stop', 'iloveyou'
]

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

def get_finger_straightness(c, base, pip, dip, tip):
    d_base_tip = np.linalg.norm(c[base] - c[tip])
    d_segments = (np.linalg.norm(c[base] - c[pip]) + 
                  np.linalg.norm(c[pip] - c[dip]) + 
                  np.linalg.norm(c[dip] - c[tip]))
    return d_base_tip / d_segments if d_segments > 0 else 1.0

def clean_dataset_and_retrain():
    # 1. Identify Outliers
    to_delete = []
    
    # We only run geometric cleaning on the alphabet signs A-Z
    ALPHABET_SIGNS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    for sign in ALPHABET_SIGNS:
        sign_dir = os.path.join(DATA_DIR, sign)
        if not os.path.exists(sign_dir):
            continue
        
        files = [f for f in os.listdir(sign_dir) if f.endswith('.npy')]
        samples = []
        features = []
        
        for f in files:
            path = os.path.join(sign_dir, f)
            raw = np.load(path)[:63]
            c = normalize_landmarks(raw).reshape(21, 3)
            
            idx_s = get_finger_straightness(c, 5, 6, 7, 8)
            mid_s = get_finger_straightness(c, 9, 10, 11, 12)
            rng_s = get_finger_straightness(c, 13, 14, 15, 16)
            pnk_s = get_finger_straightness(c, 17, 18, 19, 20)
            ti_dist = np.linalg.norm(c[4] - c[8])
            
            samples.append((f, path))
            features.append([idx_s, mid_s, rng_s, pnk_s, ti_dist])
            
        features = np.array(features)
        if len(features) == 0:
            continue
            
        # Rules-based outlier checks
        for i, (f, path) in enumerate(samples):
            idx_s, mid_s, rng_s, pnk_s, ti_dist = features[i]
            deleted = False
            
            # B must have straight fingers
            if sign == 'B':
                if idx_s < 0.85 or mid_s < 0.85 or rng_s < 0.85 or pnk_s < 0.85:
                    to_delete.append(path)
                    deleted = True
            
            # F must have bent index finger and touch thumb
            elif sign == 'F':
                if idx_s > 0.85 or ti_dist > 0.35:
                    to_delete.append(path)
                    deleted = True
                    
            # U must have index and middle straight and close
            elif sign == 'U':
                if idx_s < 0.80 or mid_s < 0.80 or rng_s > 0.80 or pnk_s > 0.80:
                    to_delete.append(path)
                    deleted = True
                else:
                    c = normalize_landmarks(np.load(path)[:63]).reshape(21, 3)
                    im_dist = np.linalg.norm(c[8] - c[12])
                    if im_dist > 0.20:
                        to_delete.append(path)
                        deleted = True
                    
            # V must have index and middle straight and spread
            elif sign == 'V':
                if idx_s < 0.80 or mid_s < 0.80 or rng_s > 0.80 or pnk_s > 0.80:
                    to_delete.append(path)
                    deleted = True
                else:
                    c = normalize_landmarks(np.load(path)[:63]).reshape(21, 3)
                    im_dist = np.linalg.norm(c[8] - c[12])
                    if im_dist < 0.15:
                        to_delete.append(path)
                        deleted = True
                    
            # W must have index, middle, ring straight and spread
            elif sign == 'W':
                if idx_s < 0.80 or mid_s < 0.80 or rng_s < 0.80 or pnk_s > 0.80:
                    to_delete.append(path)
                    deleted = True
            
            # R must have index and middle crossed
            elif sign == 'R':
                if idx_s < 0.80 or mid_s < 0.80:
                    to_delete.append(path)
                    deleted = True
            
            # Statistical anomaly (Z-score > 3.5)
            if not deleted and len(features) > 10:
                means = np.mean(features, axis=0)
                stds = np.std(features, axis=0)
                stds[stds == 0] = 1e-6
                z = np.abs((features[i] - means) / stds)
                if np.any(z > 3.5):
                    to_delete.append(path)

    # Deduplicate to_delete list
    to_delete = list(set(to_delete))
    print(f"Flagged {len(to_delete)} files for deletion.")
    
    # 2. Delete flagged files
    for p in to_delete:
        try:
            os.remove(p)
        except Exception as e:
            print(f"Error deleting {p}: {e}")
            
    print("Outlier deletion finished.")
    
    # 3. Load all data (cleaned)
    X = []
    y = []
    for sign in SIGNS:
        sign_dir = os.path.join(DATA_DIR, sign)
        if not os.path.exists(sign_dir):
            continue
        count = 0
        for file in os.listdir(sign_dir):
            if file.endswith('.npy'):
                landmarks = np.load(os.path.join(sign_dir, file))
                normalized = normalize_landmarks(landmarks[:63])
                X.append(normalized)
                y.append(sign)
                count += 1
        print(f"Loaded {count} cleaned samples for {sign}")
        
    X = np.array(X)
    y = np.array(y)
    print(f"Total cleaned dataset size: {len(X)}")
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Train Random Forest model
    print("Training optimized Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=300,        # Increased from 250 for better stability
        max_depth=30,            # Increased from 25 for better capacity
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 6. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nNew Model Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_pred, digits=4))
    
    # 7. Save the new model
    model_path = os.path.join(MODELS_DIR, "model_custom.pkl")
    labels_path = os.path.join(MODELS_DIR, "labels_custom.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Cleaned and retrained model saved to {model_path}")
    
    with open(labels_path, 'wb') as f:
        pickle.dump(list(model.classes_), f)
    print(f"Labels saved to {labels_path}")

if __name__ == '__main__':
    clean_dataset_and_retrain()
