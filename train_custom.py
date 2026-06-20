import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

SIGNS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'hello', 'thanks', 'yes', 'no', 'help',
    'please', 'sorry', 'good', 'bad', 'eat',
    'water', 'more', 'stop', 'iloveyou'
]

DATA_DIR = 'data'
MODELS_DIR = 'models'

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

print("Loading data...")
X = []
y = []

for sign in SIGNS:
    sign_dir = os.path.join(DATA_DIR, sign)
    if not os.path.exists(sign_dir):
        print(f"Skipping {sign} - folder not found")
        continue
    count = 0
    for file in os.listdir(sign_dir):
        if file.endswith('.npy'):
            landmarks = np.load(os.path.join(sign_dir, file))
            # Normalize to make translation and scale invariant
            normalized = normalize_landmarks(landmarks[:63])
            X.append(normalized)
            y.append(sign)
            count += 1
    print(f"Loaded {count} samples for {sign}")

X = np.array(X)
y = np.array(y)

print(f"Total samples loaded: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=250,
    max_depth=25,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model and labels
os.makedirs(MODELS_DIR, exist_ok=True)
model_path = os.path.join(MODELS_DIR, "model_custom.pkl")
labels_path = os.path.join(MODELS_DIR, "labels_custom.pkl")

with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {model_path}")

with open(labels_path, 'wb') as f:
    pickle.dump(list(model.classes_), f)
print(f"Labels saved to {labels_path}")
print("Training and serialization completed successfully!")
