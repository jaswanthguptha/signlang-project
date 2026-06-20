import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# All your signs
SIGNS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'hello', 'thanks', 'yes', 'no', 'help',
    'please', 'sorry', 'good', 'bad', 'eat',
    'water', 'more', 'stop', 'iloveyou'
]

DATA_DIR = 'data'

print("📂 Loading data...")
X = []
y = []

for sign in SIGNS:
    sign_dir = os.path.join(DATA_DIR, sign)
    if not os.path.exists(sign_dir):
        print(f"⚠️ Skipping {sign} - no data found")
        continue
    for file in os.listdir(sign_dir):
        if file.endswith('.npy'):
            landmarks = np.load(os.path.join(sign_dir, file))
            X.append(landmarks)
            y.append(sign)

X = np.array(X)
y = np.array(y)

print(f"✅ Loaded {len(X)} samples for {len(SIGNS)} signs")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("🤖 Training AI model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save sign labels
with open('labels.pkl', 'wb') as f:
    pickle.dump(SIGNS, f)

print("\n✅ Model saved as model.pkl")
print("✅ Labels saved as labels.pkl")
print("\n🎉 Training complete! Ready for real-time detection!")