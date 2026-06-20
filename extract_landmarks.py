import os
import zipfile
import io
import pickle
import numpy as np
import pandas as pd

ZIP_PATH = r"C:\Users\User\Downloads\asl-signs.zip"
CSV_PATH = r"C:\Users\User\.gemini\antigravity\brain\5932710d-626d-434a-bcd9-6ac056d726a4\scratch\train.csv"
DATA_DIR = r"C:\Users\User\signlang_project\data"

TARGET_CLASSES = [
    'hello', 'help', 'yes', 'no', 'thanks', 'please', 'water', 'eat', 'stop', 'iloveyou', 'good', 'bad', 'more', 'sorry',
    'father', 'mother', 'friend', 'family', 'school', 'home', 'drink', 'phone', 'go', 'come', 'wait', 'happy', 'sad',
    'angry', 'hungry', 'sick', 'what', 'where', 'who', 'need', 'hospital'
]

KAGGLE_MAP = {
    'father': 'dad',
    'mother': 'mom',
    'friend': 'boy',
    'family': 'grandpa',
    'school': 'pencil',
    'home': 'home',
    'drink': 'drink',
    'phone': 'callonphone',
    'go': 'go',
    'come': 'go', # reversed
    'wait': 'wait',
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'mad',
    'hungry': 'hungry',
    'sick': 'sick',
    'what': 'why',
    'where': 'where',
    'who': 'who',
    'need': 'have',
    'love': 'kiss',
    'like': 'like',
    'book': 'book',
    'computer': 'TV',
    'teacher': 'person',
    'student': 'child',
    'doctor': 'fireman',
    'hospital': 'room',
    'work': 'clean',
    'study': 'read',
    'money': 'penny',
    'time': 'time',
    'day': 'yesterday',
    'night': 'night',
    'morning': 'morning',
    'afternoon': 'after',
    'evening': 'sleepy',
    'week': 'before',
    'month': 'same',
    'year': 'tomorrow'
}

def resample_sequence(seq, target_len=100):
    n = len(seq)
    if n == target_len:
        return seq
    indices = np.linspace(0, n - 1, target_len)
    resampled = np.zeros((target_len, seq.shape[1]), dtype=seq.dtype)
    for i in range(seq.shape[1]):
        resampled[:, i] = np.interp(indices, np.arange(n), seq[:, i])
    return resampled

def rotate_z(seq, theta):
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

def main():
    print("Loading metadata...")
    df_meta = pd.read_csv(CSV_PATH)
    
    # 1. Clear out old seq_*.npy files first to avoid contamination
    print("Cleaning existing seq_*.npy files in data directories...")
    for cls in TARGET_CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        if os.path.exists(cls_dir):
            for f in os.listdir(cls_dir):
                if f.startswith("seq_") and f.endswith(".npy"):
                    os.remove(os.path.join(cls_dir, f))
    
    # 2. Extract from Kaggle
    print("Opening zip file...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        for cls in TARGET_CLASSES:
            cls_dir = os.path.join(DATA_DIR, cls)
            os.makedirs(cls_dir, exist_ok=True)
            
            if cls not in KAGGLE_MAP:
                print(f"Skipping Kaggle extraction for '{cls}' (local only)")
                continue
                
            kaggle_sign = KAGGLE_MAP[cls]
            cls_rows = df_meta[df_meta['sign'] == kaggle_sign]
            print(f"Extracting '{cls}' (mapped to '{kaggle_sign}') - {len(cls_rows)} available in Kaggle...")
            
            # Select first 30 rows
            selected_rows = cls_rows.head(80)
            seq_extracted = 0
            
            for idx, row in selected_rows.iterrows():
                path = row['path']
                try:
                    with z.open(path) as pf:
                        df = pd.read_parquet(io.BytesIO(pf.read()))
                        
                    # Determine active hand
                    lh_not_null = df[df['type'] == 'left_hand']['x'].notnull().sum()
                    rh_not_null = df[df['type'] == 'right_hand']['x'].notnull().sum()
                    
                    if lh_not_null == 0 and rh_not_null == 0:
                        continue
                        
                    active_hand = 'left_hand' if lh_not_null > rh_not_null else 'right_hand'
                    hand_df = df[df['type'] == active_hand].sort_values(by=['frame', 'landmark_index'])
                    
                    # Group by frame to extract coordinates
                    frames = sorted(hand_df['frame'].unique())
                    if len(frames) < 3:
                        continue
                        
                    seq_frames = []
                    for fr in frames:
                        fr_df = hand_df[hand_df['frame'] == fr].sort_values(by='landmark_index')
                        if len(fr_df) != 21:
                            # If landmark count is incorrect, skip this frame or pad
                            continue
                        coords = fr_df[['x', 'y', 'z']].values.astype(np.float32)
                        if active_hand == 'left_hand':
                            coords[:, 0] = 1.0 - coords[:, 0]
                        seq_frames.append(coords.flatten())
                        
                    if len(seq_frames) < 3:
                        continue
                        
                    # Interpolate NaNs if any
                    seq_arr = np.array(seq_frames) # shape (F, 63)
                    # Use pandas interpolation for NaNs
                    seq_df = pd.DataFrame(seq_arr)
                    seq_df = seq_df.interpolate(method='linear').ffill().bfill().fillna(0.0)
                    seq_arr = seq_df.values.astype(np.float32)
                    
                    # Resample to 100 frames
                    resampled = resample_sequence(seq_arr, 100)
                    
                    # If come, reverse in time
                    if cls == 'come':
                        resampled = resampled[::-1]
                        
                    # Save
                    np.save(os.path.join(cls_dir, f"seq_kaggle_{seq_extracted}.npy"), resampled)
                    seq_extracted += 1
                except Exception as e:
                    print(f"Error extracting {path}: {e}")
                    
            print(f"Successfully extracted {seq_extracted} Kaggle sequences for '{cls}'")

    # 3. Handle local data and Augmentations
    print("\n--- Processing Local Data & Generating Augmentations ---")
    for cls in TARGET_CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        # Check if local 0.npy exists (means we have local frame-by-frame data)
        local_exists = os.path.exists(os.path.join(cls_dir, "0.npy"))
        
        # Count currently saved Kaggle sequences
        kaggle_seqs = [f for f in os.listdir(cls_dir) if f.startswith("seq_kaggle_")]
        num_kaggle = len(kaggle_seqs)
        
        if local_exists:
            print(f"Local data found for '{cls}'. Creating local sequence...")
            # Load frames 0 to 99
            frames = []
            valid = True
            for i in range(100):
                frame_path = os.path.join(cls_dir, f"{i}.npy")
                if not os.path.exists(frame_path):
                    valid = False
                    break
                landmarks = np.load(frame_path)[:63]
                frames.append(landmarks)
                
            if valid:
                local_seq = np.array(frames, dtype=np.float32) # shape (100, 63)
                # Flip x-coordinates to make it right-handed (since collected with flipped camera)
                local_seq[:, 0::3] = 1.0 - local_seq[:, 0::3]
                np.save(os.path.join(cls_dir, "seq_local_org.npy"), local_seq)
                
                # We want a target of 55 sequences per class.
                # If we have num_kaggle sequences, we need (55 - num_kaggle - 1) augmentations.
                # If no Kaggle, we need 54 augmentations.
                target_aug = 55 - num_kaggle - 1
                if target_aug < 10:
                    target_aug = 25 # Ensure at least some augmentations
                    
                print(f"Generating {target_aug} augmentations for '{cls}'...")
                for i in range(target_aug):
                    theta = np.random.uniform(-0.15, 0.15)
                    noise_std = np.random.uniform(0.002, 0.006)
                    aug_seq = rotate_z(local_seq, theta)
                    aug_seq = add_noise(aug_seq, noise_std)
                    np.save(os.path.join(cls_dir, f"seq_local_aug_{i}.npy"), aug_seq)
            else:
                print(f"⚠️ Warning: Missing frames in local data for '{cls}'")
        else:
            print(f"No local data found for '{cls}'. Running Kaggle-only augmentations...")
            # If no local data, we augment the Kaggle sequences to reach the target of ~55 sequences.
            if num_kaggle > 0:
                target_aug = 55 - num_kaggle
                print(f"Generating {target_aug} Kaggle-based augmentations for '{cls}'...")
                for i in range(target_aug):
                    # Randomly pick a Kaggle sequence to augment
                    src_file = np.random.choice(kaggle_seqs)
                    src_seq = np.load(os.path.join(cls_dir, src_file))
                    
                    theta = np.random.uniform(-0.15, 0.15)
                    noise_std = np.random.uniform(0.002, 0.006)
                    aug_seq = rotate_z(src_seq, theta)
                    aug_seq = add_noise(aug_seq, noise_std)
                    np.save(os.path.join(cls_dir, f"seq_kaggle_aug_{i}.npy"), aug_seq)
            else:
                print(f"❌ ERROR: No data source available for class '{cls}'!")

    print("\nDataset preparation complete! Verification of sequence counts:")
    for cls in TARGET_CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        seq_files = [f for f in os.listdir(cls_dir) if f.startswith("seq_") and f.endswith(".npy")]
        print(f"  Class '{cls}': {len(seq_files)} sequence files.")

if __name__ == "__main__":
    main()
