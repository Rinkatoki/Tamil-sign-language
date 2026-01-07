import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp

# ====== CONFIG ======
DATA_ROOT = "dataset"       # main dataset folder
OUT_CSV = "landmarks_twohand.csv"
OUT_CLASSES = "label_classes.npy"
MIN_DETECTION_CONF = 0.5
# =====================

# mapping folders → Tamil vowels
num_to_tamil = {
    "1": "அ", "2": "ஆ", "3": "இ", "4": "ஈ",
    "5": "உ", "6": "ஊ", "7": "எ", "8": "ஏ",
    "9": "ஐ", "10": "ஒ", "11": "ஓ", "12": "ஔ"
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=2,
                       min_detection_confidence=MIN_DETECTION_CONF)

def process_hand(landmarks):
    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = arr[0].copy()
    arr[:, :2] -= wrist[:2]
    scale = np.linalg.norm(arr[9, :2])
    if scale < 1e-6:
        scale = 1.0
    arr[:, :2] /= scale
    return arr.flatten().tolist()  # 63 floats

rows = []
class_list = []
extracted = {}
skipped = 0
bad_files = 0
total = 0

for folder in sorted(os.listdir(DATA_ROOT), key=lambda x: (x!="background", x)):
    fpath = os.path.join(DATA_ROOT, folder)
    if not os.path.isdir(fpath) or folder.lower() == "background":
        continue
    if folder not in num_to_tamil:
        continue

    label = num_to_tamil[folder]
    class_list.append(label)
    extracted[label] = 0
    print(f"Processing {label} ({folder})...")

    for fname in tqdm(sorted(os.listdir(fpath))):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        total += 1
        img = cv2.imread(os.path.join(fpath, fname))
        if img is None:
            bad_files += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)
        if not res.multi_hand_landmarks:
            skipped += 1
            continue

        # prepare zero vectors for both hands
        left_feats = [0.0]*63
        right_feats = [0.0]*63
        has_left, has_right = 0, 0

        for i, (landmarks, handedness) in enumerate(zip(res.multi_hand_landmarks, res.multi_handedness)):
            side = handedness.classification[0].label.lower()  # "left" or "right"
            feats = process_hand(landmarks.landmark)
            if side == "left":
                left_feats = feats
                has_left = 1
            else:
                right_feats = feats
                has_right = 1

        row = [label] + left_feats + right_feats + [has_left, has_right]
        rows.append(row)
        extracted[label] += 1

if len(rows) == 0:
    print("No samples found. Check dataset.")
    raise SystemExit(1)

# Build DataFrame
n_feat = 63
cols = (["label"] +
        [f"L_f{i}" for i in range(n_feat)] +
        [f"R_f{i}" for i in range(n_feat)] +
        ["has_left", "has_right"])
df = pd.DataFrame(rows, columns=cols)
df.to_csv(OUT_CSV, index=False)
np.save(OUT_CLASSES, np.array(class_list))

hands.close()
print("\n=== DONE ===")
print(f"Total images: {total}")
print(f"Bad files: {bad_files}")
print(f"Skipped (no hands): {skipped}")
print(f"Saved: {len(df)} → {OUT_CSV}")
for k,v in extracted.items():
    print(f"  {k}: {v}")
print(f"Columns: label + 63(L) + 63(R) + has_left + has_right = {len(df.columns)}")
