import os
import cv2
import random
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

import mediapipe as mp

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ================= CONFIG =================

DATASET_PATH = "dataset"
CSV_OUTPUT = "landmarks_twohand_247.csv"
MODEL_OUTPUT = "tamil_sign_rf_247.pkl"

SAMPLES_PER_CLASS = 100
RANDOM_SEED = 42
MIN_DETECTION_CONF = 0.5

# ==========================================

random.seed(RANDOM_SEED)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=MIN_DETECTION_CONF
)

# ==========================================
# LANDMARK NORMALIZATION FUNCTION
# ==========================================

def process_hand(landmarks):

    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    # translate wrist → origin
    arr[:, :2] -= arr[0, :2]

    scale = np.linalg.norm(arr[9, :2])

    if scale < 1e-6:
        scale = 1.0

    arr[:, :2] /= scale

    return arr.flatten().tolist()


# ==========================================
# CREATE CSV
# ==========================================

rows = []
class_counts = {}

folders = sorted(os.listdir(DATASET_PATH))

print("\nExtracting landmarks...\n")

for folder in folders:

    folder_path = os.path.join(DATASET_PATH, folder)

    if not os.path.isdir(folder_path):
        continue

    if folder.lower() == "background":
        continue

    label = folder

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if len(images) == 0:
        continue

    # sample only 100 images
    if len(images) > SAMPLES_PER_CLASS:
        images = random.sample(images, SAMPLES_PER_CLASS)

    class_counts[label] = 0

    print(f"Processing class {label} ({len(images)} images)")

    for img_name in tqdm(images):

        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        res = hands.process(img_rgb)

        if not res.multi_hand_landmarks:
            continue

        left_feats = [0.0]*63
        right_feats = [0.0]*63
        has_left = 0
        has_right = 0

        for landmarks, handedness in zip(
                res.multi_hand_landmarks,
                res.multi_handedness):

            side = handedness.classification[0].label.lower()

            feats = process_hand(landmarks.landmark)

            if side == "left":
                left_feats = feats
                has_left = 1
            else:
                right_feats = feats
                has_right = 1

        row = [label] + left_feats + right_feats + [has_left, has_right]

        rows.append(row)

        class_counts[label] += 1


hands.close()

print("\nSaving CSV...\n")

columns = (
    ["label"] +
    [f"L_{i}" for i in range(63)] +
    [f"R_{i}" for i in range(63)] +
    ["has_left", "has_right"]
)

df = pd.DataFrame(rows, columns=columns)

df.to_csv(CSV_OUTPUT, index=False)

print("CSV saved:", CSV_OUTPUT)
print("Total samples:", len(df))
print("Total classes:", len(class_counts))


