# real_time_two_hands.py
import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import deque, Counter
from PIL import ImageFont, ImageDraw, Image

def draw_tamil_text(frame, text, position=(50, 50), font_size=60, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("NotoSansTamil-VariableFont_wdth,wght.ttf", font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ---------- CONFIG ----------
MODEL_PATH = "tamil_sign_model.pkl"
CAMERA_ID = 0
CONF_THRESH = 0.75
BUFFER_SIZE = 8
REQ_SAME = 6
# ----------------------------

model = joblib.load(MODEL_PATH)
print("Loaded model. Classes:", list(model.classes_))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# buffers for left and right hands separately
buffer_left = deque(maxlen=BUFFER_SIZE)
buffer_right = deque(maxlen=BUFFER_SIZE)
last_display_left = ""
last_display_right = ""

def landmarks_to_features(landmark_list, handedness_label):
    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmark_list], dtype=np.float32)
    wrist = arr[0].copy()
    arr[:, :2] -= wrist[:2]
    scale = np.linalg.norm(arr[9, :2])
    if scale < 1e-6:
        scale = 1.0
    arr[:, :2] /= scale
    feat63 = arr.flatten().tolist()
    feat63.append(float(handedness_label))
    return feat63

col_names = [f"f{i}" for i in range(64)]

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(frame_rgb)

    if res.multi_hand_landmarks and res.multi_handedness:
        for i, hand_landmarks in enumerate(res.multi_hand_landmarks):
            label = res.multi_handedness[i].classification[0].label
            handed = 1.0 if label.lower().startswith("r") else 0.0

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            feat = landmarks_to_features(hand_landmarks.landmark, handed)
            X_df = pd.DataFrame([feat], columns=col_names)

            # predict
            try:
                probs = model.predict_proba(X_df)[0]
                idx = int(np.argmax(probs))
                prob = float(probs[idx])
                pred_label = model.classes_[idx]
            except Exception:
                pred_label = None
                prob = 0.0

            # choose which buffer based on hand
            if label == "Right":
                buffer_right.append((pred_label, prob))
                buf = buffer_right
                last_display = last_display_right
                pos = (350, 100)
            else:
                buffer_left.append((pred_label, prob))
                buf = buffer_left
                last_display = last_display_left
                pos = (50, 100)

            # stabilization
            labels_only = [p for p, _ in buf if p is not None]
            display_text = ""
            if labels_only:
                most_common_label, count = Counter(labels_only).most_common(1)[0]
                probs_for_label = [pr for lbl, pr in buf if lbl == most_common_label]
                best_prob = max(probs_for_label) if probs_for_label else 0.0

                if most_common_label and count >= REQ_SAME and best_prob >= CONF_THRESH:
                    display_text = str(most_common_label)
                    if label == "Right":
                        last_display_right = display_text
                    else:
                        last_display_left = display_text

            # draw Tamil text near hand
            if display_text:
                frame = draw_tamil_text(frame, display_text, pos)

    cv2.imshow("Tamil Sign Recognition (Two Hands)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
