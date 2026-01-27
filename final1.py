# real_time_two_hands_no_npy_updated.py
import cv2
import joblib
import mediapipe as mp
import numpy as np
import time
from collections import deque, Counter
from PIL import ImageFont, ImageDraw, Image

# ---------- CONFIG ----------
MODEL_PATH = "tamil_sign_twohand_model.pkl"
CAMERA_ID = 0
CONF_THRESH = 0.75
BUFFER_SIZE = 8
REQ_SAME = 6
FONT_PATH = "NotoSansTamil-VariableFont_wdth,wght.ttf"
FONT_SIZE = 42
COOLDOWN = 0.7
# ----------------------------
delete_locked = False
space_locked = False
delete_gesture_active = False
gesture_mode = None   # None | "DELETE" | "SPACE"




model = joblib.load(MODEL_PATH)
classes = list(model.classes_)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Buffers
buffer_joint = deque(maxlen=BUFFER_SIZE)
buffer_left = deque(maxlen=BUFFER_SIZE)
buffer_right = deque(maxlen=BUFFER_SIZE)

# Text memory
text_buffer = ""
last_added = ""
last_add_time = 0
current_word = ""
word_list = []


# Load font
try:
    FONT = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except:
    FONT = None

def draw_tamil_text(frame, text, pos):
    if FONT:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        draw.text(pos, text, font=FONT, fill=(255,255,255))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return frame

def process_hand(lm):
    arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    arr[:, :2] -= arr[0, :2]
    scale = np.linalg.norm(arr[9, :2])
    if scale < 1e-6: scale = 1.0
    arr[:, :2] /= scale
    return arr.flatten().tolist()

def is_thumb_down(landmarks):
    # thumb tip lower than thumb mcp
    return landmarks[4].y > landmarks[2].y

cap = cv2.VideoCapture(CAMERA_ID)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    left_feats = [0]*63
    right_feats = [0]*63
    has_left = has_right = 0

    thumbs_down = 0

    if res.multi_hand_landmarks:
        for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
            side = handed.classification[0].label.lower()
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            if is_thumb_down(lm.landmark):
                thumbs_down += 1

            feats = process_hand(lm.landmark)
            if side == "left":
                left_feats = feats
                has_left = 1
            else:
                right_feats = feats
                has_right = 1

        now = time.time()

        # -------------------------
        # -------------------------
        # SPACE & DELETE (locked)
        # -------------------------

        # SPACE (1 thumb down)
        if thumbs_down == 1:
            if not space_locked and now - last_add_time > COOLDOWN:
                text_buffer += " "
                if current_word.strip():
                    print("Predicted word:", current_word)
                    word_list.append(current_word)
                current_word = ""

                last_add_time = now
                last_added = "SPACE"
                space_locked = True
        else:
            space_locked = False


        # DELETE (2 thumbs down) – safe, single delete
        if thumbs_down == 2:
            if not delete_gesture_active and len(text_buffer) > 0:

                # delete from text buffer
                removed = text_buffer[-1]
                text_buffer = text_buffer[:-1]

                # sync word buffers
                if removed == " ":
                    if len(word_list) > 0:
                        current_word = word_list.pop()
                    else:
                        current_word = ""
                else:
                    if len(current_word) > 0:
                        current_word = current_word[:-1]

                last_add_time = now
                last_added = "DELETE"
                delete_gesture_active = True
        else:
            delete_gesture_active = False




        # -------------------------
        # NORMAL LETTER PREDICTION
        # -------------------------
        if thumbs_down == 0:
            feat = np.array(left_feats + right_feats + [has_left, has_right], dtype=np.float32).reshape(1, -1)
            probs = model.predict_proba(feat)[0]
            idx = np.argmax(probs)
            prob = probs[idx]
            pred = classes[idx]

            buffer_joint.append((pred, prob))
            labels = [p for p,_ in buffer_joint]

            if labels:
                most, cnt = Counter(labels).most_common(1)[0]
                best_prob = max(pr for p,pr in buffer_joint if p == most)

                if (
                    cnt >= REQ_SAME and
                    best_prob >= CONF_THRESH and
                    most != last_added and
                    now - last_add_time > COOLDOWN
                ):
                    text_buffer += most
                    current_word += most

                    last_added = most
                    last_add_time = now


        # Display text buffer
        frame = draw_tamil_text(frame, text_buffer, (20, frame.shape[0] - 60))

    cv2.imshow("Tamil Sign Language Typing", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Sentence so far:", " ".join(word_list))
cap.release()
cv2.destroyAllWindows()
hands.close()
