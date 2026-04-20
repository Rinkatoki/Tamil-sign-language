import cv2
import joblib
import mediapipe as mp
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import filedialog
from collections import deque, Counter
from PIL import ImageFont, ImageDraw, Image
from gtts import gTTS
import pygame
import google.generativeai as genai




genai.configure(api_key="AIzaSyCwQvXEqB0TeFCdYWxZr698KXbXuF1G6lY")

gemini_model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")



# ---------- CONFIG ----------
MODEL_PATH = "tamil_sign_xgb_247.pkl"
LABEL_ENCODER_PATH = "label_encoder_247.pkl"
CAMERA_ID = 0
CONF_THRESH = 0.6
BUFFER_SIZE = 8
REQ_SAME = 8
FONT_PATH = "NotoSansTamil-VariableFont_wdth,wght.ttf"
FONT_SIZE = 42
COOLDOWN = 1.5
# ----------------------------

delete_gesture_active = False
space_locked = False
gesture_mode = None

thumb_buffer = deque(maxlen=8)

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

buffer_joint = deque(maxlen=BUFFER_SIZE)

text_buffer = ""
current_word = ""
word_list = []

last_added = ""
last_add_time = 0
SIGN_SWITCH_DELAY = 1.0
last_sign_time = 0



# ---------- FONT ----------
try:
    FONT = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except:
    FONT = None


def draw_tamil_text(frame, text, pos):
    if FONT:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        draw.text(pos, text, font=FONT, fill=(255, 255, 255))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame


def autocorrect_text():
    global text_buffer

    if not text_buffer.strip():
        return

    prompt = f"""
Correct only spelling mistakes in the following Tamil sentence.
Do NOT change words unless spelling is wrong.
Do NOT translate.
Return only corrected Tamil text.

Sentence:
{text_buffer}
"""

    try:
        response = gemini_model.generate_content(prompt)

        corrected = response.text.strip()

        text_buffer = corrected

        text_display.delete("1.0", tk.END)
        text_display.insert(tk.END, text_buffer)

    except Exception as e:
        print("Gemini error:", e)


def process_hand(lm):
    arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    arr[:, :2] -= arr[0, :2]
    scale = np.linalg.norm(arr[9, :2])
    if scale < 1e-6:
        scale = 1.0
    arr[:, :2] /= scale
    return arr.flatten().tolist()


def is_thumb_down(landmarks):
    #return landmarks[4].y > landmarks[2].y
    return landmarks[4].y > landmarks[2].y + 0.08


import os

pygame.mixer.init()

def speak_text():
    global text_buffer
    if text_buffer.strip():

        filename = "speech.mp3"

        tts = gTTS(text=text_buffer, lang='ta')
        tts.save(filename)

        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            continue

        pygame.mixer.music.unload()
        os.remove(filename)


# ---------- GUI ----------
root = tk.Tk()
root.title("Tamil Sign Language Assistant")

text_display = tk.Text(root, height=5, width=60, font=("Arial", 18))
text_display.pack(pady=10)


def copy_text():
    root.clipboard_clear()
    root.clipboard_append(text_buffer)


def clear_text():
    global text_buffer, current_word, word_list
    text_buffer = ""
    current_word = ""
    word_list.clear()
    text_display.delete("1.0", tk.END)


def save_text():
    file = filedialog.asksaveasfilename(defaultextension=".txt")
    if file:
        with open(file, "w", encoding="utf-8") as f:
            f.write(text_buffer)


btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="Copy", command=copy_text).pack(side="left")
tk.Button(btn_frame, text="Clear", command=clear_text).pack(side="left")
tk.Button(btn_frame, text="Save", command=save_text).pack(side="left")
tk.Button(btn_frame, text="Speak", command=speak_text).pack(side="left")
tk.Button(btn_frame, text="Auto Correct", command=autocorrect_text).pack(side="left")


# ---------- CAMERA LOOP ----------
def camera_loop():
    global text_buffer
    global last_added
    global last_add_time
    global delete_gesture_active
    global space_locked
    global gesture_mode
    global current_word
    global last_sign_time

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

        has_left = 0
        has_right = 0

        thumbs_down = 0

        if res.multi_hand_landmarks:
            for lm, handed in zip(res.multi_hand_landmarks,
                                  res.multi_handedness):

                side = handed.classification[0].label.lower()

                mp_drawing.draw_landmarks(
                    frame,
                    lm,
                    mp_hands.HAND_CONNECTIONS
                )

                if is_thumb_down(lm.landmark):
                    thumbs_down += 1

                feats = process_hand(lm.landmark)

                if side == "left":
                    left_feats = feats
                    has_left = 1
                else:
                    right_feats = feats
                    has_right = 1

            thumb_buffer.append(thumbs_down)

            stable_thumbs = Counter(
                thumb_buffer
            ).most_common(1)[0][0]

            now = time.time()

            # DELETE gesture
            if stable_thumbs == 2 and has_left and has_right:
                gesture_mode = "DELETE"

            #elif stable_thumbs == 1 and has_left and has_right and last_added != "SPACE":
            elif stable_thumbs == 1 and has_left and has_right and np.linalg.norm(left_feats) > 0.05 and np.linalg.norm(right_feats) > 0.05:
                gesture_mode = "SPACE"

            else:
                gesture_mode = None

            # DELETE
            if gesture_mode == "DELETE":
                if not delete_gesture_active and len(text_buffer) > 0:

                    removed = text_buffer[-1]
                    text_buffer = text_buffer[:-1]

                    if removed != " ":
                        current_word = current_word[:-1]

                    last_added = "DELETE"
                    last_add_time = now
                    delete_gesture_active = True
            else:
                delete_gesture_active = False

            # SPACE
            if gesture_mode == "SPACE":
                #if not space_locked: #and now - last_add_time > COOLDOWN
                if last_added != "SPACE":
                    text_buffer += " "

                    if current_word.strip():
                        word_list.append(current_word)

                    current_word = ""

                    last_added = "SPACE"
                    last_add_time = now
                    space_locked = True
            else:
                space_locked = False

            # LETTER prediction
            if gesture_mode is None and has_left and has_right:

                feat = np.array(
                    left_feats + right_feats +
                    [has_left, has_right],
                    dtype=np.float32
                ).reshape(1, -1)

                probs = model.predict_proba(feat)[0]
                boost_map = {
                    "அ": 1.25,
                    #"க": 1.30,
                    "ம்": 1.20,
                    "மா":3.0,
                    "ப்":1.30,
                    "பா":1.30
                    #"மா":1.25
                }

                for label, factor in boost_map.items():
                    if label in label_encoder.classes_:
                        idx_boost = list(label_encoder.classes_).index(label)
                        probs[idx_boost] *= factor

                probs = probs / probs.sum()
                
                

                idx = np.argmax(probs)
                prob = probs[idx]

                pred = label_encoder.inverse_transform([idx])[0]

                buffer_joint.append((pred, prob))

                labels = [p for p, _ in buffer_joint]

                if labels:

                    most, cnt = Counter(labels).most_common(1)[0]

                    best_prob = max(
                        pr for p, pr in buffer_joint if p == most
                    )

                    if (
                        cnt >= REQ_SAME
                        and best_prob >= CONF_THRESH
                        and most != last_added
                        and now - last_sign_time > SIGN_SWITCH_DELAY
                    ):
                        text_buffer += most
                        current_word += most

                        last_added = most
                        last_add_time = now
                        last_sign_time = now


        # update GUI text
        text_display.delete("1.0", tk.END)
        text_display.insert(tk.END, text_buffer)

        # display camera text
        frame = draw_tamil_text(
            frame,
            text_buffer,
            (20, frame.shape[0] - 60)
        )

        cv2.imshow("Tamil Sign Typing", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------- START THREAD ----------
threading.Thread(target=camera_loop).start()

root.mainloop()