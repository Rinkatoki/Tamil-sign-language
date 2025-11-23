# real_time_two_hands_no_npy_updated.py
import cv2
import joblib
import mediapipe as mp
import numpy as np
from collections import deque, Counter
from PIL import ImageFont, ImageDraw, Image

# ---------- CONFIG ----------
MODEL_PATH = "tamil_sign_twohand_model.pkl"   # <-- set to your saved model file
CAMERA_ID = 0
CONF_THRESH = 0.75
BUFFER_SIZE = 8
REQ_SAME = 6
FONT_PATH = "NotoSansTamil-VariableFont_wdth,wght.ttf"  # optional; fallback to cv2 if missing
FONT_SIZE = 48
# ----------------------------

# Load model (it must have .classes_)
model = joblib.load(MODEL_PATH)
if not hasattr(model, "classes_"):
    raise SystemExit("Loaded model has no attribute 'classes_'. Re-train/save with model.classes_.")
classes = list(model.classes_)
print("Loaded model. #classes:", len(classes))

# MediaPipe init
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Buffers for stabilization
buffer_left = deque(maxlen=BUFFER_SIZE)
buffer_right = deque(maxlen=BUFFER_SIZE)
buffer_joint = deque(maxlen=BUFFER_SIZE)   # for joint (both-hands) predictions
last_display_left = ""
last_display_right = ""
last_display_joint = ""

# Try to pre-load font once (avoid re-loading on every frame)
try:
    _PIL_FONT = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except Exception:
    _PIL_FONT = None

def draw_tamil_text(frame, text, position=(50,50), font_size=FONT_SIZE, color=(255,255,255)):
    """
    Draw text on frame using PIL (for Tamil). Falls back to cv2.putText if font not available.
    """
    if _PIL_FONT is not None:
        try:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            # Use preloaded font; if requested font_size differs, attempt to load sized font
            if font_size == FONT_SIZE:
                font = _PIL_FONT
            else:
                try:
                    font = ImageFont.truetype(FONT_PATH, font_size)
                except Exception:
                    font = _PIL_FONT
            draw.text(position, text, font=font, fill=color)
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception:
            # fallback to cv2 if PIL drawing fails unexpectedly
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
            return frame
    else:
        # PIL font missing — use cv2 fallback (may not render Tamil accurately)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
        return frame

def process_hand(landmarks):
    """
    Convert mediapipe landmarks -> normalized 63-float list (same normalization as extract.py)
    """
    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = arr[0].copy()
    arr[:, :2] -= wrist[:2]
    scale = np.linalg.norm(arr[9, :2])
    if scale < 1e-6:
        scale = 1.0
    arr[:, :2] /= scale
    return arr.flatten().tolist()  # 63 floats

def midpoint_between_wrists(res, frame_shape):
    """
    Given MediaPipe results and frame shape, compute pixel midpoint between left & right wrist landmarks (if both present).
    Returns (x, y) in pixels or None if not available.
    """
    left_wrist = None
    right_wrist = None
    # iterate through detected hands
    for landmarks, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
        side = handedness.classification[0].label.lower()
        wrist = landmarks.landmark[0]  # wrist normalized coords
        if side == "left":
            left_wrist = wrist
        else:
            right_wrist = wrist
    if left_wrist is not None and right_wrist is not None:
        h, w = frame_shape[0], frame_shape[1]
        mx = int(((left_wrist.x + right_wrist.x) / 2) * w)
        my = int(((left_wrist.y + right_wrist.y) / 2) * h)
        # clamp to frame
        mx = max(0, min(w-1, mx))
        my = max(0, min(h-1, my))
        return (mx, my)
    return None

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(frame_rgb)

        # defaults for this frame
        left_feats = [0.0] * 63
        right_feats = [0.0] * 63
        has_left = 0
        has_right = 0

        if res.multi_hand_landmarks and res.multi_handedness:
            # Fill features and draw landmarks
            for landmarks, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                side = handedness.classification[0].label.lower()  # 'left' or 'right'
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                feats = process_hand(landmarks.landmark)
                if side == "left":
                    left_feats = feats
                    has_left = 1
                else:
                    right_feats = feats
                    has_right = 1

            # Build the combined feature vector (exact order used during training)
            feat = np.array(left_feats + right_feats + [has_left, has_right], dtype=np.float32).reshape(1, -1)

            # Predict
            try:
                probs = model.predict_proba(feat)[0]   # shape (n_classes,)
                idx = int(np.argmax(probs))
                prob = float(probs[idx])
                pred_label = classes[idx]
            except Exception as e:
                pred_label = None
                prob = 0.0

            # Decide whether to use joint buffer (both hands) or per-hand buffers
            if has_left and has_right:
                # Both hands present -> append to joint buffer
                buffer_joint.append((pred_label, prob))

                # Stabilize joint prediction
                labels_j = [p for p, _ in buffer_joint if p is not None]
                display_joint = ""
                if labels_j:
                    most_common_label, count = Counter(labels_j).most_common(1)[0]
                    probs_for_label = [pr for lbl, pr in buffer_joint if lbl == most_common_label]
                    best_prob = max(probs_for_label) if probs_for_label else 0.0
                    if count >= REQ_SAME and best_prob >= CONF_THRESH:
                        display_joint = str(most_common_label)
                        last_display_joint = display_joint

                # Draw joint display at midpoint between wrists (or frame center)
                if display_joint:
                    mid = midpoint_between_wrists(res, frame.shape)
                    if mid is None:
                        mid = (frame.shape[1] // 2 - 40, frame.shape[0] // 2 - 20)
                    frame = draw_tamil_text(frame, display_joint, position=mid)
            else:
                # Only one hand present -> append prediction to that hand's buffer only
                if has_left:
                    buffer_left.append((pred_label, prob))
                if has_right:
                    buffer_right.append((pred_label, prob))

                # Stabilize left display
                display_left = ""
                labels_l = [p for p, _ in buffer_left if p is not None]
                if labels_l:
                    most_common_label, count = Counter(labels_l).most_common(1)[0]
                    probs_for_label = [pr for lbl, pr in buffer_left if lbl == most_common_label]
                    best_prob = max(probs_for_label) if probs_for_label else 0.0
                    if count >= REQ_SAME and best_prob >= CONF_THRESH:
                        display_left = str(most_common_label)
                        last_display_left = display_left

                # Stabilize right display
                display_right = ""
                labels_r = [p for p, _ in buffer_right if p is not None]
                if labels_r:
                    most_common_label, count = Counter(labels_r).most_common(1)[0]
                    probs_for_label = [pr for lbl, pr in buffer_right if lbl == most_common_label]
                    best_prob = max(probs_for_label) if probs_for_label else 0.0
                    if count >= REQ_SAME and best_prob >= CONF_THRESH:
                        display_right = str(most_common_label)
                        last_display_right = display_right

                # Draw per-hand labels in fixed positions (or compute bounding boxes if you prefer)
                if display_left:
                    frame = draw_tamil_text(frame, display_left, position=(30, 80))
                if display_right:
                    frame = draw_tamil_text(frame, display_right, position=(frame.shape[1] - 220, 80))

        # Show frame
        cv2.imshow("Tamil Sign Recognition (Two Hands)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
