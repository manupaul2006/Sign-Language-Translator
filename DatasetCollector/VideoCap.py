import cv2
import mediapipe as mp
import os
import json
import math
import threading
from datetime import datetime

try:
    import winsound
except ImportError:
    winsound = None

# ---------------- CONFIG ----------------
OUTPUT_DIR = "dataset"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
LANDMARK_DIR = os.path.join(OUTPUT_DIR, "landmarks")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LANDMARK_DIR, exist_ok=True)

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MAX_HANDS = 2
STABILITY_FRAMES = 5          # Pose must remain similar for this many frames
CAPTURE_COOLDOWN_FRAMES = 8   # Prevent rapid repeated captures
SIMILARITY_THRESHOLD = 0.055  # Lower = more strict duplicate filtering
MIN_HAND_SIZE = 0.08          # Reject tiny/distant hands
PADDING = 35

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    model_complexity=0,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
)

# ---------------- HELPERS ----------------
def beep():
    if winsound:
        try:
            winsound.Beep(1500, 100)
        except RuntimeError:
            pass


def normalize_hand(hand_landmarks):
    """Return translation/scale-normalized 21-point hand landmarks."""
    pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    wrist = pts[0]
    translated = [(x - wrist[0], y - wrist[1], z - wrist[2]) for x, y, z in pts]

    scale = max(math.sqrt(x * x + y * y + z * z) for x, y, z in translated)
    if scale < 1e-6:
        scale = 1.0

    return [[x / scale, y / scale, z / scale] for x, y, z in translated]


def hand_signature(results):
    """Create an ordered, compact signature for the current pose."""
    if not results.multi_hand_landmarks:
        return None

    items = []
    for i, hand in enumerate(results.multi_hand_landmarks):
        label = results.multi_handedness[i].classification[0].label
        items.append((label, normalize_hand(hand)))

    items.sort(key=lambda item: item[0])
    return items


def pose_distance(sig_a, sig_b):
    if sig_a is None or sig_b is None:
        return float("inf")
    if len(sig_a) != len(sig_b):
        return float("inf")

    total = 0.0
    count = 0
    for (label_a, hand_a), (label_b, hand_b) in zip(sig_a, sig_b):
        if label_a != label_b:
            return float("inf")
        for p1, p2 in zip(hand_a, hand_b):
            total += math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
            count += 1
    return total / max(count, 1)


def get_bbox(results, frame_shape):
    h, w = frame_shape[:2]
    xs, ys = [], []
    for hand in results.multi_hand_landmarks:
        for lm in hand.landmark:
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

    x_min = max(0, min(xs) - PADDING)
    y_min = max(0, min(ys) - PADDING)
    x_max = min(w, max(xs) + PADDING)
    y_max = min(h, max(ys) + PADDING)
    return x_min, y_min, x_max, y_max


def hands_are_large_enough(results):
    if not results.multi_hand_landmarks:
        return False
    for hand in results.multi_hand_landmarks:
        xs = [lm.x for lm in hand.landmark]
        ys = [lm.y for lm in hand.landmark]
        if (max(xs) - min(xs)) < MIN_HAND_SIZE or (max(ys) - min(ys)) < MIN_HAND_SIZE:
            return False
    return True


def save_sample(frame, results, label, sample_number):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_label = "_".join(label.strip().split()) or "UNLABELED"
    image_folder = os.path.join(IMAGE_DIR, safe_label)
    landmark_folder = os.path.join(LANDMARK_DIR, safe_label)
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(landmark_folder, exist_ok=True)

    x_min, y_min, x_max, y_max = get_bbox(results, frame.shape)
    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None

    base = f"{sample_number:04d}_{timestamp}"
    image_path = os.path.join(image_folder, base + ".jpg")
    json_path = os.path.join(landmark_folder, base + ".json")

    if not cv2.imwrite(image_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        return None

    landmark_data = {
        "label": label,
        "timestamp": timestamp,
        "image": os.path.relpath(image_path, OUTPUT_DIR),
        "image_bbox": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        },
        "hands": [],
    }

    for i, hand in enumerate(results.multi_hand_landmarks):
        handedness = results.multi_handedness[i].classification[0]
        landmark_data["hands"].append({
            "handedness": handedness.label,
            "confidence": round(handedness.score, 4),
            "landmarks_normalized": normalize_hand(hand),
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(landmark_data, f, indent=2)

    return image_path


def count_samples(label):
    folder = os.path.join(IMAGE_DIR, "_".join(label.strip().split()) or "UNLABELED")
    if not os.path.isdir(folder):
        return 0
    return len([name for name in os.listdir(folder) if name.lower().endswith(".jpg")])


# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

label = "A"
stable_count = 0
cooldown = 0
last_captured_signature = None
current_signature = None
status = "Show a sign and hold it steady"
manual_flash = 0

print("\n=== Sign Language Dataset Collector ===")
print("Keys: A-Z = label | SPACE = manual capture | R = reset duplicate lock | Q = quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    clean_frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    display = clean_frame.copy()
    captured_this_frame = False

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display, hand, mp_hands.HAND_CONNECTIONS)

        if hands_are_large_enough(results):
            new_signature = hand_signature(results)
            distance = pose_distance(current_signature, new_signature)

            if current_signature is not None and distance < SIMILARITY_THRESHOLD:
                stable_count += 1
            else:
                stable_count = 1
                current_signature = new_signature

            if cooldown > 0:
                cooldown -= 1

            # Automatic capture only for a new, stable pose.
            if (
                stable_count >= STABILITY_FRAMES
                and cooldown == 0
                and pose_distance(last_captured_signature, new_signature) >= SIMILARITY_THRESHOLD
            ):
                sample_number = count_samples(label) + 1
                saved = save_sample(clean_frame, results, label, sample_number)
                if saved:
                    last_captured_signature = new_signature
                    cooldown = CAPTURE_COOLDOWN_FRAMES
                    manual_flash = 6
                    captured_this_frame = True
                    threading.Thread(target=beep, daemon=True).start()
                    status = f"AUTO SAVED  {label}  |  sample {sample_number}"
        else:
            stable_count = 0
            status = "Move closer - hand is too small"
    else:
        stable_count = 0
        current_signature = None
        status = f"Label: {label} | Show a sign"

    # ---------------- UI ----------------
    if manual_flash > 0:
        manual_flash -= 1
        cv2.rectangle(display, (0, 0), (display.shape[1] - 1, display.shape[0] - 1), (0, 255, 0), 8)

    sample_count = count_samples(label)
    cv2.rectangle(display, (0, 0), (display.shape[1], 92), (0, 0, 0), cv2.FILLED)
    cv2.putText(display, f"LABEL: {label}   SAMPLES: {sample_count}", (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(display, f"STABILITY: {stable_count}/{STABILITY_FRAMES}   HANDS: {len(results.multi_hand_landmarks or [])}",
                (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(display, status[:75], (12, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow("Sign Language Dataset Collector", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("r"):
        last_captured_signature = None
        status = "Duplicate lock reset"
    elif key == 32:  # SPACE
        if results.multi_hand_landmarks and hands_are_large_enough(results):
            sample_number = count_samples(label) + 1
            saved = save_sample(clean_frame, results, label, sample_number)
            if saved:
                last_captured_signature = hand_signature(results)
                cooldown = CAPTURE_COOLDOWN_FRAMES
                manual_flash = 6
                threading.Thread(target=beep, daemon=True).start()
                status = f"MANUAL SAVED  {label}  |  sample {sample_number}"
        else:
            status = "Cannot capture: no valid hand detected"
    elif (ord("A") <= key <= ord("Z")) or (ord('a') <= key <= ord('z')):
        label = chr(key).capitalize()
        last_captured_signature = None
        stable_count = 0
        status = f"Label changed to {label}"

cap.release()
cv2.destroyAllWindows()
hands.close()
