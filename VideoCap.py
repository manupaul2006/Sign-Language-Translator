import cv2
import mediapipe as mp
import os
import winsound
import threading
from datetime import datetime
import numpy as np

# ---------------------------------------------------------
# Automatic Sign-Language Capture
# Captures ANY stable hand pose, not just thumbs-up.
# Supports up to 2 hands.
# ---------------------------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
)

output_dir = "captured_signs"
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Settings
# -------------------------
STABLE_FRAMES = 5          # Frames the pose must remain stable before capture
POSE_CHANGE_THRESHOLD = 0.12  # Larger = less sensitive to pose changes
NEW_SIGN_DELAY = 8         # Small delay after a capture to avoid duplicates
PADDING = 50

capture_cooldown = 0
stable_count = 0
last_pose = None
last_captured_pose = None
status_message = "Show any sign..."


def play_beep():
    try:
        winsound.Beep(1500, 100)
    except Exception:
        pass


def get_hand_features(hand_landmarks):
    """Return wrist-relative, scale-normalized x/y/z features."""
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )

    # Put the wrist at the origin.
    points -= points[0]

    # Normalize by hand size so distance from the camera matters less.
    scale = np.linalg.norm(points[9])  # wrist -> middle MCP
    if scale < 1e-6:
        scale = 1.0
    points /= scale

    return points.flatten()


def build_pose(results):
    """Create one comparable feature vector for all detected hands."""
    if not results.multi_hand_landmarks:
        return None

    hand_features = []

    # Sort hands using wrist x-position for more consistent ordering.
    indexed_hands = list(enumerate(results.multi_hand_landmarks))
    indexed_hands.sort(key=lambda item: item[1].landmark[0].x)

    for _, hand in indexed_hands:
        hand_features.append(get_hand_features(hand))

    # Fixed-size representation: 2 hands maximum.
    # Missing second hand is filled with zeros.
    if len(hand_features) == 1:
        hand_features.append(np.zeros(63, dtype=np.float32))

    return np.concatenate(hand_features[:2])


def pose_distance(pose_a, pose_b):
    if pose_a is None or pose_b is None:
        return float("inf")
    return float(np.mean(np.abs(pose_a - pose_b)))


def capture_sign(clean_frame, results):
    """Crop all detected hands and save the sign image."""
    h, w, _ = clean_frame.shape

    x_values = []
    y_values = []

    for hand in results.multi_hand_landmarks:
        for lm in hand.landmark:
            x_values.append(int(lm.x * w))
            y_values.append(int(lm.y * h))

    if not x_values or not y_values:
        return False, ""

    x_min = max(0, min(x_values) - PADDING)
    x_max = min(w, max(x_values) + PADDING)
    y_min = max(0, min(y_values) - PADDING)
    y_max = min(h, max(y_values) + PADDING)

    hand_crop = clean_frame[y_min:y_max, x_min:x_max]

    if hand_crop.size == 0:
        return False, ""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(output_dir, f"sign_{timestamp}.jpg")

    if not cv2.imwrite(filename, hand_crop):
        return False, ""

    threading.Thread(target=play_beep, daemon=True).start()
    return True, filename


# -------------------------
# Webcam
# -------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    clean_frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    display_frame = clean_frame.copy()
    current_pose = build_pose(results)

    if current_pose is not None:
        # Check whether the hand pose is staying still.
        if last_pose is not None:
            movement = pose_distance(current_pose, last_pose)

            if movement < 0.035:
                stable_count += 1
            else:
                stable_count = 0
        else:
            stable_count = 1

        last_pose = current_pose.copy()

        # Wait until the pose is stable, then capture only if it is a NEW sign.
        if capture_cooldown > 0:
            capture_cooldown -= 1

        elif stable_count >= STABLE_FRAMES:
            is_new_sign = (
                last_captured_pose is None
                or pose_distance(current_pose, last_captured_pose) > POSE_CHANGE_THRESHOLD
            )

            if is_new_sign:
                captured, filename = capture_sign(clean_frame, results)

                if captured:
                    last_captured_pose = current_pose.copy()
                    stable_count = 0
                    capture_cooldown = NEW_SIGN_DELAY
                    status_message = f"CAPTURED: {os.path.basename(filename)}"

        if capture_cooldown == 0 and stable_count < STABLE_FRAMES:
            status_message = f"Detecting sign... {stable_count}/{STABLE_FRAMES}"

        elif capture_cooldown == 0 and stable_count >= STABLE_FRAMES:
            status_message = "Sign detected - change pose for next sign"

        # Draw landmarks only on display.
        for hand in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                display_frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

    else:
        last_pose = None
        stable_count = 0
        status_message = "Show any sign..."

        if capture_cooldown > 0:
            capture_cooldown -= 1

    # UI
    cv2.rectangle(
        display_frame,
        (0, 0),
        (display_frame.shape[1], 45),
        (0, 0, 0),
        cv2.FILLED,
    )

    cv2.putText(
        display_frame,
        status_message,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    # Green border briefly after capture.
    if capture_cooldown > 0:
        cv2.rectangle(
            display_frame,
            (0, 0),
            (display_frame.shape[1] - 1, display_frame.shape[0] - 1),
            (0, 255, 0),
            8,
        )

    cv2.imshow("Automatic Sign Capture", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # Space = manual capture of the current sign.
    # Useful for dynamic signs where the meaning depends on movement.
    if key == 32 and results.multi_hand_landmarks:
        captured, filename = capture_sign(clean_frame, results)
        if captured:
            last_captured_pose = current_pose.copy() if current_pose is not None else None
            stable_count = 0
            capture_cooldown = NEW_SIGN_DELAY
            status_message = f"MANUAL CAPTURED: {os.path.basename(filename)}"

cap.release()
hands.close()
cv2.destroyAllWindows()
