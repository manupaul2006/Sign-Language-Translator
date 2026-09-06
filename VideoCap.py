import cv2
import mediapipe as mp

# 1. Initialize MediaPipe Hands model and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configure the hands model
hands = mp_hands.Hands(
    static_image_mode=False,        # False for video stream
    max_num_hands=1,                # Maximum hands to detect
    min_detection_confidence=0.5,   # Minimum confidence for initial detection
    min_tracking_confidence=0.5     # Minimum confidence to maintain tracking
)

# 2. Open the default webcam (index 0)
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 3. Preprocess the frame
    # Flip horizontally for a natural "selfie" view
    frame = cv2.flip(frame, 1)
    
    # OpenCV uses BGR by default, but MediaPipe requires RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 4. Process the frame to detect hands
    results = hands.process(rgb_frame)

    # 5. Draw landmarks if any hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the skeleton connecting the 21 points
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

            # Optional: Extract specific point coordinates (e.g., Index Finger Tip is point 8)
            h, w, c = frame.shape
            index_finger_tip = hand_landmarks.landmark[8]
            
            # Convert normalized coordinates (0.0 to 1.0) back to pixel dimensions
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            
            # Draw a prominent circle on the index fingertip
            cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

    # 6. Display the final annotated frame
    cv2.imshow('Hand Tracking', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()