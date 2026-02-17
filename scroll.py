import cv2
import mediapipe as mp
import pyautogui
import math
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

scroll_speed = 100
click_threshold = 35

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

click_cooldown = 0

def finger_up(tip, pip):
    return tip.y < pip.y

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        index_tip = hand[8]
        index_pip = hand[6]

        middle_tip = hand[12]
        middle_pip = hand[10]

        thumb_tip = hand[4]

        index_up = finger_up(index_tip, index_pip)
        middle_up = finger_up(middle_tip, middle_pip)

        # ===== SCROLL LOGIC =====
        if index_up and not middle_up:
            pyautogui.scroll(-scroll_speed)  # 1 jari → bawah
            cv2.putText(frame, "SCROLL DOWN", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        elif index_up and middle_up:
            pyautogui.scroll(scroll_speed)   # 2 jari → atas
            cv2.putText(frame, "SCROLL UP", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # ===== CLICK =====
        x1 = int(index_tip.x * w)
        y1 = int(index_tip.y * h)

        x2 = int(thumb_tip.x * w)
        y2 = int(thumb_tip.y * h)

        distance = math.hypot(x2 - x1, y2 - y1)

        if distance < click_threshold:
            if time.time() - click_cooldown > 1:
                pyautogui.click()
                click_cooldown = time.time()
                cv2.putText(frame, "CLICK", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

    cv2.imshow("1 Finger Down | 2 Finger Up", frame)

    key = cv2.waitKey(1)
    if key == ord('q') or cv2.getWindowProperty("1 Finger Down | 2 Finger Up", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
