import cv2
import mediapipe as mp
import os
from gtts import gTTS
import pygame
import time
import numpy as np
from hashlib import sha1

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


os.makedirs("audio_cache", exist_ok=True)
audio_available = True
try:
    pygame.mixer.init()
except Exception as e:
    print("Pygame mixer init failed (audio disabled):", e)
    audio_available = False

def tts_cache_filename(text: str) -> str:
    """Return a safe cache filename for the given text (uses sha1).

    We avoid raw text as filename to prevent invalid characters and very long
    filenames on Windows. The cache key is a short SHA1 prefix.
    """
    key = sha1(text.encode('utf-8')).hexdigest()[:16]
    return os.path.join("audio_cache", f"{key}.mp3")


def speak(text: str):
    """Text to speech Bahasa Indonesia (cached)."""
    try:
        filename = tts_cache_filename(text)
        if not os.path.exists(filename):
            tts = gTTS(text=text, lang='id')
            tts.save(filename)

        if not audio_available or not pygame.mixer.get_init():
            return

        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
    except Exception as e:
        print("Error saat memutar suara:", e)

hands_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
print("Kamera dibuka:", cap.isOpened())
if not cap.isOpened():
    print("Peringatan: tangkapan kamera tidak dapat dibuka.")

def finger_status(hand_landmarks, handedness: str = "Unknown"):
    """Return list of 5 ints (1=open, 0=closed) for [thumb, index, middle, ring, pinky].

    handedness: 'Left' or 'Right' as reported by MediaPipe. Thumb logic depends
    on handedness because X coordinates invert for left/right hands.
    """
    finger_states = []
    tips = [4, 8, 12, 16, 20]

    try:
        thumb_tip = hand_landmarks.landmark[tips[0]]
        thumb_ip = hand_landmarks.landmark[tips[0] - 1]
        if handedness.lower().startswith('right'):
            thumb_open = thumb_tip.x < thumb_ip.x
        else:
            thumb_open = thumb_tip.x > thumb_ip.x
        finger_states.append(1 if thumb_open else 0)
    except Exception:
        finger_states.append(0)

    for idx in range(1, 5):
        try:
            tip = hand_landmarks.landmark[tips[idx]]
            pip = hand_landmarks.landmark[tips[idx] - 2]
            finger_states.append(1 if tip.y < pip.y else 0)
        except Exception:
            finger_states.append(0)
    return finger_states

def is_palm_facing_camera_advanced(img, hand_landmarks):
    h, w, _ = img.shape

    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
    x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, w)
    y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, h)
    hand_region = img[y_min:y_max, x_min:x_max]

    brightness_score = 0
    if hand_region.size > 0:
        gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        brightness_score = 1 if brightness > 110 else 0

    wrist = hand_landmarks.landmark[0]
    middle_tip = hand_landmarks.landmark[12]
    depth_score = 1 if middle_tip.z < wrist.z else 0

    p0 = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
    p5 = np.array([hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z])
    p17 = np.array([hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z])
    v1 = p5 - p0
    v2 = p17 - p0
    normal = np.cross(v1, v2)
    normal_score = 1 if normal[2] > 0 else 0

    total_score = brightness_score + depth_score + normal_score
    confidence = (total_score / 3) * 100
    palm_detected = total_score >= 2

    return palm_detected, confidence

last_gesture = ""
last_time = 0
cooldown = 2.0
first_hand_detected = False
first_detect_time = 0
initial_delay = 3

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    gesture_text = ""
    palm_status = "Tidak ada tangan terdeteksi"

    if results.multi_hand_landmarks:
        if not first_hand_detected:
            first_hand_detected = True
            first_detect_time = time.time()
            print("Tangan pertama terdeteksi, tunggu 3 detik sebelum mulai...")
            speak("Tangan terdeteksi, mohon tunggu tiga detik.")
        
        if time.time() - first_detect_time < initial_delay:
            remaining = int(initial_delay - (time.time() - first_detect_time))
            cv2.putText(img, f"Menunggu {remaining} detik...", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            cv2.imshow("Hand Gesture Recognition - Dual Hand Fixed 👋", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        h, w, c = img.shape
        hand_data = []

        for idx, handLms in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            is_palm, conf = is_palm_facing_camera_advanced(img, handLms)

            handedness = "Unknown"
            try:
                if results.multi_handedness and len(results.multi_handedness) > idx:
                    handedness = results.multi_handedness[idx].classification[0].label
            except Exception:
                handedness = "Unknown"

            fingers = finger_status(handLms, handedness)
            hand_data.append({'is_palm': is_palm, 'fingers': fingers, 'conf': conf, 'handedness': handedness})

        num_hands = len(hand_data)

        # Gesture mapping
        gesture_map = {
            (0,0,0,0,0): "kelass!",
            (0,1,1,1,1): "Salam kenal ya!",
            (1,0,0,0,0): "Sip!",
            (0,1,0,0,0): "nama saya...",
            (0,1,1,0,0): "Abdu Nizam Al latief",
            (0,1,1,1,0): "Terima kasih!",
            (1,1,1,1,1): "Halo semuanya!",
            (1,1,0,0,1): "EZ!",
        }

        gesture_candidates = []
        for hand in hand_data:
            fingers_tuple = tuple(hand['fingers'])
            gesture = gesture_map.get(fingers_tuple, None)
            gesture_candidates.append(gesture)

        if num_hands == 2:
            valid_gestures = [g for g in gesture_candidates if g is not None]
            if valid_gestures and all(g == valid_gestures[0] for g in valid_gestures):
                gesture_text = valid_gestures[0]
                if gesture_text == "Halo semuanya!":
                    gesture_text = "Absolute Cinema 🎬"

        if num_hands == 2:
            palm_count = sum(1 for h in hand_data if h['is_palm'])
            if palm_count >= 1:
                open_hands = [sum(h['fingers']) for h in hand_data]
                if all(f >= 4 for f in open_hands):
                    palm_status = f"2 Telapak Tangan ({palm_count}/2 terdeteksi)"
                else:
                    palm_status = f"2 Tangan terdeteksi (Telapak: {palm_count}/2)"
            else:
                palm_status = "2 tangan tapi bukan telapak"
        
        elif num_hands == 1:
            hand = hand_data[0]
            fingers = hand['fingers']
            conf = hand['conf']
            palm_status = f"1 Tangan ({'Telapak' if hand['is_palm'] else 'Punggung'}) {conf:.0f}%"

    current_time = time.time()
    is_audio_playing = False
    if audio_available and pygame.mixer.get_init():
        try:
            is_audio_playing = pygame.mixer.music.get_busy()
        except Exception:
            is_audio_playing = False

    if gesture_text and gesture_text != last_gesture and (current_time - last_time) > cooldown and not is_audio_playing:
        last_gesture = gesture_text
        last_time = current_time
        speak(gesture_text)

    cv2.putText(img, palm_status, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    if gesture_text:
        cv2.putText(img, gesture_text, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
    else:
        cv2.putText(img, "Tunjukkan telapak tangan ", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

    cv2.imshow("Hand Gesture Recognition - Dual Hand Fixed ", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
if audio_available:
    try:
        pygame.mixer.quit()
    except Exception:
        pass
try:
    hands_detector.close()
except:
    pass
