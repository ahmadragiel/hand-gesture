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
    key = sha1(text.encode('utf-8')).hexdigest()[:16]
    return os.path.join("audio_cache", f"{key}.mp3")


def speak(text: str):
    """Text to speech Bahasa Indonesia (cached)."""
    clean_text = text.encode('ascii', 'ignore').decode('ascii').strip()
    if not clean_text:
        clean_text = text
    try:
        filename = tts_cache_filename(clean_text)
        if not os.path.exists(filename):
            tts = gTTS(text=clean_text, lang='id')
            tts.save(filename)
        if not audio_available or not pygame.mixer.get_init():
            return
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
    except Exception as e:
        print("Error saat memutar suara:", e)


hands_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# ── Coba buka kamera, fallback ke index lain kalau gagal ──────────────────
cap = None
for cam_index in range(4):
    _cap = cv2.VideoCapture(cam_index)
    if _cap.isOpened():
        ret, _ = _cap.read()
        if ret:
            cap = _cap
            print(f"Kamera dibuka di index {cam_index}")
            break
        _cap.release()

if cap is None:
    print("ERROR: Tidak ada kamera yang bisa dibuka. Program berhenti.")
    exit(1)


def finger_status(hand_landmarks, handedness: str = "Unknown"):
    """Return list of 5 ints (1=open, 0=closed) [thumb, index, middle, ring, pinky]."""
    finger_states = []
    tips = [4, 8, 12, 16, 20]

    try:
        thumb_tip = hand_landmarks.landmark[tips[0]]
        thumb_ip  = hand_landmarks.landmark[tips[0] - 1]
        if handedness.lower().startswith('right'):
            thumb_open = thumb_tip.x > thumb_ip.x
        else:
            thumb_open = thumb_tip.x < thumb_ip.x
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


def is_palm_facing_camera(img, hand_landmarks):
    h, w, _ = img.shape
    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
    x_min = max(min(x_coords) - 20, 0)
    x_max = min(max(x_coords) + 20, w)
    y_min = max(min(y_coords) - 20, 0)
    y_max = min(max(y_coords) + 20, h)
    hand_region = img[y_min:y_max, x_min:x_max]

    brightness_score = 0
    if hand_region.size > 0:
        gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        brightness_score = 1 if np.mean(gray) > 110 else 0

    wrist      = hand_landmarks.landmark[0]
    middle_tip = hand_landmarks.landmark[12]
    depth_score = 1 if middle_tip.z < wrist.z else 0

    p0  = np.array([hand_landmarks.landmark[0].x,  hand_landmarks.landmark[0].y,  hand_landmarks.landmark[0].z])
    p5  = np.array([hand_landmarks.landmark[5].x,  hand_landmarks.landmark[5].y,  hand_landmarks.landmark[5].z])
    p17 = np.array([hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z])
    normal = np.cross(p5 - p0, p17 - p0)
    normal_score = 1 if normal[2] > 0 else 0

    total_score   = brightness_score + depth_score + normal_score
    confidence    = (total_score / 3) * 100
    palm_detected = total_score >= 2
    return palm_detected, confidence


# ══════════════════════════════════════════════════════════════════════════════
# GESTURE MAP
# Jari urutan: [thumb, index, middle, ring, pinky]  1=buka 0=tutup
# ══════════════════════════════════════════════════════════════════════════════

# 1 tangan
single_hand_gestures = {
    (0, 0, 0, 0, 0): "Halo!",
    (0, 1, 0, 0, 0): "Nama saya...",
    (0, 1, 1, 0, 0): "Ahmad Ragiel Zaini",
    (0, 1, 0, 0, 1): "Senang bertemu kalian!",
    (1, 0, 0, 0, 0): "Sip!",
    (1, 1, 1, 1, 1): "Halo semuanya!",
    (1, 1, 0, 0, 1): "EZ!",
    (0, 1, 1, 1, 1): "Salam kenal ya!",
    (0, 1, 1, 1, 0): "Terima kasih!",
}

# 2 tangan kombinasi — key: (left_fingers, right_fingers)
dual_hand_gestures = {
    ((0,0,0,0,0), (0,1,0,0,0)): "Perkenalkan, nama saya...",
    ((0,1,1,0,0), (0,1,1,0,0)): "Ahmad Ragiel Zaini!",
    ((1,0,0,0,0), (1,1,1,1,1)): "Terima kasih sudah mendengarkan!",
    ((1,1,1,1,1), (1,1,1,1,1)): "Semoga kita bisa berteman!",
    ((0,1,0,0,0), (1,0,0,0,0)): "Saya siap belajar bersama kalian!",
    ((0,1,1,0,0), (0,1,0,0,0)): "Hai, apa kabar?",
    ((1,1,1,1,1), (0,0,0,0,0)): "Yuk kita mulai!",
    ((0,1,0,0,1), (0,1,1,0,0)): "Dengan senang hati berkenalan!",
}


def detect_gesture(hand_data):
    """
    Deteksi gesture dari hand_data.
    Return: (gesture_text, mode)  mode = 'single' | 'dual' | ''
    """
    num = len(hand_data)

    if num == 1:
        fingers = tuple(hand_data[0]['fingers'])
        text = single_hand_gestures.get(fingers)
        return (text or "", 'single' if text else '')

    elif num == 2:
        # Pisahkan tangan kiri & kanan
        left_hand  = None
        right_hand = None
        for h in hand_data:
            if h['handedness'].lower().startswith('left'):
                left_hand = h
            elif h['handedness'].lower().startswith('right'):
                right_hand = h

        # Fallback: sort berdasarkan posisi x di layar
        if left_hand is None or right_hand is None:
            sorted_hands = sorted(hand_data, key=lambda h: h['cx'])
            left_hand  = sorted_hands[0]
            right_hand = sorted_hands[1]

        lf = tuple(left_hand['fingers'])
        rf = tuple(right_hand['fingers'])

        # Cek dual gesture dulu (prioritas)
        dual_text = dual_hand_gestures.get((lf, rf))
        if dual_text:
            return dual_text, 'dual'

        # Fallback ke single gesture (prioritas tangan kanan)
        text = single_hand_gestures.get(rf) or single_hand_gestures.get(lf)
        return (text or "", 'single' if text else '')

    return "", ""


# ── State variables ────────────────────────────────────────────────────────
last_gesture        = ""
last_time           = 0.0
cooldown            = 2.0
first_hand_detected = False
hand_was_absent     = False
first_detect_time   = 0.0
initial_delay       = 3

# ── Panduan gesture ────────────────────────────────────────────────────────
GUIDE_LINES = [
    "=== 1 TANGAN ===",
    "Kepalan (00000)     : Halo!",
    "Telunjuk (01000)    : Nama saya...",
    "Peace (01100)       : Ahmad Ragiel Zaini",
    "Rock (01001)        : Senang bertemu!",
    "Jempol (10000)      : Sip!",
    "Buka semua (11111)  : Halo semuanya!",
    "Shaka (10001)       : EZ!",
    "4 jari (01111)      : Salam kenal!",
    "=== 2 TANGAN ===",
    "Kepalan + Telunjuk         : Perkenalkan...",
    "Peace + Peace       : Ahmad Ragiel Zaini!",
    "Jempol + Buka        : Terima kasih!",
    "Buka + Buka         : Semoga berteman!",
    "Telunjuk + Jempol        : Siap belajar!",
    "Peace + Telunjuk        : Hai apa kabar?",
    "Buka + Kepalan         : Yuk kita mulai!",
    "Rock + Peace        : Senang berkenalan!",
    "=== KONTROL ===",
    "'g' = toggle panduan",
    "'q' = keluar",
]
show_guide = True

# ──────────────────────────────────────────────────────────────────────────
while True:
    success, img = cap.read()
    if not success:
        break

    img     = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)

    gesture_text = ""
    gesture_mode = ""
    palm_status  = "Tidak ada tangan terdeteksi"
    hand_data    = []

    if results.multi_hand_landmarks:
        # Reset delay kalau tangan baru muncul / sempat hilang
        if hand_was_absent or not first_hand_detected:
            first_hand_detected = True
            hand_was_absent     = False
            first_detect_time   = time.time()
            speak("Tangan terdeteksi, mohon tunggu tiga detik.")

        elapsed = time.time() - first_detect_time
        if elapsed < initial_delay:
            remaining = int(initial_delay - elapsed) + 1
            cv2.putText(img, f"Bersiap... {remaining}", (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            cv2.imshow("Hand Gesture - Perkenalan Ahmad Ragiel Zaini", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Proses tiap tangan
        h_img, w_img, _ = img.shape
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            is_palm, conf = is_palm_facing_camera(img, handLms)

            handedness = "Unknown"
            try:
                if results.multi_handedness and len(results.multi_handedness) > idx:
                    handedness = results.multi_handedness[idx].classification[0].label
            except Exception:
                pass

            fingers = finger_status(handLms, handedness)
            cx = int(np.mean([lm.x for lm in handLms.landmark]) * w_img)

            hand_data.append({
                'is_palm':    is_palm,
                'fingers':    fingers,
                'conf':       conf,
                'handedness': handedness,
                'cx':         cx,
            })

        num_hands = len(hand_data)

        # Deteksi gesture
        gesture_text, gesture_mode = detect_gesture(hand_data)

        # Palm status label
        if num_hands == 2:
            palm_count = sum(1 for h in hand_data if h['is_palm'])
            open_hands = [sum(h['fingers']) for h in hand_data]
            if all(f >= 4 for f in open_hands):
                palm_status = f"2 Telapak ({palm_count}/2 terdeteksi)"
            else:
                palm_status = f"2 Tangan (Telapak: {palm_count}/2)"
        elif num_hands == 1:
            h0 = hand_data[0]
            palm_status = f"1 Tangan ({'Telapak' if h0['is_palm'] else 'Punggung'}) {h0['conf']:.0f}%"

    else:
        if first_hand_detected:
            hand_was_absent = True

    # ── Audio trigger ──────────────────────────────────────────────────────
    current_time     = time.time()
    is_audio_playing = False
    if audio_available and pygame.mixer.get_init():
        try:
            is_audio_playing = pygame.mixer.music.get_busy()
        except Exception:
            pass

    if (gesture_text
            and gesture_text != last_gesture
            and (current_time - last_time) > cooldown
            and not is_audio_playing):
        last_gesture = gesture_text
        last_time    = current_time
        speak(gesture_text)

    # ── HUD utama ──────────────────────────────────────────────────────────
    cv2.rectangle(img, (0, 0), (img.shape[1], 120), (0, 0, 0), -1)
    cv2.rectangle(img, (0, 0), (img.shape[1], 120), (50, 50, 50), 1)

    cv2.putText(img, palm_status, (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    # Badge mode
    if gesture_mode == 'dual':
        badge_color = (255, 130, 0)
        badge_label = "[ DUAL HAND ]"
    elif gesture_mode == 'single':
        badge_color = (0, 210, 100)
        badge_label = "[ SINGLE HAND ]"
    else:
        badge_color = (120, 120, 120)
        badge_label = ""

    if badge_label:
        cv2.putText(img, badge_label, (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, badge_color, 2)

    if gesture_text:
        cv2.putText(img, gesture_text, (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 80), 3)
    else:
        cv2.putText(img, "Tunjukkan gesture  |  'g' panduan  |  'q' keluar",
                    (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 170, 170), 1)

    # Debug finger state di pojok bawah kiri
    for i, hand in enumerate(hand_data):
        side = hand['handedness'][0] if hand['handedness'] != 'Unknown' else '?'
        dbg  = f"H{i+1}({side}): {hand['fingers']}"
        cv2.putText(img, dbg, (15, img.shape[0] - 12 - i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 255), 1)

    # ── Panduan gesture (toggle 'g') ───────────────────────────────────────
    if show_guide:
        panel_w = 320
        panel_x = img.shape[1] - panel_w - 10
        panel_y = 128
        line_h  = 21
        panel_h = len(GUIDE_LINES) * line_h + 16

        # Background semi-transparan
        overlay = img.copy()
        cv2.rectangle(overlay, (panel_x - 8, panel_y - 8),
                      (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        cv2.rectangle(img, (panel_x - 8, panel_y - 8),
                      (panel_x + panel_w, panel_y + panel_h), (60, 60, 60), 1)

        for i, line in enumerate(GUIDE_LINES):
            if line.startswith('==='):
                color = (0, 220, 255)
                thick = 1
            elif line.startswith("'"):
                color = (160, 160, 160)
                thick = 1
            else:
                color = (220, 220, 220)
                thick = 1
            cv2.putText(img, line, (panel_x, panel_y + i * line_h + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, thick)

    cv2.imshow("Hand Gesture - Perkenalan Ahmad Ragiel Zaini", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('g'):
        show_guide = not show_guide

# ── Cleanup ────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
if audio_available:
    try:
        pygame.mixer.quit()
    except Exception:
        pass
try:
    hands_detector.close()
except Exception:
    pass