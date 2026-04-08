import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Kamera berhasil dibuka")
    ret, frame = cap.read()
    if ret:
        print("Frame berhasil ditangkap")
    else:
        print("Gagal menangkap frame")
    cap.release()
else:
    print("Gagal membuka kamera")
