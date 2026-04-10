import cv2

def test_kamera(max_index=3):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            print(f"[index {i}] Gagal membuka kamera")
            continue

        ret, frame = cap.read()
        if not ret:
            print(f"[index {i}] Kamera terbuka tapi gagal capture frame")
            cap.release()
            continue

        # Info kamera
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[index {i}] ✅ Kamera OK — Resolusi: {w}x{h}, FPS: {fps}")

        # Preview live (tekan 'q' untuk lanjut ke kamera berikutnya)
        print(f"[index {i}] Preview aktif — tekan 'q' untuk lanjut")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(f"Kamera index {i}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

test_kamera()