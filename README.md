# 🖐️ Hand Gesture Recognition — Perkenalan Nama

Project computer vision berbasis **gesture tangan** untuk memperkenalkan diri secara interaktif menggunakan webcam. Setiap gesture tangan yang ditunjukkan akan memunculkan teks dan suara dalam Bahasa Indonesia secara real-time.

---

## 📸 Demo

> Tunjukkan gesture tangan ke kamera → teks muncul di layar → suara otomatis berbunyi

---

## ✨ Fitur

- 🖐️ Deteksi hingga **2 tangan** secara bersamaan
- 🔊 **Text-to-Speech** otomatis (Bahasa Indonesia) dengan sistem cache audio
- 🤏 Deteksi **8 gesture** berbeda dengan mapping ke kalimat perkenalan
- 📷 **Auto-detect kamera** — otomatis fallback ke index kamera lain jika gagal
- ⏱️ **Cooldown system** agar audio tidak spam
- 🧠 Palm detection cerdas menggunakan 3 metode: brightness, depth, dan normal vector
- 🐛 Debug overlay: tampilkan status jari per tangan secara real-time

---

## 🤟 Daftar Gesture
ini adalah contoh, Setiap kata bisa diganti sesuai keinginan
| Gesture (Thumb, Index, Middle, Ring, Pinky) | Arti |
|---|---|
| ✊ Semua jari tertutup `(0,0,0,0,0)` | *"kelass!"* |
| 🤘 Jempol + telunjuk `(1,1,0,0,0)` | *"EZ!"* |
| 👍 Jempol saja `(1,0,0,0,0)` | *"Sip!"* |
| ☝️ Telunjuk saja `(0,1,0,0,0)` | *"nama saya..."* |
| ✌️ Telunjuk + tengah `(0,1,1,0,0)` | *"Ahmad Ragiel Zaini"* |
| 🤟 Telunjuk + tengah + manis `(0,1,1,1,0)` | *"Terima kasih!"* |
| 🤙 4 jari kecuali jempol `(0,1,1,1,1)` | *"Salam kenal ya!"* |
| 🖐️ Semua jari terbuka `(1,1,1,1,1)` | *"Halo semuanya!"* |

---

## 🛠️ Teknologi yang Digunakan

- [Python 3.8+](https://www.python.org/)
- [OpenCV](https://opencv.org/) — capture & display kamera
- [MediaPipe](https://mediapipe.dev/) — deteksi landmark tangan
- [gTTS](https://pypi.org/project/gTTS/) — Google Text-to-Speech
- [Pygame](https://www.pygame.org/) — playback audio
- [NumPy](https://numpy.org/) — kalkulasi vektor palm detection

---

## ⚙️ Instalasi

### 1. Clone repository
```bash
git clone https://github.com/ahmadragiel/hand-gesture-perkenalan.git
cd hand-gesture-perkenalan
```

### 2. Install dependencies
```bash
pip install opencv-python mediapipe gtts pygame numpy
```

### 3. Jalankan program
```bash
python gerakan.py
```

> Pastikan webcam terhubung sebelum menjalankan program.

---

## 📁 Struktur Project

```
hand-gesture-perkenalan/
│
├── gerakan.py          # Main program — deteksi gesture & TTS
├── test_kamera.py      # Utility untuk test koneksi kamera
├── audio_cache/        # Cache file audio TTS (auto-generated)
└── README.md
```

---

## 🚀 Cara Penggunaan

1. Jalankan `gerakan.py`
2. Arahkan tangan ke kamera
3. Tunggu **3 detik** setelah tangan terdeteksi (countdown muncul di layar)
4. Tunjukkan gesture sesuai tabel di atas
5. Teks dan suara akan muncul otomatis
6. Tekan **`q`** untuk keluar

---

## 📝 Catatan

- Pastikan pencahayaan ruangan **cukup terang** agar deteksi lebih akurat
- Gunakan **latar belakang kontras** dengan warna kulit untuk hasil terbaik
- File audio di-cache di folder `audio_cache/` — bisa dihapus manual jika ingin generate ulang
- Program otomatis mencoba index kamera `0–3` jika kamera utama tidak terdeteksi

---

## 👤 Author

**Ahmad Ragiel Zaini**  

---