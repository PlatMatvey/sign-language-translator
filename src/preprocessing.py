import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os

# Скачиваем модель если нет
model_path = 'models/hand_landmarker.task'
if not os.path.exists(model_path):
    print("Скачиваем модель...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        model_path
    )
    print("Модель скачана!")

# Настраиваем детектор рук через новый API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Настройки детектора
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = HandLandmarker.create_from_options(options)

# Связи между точками пальцев
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),     
    (0, 5), (5, 6), (6, 7), (7, 8),     
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

def process_frame(frame):
    # Конвертируем кадр для MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Ищем руку
    results = detector.detect(mp_image)
    landmarks = None

    if results.hand_landmarks:
        hand = results.hand_landmarks[0]

        # Извлекаем 21 точку (x, y, z)
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand])

        h, w = frame.shape[:2]

        # Рисуем соединения
        for start, end in connections:
            x1, y1 = int(hand[start].x * w), int(hand[start].y * h)
            x2, y2 = int(hand[end].x * w), int(hand[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Рисуем точки поверх линий
        for lm in hand:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

    return frame, landmarks