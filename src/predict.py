import cv2
import pygame
import numpy as np
import tensorflow as tf
from preprocessing import process_frame

# Загрузка модели и классов
model = tf.keras.models.load_model(
    "models/sign_language_model.h5"
)

classes = np.load(
    "data/processed/classes.npy",
    allow_pickle=True
)

# Камера
cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Окно
pygame.init()

scale = 1.5
screen = pygame.display.set_mode(
    (int(width * scale), int(height * scale))
)
pygame.display.set_caption(
    "Sign Language Recognition"
)
running = True

while running:
    # Обработка событий
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    success, frame = cam.read()
    if not success:
        break

    # Поиск руки
    frame, landmarks = process_frame(frame)
    if landmarks is not None:
        # Подготовка данных
        data = landmarks.flatten().reshape(1, -1)
        # Предсказание
        prediction = model.predict(
            data,
            verbose=0
        )

        class_id = np.argmax(prediction)
        letter = classes[class_id]
        confidence = prediction[0][class_id]

        # Вывод результата
        cv2.putText(
            frame,
            f"{letter} ({confidence:.2f})",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
    # Отображение кадра
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )
    surface = pygame.surfarray.make_surface(
        np.rot90(frame_rgb)
    )
    surface = pygame.transform.scale(
        surface,
        (int(width * scale), int(height * scale))
    )
    screen.blit(
        pygame.transform.flip(surface, True, False),
        (0, 0)
    )

    pygame.display.update()

# Очистка ресурсов
cam.release()
pygame.quit()