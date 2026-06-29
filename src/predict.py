import cv2
import pygame
import numpy as np
import tensorflow as tf
from src.preprocessing import process_frame
from src.utils import predict_gesture, is_confident
from src.config import *

def main():
    # Загрузка модели и классов
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)

    # Инициализация камеры
    cam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Создание окна
    pygame.init()
    screen = pygame.display.set_mode(
        (int(width * WINDOW_SCALE),
         int(height * WINDOW_SCALE))
    )

    pygame.display.set_caption(WINDOW_TITLE)
    running = True
    while running:

        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Получение кадра
        success, frame = cam.read()

        if not success:
            break

        # Поиск руки
        frame, landmarks = process_frame(frame)
        if landmarks is not None:

            # Распознавание жеста
            letter, confidence = predict_gesture(
                model,
                landmarks,
                classes
            )
            if is_confident(confidence, CONFIDENCE_THRESHOLD):

                cv2.putText(
                    frame,
                    f"{letter} ({confidence:.2f})",
                    TEXT_POSITION,
                    FONT,
                    FONT_SCALE,
                    FONT_COLOR,
                    FONT_THICKNESS
                )

        # Отображение изображения
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surface = pygame.surfarray.make_surface(
            np.rot90(frame_rgb)
        )
        surface = pygame.transform.scale(
            surface,
            (
                int(width * WINDOW_SCALE),
                int(height * WINDOW_SCALE)
            )
        )
        screen.blit(
            pygame.transform.flip(surface, True, False),
            (0, 0)
        )
        pygame.display.update()
    cam.release()
    pygame.quit()

if __name__ == "__main__":
    main()