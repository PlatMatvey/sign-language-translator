import cv2
import pygame
import numpy as np
from src.preprocessing import process_frame

# Открываем камеру
cam = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Получаем размер кадра
w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Настраиваем запись видео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (w, h))

# Инициализируем окно
pygame.init()
scale = 1.5
screen = pygame.display.set_mode((int(w * scale), int(h * scale)))
pygame.display.set_caption("Камера")

while True:
    # Проверяем закрытие окна
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cam.release()
            out.release()
            pygame.quit()
            exit()

    # Читаем кадр с камеры
    ret, frame = cam.read()
    if not ret:
        break

    # Обрабатываем кадр через MediaPipe
    frame, landmarks = process_frame(frame)

    # Выводим landmarks в консоль если рука найдена
    if landmarks is not None:
        print(f"Рука найдена! Точек: {len(landmarks)}")

    # Записываем кадр в файл
    out.write(frame)

    # Отображаем кадр на экране
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    surf = pygame.surfarray.make_surface(np.rot90(frame_rgb))
    surf = pygame.transform.scale(surf, (int(w * scale), int(h * scale)))
    screen.blit(pygame.transform.flip(surf, True, False), (0, 0))
    pygame.display.update()