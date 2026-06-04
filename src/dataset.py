import os
import numpy as np
from preprocessing import process_frame
import cv2

# Пути к данным
TRAIN_PATH = 'data/raw/archive/asl_alphabet_train/asl_alphabet_train'
PROCESSED_PATH = 'data/processed'

def extract_landmarks():
    """Извлекает landmarks из всех фото и сохраняет в файлы"""
    
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    
    labels = sorted(os.listdir(TRAIN_PATH))
    print(f"Найдено классов: {len(labels)}")

    all_landmarks = []
    all_labels = []

    for label in labels:
        label_path = os.path.join(TRAIN_PATH, label)
        images = os.listdir(label_path)
        count = 0

        for img_name in images:
            img_path = os.path.join(label_path, img_name)
            frame = cv2.imread(img_path)

            if frame is None:
                continue

            # Извлекаем landmarks
            _, landmarks = process_frame(frame)

            if landmarks is not None:
                all_landmarks.append(landmarks.flatten())
                all_labels.append(label)
                count += 1

        print(f"{label}: {count} примеров")

    # Сохраняем данные
    np.save(f'{PROCESSED_PATH}/landmarks.npy', np.array(all_landmarks))
    np.save(f'{PROCESSED_PATH}/labels.npy', np.array(all_labels))
    print("Готово! Данные сохранены в data/processed/")

if __name__ == '__main__':
    extract_landmarks()