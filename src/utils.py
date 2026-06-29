import numpy as np

def prepare_input(landmarks):
    # Подготавливаем landmarks для модели
    return landmarks.flatten().reshape(1, -1)

def predict_gesture(model, landmarks, classes):
    # Подготавливаем входные данные
    data = prepare_input(landmarks)

    # Выполняем предсказание
    prediction = model.predict(data, verbose=0)

    # Находим наиболее вероятный класс
    class_index = np.argmax(prediction)

    # Получаем букву и вероятность
    letter = classes[class_index]
    confidence = float(prediction[0][class_index])

    return letter, confidence

def is_confident(confidence, threshold):
    # Проверяем уверенность модели
    return confidence >= threshold