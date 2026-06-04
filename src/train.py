import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import os

# Пути
PROCESSED_PATH = 'data/processed'
MODEL_PATH = 'models/sign_language_model.h5'

# Загружаем данные
print("Загружаем данные...")
X = np.load(f'{PROCESSED_PATH}/landmarks.npy')
y = np.load(f'{PROCESSED_PATH}/labels.npy')

print(f"Примеров: {len(X)}, классов: {len(set(y))}")

# Кодируем метки (A=0, B=1, ...)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
np.save(f'{PROCESSED_PATH}/classes.npy', encoder.classes_)

# Нормализуем данные
X = X / X.max()

# Разделяем на обучение и тест
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Строим модель
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(63,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Обучаем
print("Обучаем модель...")
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Сохраняем
os.makedirs('models', exist_ok=True)
model.save(MODEL_PATH)
print(f"Модель сохранена в {MODEL_PATH}")