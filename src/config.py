# Пути к модели и данным
MODEL_PATH = "models/sign_language_model.h5"
CLASSES_PATH = "data/processed/classes.npy"

# Настройки камеры
CAMERA_INDEX = 0
WINDOW_SCALE = 1.5
WINDOW_TITLE = "Sign Language Recognition"

# Настройки распознавания
CONFIDENCE_THRESHOLD = 0.90

# Параметры отображения текста
TEXT_POSITION = (20, 50)
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_COLOR = (0, 255, 0)
FONT_THICKNESS = 2