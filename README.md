# Sign Language Translator

A real-time sign language recognition system developed using **Python**, **TensorFlow**, **MediaPipe**, and **OpenCV**.

The application detects a user's hand through a webcam, extracts hand landmarks using MediaPipe, and recognizes sign language gestures with a trained neural network.

---

## Features

* Real-time hand detection
* Hand landmark extraction using MediaPipe
* Sign language gesture recognition
* Neural network trained with TensorFlow/Keras
* Live prediction with confidence score
* Simple and modular project structure

---

## Technologies

* Python 3
* TensorFlow / Keras
* OpenCV
* MediaPipe
* NumPy
* Pygame
* Scikit-learn

---

## Project Structure

```
SignLanguageTranslator/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── labels/
│
├── models/
│   └── sign_language_model.h5
│
├── src/
│   ├── camera.py
│   ├── config.py
│   ├── dataset.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/SignLanguageTranslator.git
```

Open the project directory:

```bash
cd SignLanguageTranslator
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the application:

```bash
python3 main.py
```

The webcam will open automatically.

Show a supported hand gesture in front of the camera, and the application will display the predicted letter together with the confidence score.

---

## Model

The neural network was trained using hand landmark coordinates extracted by MediaPipe.

* Input size: **63 features** (21 hand landmarks × 3 coordinates)
* Hidden layers: **256 → 128 → 64 neurons**
* Output: **Softmax classification**
* Optimizer: **Adam**
* Loss function: **Sparse Categorical Crossentropy**

Model accuracy on the test dataset: **97%**

---

## Future Improvements

* Support for full words and sentences
* Recognition of dynamic gestures
* Improved graphical user interface
* Higher robustness under different lighting conditions
* Extended gesture dataset

---

## Author

Developed as an educational machine learning and computer vision project.