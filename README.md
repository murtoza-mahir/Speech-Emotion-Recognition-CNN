# 🎙️ Abhas — Audio Emotion Recognition from Speech Signals (CNN)

## 📌 Project Overview

**Abhas** is a deep learning project that classifies **human emotions from audio speech signals** using a **Convolutional Neural Network (CNN)**. The model processes raw `.wav` audio files from the **RAVDESS dataset**, extracts **MFCC (Mel-Frequency Cepstral Coefficients)** features, and classifies speech into 4 emotional states: **Happy, Sad, Angry, and Neutral**.

The name *Abhas* means "feeling" or "sense" in Bengali — reflecting the project's goal of teaching machines to sense human emotion from voice.

---

## 🎯 What This Project Does

- Loads and processes the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset
- Extracts **40 MFCC features** from each audio file using Librosa
- Pads/truncates features to a fixed length (174 frames) for consistent input shape
- Builds and trains a **CNN model** with 2 Conv2D layers, MaxPooling, Dropout, and Dense layers
- Evaluates model accuracy on a held-out test set
- Predicts emotion from any new `.wav` audio file

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| TensorFlow / Keras | Deep learning framework |
| Librosa | Audio feature extraction |
| NumPy / Pandas | Data manipulation |
| Scikit-learn | Label encoding & data splitting |
| Matplotlib | Training curve visualization |
| Google Colab | Training environment |

---

## 🏗️ Model Architecture

```
Input (40 MFCCs × 174 frames × 1 channel)
    ↓
Conv2D (32 filters, 3×3, ReLU)
    ↓
MaxPooling2D (2×2)
    ↓
Dropout (0.4)
    ↓
Conv2D (64 filters, 3×3, ReLU)
    ↓
MaxPooling2D (2×2)
    ↓
Dropout (0.4)
    ↓
Flatten
    ↓
Dense (128, ReLU, L2 regularization)
    ↓
Dropout (0.4)
    ↓
Dense (4, Softmax) → [Happy, Sad, Angry, Neutral]
```

---

## 🎵 Dataset — RAVDESS

- **Full name:** Ryerson Audio-Visual Database of Emotional Speech and Song
- **Emotions used:** Happy, Sad, Angry, Neutral (4 of 8 available)
- **Format:** `.wav` audio files
- **Actors:** 24 professional actors (12 male, 12 female)
- **Download:** [Kaggle RAVDESS Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

---

## 📂 Repository Structure

```
abhas-audio-emotion-recognition/
│
├── Emotion_Recognition_from_Audio_(speech_signals).ipynb   # Main notebook
├── README.md                                                 # Project documentation
└── RAVDESS/                                                  # Dataset folder (not included)
    ├── Actor_01/
    ├── Actor_02/
    └── ...
```

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**
2. Download the RAVDESS dataset from Kaggle
3. Upload and unzip to `/content/RAVDESS/` in Colab
4. Run all cells from top to bottom
5. Use `predict_emotion(file_path)` to test on any `.wav` file

---

## 💡 Key Concepts Demonstrated

- Audio feature extraction using MFCC (Mel-Frequency Cepstral Coefficients)
- CNN architecture for 2D feature map classification
- L2 regularization and Dropout to prevent overfitting
- EarlyStopping callback for optimal training
- Multi-class emotion classification from raw audio

---

## 👤 Author

**Md Murtoza Mahir**
Data Analyst | ML Engineer | Power BI | Python | SQL
📧 murtozamahir.info@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/murtoza-mahir)
🌐 [Portfolio](https://murtoza-mahir.github.io)
