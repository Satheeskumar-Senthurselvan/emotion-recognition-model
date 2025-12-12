# Emotion Recognition Model

## 🌟 Project Overview

This project is a real-time emotion recognition system using deep learning. It leverages a ResNet50-based convolutional neural network trained on facial expression data to detect and classify emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) from webcam input. The system can map detected emotions to stress levels and provide interventions such as opening calming Spotify playlists.

## ✨ Features

* **Real-time Emotion Recognition:** Detects 7 different emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) from live webcam feed
* **Stress Level Mapping:** Maps detected emotions to stress levels (None, Low, Moderate, High)
* **Model Training Pipeline:** Complete training script with data augmentation, transfer learning, and fine-tuning
* **Model Evaluation:** Comprehensive evaluation with confusion matrix, classification report, and sample predictions
* **Interactive Interventions:** Automatic Spotify playlist intervention when stress is detected
* **Configurable Setup:** Environment-based configuration for easy customization

## 🚀 Technologies Used

**Machine Learning:**
* Python 3.9+
* TensorFlow / Keras
* ResNet50 (pre-trained on ImageNet)
* OpenCV (for webcam access and face detection)
* scikit-learn (for evaluation metrics)

**Data Processing:**
* NumPy
* Matplotlib (for visualization)
* Seaborn (for confusion matrix visualization)

**Configuration:**
* python-dotenv (for environment variable management)

**Development:**
* Git (for version control)

## 🛠️ Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed:

* [Python 3.9+](https://www.python.org/downloads/)
* [Git](https://git-scm.com/downloads)
* (Recommended) A virtual environment tool (venv, virtualenv, or conda)
* A webcam (for real-time emotion detection)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Satheeskumar-Senthurselvan/emotion-recognition-model.git
    cd emotion-recognition-model
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn python-dotenv
    ```
    
    Or create a `requirements.txt` file with:
    ```
    tensorflow>=2.10.0
    opencv-python>=4.6.0
    numpy>=1.23.0
    matplotlib>=3.6.0
    seaborn>=0.12.0
    scikit-learn>=1.1.0
    python-dotenv>=0.21.0
    ```
    
    Then install:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    ```bash
    cp .env.example .env
    ```
    
    Edit `.env` file and configure the paths to match your system:
    - Dataset paths (`DATASET_TRAIN_DIR`, `DATASET_VAL_DIR`, `DATASET_TEST_DIR`)
    - Model paths (`MODEL_PATH`, `MODEL_SAVE_PATH`, `FINAL_MODEL_PATH`)
    - Spotify playlist URL (if using intervention feature)
    - Other configuration parameters (batch size, epochs, etc.)

### Running the Application

Make sure your virtual environment is activated before running any scripts.

#### Training the Model

Train a new emotion recognition model:
```bash
python train_emotion_model.py
```

This will:
- Load and preprocess training/validation datasets
- Train a ResNet50-based model with transfer learning
- Save checkpoints and the final model
- Display training/validation accuracy and loss plots

#### Evaluating the Model

Evaluate the trained model on test data:
```bash
python evaluate_model.py
```

This will:
- Load the trained model
- Run predictions on test dataset
- Generate a classification report
- Display a confusion matrix
- Show sample predictions with visualizations

#### Real-time Emotion Detection

Run real-time emotion detection from webcam:
```bash
python emotion_detection_webcam.py
```

Features:
- Detects faces in real-time from webcam feed
- Predicts emotion and maps to stress level
- Displays emotion and stress level on video feed
- Press 'q' to quit

**macOS Camera Permission Note:**
If you encounter "not authorized to capture video" error on macOS, you need to grant camera access:
1. Open **System Settings** → **Privacy & Security** → **Camera**
2. Enable camera access for your terminal app (Terminal, iTerm2, etc.)
3. Run the script again after granting permissions

#### Emotion Detection with Intervention

Run emotion detection with automatic Spotify intervention:
```bash
python emotion_intervention_spotify.py
```

Features:
- All features from `emotion_detection_webcam.py`
- Automatically opens Spotify playlist when stress is detected
- Playlist URL can be configured in `.env` file

## 📂 Project Structure

```
emotion-recognition-model/
├── ml-model/                           # Machine Learning data and models
│   ├── dataset-5.5/                   # Dataset version 5.5
│   │   ├── train/                     # Training images (7 emotion classes)
│   │   └── val/                       # Validation images (7 emotion classes)
│   ├── dataset-6.2/                   # Dataset version 6.2
│   │   ├── train/                     # Training images (7 emotion classes)
│   │   ├── val/                       # Validation images (7 emotion classes)
│   │   └── test/                      # Test images (7 emotion classes)
│   ├── emotion-recognition-model-5.5.keras  # Pre-trained model (v5.5)
│   └── emotion-recognition-model-6.2.keras  # Pre-trained model (v6.2)
│
├── train_emotion_model.py             # Train the emotion recognition model
├── evaluate_model.py                  # Evaluate model on test data
├── emotion_detection_webcam.py        # Real-time emotion detection from webcam
├── emotion_intervention_spotify.py    # Emotion detection with Spotify intervention
│
├── .env.example                       # Environment variables template
├── .env                               # Your local environment variables (not in git)
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
└── README.md                          # This file
```

## 🔧 Configuration

All configuration is managed through environment variables. Copy `.env.example` to `.env` and customize:

- **Dataset Paths**: Point to your dataset directories
- **Model Paths**: Specify where to save/load models
- **Training Parameters**: Batch size, epochs, image size, etc.
- **Spotify Integration**: Playlist URL for interventions

See `.env.example` for all available configuration options.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'feat: Add new feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
*(If you don't have a LICENSE.md file, you might want to add one. GitHub can help you choose one when you create a new file.)*

## 📧 Contact

Satheeskumar Senthurselvan - [https://github.com/Satheeskumar-Senthurselvan] - [satheeskumar.dev@gmail.com]

Project Link: [https://github.com/Satheeskumar-Senthurselvan/emotion-recognition-model](https://github.com/Satheeskumar-Senthurselvan/emotion-recognition-model)

## 📝 Model Details

- **Architecture**: ResNet50 (transfer learning)
- **Input Size**: 96x96 RGB images
- **Classes**: 7 emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- **Training**: Transfer learning with frozen base layers, then fine-tuning
- **Data Augmentation**: Rotation, zoom, shifts, horizontal flip

## 🎯 Emotion to Stress Mapping

- **High Stress**: Angry, Disgust, Fear
- **Moderate Stress**: Sad
- **Low Stress**: Surprise
- **No Stress**: Happy, Neutral

