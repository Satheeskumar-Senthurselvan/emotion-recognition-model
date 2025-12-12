import cv2
import numpy as np
import webbrowser
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load model and face detector
MODEL_PATH = os.getenv('MODEL_PATH', 'ml-model/emotion-recognition-model-6.2.keras')
model = load_model(MODEL_PATH)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion and stress mapping
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
stress_map = {
    'Angry': 'High',
    'Disgust': 'High',
    'Fear': 'High',
    'Sad': 'Moderate',
    'Surprise': 'Low',
    'Happy': 'None',
    'Neutral': 'None'
}

# Spotify calm playlist link - Load from environment variable
spotify_playlist_url = os.getenv('SPOTIFY_PLAYLIST_URL', 'https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0')
spotify_opened = False

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Webcam stream started with Spotify intervention...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    stress_detected = False

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (96, 96))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = img_to_array(face) / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face, verbose=0)[0]
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]
        stress_level = stress_map.get(emotion, "None")

        label = f"{emotion} ({stress_level} Stress)"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if stress_level in ['High', 'Moderate']:
            stress_detected = True

    # Launch Spotify Calm Playlist
    if stress_detected and not spotify_opened:
        print("[INFO] Opening Spotify playlist for calm music...")
        webbrowser.open(spotify_playlist_url)
        spotify_opened = True

    if not stress_detected:
        spotify_opened = False  # reset when no stress detected

    cv2.imshow("Stress Detection - Spotify Intervention", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
