import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('models/final_stress_model.keras')

# Define class labels (adjust if different)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Map emotions to stress level (optional logic)
stress_map = {
    'Angry': 'High Stress',
    'Disgust': 'High Stress',
    'Fear': 'High Stress',
    'Sad': 'Moderate Stress',
    'Happy': 'No Stress',
    'Surprise': 'Low Stress',
    'Neutral': 'No Stress'
}

# Initialize OpenCV face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
video_capture = cv2.VideoCapture(0)

print("[INFO] Starting webcam stream...")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Preprocess face for model
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (96, 96))  # must match model input
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = img_to_array(face)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        # Predict emotion
        prediction = model.predict(face, verbose=0)[0]
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]
        stress_level = stress_map.get(emotion, "Unknown")

        # Display predicted emotion and stress level
        label = f"{emotion} ({stress_level})"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Stress Detection - Webcam", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
print("[INFO] Webcam stream ended.")
