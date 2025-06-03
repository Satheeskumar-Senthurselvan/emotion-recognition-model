import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

os.system("find /Users/satheeskumar/Technologies/FER/dataset/ -name '.DS_Store' -type f -delete")

IMG_SIZE = (96, 96)
BATCH_SIZE = 32
NUM_CLASSES = 7
MODEL_PATH = 'models/final_stress_model.keras'
TEST_DIR = 'dataset1/test'  
CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

model = load_model(MODEL_PATH)
print(f"Loaded model from: {MODEL_PATH}")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(" Predicting...")
pred_probs = model.predict(test_generator)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_generator.classes

# Classification Report
print("\n Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

# Confusion Matrix
print("\n Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

#Sample Predictions
def show_sample_predictions(generator, y_pred, num_samples=6):
    class_indices = {v: k for k, v in generator.class_indices.items()}
    x_batch, y_batch = next(generator)
    plt.figure(figsize=(15, 6))
    for i in range(num_samples):
        ax = plt.subplot(2, num_samples // 2, i + 1)
        img = x_batch[i]
        true_idx = np.argmax(y_batch[i])
        pred_idx = np.argmax(model.predict(img[np.newaxis, ...]))
        plt.imshow(img)
        plt.title(f"True: {class_indices[true_idx]}\nPred: {class_indices[pred_idx]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

print("\n Showing sample predictions:")
# Use a new generator with shuffle=True for random images
sample_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)
show_sample_predictions(sample_generator, y_pred)
