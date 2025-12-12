import tensorflow.keras.backend as K
import gc
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

K.clear_session()
gc.collect()

# Clean up .DS_Store files
dataset_cleanup_path = os.getenv('DATASET_CLEANUP_PATH', 'ml-model/')
if os.path.exists(dataset_cleanup_path):
    os.system(f"find {dataset_cleanup_path} -name '.DS_Store' -type f -delete")
os.system("find . -type d -name '__pycache__' -exec rm -r {} +")


# Configurations - Load from environment variables with defaults
img_size_str = os.getenv('IMG_SIZE', '96,96')
IMG_SIZE = tuple(map(int, img_size_str.split(',')))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
NUM_CLASSES = int(os.getenv('NUM_CLASSES', '7'))
EPOCHS = int(os.getenv('EPOCHS', '50'))
TRAIN_DIR = os.getenv('DATASET_TRAIN_DIR', 'ml-model/dataset-6.2/train')
VAL_DIR = os.getenv('DATASET_VAL_DIR', 'ml-model/dataset-6.2/val')
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', 'ml-model/stress_fer_resnet50.keras')

# Data Augmentation and Loading
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
)

# Load base model with pretrained weights, exclude top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Freeze all base layers initially
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

# Train frozen model head for initial 5 epochs
initial_epochs = 5
history = model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop]
)

# Unfreeze last 40 layers of ResNet50 for fine-tuning
for layer in base_model.layers[-80:]:
    layer.trainable = True

# Recompile with lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune for remaining epochs
fine_tune_epochs = EPOCHS - initial_epochs
history_fine = model.fit(
    train_generator,
    epochs=fine_tune_epochs,
    validation_data=val_generator,
    #callbacks=[checkpoint, early_stop]
    callbacks=[checkpoint]
)

# Save final model in Keras native format
FINAL_MODEL_PATH = os.getenv('FINAL_MODEL_PATH', 'ml-model/final_stress_model.keras')
model.save(FINAL_MODEL_PATH)
print(train_generator.class_indices)

# Plot training & validation accuracy
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()