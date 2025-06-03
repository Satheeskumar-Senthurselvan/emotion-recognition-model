from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException # <-- Added HTTPException

# Load environment variables from .env file
load_dotenv()

# Load your trained model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ml-model', 'models', 'final_stress_model.keras')
print(f"Attempting to load model from: {model_path}")

try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    model.summary() # This should now work!

    # Get class labels from your training script (e.g., train_generator.class_indices)
    # Ensure this order matches how your model was trained
    CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    IMG_SIZE = (96, 96) # Must match your model's input size
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"Error loading model in shell after upgrade: {e}")
    import traceback
    traceback.print_exc()

app = FastAPI()

# --- ADD CORS MIDDLEWARE HERE ---
origins = [
    "http://localhost",
    "http://localhost:3000", # Default Create React App port (if not using Vite)
    "http://localhost:5173", # Default Vite React port
    # Add any other frontend URLs if deployed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- END CORS MIDDLEWARE ---

class PredictionResult(BaseModel):
    emotion: str
    confidence: float
    all_confidences: dict

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Emotion Recognition API!"}

@app.post("/predict/image/", response_model=PredictionResult)
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        # If model loading failed at startup, return an error
        raise HTTPException(status_code=500, detail="Machine learning model failed to load.")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(IMG_SIZE)
        image_array = np.array(image) / 255.0 # Normalize pixels (0-1)
        image_array = np.expand_dims(image_array, axis=0) # Add batch dimension

        # Make prediction
        predictions = model.predict(image_array)[0] # Get probabilities for the single image

        # Get predicted emotion and confidence
        predicted_index = np.argmax(predictions)
        predicted_emotion = CLASS_NAMES[predicted_index]
        confidence = float(predictions[predicted_index])

        # Prepare all confidences for the response
        all_confidences = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}

        return PredictionResult(
            emotion=predicted_emotion,
            confidence=confidence,
            all_confidences=all_confidences
        )
    except Exception as e:
        # Catch any processing errors and return a proper error response
        print(f"Error during prediction: {e}") # Log the actual error for debugging
        raise HTTPException(status_code=500, detail=f"Image processing or prediction failed: {e}")

# You would typically run the app using 'uvicorn app:app --reload' from the terminal
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)