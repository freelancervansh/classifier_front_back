from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import requests
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

# Load the model
MODEL_PATH = "/Users/vansh/Documents/assignment/model/front_back_classifier_mobilenetv2.h5"
model = load_model(MODEL_PATH)

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Class labels
class_labels = {0: "back", 1: "front"}

# Input data model
class ImageURL(BaseModel):
    url: str

@app.post("/predict/")
async def predict_image(image_data: ImageURL):
    """
    Endpoint to predict the class of an image from a URL.
    """
    try:
        # Fetch the image from the URL
        response = requests.get(image_data.url)
        response.raise_for_status()

        # Load the image from the URL
        image = load_img(BytesIO(response.content), target_size=(IMG_HEIGHT, IMG_WIDTH))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0

        # Make a prediction
        prediction = model.predict(image)
        predicted_class = class_labels[int(prediction[0] > 0.5)]

        return {"class": predicted_class, "confidence": float(prediction[0])}
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch image from URL: {str(e)}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

if __name__ == "__main__":
    # Run the FastAPI app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
