from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras

app = FastAPI()

CLASS_NAMES = ['Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_healthy']

MODEL = keras.models.load_model('../models/1')


@app.get("/ping")
async def ping():
    return "pong"


def image_from_file(file) -> np.array:
    image = np.array(Image.open(BytesIO(file)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = image_from_file(await file.read())
    # add batch dimension to image array (256x256x3) -> (1x256x256x3)
    image_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(image_batch)

    """
    Since we have softmax activation in the last layer of the model,
    we can use the argmax function to get the index of the highest value in the array.
    we can then use the index to get the class name.

    for confidence, we can use the max function to get the highest value in the array.
    """
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence_prob = np.max(predictions[0])

    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence_prob)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
