from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ['Early blight', 'Late blight', 'Healthy']

endpoint = "http://localhost:8001/v1/models/tomatoes_model:predict"


@app.get("/ping")
async def ping():
    return "pong"


def image_from_file(file) -> np.array:
    image = np.array(Image.open(BytesIO(file)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = image_from_file(await file.read())
    img_batch = np.expand_dims(image, axis=0)
    response = requests.post(endpoint, json={"instances": img_batch.tolist()})
    prediction = np.array(response.json()['predictions'][0])
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

# To run do the following:
# $ uvicorn main:app --reload
