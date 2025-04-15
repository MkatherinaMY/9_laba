from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка модели
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.h5')
model = tf.keras.models.load_model(model_path)
class_names = ['cat', 'dog', 'panda']


# Предобработка изображения
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return np.expand_dims(image, axis=0)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Чтение и декодирование изображения
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # Предобработка
    processed_image = preprocess_image(image)

    # Предсказание
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]

    return {
        "class": predicted_class,
        "probabilities": {class_names[i]: float(predictions[0][i]) for i in range(3)}
    }