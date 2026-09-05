from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from PIL import Image
from PIL import UnidentifiedImageError
from ml.predict import predict
import io

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict_digit(image: UploadFile = File(...)):

    image_bytes = await image.read()
    img_bytes = len(image_bytes)

    if img_bytes < 1 or img_bytes > 5242880:
        return {"message": "Image size exceeds limit (max 5MB)"}

    try:
        unpacked_img = io.BytesIO(image_bytes)
        pil_img = Image.open(unpacked_img)

    except UnidentifiedImageError:
        return {"message": "Invalid Image"}

    width, height = pil_img.size
    if width > 1920 or height > 1080:
        return {"message": "Image size exceeds limit (max 1920x1080)"}

    return {"prediction": predict(pil_img)}