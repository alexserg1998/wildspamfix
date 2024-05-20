from fastapi import FastAPI, File, UploadFile, Form
from typing import List, Optional
import os
import torch
from pydantic import BaseModel
from predictor import SwinSpamPredictor


class FolderPathRequest(BaseModel):
    folder_path: str
    batch_size: int
    threshold: Optional[float] = 0.5


class ImageRequest(BaseModel):
    threshold: Optional[float] = 0.5


app = FastAPI()
predictor = SwinSpamPredictor()


@app.post("/swin_predict_image")
async def swin_predict(item: FolderPathRequest) -> List[dict]:
    folder_path = item.folder_path
    batch_size = item.batch_size
    threshold = item.threshold
    predictions = []

    if folder_path:
        spam_probabilities = predictor.process_folder(folder_path, batch_size)
        for path, prob in zip(os.listdir(folder_path), spam_probabilities):
            predict = 'spam' if prob >= threshold else 'not spam'
            predictions.append({'data': path, 'predict': predict, 'predict proba': prob})
    else:
        return [{"error": "Необходимо предоставить или папку с изображениями, или изображения в форме данных"}]
    return predictions


@app.post("/swin_predict_files")
async def swin_predict(files: List[UploadFile] = File(None), threshold: float = Form(0.5)) -> List[dict]:
    images = []

    for file in files:
        contents = await file.read()
        image = predictor.transform_image(contents)
        images.append(image)

    images = torch.stack(images)
    spam_probabilities = predictor.predict_spam(images)
    predictions = []

    for file, prob in zip(files, spam_probabilities):
        predict = 'spam' if prob >= threshold else 'not spam'
        predictions.append({'data': file.filename, 'predict': predict, 'predict proba': prob})

    return predictions
