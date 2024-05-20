from fastapi import FastAPI, File, UploadFile
from typing import List, Optional
import os
import torch
from pydantic import BaseModel
from predictor import SwinSpamPredictor


class FolderPathRequest(BaseModel):
    folder_path: Optional[str] = None
    batch_size: int = 16


app = FastAPI()
predictor = SwinSpamPredictor()


@app.post("/swin_predict_image")
async def swin_predict(item: Optional[FolderPathRequest] = None) -> List[dict]:
    folder_path = item.folder_path
    batch_size = item.batch_size
    predictions = []
    if folder_path:
        spam_probabilities = predictor.process_folder(folder_path, batch_size)
        for path, prob in zip(os.listdir(folder_path), spam_probabilities):
            predictions.append({'data': path, 'predict': prob})
    else:
        return [{"error": "Необходимо предоставить или папку с изображениями, или изображения в форме данных"}]
    return predictions


@app.post("/swin_predict_files")
async def swin_predict(files: List[UploadFile] = File(None)) -> List[dict]:
    images = []
    for file in files:
        contents = await file.read()
        image = predictor.transform_image(contents)
        images.append(image)
    images = torch.stack(images)
    spam_probabilities = predictor.predict_spam(images)
    predictions = [{'data': file.filename, 'predict': prob} for file, prob in zip(files, spam_probabilities)]
    return predictions
