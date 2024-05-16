from fastapi import FastAPI, File, UploadFile
from typing import List, Optional
import torch
from PIL import Image
from io import BytesIO
import os
from utils import softmax, test_transform
from predict import SwinSpamModel, SpamDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from pydantic import BaseModel


class FolderPathRequest(BaseModel):
    folder_path: str = None
    batch_size: int = 16

app = FastAPI()
swin_model = SwinSpamModel()
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def predict_spam(images: torch.Tensor) -> List[float]:
    swin_model.to(device).eval()
    with torch.no_grad():
        inputs = images.to(device)
        output = swin_model(inputs)
    return softmax(output.detach().to('cpu'))[:, 1].item()


def process_folder(folder_path: str, batch_size: int) -> List[float]:
    dataset = SpamDataset(root_dir=folder_path, transform=test_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    swin_model.to(device).eval()
    predicted_classes = []
    predicted_prob = []
    with tqdm(total=len(dataloader)) as pbar:
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                inputs = batch
                inputs = inputs.to(device)

                output = swin_model(inputs)

                _, predicted = torch.max(output.detach(), 1)
                predicted_classes.append(predicted)

                predicted_prob.append(softmax(output.detach()).tolist())
                pbar.update(1)

    predicted_prob = [item[1] for sublist in predicted_prob for item in sublist]
    return predicted_prob


@app.post("/swin_predict_image")
async def swin_predict(item: Optional[FolderPathRequest] = None):
    folder_path = item.folder_path
    batch_size = item.batch_size
    predictions = []
    if folder_path:
        spam_probabilities = process_folder(folder_path, batch_size)
        for path, prob in zip(os.listdir(folder_path), spam_probabilities):
            predictions.append({'data': path, 'predict': prob})
    else:
        return {"error": "Необходимо предоставить или папку с изображениями, или изображения в форме данных"}
    return predictions


@app.post("/swin_predict_files")
async def swin_predict(files: List[UploadFile] = File(None)):
    images = []
    for file in files:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image = test_transform(image)
        images.append(image)
    images = torch.stack(images)
    spam_probabilities = predict_spam(images)
    predictions = [{'data': files[0].filename, 'predict': spam_probabilities}]
    return predictions
