import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import softmax, test_transform
from spam_model import SwinSpamModel
from spam_dataset import SpamDataset
from PIL import Image
from io import BytesIO
from typing import List


class SwinSpamPredictor:
    def __init__(self) -> None:
        self.swin_model = SwinSpamModel()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.swin_model.to(self.device)

    def predict_spam(self, images: torch.Tensor) -> List[float]:
        self.swin_model.eval()
        with torch.no_grad():
            inputs = images.to(self.device)
            output = self.swin_model(inputs)
        return softmax(output.detach().to('cpu'))[:, 1].tolist()

    def process_folder(self, folder_path: str, batch_size: int) -> List[float]:
        dataset = SpamDataset(root_dir=folder_path, transform=test_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.swin_model.eval()
        predicted_prob = []
        with tqdm(total=len(dataloader)) as pbar:
            with torch.no_grad():
                for batch in dataloader:
                    inputs = batch.to(self.device)
                    output = self.swin_model(inputs)
                    predicted_prob.append(softmax(output.detach()).tolist())
                    pbar.update(1)
        predicted_prob = [item[1] for sublist in predicted_prob for item in sublist]
        return predicted_prob

    def transform_image(self, image_bytes: bytes) -> torch.Tensor:
        image = Image.open(BytesIO(image_bytes))
        image = test_transform(image)
        return image
