import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Optional, Any


class SpamDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[Any] = None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
                            if fname.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image
