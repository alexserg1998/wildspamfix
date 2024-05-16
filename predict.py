import torch
import timm
import os
from PIL import Image
import torch.nn as nn
from torch.utils.data import  Dataset


class SwinSpamModel(nn.Module):
    def __init__(self, num_classes=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_swin_smooth = timm.create_model('swin_base_patch4_window7_224.ms_in22k', pretrained=False)
        if hasattr(self.model_swin_smooth, 'head'):
            in_features = self.model_swin_smooth.head.fc.in_features
            self.model_swin_smooth.head.fc = torch.nn.Linear(in_features, num_classes)
        else:
            print("The model does not have a 'head' attribute. Please check the model architecture.")

        checkpoint = torch.load("swin_base_smoothing_sum_cross_entropy_full.pt", map_location='cpu')
        self.model_swin_smooth.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, image):
        return self.model_swin_smooth(image)


class SpamDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
                            if fname.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image