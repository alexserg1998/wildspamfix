import torch
import timm
import torch.nn as nn
from typing import Any


class SwinSpamModel(nn.Module):
    def __init__(self, num_classes: int = 2, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_swin_smooth = timm.create_model('swin_base_patch4_window7_224.ms_in22k', pretrained=False)
        if hasattr(self.model_swin_smooth, 'head'):
            in_features = self.model_swin_smooth.head.fc.in_features
            self.model_swin_smooth.head.fc = torch.nn.Linear(in_features, num_classes)
        else:
            print("The model does not have a 'head' attribute. Please check the model architecture.")

        checkpoint = torch.load("./app/swin_base_smoothing_sum_cross_entropy_full.pt", map_location='cpu')
        self.model_swin_smooth.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model_swin_smooth(image)
