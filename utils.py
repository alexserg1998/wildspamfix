import torch
from PIL import Image
import io
from torchvision import transforms
from typing import List


def softmax(x: torch.Tensor) -> torch.Tensor:
    exp_x = torch.exp(x)
    sum_exp_x = torch.sum(exp_x, dim=1, keepdim=True)
    softmax_output = exp_x / sum_exp_x
    return softmax_output


def image_to_byte_array(image: Image.Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


mean: List[float] = [0.5197998285293579, 0.4744606018066406, 0.4502107799053192]
std: List[float] = [0.21492072939872742, 0.21317192912101746, 0.21331869065761566]
width: int = 224
height: int = 224

test_transform = transforms.Compose([
    transforms.Resize((width, height), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
