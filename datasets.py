import os
import torch
import json
from torch.utils.data import Dataset
from PIL import Image

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from torch.utils.data import Dataset
import clip

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class S2C_Dataset_test(Dataset):
    def __init__(self, data_path, npx = 224):
        # load json
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.preprocess = _transform(n_px=npx)

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # dict_item: img_worse, img_better, text_ids, text_mask
        dict_item = self.handle_data(self.data[index])
        return dict_item
        
    def handle_data(self, item):
        dict_item = {}
        simple_image_path = item['simple_img']
        complex_image_path = item['complex_img']

        simple_image = Image.open(simple_image_path)
        complex_image = Image.open(complex_image_path)
        simple_image = self.preprocess(simple_image)
        complex_image = self.preprocess(complex_image)

        dict_item["img_better"] = complex_image
        dict_item["img_worse"] = simple_image 
        dict_item["text"] = item['simple']
        dict_item['clip_text'] = clip.tokenize(item["simple"], truncate=True)

        return dict_item


class ImageReward_Dataset_test(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]




        
        
