import os
import json
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

class S2C_Dataset_test(Dataset):
    def __init__(self, data_path):
        # load json
        with open(data_path, "r") as f:
            self.data = json.load(f)
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # item: img_worse, img_better, text_ids, text_mask
        item = self.data[index]

        text = item['simple']
        img_worse_path = item['simple_img']
        img_better_path = item['complex_img']
        
        return text, img_worse_path, img_better_path








        
        
