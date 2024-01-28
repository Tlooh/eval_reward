"""
methods: Welford算法

ref: 
* https://blog.csdn.net/midnight_DJ/article/details/119450244

"""
import os
import json
import argparse

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from PIL import Image

import pdb


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def image_transform():
    return Compose([
        _convert_image_to_rgb,
        ToTensor(),
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class S2C_Dataset_test(Dataset):
    def __init__(self, data_path):
        # load json
        with open(data_path, "r") as f:
            self.data = json.load(f)
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        item = self.data[index]
        simple_imgs = item['simple_img']
        complex_imgs = item['complex_img']
        # print(simple_imgs)
        images = [simple_imgs, complex_imgs]
        return images



def main(args):

    test_dataset = S2C_Dataset_test(args.json_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 初始化均值和方差的累积器
    mean_accumulator = torch.zeros(3)  # 3通道图像的均值
    var_accumulator = torch.zeros(3)  # 3通道图像的方差
    num_samples = len(test_loader)
    global_step = 0

    preprocess = image_transform()
    for step, imgs in tqdm(enumerate(test_loader), total = len(test_loader), desc = "calulate"):
        
        global_step += 1
        # print(global_step)
        imgs_list = []
        for subimgs in imgs:
            imgs_list.extend(subimgs)

        num_samples = len(imgs_list) # length: bsz * 2
        
        pil_imgs = [Image.open(img) for img in imgs_list]
        imgs_tensor_list = [preprocess(img) for img in pil_imgs]
        imgs_tensor = torch.stack(imgs_tensor_list) # [64, 3, 512, 512]

        
        # 计算当前 batch 的均值和方差
        batch_mean = torch.mean(imgs_tensor, dim=(0, 2, 3))
        batch_var = torch.var(imgs_tensor, dim=(0, 2, 3))

        # print(batch_mean)
        # print(batch_var)

        # 更新累积器
        mean_accumulator += batch_mean
        var_accumulator += batch_var

    # 计算最终均值和方差
    dataset_mean = mean_accumulator / len(test_loader)
    dataset_variance = var_accumulator / len(test_loader)
    dataset_std = torch.sqrt(dataset_variance)

    print("Dataset Mean:", dataset_mean)
    print("Dataset Variance:", dataset_variance)
    print("Dataset Standard Deviation (Std):", dataset_std)

    return 




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training CLIP Reward")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for DataLoader")
    parser.add_argument("--json_path", type=str, default="/data/liutao/mac8/json/test_11496.json", help="Directory to load images")

    args = parser.parse_args()
    main(args)




# Dataset Mean: tensor([0.4501, 0.3660, 0.3203])
# Dataset Variance: tensor([0.0761, 0.0608, 0.0559])
# Dataset Standard Deviation (Std): tensor([0.2759, 0.2465, 0.2365])

