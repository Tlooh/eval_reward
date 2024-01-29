import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

from PIL import Image
from tqdm import tqdm
import clip
import json
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from utils import *
from CLIP.model import clip_pretrain


# ViT-L/14 
kwargs = {'embed_dim': 768, 'image_resolution': 224, 'vision_layers': 24, 'vision_width': 1024, 'vision_patch_size': 14, 'context_length': 77, 'vocab_size': 49408, 'transformer_width': 768, 'transformer_heads': 12, 'transformer_layers': 12}


"""
clip_name: 
* 若为 ViT-L/14 ,则加载预训练模型
* 若为权重 /path/to/bs32_lr=5e-06_sig1.pt, 则是加载自己训练的权重

"""

class CLIPReward(nn.Module):
    def __init__(self, clip_name = None, device = 'cpu'):
        super().__init__()
        self.device = device
        self.clip_model = clip_pretrain(pretrained=clip_name, **kwargs)

        # TODO: 计算 mean 、 std
    
    def forward(self, batch_data):
        # encode data, return shape [bsz]
        better_score, worse_score = self.encode_pair(batch_data)

        # 拼接成 [bsz, 2]
        rewards = torch.cat([better_score.unsqueeze(1),worse_score.unsqueeze(1)], dim=1)

        return rewards

    
    def encode_pair(self, batch_data):
        # images：pil -> tensor
        imgs_better, imgs_worse , text_tokens= batch_data['img_better'], batch_data['img_worse'], batch_data['clip_text']

        # move to device
        imgs_better = imgs_better.to(self.device) # [batch_size, C, H, W]
        imgs_worse = imgs_worse.to(self.device) # [batch_size, C, H, W]
        #[batch_size, 1, 77] ->  [batch_size, 77]
        text_tokens = text_tokens.squeeze(dim = 1).to(self.device) 
        
        # encode images, texts
        emb_better = self.clip_model.encode_image(imgs_better)
        emb_worse = self.clip_model.encode_image(imgs_worse)
        emb_text = self.clip_model.encode_text(text_tokens)

        # normalized features
        emb_better = emb_better / emb_better.norm(dim=1, keepdim=True)
        emb_worse = emb_worse / emb_worse.norm(dim=1, keepdim = True)
        emb_text = emb_text / emb_text.norm(dim=1, keepdim = True)

        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        better_score = logit_scale * emb_better @ emb_text.t() # [bsz, bsz]
        worse_score = logit_scale * emb_worse @ emb_text.t() # [bsz, bsz]
        
        # 取出对角线上的元素
        better_score = torch.diag(better_score)
        worse_score = torch.diag(worse_score)

        return better_score, worse_score


def CLIPReward_load(weight, device):
    model = CLIPReward(clip_name=weight, device=device).to(device)
    print("checkpoint loaded")
    model.eval()
    return model
    
