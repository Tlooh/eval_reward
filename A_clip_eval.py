import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from PIL import Image
import clip
import torch
import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

CLIP_DIMS = {"ViT-L/14":768,}


"""----------------- CLIP-V2 -------------------"""

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


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # initial MLP param
        # for name, param in self.layers.named_parameters():
        #     if 'weight' in name:
        #         nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
        #     if 'bias' in name:
        #         nn.init.constant_(param, val=0)
        
    def forward(self, input):
        return self.layers(input)




class CLIPReward_v1(nn.Module):
    def __init__(self, clip_name = "ViT-L/14", device = 'cpu'):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(clip_name, device = device)
        self.mlp = MLP(input_size=CLIP_DIMS[clip_name])
        # self.linear = nn.Linear(CLIP_DIMS[clip_name], 1)

        self.mean =  -0.03562733882188501
        self.std =  0.000721483093727557


    def score(self, prompt, image):
        if  isinstance(image, str):
            # preprocess image
            image = self.preprocess(Image.open(image)).unsqueeze(0).to(self.device)
        # normalize and encode
        image_emb = F.normalize(self.clip_model.encode_image(image)).float()
        # mlp
        reward = self.mlp(image_emb)
        # print("reward: ", reward)
        # reward = (reward - self.mean) / self.std
        # print("reward normalize: ", reward)

        return reward.detach().cpu().numpy().item()

    def scores_list(self, prompts, images):
        rm_scores = []
        for prompt, image in zip(prompts, images):
            rm_scores.append(self.score(prompt, image))
        return rm_scores
    
    def score_pth(self, batch_data):
        text_ids, text_mask, img_better, img_worse = batch_data['text_ids'], batch_data['text_mask'], batch_data['img_better'], batch_data['img_worse']

        img_better = img_better.to(self.device) # [batch_size, C, H, W]
        img_worse = img_worse.to(self.device) # [batch_size, C, H, W]

        emb_better = F.normalize(self.clip_model.encode_image(img_better)).float() #[batch_size, 768]
        emb_worse = F.normalize(self.clip_model.encode_image(img_worse)).float() #[batch_size, 768]

        # mlp projector
        reward_better = self.mlp(emb_better)
        reward_worse = self.mlp(emb_worse)
        rewards = torch.concat((reward_better, reward_worse), dim=1)

        return rewards



def CLIPReward_v1_load(weight_path, device):
    print('load checkpoint from %s'%weight_path)
    state_dict = torch.load(weight_path, map_location='cpu')

    model = CLIPReward_v1(clip_name='ViT-L/14', device=device).to(device)
    msg = model.load_state_dict(state_dict, strict=False)
    print("checkpoint loaded")
    model.eval()

    return model


"""----------------- CLIP-V2 -------------------"""

class CLIPReward_v2(nn.Module):
    def __init__(self, clip_name = "ViT-L/14", device = 'cpu'):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(clip_name, device = device)
        

    def forward(self, batch_data):
        # encode data, return shape [bsz]
        better_score, worse_score = self.encode_pair(batch_data)

        # 拼接成 [bsz, 2]
        rewards = torch.cat([better_score.unsqueeze(1),worse_score.unsqueeze(1)], dim=1)

        return rewards

    
    def encode_pair(self, batch_data):
        # images：pil -> tensor
        img_better, img_worse , text_tokens= batch_data['img_better'], batch_data['img_worse'], batch_data['clip_text']

        # move to device
        img_better = img_better.to(self.device) # [batch_size, C, H, W]
        img_worse = img_worse.to(self.device) # [batch_size, C, H, W]
        #[batch_size, 1, 77] ->  [batch_size, 77]
        text_tokens = text_tokens.squeeze(dim = 1).to(self.device) 
        
        # encode images, texts
        emb_better = self.clip_model.encode_image(img_better)
        emb_worse = self.clip_model.encode_image(img_worse)
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



def CLIPReward_v2_load(weight_path, device):
    print('load checkpoint from %s'%weight_path)
    state_dict = torch.load(weight_path, map_location='cpu')
    print(state_dict.keys())
    model = CLIPReward_v2(clip_name=weight_path, device=device).to(device)
    print("checkpoint loaded")
    model.eval()

    return model



