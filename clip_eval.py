import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import clip
import torch
import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

CLIP_DIMS = {"ViT-L/14":768,}

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




class CLIPReward(nn.Module):
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


class CLIPScore(nn.Module):
    def __init__(self, clip_name = "ViT-L/14", device = 'cpu'):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(clip_name, device = device)
        self.mlp = MLP(input_size=CLIP_DIMS[clip_name])

    
    def score(self, prompt, image):
        # support image_path:str or image:Image
        if isinstance(image, str):
            image_path = image
            pil_image = Image.open(image_path)
        elif isinstance(image, Image.Image):
            pil_image = image

        # print(prompt)
        # print(image)
        # text encode
        text = clip.tokenize(prompt, truncate=True).to(self.device) # [1, 77]
        text_features = F.normalize(self.model.encode_text(text)) # [1, 768]
        # text_features = self.model.encode_text(text) # [1, 768]


        # image encode
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device) # [1, 3, 224, 224]
        image_features = F.normalize(self.model.encode_image(image)) # [1, 768]
        # image_features = self.model.encode_image(image) # [1, 768]
        
        logit_scale = self.model.logit_scale.exp()
        # similarity = torch.sum(torch.mul(text_features, image_features), dim=1, keepdim=True)
        similarity = logit_scale * image_features @ text_features.T
        
        # print("图像和文本的相似性得分:", similarity.detach().cpu().numpy().item())

        return similarity.detach().cpu().numpy().item()
    
    def scores_list(self, prompts, images):
        clip_scores = []
        for prompt, image in zip(prompts, images):
            clip_scores.append(self.score(prompt, image))
        # for i in range(len(prompts)):
        #     clip_scores.append(self.score(prompts[i], images[i]))

        return clip_scores


def CLIPReward_load(weight_path, device):
    print('load checkpoint from %s'%weight_path)
    state_dict = torch.load(weight_path, map_location='cpu')

    model = CLIPReward(clip_name='ViT-L/14', device=device).to(device)
    msg = model.load_state_dict(state_dict, strict=False)
    print("checkpoint loaded")
    model.eval()

    return model

