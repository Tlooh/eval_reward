import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from blip.blip_pretrain import BLIP_Pretrain
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
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)
        
    def forward(self, input):
        return self.layers(input)


class ImageReward(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        self.blip = BLIP_Pretrain(image_size=224, vit='large')
        self.preprocess = _transform(224)
        self.mlp = MLP(768)
        
        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072


    def score_gard(self, prompt_ids, prompt_attention_mask, image):

        image_embeds = self.blip.visual_encoder(image)
        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(prompt_ids,
                                                    attention_mask = prompt_attention_mask,
                                                    encoder_hidden_states = image_embeds,
                                                    encoder_attention_mask = image_atts,
                                                    return_dict = True,
                                                 )
        
        txt_features = text_output.last_hidden_state[:,0,:] # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std
        
        return rewards


    def score(self, prompt, image):
        
        if (type(image).__name__=='list'):
            _, rewards = self.inference_rank(prompt, image)
            return rewards
            
        # text encode
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
        
        # image encode
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str):
            if os.path.isfile(image):
                pil_image = Image.open(image)
        else:
            raise TypeError(r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.')
            
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_embeds = self.blip.visual_encoder(image)
        
        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(text_input.input_ids,
                                                attention_mask = text_input.attention_mask,
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,
                                                return_dict = True,
                                            )
        
        txt_features = text_output.last_hidden_state[:,0,:].float() # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std
        
        return rewards.detach().cpu().numpy().item()

    def scores_list(self, prompts, images):
        rm_scores = []
        for prompt, image in zip(prompts, images):
            rm_scores.append(self.score(prompt, image))
        
        return rm_scores


    def score_pth(self, batch_data):
        text_ids, text_mask, img_better, img_worse = batch_data['text_ids'], batch_data['text_mask'], batch_data['img_better'], batch_data['img_worse']
        text_ids = text_ids.view(text_ids.shape[0], -1).to(self.device) # [batch_size, seq_len]
        text_mask = text_mask.view(text_mask.shape[0], -1).to(self.device) # [batch_size, seq_len]
        img_better = img_better.to(self.device) # [batch_size, C, H, W]
        img_worse = img_worse.to(self.device) # [batch_size, C, H, W]
        
        # encode better emb
        image_embeds_better = self.blip.visual_encoder(img_better)
        image_atts_better = torch.ones(image_embeds_better.size()[:-1], dtype=torch.long).to(self.device)
        emb_better = self.blip.text_encoder(text_ids,
                                            attention_mask = text_mask,
                                            encoder_hidden_states = image_embeds_better,
                                            encoder_attention_mask = image_atts_better,
                                            return_dict = True,
                                           ).last_hidden_state # [batch_size, seq_len, feature_dim]
        emb_better = emb_better[:, 0, :].float()
        
        # encode worse emb
        image_embeds_worse = self.blip.visual_encoder(img_worse)
        image_atts_worse = torch.ones(image_embeds_worse.size()[:-1], dtype=torch.long).to(self.device)
        emb_worse = self.blip.text_encoder(text_ids,
                                            attention_mask = text_mask,
                                            encoder_hidden_states = image_embeds_worse,
                                            encoder_attention_mask = image_atts_worse,
                                            return_dict = True,
                                           ).last_hidden_state
        emb_worse = emb_worse[:, 0, :].float()
        
        # mlp projector
        reward_better = self.mlp(emb_better)
        reward_worse = self.mlp(emb_worse)
        rewards = torch.concat((reward_better, reward_worse), dim=1)

        return rewards

    def inference_rank(self, prompt, generations_list):
        
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
        
        txt_set = []
        for generation in generations_list:
            # image encode
            if isinstance(generation, Image.Image):
                pil_image = generation
            elif isinstance(generation, str):
                if os.path.isfile(generation):
                    pil_image = Image.open(generation)
            else:
                raise TypeError(r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.')
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_embeds = self.blip.visual_encoder(image)
            
            # text encode cross attention with image
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
            text_output = self.blip.text_encoder(text_input.input_ids,
                                                    attention_mask = text_input.attention_mask,
                                                    encoder_hidden_states = image_embeds,
                                                    encoder_attention_mask = image_atts,
                                                    return_dict = True,
                                                )
            txt_set.append(text_output.last_hidden_state[:,0,:])
            
        txt_features = torch.cat(txt_set, 0).float() # [image_num, feature_dim]
        rewards = self.mlp(txt_features) # [image_num, 1]
        rewards = (rewards - self.mean) / self.std
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1
        
        return indices.detach().cpu().numpy().tolist(), rewards.detach().cpu().numpy().tolist()


def ImageReward_load(weight_path, device):
    print('load checkpoint from %s'%weight_path)
    state_dict = torch.load(weight_path, map_location='cpu')

    model = ImageReward(device=device).to(device)
    msg = model.load_state_dict(state_dict, strict=False)
    print("checkpoint loaded")
    model.eval()

    return model


