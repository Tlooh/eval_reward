"""
自己训练的 reward, 在 ImageReward 数据集上看拟合效果

"""
import os
import json
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils import *
from clip_eval import CLIPReward_load
from rm_eval import ImageReward_load

import pdb



class ImageReward_Dataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a test script.")
    parser.add_argument(
        "--data_path",
        default="/data/liutao/mac8/pair_store_clip/test.pth",
        type=str,
        help="Path to the test pth",
    )
    parser.add_argument(
        "--benchmark",
        default="ImageReward",
        type=str,
        help="ImageReward, Aesthetic, BLIP or CLIP, splitted with comma(,) if there are multiple benchmarks.",
    )
    parser.add_argument(
        "--result_dir",
        default="./benchmark",
        type=str,
        help="Path to the metric results directory",
    )
    
    parser.add_argument(
        "--rm_path",
        default="/data/liutao/checkpoints/ImageReward/ImageReward.pt",
        type=str,
        help="Path to place downloaded reward model in.",
    )
    parser.add_argument(
        "--gpu_id",
        default=7,
        type=str,
        help="GPU ID(s) to use for CUDA.",
    )
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    return args


def loss_func(reward):
    
    target = torch.zeros(reward.shape[0], dtype=torch.long).to(reward.device)
    loss_list = F.cross_entropy(reward, target, reduction='none')
    loss = torch.mean(loss_list)
    
    reward_diff = reward[:, 0] - reward[:, 1]
    acc = torch.mean((reward_diff > 0).clone().detach().float())
    
    return loss, loss_list, acc


def main(args):
    device = torch.device(f"cuda:{args.gpu_id}")
    test_dataset = ImageReward_Dataset(args.data_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # load model
    if args.benchmark == 'CLIP':
        reward_model = CLIPReward_load(args.rm_path, device=device)
    elif args.benchmark == 'ImageReward':
        reward_model = ImageReward_load(args.rm_path, device=device)
    
    test_loss = []
    acc_list = []
    with torch.no_grad():
        for i, batch_data in tqdm(enumerate(test_loader), total = len(test_loader)):
            rewards = reward_model.score_pth(batch_data)
            # print(rewards)
            # break
            _, loss_list, acc = loss_func(rewards)
            test_loss.append(loss_list)
            acc_list.append(acc.item())

    test_loss = torch.cat(test_loss, 0)
    print('Test Loss %6.5f | Acc %6.4f' % (torch.mean(test_loss), sum(acc_list) / len(acc_list)))
        



        
    

if __name__ == "__main__":
    args = parse_args()

    args.result_dir = args.result_dir + '/' + args.benchmark + '/' 
    if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
    
    # add log
    ckpt_name = args.rm_path.split("/")[-1].split(".")[0]
    log_path = os.path.join(args.result_dir, 'log_{}_{}.log'.format(str(args.benchmark), str(ckpt_name)))
    args.log = logger_config(log_path)

    main(args)


"""
# 测试 ImageReward
python ImageReward_Data_test.py
* Test Loss 0.61825 | Acc 0.6516

# 测试 CLIP
python ImageReward_Data_test.py --benchmark "CLIP" --rm_path /data/liutao/checkpoints/ClipReward/bs32_lr=5e-4.pt
Test Loss 0.69253 | Acc 0.5172


"""