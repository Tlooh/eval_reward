

import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
import sys
import argparse
from tqdm import tqdm
from datasets import S2C_Dataset_test
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from utils import *
from clip_eval import CLIPReward_load
from rm_eval import ImageReward_load


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a test script.")
    parser.add_argument(
        "--prompts_json_path",
        default="/data/liutao/mac8/json/test_11496.json",
        type=str,
        help="Path to the prompts json list file, each item of which is a dict with keys `id` and `prompt`.",
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
        default="/data/liutao/checkpoints/blipreward/blip_reward_bs64_fix=0.5_lr=1e-05cosine/best_lr=1e-05.pt",
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



def main(args):
    device = torch.device(f"cuda:{args.gpu_id}")
    # load data
    s2c_data = S2C_Dataset_test(args.prompts_json_path)
    s2c_test_loader = DataLoader(s2c_data, batch_size=args.batch_size)

    # load model
    if args.benchmark == 'CLIP':
            reward_model = CLIPReward_load(args.rm_path, device=device)
    elif args.benchmark == 'ImageReward':
        reward_model = ImageReward_load(args.rm_path, device=device)
    
    # start Testing
    num_all = len(s2c_test_loader)
    num_acc = 0

    for i, data in tqdm(enumerate(s2c_test_loader), total = num_all):
        texts, sim_imgs, com_imgs = data        
        sim_score = reward_model.scores_list(texts, sim_imgs)
        com_score = reward_model.scores_list(texts, com_imgs)
        
        count_acc = sum(x < y for x, y in zip(sim_score, com_score))
        num_acc += count_acc
        # print(sim_imgs)
        # print(com_imgs)
        # print(sim_score)
        # print(com_score)
        iter_samples = (i + 1) * args.batch_size
        avg_acc = count_acc / args.batch_size
        total_acc = num_acc / iter_samples
        print(f'iter {i} | acc: {avg_acc} | all_acc: {total_acc}')

        break
        
    
    

if __name__ == "__main__":
    args = parse_args()
    main(args)

# test rm

"""
# 测试 ImageReward
python test_model.py

# 测试 CLIP
python test_model.py --benchmark "CLIP" --rm_path /data/liutao/checkpoints/ClipReward/bs32_lr=5e-4.pt

python main.py --benchmark "CLIP" --rm_path /data/liutao/checkpoints/ClipReward/bs32_lr=5e-4.pt
"""
