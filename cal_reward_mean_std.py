import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
import sys
import argparse
import math
from tqdm import tqdm
from datasets import S2C_Dataset_test
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from utils import *
from clip_eval import CLIPReward_load
from rm_eval import ImageReward_load



def init_seeds(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
       cudnn.deterministic = True
       cudnn.benchmark = False
    else:  # faster, less reproducible
       cudnn.deterministic = False
       cudnn.benchmark = True



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
        default=0,
        type=str,
        help="GPU ID(s) to use for CUDA.",
    )
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    return args



def main(args):
    init_seeds(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}")
    # load data
    s2c_data = S2C_Dataset_test(args.prompts_json_path)
    s2c_test_loader = DataLoader(s2c_data, batch_size=args.batch_size)

    # load model
    if args.benchmark == 'CLIP':
            reward_model = CLIPReward_load(args.rm_path, device=device)
    elif args.benchmark == 'ImageReward':
        reward_model = ImageReward_load(args.rm_path, device=device)
    
    reward_model.eval()
    # start Testing
    num_all = len(s2c_test_loader)


    # 记录所有的 reward score
    scores_accumulator = []  
   
    
    # 计算均值
    with torch.no_grad():
        for i, data in tqdm(enumerate(s2c_test_loader), total = num_all):
            texts, sim_imgs, com_imgs = data        
            sim_score = reward_model.scores_list(texts, sim_imgs)
            com_score = reward_model.scores_list(texts, com_imgs)
            
            scores_accumulator.extend(sim_score)
            scores_accumulator.extend(com_score)
            # print(scores_accumulator)

            # print(sim_score)
            print(f"Step: {i} | len: {len(scores_accumulator)} | mean:{sum(scores_accumulator) / len(scores_accumulator)}")
        
        # break
    # 均值
    mean = sum(scores_accumulator) / len(scores_accumulator)
    # 计算平方差和
    squared_diff_sum = sum((x - mean) ** 2 for x in scores_accumulator)
    # 计算方差（样本方差，使用 n - 1）
    variance = squared_diff_sum / (len(scores_accumulator) - 1)

    variance_2 = squared_diff_sum / (len(scores_accumulator))
    std_2 = math.sqrt(variance_2)
    print("var2",variance_2)
    print("std_2",std_2)
        
    # 标准差
    std = math.sqrt(variance)

    print("Dataset Mean:", mean)
    print("Dataset Variance:", variance)
    print("Dataset Standard Deviation (Std):", std)



if __name__ == "__main__":
    args = parse_args()
    main(args)

# test rm

"""
# 测试 ImageReward
python cal_reward_mean_std.py

# 测试 CLIP
python cal_reward_mean_std.py --benchmark "CLIP" --rm_path /data/liutao/checkpoints/ClipReward/bs32_lr=5e-4.pt

1. 第一次跑
CLIP reward(无偏估计)
* Dataset Mean: -0.03562733882188501
* Dataset Variance: 5.205378545346869e-07
* Dataset Standard Deviation (Std): 0.000721483093727557

# model2(无偏估计)
Dataset Mean: 0.02702386127237202
Dataset Variance: 0.0004931982316955882
Dataset Standard Deviation (Std): 0.022208066815812406

2. 第二次跑
var2 5.205038385319706e-07
std_2 0.0007214595196765863
Dataset Mean: -0.035627373790714525
Dataset Variance: 5.205264779925652e-07
Dataset Standard Deviation (Std): 0.0007214752095481626

var2 0.0004931767808330405
std_2 0.022207583858516452
Dataset Mean: 0.02702386127237202
Dataset Variance: 0.0004931982316955882
Dataset Standard Deviation (Std): 0.022208066815812406


var2 7.0218404998303185
std_2 2.649875563084108
Dataset Mean: 7.055357851639096
Dataset Variance: 7.022145916754325
Dataset Standard Deviation (Std): 2.6499331909982797

"""
