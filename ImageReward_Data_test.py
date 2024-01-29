"""
在 ImageReward 数据集上看拟合效果

测试模型：
* ImageReward
* CLIP
* 自己训练的 reward

改编自 main.py
"""

import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
import sys
import argparse
from tqdm import tqdm
from datasets import ImageReward_Dataset_test
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import *

from CLIPReward import CLIPReward_load
from ImageReward import ImageReward_load

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
        help="ImageReward, Aesthetic, BLIP or CLIP, CLIP_v1, CLIP_v2, splitted with comma(,) if there are multiple benchmarks.",
    )
    parser.add_argument(
        "--result_dir",
        default="./benchmark_ImageReward",
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
        default=3,
        type=str,
        help="GPU ID(s) to use for CUDA.",
    )
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    args.filename = os.path.basename(__file__)

    return args


def print_basic_info(args):
    # Log argument values
    log = args.log
    log.info(f'---------------- Testing: {args.benchmark} ----------------')
    log.info(f'reward model: {args.rm_path}')
    log.info(f'result_dir: {args.result_dir}')
    log.info(f'run file: {args.filename}')
    log.info(f'Do eval on ImageReward Dataset ……')


def cal_acc(rewards):
    # rewards: [bsz, 2]
    rewards_diff = rewards[:, 0] - rewards[:, 1]
    num_acc = torch.sum((rewards_diff > 0).clone().detach())

    return num_acc


def main(args):
    device = torch.device(f"cuda:{args.gpu_id}")
    # load data
    test_dataset = ImageReward_Dataset_test(args.data_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # load model
    if args.benchmark == 'CLIP':
        reward_model = CLIPReward_load(weight = 'ViT-L/14', device=device)
    elif args.benchmark == 'CLIP_v2':
        reward_model = CLIPReward_load(args.rm_path, device=device)
    elif args.benchmark == 'ImageReward':
        reward_model = ImageReward_load(args.rm_path, device=device)
    
    reward_model.eval()
    # start Testing
    num_all = len(test_dataset)
    num_acc = 0

    with torch.no_grad():
        for i, batch_data in tqdm(enumerate(test_loader), total = len(test_loader)):
            if args.benchmark == 'ImageReward':
                rewards = reward_model.score_pth(batch_data) #[bsz, 2]
            else:
                rewards = reward_model(batch_data) #[bsz, 2]
            
            count_acc = cal_acc(rewards)
            # print(rewards)
            # print(count_acc)
            # return 
            num_acc += count_acc
            iter_samples = (i + 1) * args.batch_size

            avg_acc = count_acc / args.batch_size
            total_avg_acc = num_acc / iter_samples

            args.log.info(f'iter {i} | acc: {avg_acc} | all_acc: {total_avg_acc}')
        
        
    args.log.info(f'---------------- Result ----------------')
    args.log.info(f"样本总数: {num_all}")
    args.log.info(f"一致性样本数: {num_acc}")
    args.log.info(f"一致率: {num_acc / num_all}")


if __name__ == "__main__":
    args = parse_args()

    args.result_dir = args.result_dir + '/' + args.benchmark + '/' 
    if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        
    # add log
    ckpt_name = args.rm_path.split("/")[-1].split(".")[0]
    if args.benchmark == 'CLIP':
        ckpt_name = 'ViT-L-14'
    log_path = os.path.join(args.result_dir, 'log_{}_{}.log'.format(str(args.benchmark), str(ckpt_name)))
    args.log = logger_config(log_path)

    print_basic_info(args)
    main(args)

"""
# 测试 ImageReward
python ImageReward_Data_test.py 

# 测试 CLIP
python ImageReward_Data_test.py --benchmark "CLIP" 

# 测试 Custome CLIP(自己训练的)
python ImageReward_Data_test.py --benchmark "CLIP_v2" --rm_path /data/liutao/checkpoints/ClipReward/bs32_lr=5e-06_sig.pt --gpu_id 3

"""