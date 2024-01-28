import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
import sys
import argparse
from tqdm import tqdm
from datasets import S2C_Dataset_test
import torch
from torch.utils.data import DataLoader
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
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    return args


def print_basic_info(args):
    # Log argument values
    log = args.log
    log.info(f'---------------- Testing: {args.benchmark} ----------------')
    log.info(f'reward model: {args.rm_path}')
    log.info(f'result_dir: {args.result_dir}')



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
    num_all = len(s2c_data)
    num_acc = 0

    for i, data in tqdm(enumerate(s2c_test_loader), total = len(s2c_test_loader)):
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
    log_path = os.path.join(args.result_dir, 'log_{}_{}.log'.format(str(args.benchmark), str(ckpt_name)))
    args.log = logger_config(log_path)

    print_basic_info(args)
    main(args)


"""
# 测试 ImageReward
python main.py

# 测试 CLIP
python main.py --benchmark "CLIP" --rm_path /data/liutao/checkpoints/ClipReward/bs32_lr=5e-4.pt


"""