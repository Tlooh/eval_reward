import time
import numpy as np
import random
import torch.nn.functional as F
import torch
from PIL import Image
import pdb
import logging
from logging import handlers


def logger_config(log_path, level=logging.INFO,fmt='%(asctime)s | %(levelname)s: %(message)s'):
    logger = logging.getLogger(log_path)
    format_str = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M') #设置日志格式
    logger.setLevel(level = level) #设置日志级别
    console = logging.StreamHandler() #往屏幕上输出
    console.setFormatter(format_str) #设置屏幕上显示的格式
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setFormatter(format_str)#设置文件里写入的格式
    logger.addHandler(console) #把对象加到logger里
    logger.addHandler(handler)

    return logger