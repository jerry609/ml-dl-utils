# utils.py
import torch
import numpy as np
import random


def set_random_seeds(seed: int = 42):
    """设置随机种子确保实验可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """获取计算设备"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
