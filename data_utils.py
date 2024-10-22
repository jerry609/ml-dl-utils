# data_utils.py
import torch
import numpy as np
from torchvision import transforms
from typing import Tuple


def get_augmentation_pipeline(strength: str = 'medium', img_size: int = 224) -> transforms.Compose:
    """根据强度返回数据增强流水线"""
    if strength == 'light':
        augmentation = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    elif strength == 'medium':
        augmentation = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
    elif strength == 'strong':
        augmentation = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError(f"Unknown augmentation strength: {strength}")
    return augmentation


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2, device: torch.device) -> Tuple:
    """执行 Mixup 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
