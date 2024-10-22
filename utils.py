# utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from torchvision import transforms
import json
import os
from pathlib import Path


def set_random_seeds(seed=42):
    """设置所有随机种子，确保实验可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """获取计算设备"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def create_transforms(img_size=224, augmentation_strength='medium'):
    """
    创建数据转换管道

    Args:
        img_size: 图像大小
        augmentation_strength: 数据增强强度 ('light', 'medium', 'strong')

    Returns:
        train_transform, val_test_transform
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # 基础转换
    base_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ]

    # 根据增强强度选择数据增强策略
    augmentation_config = {
        'light': [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ],
        'medium': [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ],
        'strong': [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
            transforms.RandomPerspective(),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        ]
    }

    train_transform = transforms.Compose(
        augmentation_config[augmentation_strength] + base_transforms
    )
    val_test_transform = transforms.Compose(base_transforms)

    return train_transform, val_test_transform


def mixup_data(x, y, device, alpha=0.2):
    """
    执行Mixup数据增强

    Args:
        x: 输入数据
        y: 标签
        device: 计算设备
        alpha: mixup强度参数

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    """
    保存模型检查点

    Args:
        state: 要保存的状态字典
        is_best: 是否是最佳模型
        checkpoint_dir: 保存目录
        filename: 文件名
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    filepath = checkpoint_dir / filename
    torch.save(state, filepath)

    if is_best:
        best_filepath = checkpoint_dir / 'model_best.pth'
        torch.save(state, best_filepath)


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None):
    """
    加载模型检查点

    Args:
        model: 模型实例
        checkpoint_path: 检查点路径
        optimizer: 优化器实例（可选）
        scheduler: 学习率调度器实例（可选）

    Returns:
        加载的epoch和其他相关信息
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint.get('epoch', 0), checkpoint.get('best_val_acc', 0)


def create_optimizer(model, opt_type='adam', lr=1e-3, weight_decay=1e-4):
    """
    创建优化器

    Args:
        model: 模型实例
        opt_type: 优化器类型 ('adam', 'sgd', 'adamw')
        lr: 学习率
        weight_decay: 权重衰减

    Returns:
        优化器实例
    """
    optimizers = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
        'adamw': torch.optim.AdamW
    }

    if opt_type.lower() not in optimizers:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

    optimizer_class = optimizers[opt_type.lower()]

    if opt_type.lower() == 'sgd':
        return optimizer_class(model.parameters(), lr=lr,
                               momentum=0.9, weight_decay=weight_decay)
    else:
        return optimizer_class(model.parameters(), lr=lr,
                               weight_decay=weight_decay)


def create_scheduler(optimizer, scheduler_type='plateau', **kwargs):
    """
    创建学习率调度器

    Args:
        optimizer: 优化器实例
        scheduler_type: 调度器类型 ('plateau', 'cosine', 'onecycle')
        **kwargs: 调度器的其他参数

    Returns:
        调度器实例
    """
    schedulers = {
        'plateau': lambda: ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 10),
            verbose=True
        ),
        'cosine': lambda: CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 0)
        ),
        'onecycle': lambda: OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', optimizer.param_groups[0]['lr'] * 10),
            epochs=kwargs.get('epochs', 100),
            steps_per_epoch=kwargs.get('steps_per_epoch', 100)
        )
    }

    if scheduler_type.lower() not in schedulers:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return schedulers[scheduler_type.lower()]()


def plot_metrics(metrics, save_path=None):
    """
    绘制训练指标图表

    Args:
        metrics: 包含训练历史的字典
        save_path: 保存路径（可选）
    """
    plt.style.use('seaborn')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 损失曲线
    axes[0, 0].plot(metrics['train_loss'], label='Train')
    axes[0, 0].plot(metrics['val_loss'], label='Validation')
    if 'test_loss' in metrics:
        axes[0, 0].plot(metrics['test_loss'], label='Test')
    axes[0, 0].set_title('Loss History')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    # 准确率曲线
    axes[0, 1].plot(metrics['train_acc'], label='Train')
    axes[0, 1].plot(metrics['val_acc'], label='Validation')
    if 'test_acc' in metrics:
        axes[0, 1].plot(metrics['test_acc'], label='Test')
    axes[0, 1].set_title('Accuracy History')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()

    # 学习率曲线
    if 'learning_rate' in metrics:
        axes[1, 0].plot(metrics['learning_rate'])
        axes[1, 0].set_title('Learning Rate History')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')

    # 保留一个子图用于未来扩展
    axes[1, 1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_training_history(metrics, filepath):
    """
    保存训练历史到JSON文件

    Args:
        metrics: 训练指标字典
        filepath: 保存路径
    """
    # 将numpy数组转换为列表
    metrics_json = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in metrics.items()
    }

    with open(filepath, 'w') as f:
        json.dump(metrics_json, f, indent=4)


def load_training_history(filepath):
    """
    从JSON文件加载训练历史

    Args:
        filepath: JSON文件路径

    Returns:
        训练历史字典
    """
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    return metrics


class AverageMeter:
    """跟踪指标平均值的工具类"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0