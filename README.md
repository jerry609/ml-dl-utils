## 目录

- [简介](#简介)
- [模块概述](#模块概述)
  - [config.py](#configpy)
  - [data_utils.py](#data_utilspy)
  - [utils.py](#utilspy)
  - [optimizers.py](#optimizerspy)
  - [schedulers.py](#schedulerspy)
  - [callbacks.py](#callbackspy)
  - [trainer.py](#trainerpy)
  - [main.py](#mainpy)
- [使用指南](#使用指南)
  - [安装依赖](#安装依赖)
  - [准备数据](#准备数据)
  - [编写训练脚本](#编写训练脚本)
  - [运行训练](#运行训练)
- [自定义与扩展](#自定义与扩展)
  - [添加新的优化器](#添加新的优化器)
  - [添加新的学习率调度器](#添加新的学习率调度器)
  - [自定义回调函数](#自定义回调函数)
- [项目结构](#项目结构)
- [许可证](#许可证)

## 简介

本深度学习训练框架旨在提供一个模块化、可扩展的训练流程，适用于图像分类等任务。通过将配置、数据处理、模型训练、优化器、调度器、回调等功能模块化，降低了代码的耦合度，提高了可维护性和可扩展性。

## 模块概述

### config.py

- **功能**：提供配置管理类，用于读取、更新和保存训练配置。
- **主要内容**：
  - `Config` 类：用于加载和保存配置。
  - `create_default_config` 函数：创建默认的配置实例，可根据需要进行修改。

### data_utils.py

- **功能**：提供数据增强和数据处理的实用函数。
- **主要内容**：
  - `get_augmentation_pipeline` 函数：根据指定的强度返回数据增强流水线。
  - `mixup_data` 函数：实现 Mixup 数据增强方法。

### utils.py

- **功能**：提供常用的实用函数，如设置随机种子、获取计算设备等。
- **主要内容**：
  - `set_random_seeds` 函数：设置随机种子，确保实验的可重复性。
  - `get_device` 函数：获取当前可用的计算设备（CPU 或 GPU）。

### optimizers.py

- **功能**：封装优化器的创建逻辑。
- **主要内容**：
  - `create_optimizer` 函数：根据配置创建优化器实例。

### schedulers.py

- **功能**：封装学习率调度器的创建逻辑。
- **主要内容**：
  - `create_scheduler` 函数：根据配置创建学习率调度器实例。

### callbacks.py

- **功能**：实现训练过程中的回调机制，如早停、模型检查点、日志记录等。
- **主要内容**：
  - `Callback` 基类：所有回调类的父类，定义了基本的方法接口。
  - `EarlyStopping` 类：实现早停功能，当验证指标不再提升时停止训练。
  - `ModelCheckpoint` 类：在训练过程中保存模型的检查点。
  - `HistoryLogger` 类：记录训练历史并生成可视化图表。

### trainer.py

- **功能**：核心的训练管理器，负责模型的训练、验证和测试流程。
- **主要内容**：
  - `Trainer` 类：封装了训练循环，支持混合精度训练、数据增强、回调机制等。

### main.py

- **功能**：训练脚本示例，展示如何使用上述模块进行模型训练。
- **主要内容**：
  - `example_usage` 函数：演示了完整的训练流程，包括配置创建、模型定义、数据加载、训练启动等。

## 使用指南

### 安装依赖

确保您的环境安装了以下依赖：

- Python 3.7+
- PyTorch 1.7+
- torchvision
- matplotlib
- numpy
- tqdm

使用以下命令安装所需的 Python 包：

```bash
pip install torch torchvision matplotlib numpy tqdm
```

### 准备数据

您需要准备自己的数据集，并创建相应的 `DataLoader`。以下是一个简单的示例：

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义数据增强和预处理
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 加载数据集
train_dataset = datasets.ImageFolder(root='path_to_train_data', transform=train_transforms)
val_dataset = datasets.ImageFolder(root='path_to_val_data', transform=train_transforms)
test_dataset = datasets.ImageFolder(root='path_to_test_data', transform=train_transforms)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
```

### 编写训练脚本

在 `main.py` 中，您可以根据自己的需求修改训练脚本：

```python
from config import create_default_config
from trainer import Trainer
from callbacks import EarlyStopping, ModelCheckpoint, HistoryLogger
from torchvision import models
import torch.nn as nn

def main():
    # 创建配置
    config = create_default_config(
        model_name='your_model_name',
        learning_rate=1e-3,
        batch_size=32,
        epochs=50,
        augmentation_strength='strong',
        optimizer={'type': 'adamw', 'weight_decay': 1e-4},
        scheduler={'type': 'onecycle', 'max_lr': 1e-2}
    )

    # 定义模型
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 请替换 num_classes 为您的类别数

    # 创建数据加载器
    # 请确保已经定义了 train_loader, val_loader, test_loader

    # 定义回调函数
    callbacks = [
        EarlyStopping(patience=config.early_stopping_patience, min_lr=config.min_lr),
        ModelCheckpoint(checkpoint_dir=config.checkpoint_dir, max_checkpoints=config.max_checkpoints),
        HistoryLogger(checkpoint_dir=config.checkpoint_dir)
    ]

    # 创建 Trainer 并开始训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        callbacks=callbacks
    )

    best_model = trainer.train()

    # 保存最终结果
    final_metrics = {
        'best_val_acc': trainer.best_val_acc,
        'final_test_acc': trainer.metrics['test_acc'][-1],
        'total_epochs': len(trainer.metrics['train_loss'])
    }

    metrics_path = config.checkpoint_dir / 'final_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
```

### 运行训练

在终端中运行训练脚本：

```bash
python main.py
```

训练过程中的日志、模型检查点和训练历史图表将保存在 `checkpoints/your_model_name` 目录下。

## 自定义与扩展

### 添加新的优化器

在 `optimizers.py` 中的 `create_optimizer` 函数中，您可以添加新的优化器：

```python
optimizers = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
    'adamw': torch.optim.AdamW,
    'rmsprop': torch.optim.RMSprop,  # 新增优化器
}
```

然后在配置中指定：

```python
optimizer={'type': 'rmsprop', 'lr': 1e-3, 'weight_decay': 1e-4}
```

### 添加新的学习率调度器

在 `schedulers.py` 中的 `create_scheduler` 函数中，您可以添加新的调度器：

```python
schedulers = {
    'plateau': ...,
    'cosine': ...,
    'onecycle': ...,
    'step': lambda: torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_config.get('step_size', 30),
        gamma=scheduler_config.get('gamma', 0.1)
    ),
}
```

然后在配置中指定：

```python
scheduler={'type': 'step', 'step_size': 30, 'gamma': 0.1}
```

### 自定义回调函数

您可以创建自己的回调类，继承自 `callbacks.py` 中的 `Callback` 基类：

```python
class CustomCallback(Callback):
    def on_epoch_end(self, trainer):
        # 自定义逻辑
        pass
```

然后在创建 `Trainer` 时，将其添加到回调列表中：

```python
callbacks = [
    ...,
    CustomCallback(),
]
```

## 项目结构

```
├── config.py           # 配置管理
├── data_utils.py       # 数据处理与增强
├── utils.py            # 工具函数
├── optimizers.py       # 优化器创建
├── schedulers.py       # 学习率调度器创建
├── callbacks.py        # 回调机制
├── trainer.py          # 训练管理器
├── main.py             # 训练脚本
├── checkpoints/        # 保存模型和日志的目录
└── README.md           # 项目说明文档
```

## 许可证

本项目遵循 MIT 许可证，详细内容请参见 [LICENSE](LICENSE) 文件。

---

如有任何问题或建议，欢迎提交 issue 或 pull request。