# utils.py 使用文档

`utils.py` 是一个辅助函数库，包含机器学习训练中常见的工具，如设置随机种子、数据增强、保存和加载检查点、优化器创建、学习率调度器设置、训练指标记录与绘制等。

## 目录
- [环境依赖](#环境依赖)
- [函数说明](#函数说明)
  - [set_random_seeds(seed=42)](#set_random_seeds)
  - [get_device()](#get_device)
  - [create_transforms(img_size=224, augmentation_strength='medium')](#create_transforms)
  - [mixup_data(x, y, device, alpha=0.2)](#mixup_data)
  - [save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth')](#save_checkpoint)
  - [load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None)](#load_checkpoint)
  - [create_optimizer(model, opt_type='adam', lr=1e-3, weight_decay=1e-4)](#create_optimizer)
  - [create_scheduler(optimizer, scheduler_type='plateau', **kwargs)](#create_scheduler)
  - [plot_metrics(metrics, save_path=None)](#plot_metrics)
  - [save_training_history(metrics, filepath)](#save_training_history)
  - [load_training_history(filepath)](#load_training_history)
  - [AverageMeter](#AverageMeter)

## 环境依赖

在使用该模块之前，请确保安装了以下库：

```bash
pip install torch torchvision numpy matplotlib
```

## 函数说明

### set_random_seeds
```python
set_random_seeds(seed=42)
```
设置所有随机种子，确保实验可重复性。

参数:
- `seed` (int): 随机种子，默认值为 42。

### get_device
```python
get_device()
```
获取当前可用的计算设备（GPU 或 CPU）。

返回:
- `device` (torch.device): 可用的设备对象。

### create_transforms
```python
create_transforms(img_size=224, augmentation_strength='medium')
```
创建数据增强与标准化的转换管道。

参数:
- `img_size` (int): 输入图像大小，默认为 224。
- `augmentation_strength` (str): 数据增强强度，取值范围为 `'light'`, `'medium'`, `'strong'`。默认值为 `'medium'`。

返回:
- `train_transform`: 训练数据的转换。
- `val_test_transform`: 验证和测试数据的转换。

### mixup_data
```python
mixup_data(x, y, device, alpha=0.2)
```
实现 Mixup 数据增强。

参数:
- `x` (torch.Tensor): 输入数据。
- `y` (torch.Tensor): 标签数据。
- `device` (torch.device): 计算设备。
- `alpha` (float): Mixup 参数，控制数据混合程度，默认值为 0.2。

返回:
- `mixed_x`, `y_a`, `y_b`, `lam`: 混合后的数据及对应的标签。

### save_checkpoint
```python
save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth')
```
保存模型检查点。

参数:
- `state` (dict): 模型状态字典。
- `is_best` (bool): 是否是当前最佳模型。
- `checkpoint_dir` (str): 检查点保存目录。
- `filename` (str): 文件名，默认为 `checkpoint.pth`。

### load_checkpoint
```python
load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None)
```
加载模型检查点。

参数:
- `model` (torch.nn.Module): 模型实例。
- `checkpoint_path` (str): 检查点路径。
- `optimizer` (torch.optim.Optimizer, 可选): 优化器实例。
- `scheduler` (torch.optim.lr_scheduler, 可选): 学习率调度器实例。

返回:
- `epoch` (int): 加载的 epoch 数。
- `best_val_acc` (float): 最佳验证准确率。

### create_optimizer
```python
create_optimizer(model, opt_type='adam', lr=1e-3, weight_decay=1e-4)
```
创建优化器。

参数:
- `model` (torch.nn.Module): 模型实例。
- `opt_type` (str): 优化器类型，支持 `'adam'`, `'sgd'`, `'adamw'`。
- `lr` (float): 学习率，默认值为 `1e-3`。
- `weight_decay` (float): 权重衰减系数，默认值为 `1e-4`。

返回:
- `optimizer` (torch.optim.Optimizer): 优化器实例。

### create_scheduler
```python
create_scheduler(optimizer, scheduler_type='plateau', **kwargs)
```
创建学习率调度器。

参数:
- `optimizer` (torch.optim.Optimizer): 优化器实例。
- `scheduler_type` (str): 调度器类型，支持 `'plateau'`, `'cosine'`, `'onecycle'`。
- `**kwargs`: 调度器的其他参数。

返回:
- `scheduler` (torch.optim.lr_scheduler): 调度器实例。

### plot_metrics
```python
plot_metrics(metrics, save_path=None)
```
绘制训练过程中的指标曲线（损失、准确率、学习率等）。

参数:
- `metrics` (dict): 训练历史数据字典，包含 `train_loss`, `val_loss`, `train_acc`, `val_acc` 等。
- `save_path` (str, 可选): 图表保存路径。如果未指定，将显示图表。

### save_training_history
```python
save_training_history(metrics, filepath)
```
保存训练历史到 JSON 文件。

参数:
- `metrics` (dict): 训练历史字典。
- `filepath` (str): 保存路径。

### load_training_history
```python
load_training_history(filepath)
```
从 JSON 文件加载训练历史。

参数:
- `filepath` (str): JSON 文件路径。

返回:
- `metrics` (dict): 训练历史字典。

### AverageMeter
`AverageMeter` 是一个帮助类，用于记录和计算平均值，如损失、准确率等。

```python
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有计数器"""
    
    def update(self, val, n=1):
        """更新计数器"""
```

## 贡献者
此工具库由开发者创建，欢迎贡献和改进。