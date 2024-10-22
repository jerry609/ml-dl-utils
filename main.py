# main.py
from config import create_default_config
from trainer import Trainer
from callbacks import EarlyStopping, ModelCheckpoint, HistoryLogger
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader

def example_usage():
    """训练框架使用示例"""
    config = create_default_config(
        model_name='resnet18_catdog',
        learning_rate=1e-3,
        batch_size=32,
        epochs=50,
        augmentation_strength='strong',
        optimizer={'type': 'adamw', 'weight_decay': 1e-4},
        scheduler={'type': 'onecycle', 'max_lr': 1e-2}
    )

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)

    train_loader = DataLoader(...)  # 您的训练数据加载器
    val_loader = DataLoader(...)    # 您的验证数据加载器
    test_loader = DataLoader(...)   # 您的测试数据加载器

    callbacks = [
        EarlyStopping(patience=config.early_stopping_patience, min_lr=config.min_lr),
        ModelCheckpoint(checkpoint_dir=config.checkpoint_dir, max_checkpoints=config.max_checkpoints),
        HistoryLogger(checkpoint_dir=config.checkpoint_dir)
    ]

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        callbacks=callbacks
    )

    best_model = trainer.train()

    final_metrics = {
        'best_val_acc': trainer.best_val_acc,
        'final_test_acc': trainer.metrics['test_acc'][-1],
        'total_epochs': len(trainer.metrics['train_loss'])
    }

    metrics_path = config.checkpoint_dir / 'final_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)

    return best_model, final_metrics
