# trainer.py
import torch
import numpy as np
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, List
from utils import set_random_seeds, get_device
from optimizers import create_optimizer
from schedulers import create_scheduler
from data_utils import mixup_data
from callbacks import Callback
from collections import defaultdict


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


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader,
            config: Any,
            callbacks: List[Callback] = None
    ):
        self.config = config
        set_random_seeds(self.config.seed)
        self.device = get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = create_optimizer(self.model.parameters(), self.config.optimizer)
        self.scheduler = create_scheduler(
            self.optimizer, self.config.scheduler, len(self.train_loader), self.config.epochs
        )

        self.mixed_precision = self.config.mixed_precision
        self.scaler = GradScaler() if self.mixed_precision else None

        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.stop_training = False
        self.is_best = False

        self.metrics = defaultdict(list)
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.callbacks = callbacks or []

    def train(self):
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            print(f'\nEpoch {epoch + 1}/{self.config.epochs}')

            train_loss, train_acc = self._run_epoch(self.train_loader, mode='train')
            self.metrics['train_loss'].append(train_loss)
            self.metrics['train_acc'].append(train_acc)

            val_loss, val_acc = self._run_epoch(self.val_loader, mode='val')
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_acc'].append(val_acc)

            test_loss, test_acc = self._run_epoch(self.test_loader, mode='test')
            self.metrics['test_loss'].append(test_loss)
            self.metrics['test_acc'].append(test_acc)

            self.current_lr = self.optimizer.param_groups[0]['lr']
            self.metrics['learning_rate'].append(self.current_lr)

            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            print(f'Learning Rate: {self.current_lr:.6f}')

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            self.is_best = val_acc > self.best_val_acc
            if self.is_best:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict()

            for callback in self.callbacks:
                callback.on_epoch_end(self)

            if self.stop_training:
                print("Early stopping triggered!")
                break

        for callback in self.callbacks:
            callback.on_train_end(self)

        self.model.load_state_dict(self.best_model_state)
        return self.model

    def _run_epoch(self, loader, mode='train'):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        epoch_loss = AverageMeter()
        epoch_acc = AverageMeter()

        with torch.set_grad_enabled(mode == 'train'):
            for data, target in tqdm(loader, desc=f'{mode.capitalize()} Epoch {self.current_epoch + 1}'):
                data, target = data.to(self.device), target.to(self.device)
                batch_size = target.size(0)

                if mode == 'train' and self.config.augmentation.use_mixup:
                    data, targets_a, targets_b, lam = mixup_data(
                        data, target, self.config.augmentation.mixup_alpha, self.device
                    )

                if mode == 'train':
                    self.optimizer.zero_grad(set_to_none=True)

                if self.mixed_precision:
                    with autocast():
                        output = self.model(data)
                        if mode == 'train' and self.config.augmentation.use_mixup:
                            loss = lam * self.criterion(output, targets_a) + \
                                   (1 - lam) * self.criterion(output, targets_b)
                        else:
                            loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    if mode == 'train' and self.config.augmentation.use_mixup:
                        loss = lam * self.criterion(output, targets_a) + \
                               (1 - lam) * self.criterion(output, targets_b)
                    else:
                        loss = self.criterion(output, target)

                if mode == 'train':
                    if self.mixed_precision:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                epoch_loss.update(loss.item(), batch_size)
                epoch_acc.update(100. * correct / batch_size, batch_size)

        return epoch_loss.avg, epoch_acc.avg
