# schedulers.py
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Any, Dict
import torch


def create_scheduler(optimizer: Optimizer, scheduler_config: Dict, train_loader_len: int, epochs: int) -> _LRScheduler:
    scheduler_type = scheduler_config.get('type', 'plateau').lower()

    schedulers = {
        'plateau': lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.1),
            patience=scheduler_config.get('patience', 10),
            verbose=True
        ),
        'cosine': lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', 100),
            eta_min=scheduler_config.get('eta_min', 0)
        ),
        'onecycle': lambda: torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_config.get('max_lr', optimizer.param_groups[0]['lr'] * 10),
            epochs=epochs,
            steps_per_epoch=train_loader_len
        )
    }

    if scheduler_type not in schedulers:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return schedulers[scheduler_type]()
