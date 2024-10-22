# optimizers.py
import torch
from torch.optim import Optimizer
from typing import Dict


def create_optimizer(model_parameters, opt_config: Dict) -> Optimizer:
    opt_type = opt_config.get('type', 'adam').lower()
    lr = opt_config.get('lr', 1e-3)
    weight_decay = opt_config.get('weight_decay', 1e-4)

    optimizers = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
        'adamw': torch.optim.AdamW
    }

    if opt_type not in optimizers:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

    optimizer_class = optimizers[opt_type]

    if opt_type == 'sgd':
        return optimizer_class(
            model_parameters,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    return optimizer_class(
        model_parameters,
        lr=lr,
        weight_decay=weight_decay
    )
