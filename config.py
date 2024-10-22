# config.py
import json
from pathlib import Path
from typing import Any, Dict


class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self.__dict__.update(config_dict)

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)

    def to_json(self, json_path: str):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)


def create_default_config(
        model_name: str,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        augmentation_strength: str = 'medium',
        **kwargs
) -> Config:
    config_dict = {
        'model_name': model_name,
        'checkpoint_dir': f'checkpoints/{model_name}',
        'seed': 42,
        'epochs': epochs,
        'batch_size': batch_size,
        'mixed_precision': True,
        'optimizer': {
            'type': 'adamw',
            'lr': learning_rate,
            'weight_decay': 1e-4
        },
        'scheduler': {
            'type': 'cosine',
            'T_max': epochs,
            'eta_min': 1e-6
        },
        'augmentation': {
            'strength': augmentation_strength,
            'img_size': 224,
            'use_mixup': True,
            'mixup_alpha': 0.2
        },
        'early_stopping_patience': 10,
        'min_lr': 1e-6,
        'save_frequency': 5,
        'max_checkpoints': 3
    }
    config_dict.update(kwargs)
    return Config(config_dict)
