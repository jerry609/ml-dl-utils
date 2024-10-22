# callbacks.py
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any


class Callback:
    def on_epoch_end(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass


class EarlyStopping(Callback):
    def __init__(self, patience: int = 5, min_lr: float = 1e-6):
        self.patience = patience
        self.min_lr = min_lr
        self.counter = 0

    def on_epoch_end(self, trainer):
        if trainer.current_lr < self.min_lr:
            trainer.stop_training = True
        if not trainer.is_best:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.stop_training = True
        else:
            self.counter = 0


class ModelCheckpoint(Callback):
    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.saved_checkpoints = []

    def on_epoch_end(self, trainer):
        state = {
            'epoch': trainer.current_epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'best_val_acc': trainer.best_val_acc,
            'config': trainer.config.__dict__,
            'metrics': trainer.metrics
        }
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{trainer.current_epoch}.pth'
        torch.save(state, checkpoint_path)
        self.saved_checkpoints.append(checkpoint_path)

        if trainer.is_best:
            best_model_path = self.checkpoint_dir / 'model_best.pth'
            torch.save(state, best_model_path)

        if len(self.saved_checkpoints) > self.max_checkpoints:
            old_checkpoint = self.saved_checkpoints.pop(0)
            old_checkpoint.unlink(missing_ok=True)


class HistoryLogger(Callback):
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, trainer):
        self._save_training_history(trainer)
        self._plot_training_history(trainer)

    def _save_training_history(self, trainer):
        history_path = self.checkpoint_dir / 'training_history.json'
        metrics_json = {
            k: v if isinstance(v, list) else v.tolist() if hasattr(v, 'tolist') else v
            for k, v in trainer.metrics.items()
        }
        with open(history_path, 'w') as f:
            json.dump(metrics_json, f, indent=4)

    def _plot_training_history(self, trainer):
        plt.style.use('seaborn')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 损失曲线
        axes[0, 0].plot(trainer.metrics['train_loss'], label='Train')
        axes[0, 0].plot(trainer.metrics['val_loss'], label='Validation')
        axes[0, 0].plot(trainer.metrics['test_loss'], label='Test')
        axes[0, 0].set_title('Loss History')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 准确率曲线
        axes[0, 1].plot(trainer.metrics['train_acc'], label='Train')
        axes[0, 1].plot(trainer.metrics['val_acc'], label='Validation')
        axes[0, 1].plot(trainer.metrics['test_acc'], label='Test')
        axes[0, 1].set_title('Accuracy History')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 学习率曲线
        axes[1, 0].plot(trainer.metrics['learning_rate'])
        axes[1, 0].set_title('Learning Rate History')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)

        # 保留一个子图用于未来扩展
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
