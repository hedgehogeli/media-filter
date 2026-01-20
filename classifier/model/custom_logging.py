import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
import copy
from datetime import datetime
import os
import json
import csv
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms.v2 as transforms
from PIL import Image
import pandas as pd

class Logger:
    """
    Combined logger that writes to both TensorBoard and CSV files
    """
    def __init__(self, log_dir="./logs", experiment_name=None):
        # Create log directory
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.tb_writer = SummaryWriter(self.log_dir)
        
        # Initialize CSV loggers
        self.train_csv_path = os.path.join(self.log_dir, "train_metrics.csv")
        self.val_csv_path = os.path.join(self.log_dir, "val_metrics.csv")
        self.system_csv_path = os.path.join(self.log_dir, "system_metrics.csv")
        
        # Initialize CSV headers
        self._init_csv_files()
        
        # Store config
        self.config_path = os.path.join(self.log_dir, "config.json")
        
        print(f"Logging to: {self.log_dir}")
    
    def _init_csv_files(self):
        """Initialize CSV files with headers"""
        # Training metrics
        with open(self.train_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['loss', 'cls_loss', 'cons_loss', 'mod_cls_loss', 'mod_cons_loss'])
            
        # Validation metrics
        with open(self.val_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'accuracy', 'loss', 'recall_bad', 'recall_neutral', 
                           'recall_good', 'precision_bad', 'precision_neutral', 
                           'precision_good', 'f1_bad', 'f1_neutral', 'f1_good'])
        
        # System metrics
        with open(self.system_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'images_per_second', 'data_load_ms', 'gpu_compute_ms', 
                           'queue_size', 'gpu_memory_mb'])
    
    def log_config(self, config):
        """Save configuration to JSON file"""
        config_dict = vars(config) if hasattr(config, '__dict__') else config
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def log_scalar(self, tag, value, step):
        """Log a scalar value"""
        self.tb_writer.add_scalar(tag, value, step)
    
    def log_metrics(self, metrics, step=None):
        """Log multiple metrics at once"""
        for key, value in metrics.items():
            if step is not None:
                self.tb_writer.add_scalar(key, value, step)
            else:
                # For metrics without step (like epoch-end validation)
                self.tb_writer.add_scalar(key, value)
    
    def log_train_step(self, step, epoch, metrics):
        """Log training step metrics to CSV"""
        with open(self.train_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                step, epoch,
                metrics.get('loss', 0),
                metrics.get('cls_loss', 0),
                metrics.get('cons_loss', 0),
                metrics.get('mod_cls_loss', 0),
                metrics.get('mod_cons_loss', 0)
            ])
    
    def log_validation(self, epoch, metrics):
        """Log validation metrics to CSV"""
        with open(self.val_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                metrics.get('val/accuracy', 0),
                metrics.get('val/loss', 0),
                metrics.get('val/recall_bad', 0),
                metrics.get('val/recall_neutral', 0),
                metrics.get('val/recall_good', 0),
                metrics.get('val/precision_bad', 0),
                metrics.get('val/precision_neutral', 0),
                metrics.get('val/precision_good', 0),
                metrics.get('val/f1_bad', 0),
                metrics.get('val/f1_neutral', 0),
                metrics.get('val/f1_good', 0)
            ])
    
    def log_system_metrics(self, step, metrics):
        """Log system performance metrics to CSV"""
        with open(self.system_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                metrics.get('images_per_second', 0),
                metrics.get('data_load_ms', 0),
                metrics.get('gpu_compute_ms', 0),
                metrics.get('queue_size', 0),
                metrics.get('gpu_memory_mb', 0)
            ])
    
    def log_image(self, tag, image, step):
        """Log an image (for confusion matrices, etc.)"""
        self.tb_writer.add_figure(tag, image, step)
    
    def log_confusion_matrix(self, cm, class_names, epoch):
        """Log confusion matrix as both image and raw data"""
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names, ax=ax)
        ax.set_title(f'Confusion Matrix - Epoch {epoch}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # Log to TensorBoard
        # self.tb_writer.add_figure('confusion_matrix', fig, epoch)
        
        # Save as image
        cm_path = os.path.join(self.log_dir, f'confusion_matrix_epoch_{epoch}.png')
        fig.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Save raw data as CSV
        cm_csv_path = os.path.join(self.log_dir, f'confusion_matrix_epoch_{epoch}.csv')
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_csv_path)
    
    def close(self):
        """Close the logger"""
        self.tb_writer.close()
    
    def get_log_dir(self):
        """Get the log directory path"""
        return self.log_dir


class PerformanceMonitor:
    """Monitor training performance and statistics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.loss_history = []
        self.accuracy_history = []
        self.lr_history = []
        self.best_accuracy = 0
        self.epochs_without_improvement = 0
        self.training_start_time = None
        self.images_processed = 0
        self.processing_times = []
    
    def update_images_processed(self, batch_size, processing_time):
        self.images_processed += batch_size
        self.processing_times.append(batch_size / processing_time)
    
    def get_avg_images_per_second(self):
        if self.processing_times:
            return np.mean(self.processing_times[-100:])
        return 0
    
def get_gpu_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def plot_training_curves(log_dir):
    """
    Plot training curves from CSV logs
    """
    # Read CSV files
    train_df = pd.read_csv(os.path.join(log_dir, "train_metrics.csv"))
    val_df = pd.read_csv(os.path.join(log_dir, "val_metrics.csv"))
    system_df = pd.read_csv(os.path.join(log_dir, "system_metrics.csv"))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves', fontsize=16)
    
    # Plot 1: Loss curves
    ax = axes[0, 0]
    ax.plot(train_df['step'], train_df['loss'], label='Total Loss', alpha=0.7)
    ax.plot(train_df['step'], train_df['cls_loss'], label='Classification Loss', alpha=0.7)
    ax.plot(train_df['step'], train_df['consistency_loss'], label='Consistency Loss', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation metrics
    ax = axes[0, 1]
    ax.plot(val_df['epoch'], val_df['accuracy'], 'o-', label='Accuracy', markersize=8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Per-class F1 scores
    ax = axes[1, 0]
    ax.plot(val_df['epoch'], val_df['f1_bad'], 'o-', label='Bad', markersize=6)
    ax.plot(val_df['epoch'], val_df['f1_neutral'], 'o-', label='Neutral', markersize=6)
    ax.plot(val_df['epoch'], val_df['f1_good'], 'o-', label='Good', markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: System performance
    ax = axes[1, 1]
    ax.plot(system_df['step'], system_df['images_per_second'], alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Images/Second')
    ax.set_title('Training Throughput')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.show()