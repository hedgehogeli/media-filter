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


from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from typing import Tuple, List, Optional
import warnings
from tqdm import tqdm
import torchvision.transforms.v2 as transforms

# Windows-specific multiprocessing setup
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

to_tensor = transforms.Compose([
            transforms.ToImage(), 
            transforms.ToDtype(torch.float32, scale=True),
        ])

class ImageDataset(Dataset):
    """
    Basic dataset class for loading images with transforms
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.transform = transform
        
        # Pre-compute file paths for faster access
        self.file_paths = [os.path.join(self.root_dir, self.annotations.iloc[i, 0]) 
                          for i in range(len(self.annotations))]
        self.labels = [self.annotations.iloc[i, 1] for i in range(len(self.annotations))]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            img_path = self.file_paths[idx]
            image = Image.open(img_path).convert('RGB')
            
            # Get label (adding 1 to correspond to indices, 0=BAD, 1=UNLABELED, 2=GOOD)
            original_label = self.labels[idx]
            label = torch.tensor(original_label + 1, dtype=torch.long)
            if self.transform:
                image = self.transform(image)
                
            return image, label
            
        except Exception as e:
            # print(f"Error2 loading image {self.file_paths[idx]}: {e}")
            # Return dummy data in case of error
            if self.transform:
                # Create a dummy image that matches expected dimensions
                dummy_image = torch.zeros(3, 224, 224)
            else:
                dummy_image = torch.zeros(3, 256, 256)
            return dummy_image, torch.tensor(1, dtype=torch.long) # 1 for unlabeled


class StandardImageDataset(Dataset):
    """
    Standard dataset that returns two augmented versions for Mean Teacher
    No optimization, threading, or prefetching
    """
    def __init__(self, csv_file, root_dir, weak_transform=None, strong_transform=None, device=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.device = device

        self.transform_time = 0.0
        self.count = 0
        self.img_time = 0.0

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        
        try:
            # Load image as PIL
            img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
            image = Image.open(img_path).convert('RGB')
            
            image_tensor = to_tensor(image).to(self.device)
            weak_image = self.weak_transform(image_tensor) if self.weak_transform else image_tensor
            strong_image = self.strong_transform(image_tensor) if self.strong_transform else image_tensor

            original_label = self.annotations.iloc[idx, 1]
            label = torch.tensor(original_label + 1, dtype=torch.long)
            
            return weak_image, strong_image, label
        
        except Exception as e:
            # print(f"Error loading image {img_path}: {e}")
            # Return dummy data (on GPU as well to keep consistency)
            dummy = torch.zeros(3, 224, 224).to('cuda')
            return dummy, dummy, torch.tensor(1, dtype=torch.long) # 1 for unlabeled


class DataPrefetcher:
    """
    CUDA-aware data prefetcher that transfers data to GPU asynchronously
    """
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.iter = None
    
    def __iter__(self):
        self.iter = iter(self.loader)
        self.preload()
        return self
    
    def preload(self):
        try:
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    self.next_data = next(self.iter)
                    # Transfer to GPU asynchronously
                    if isinstance(self.next_data, (list, tuple)):
                        self.next_data = [x.to(self.device, non_blocking=True) 
                                         if isinstance(x, torch.Tensor) else x 
                                         for x in self.next_data]
                    else:
                        self.next_data = self.next_data.to(self.device, non_blocking=True)
            else:
                self.next_data = next(self.iter)
                if isinstance(self.next_data, (list, tuple)):
                    self.next_data = [x.to(self.device) 
                                     if isinstance(x, torch.Tensor) else x 
                                     for x in self.next_data]
                else:
                    self.next_data = self.next_data.to(self.device)
        except StopIteration:
            self.next_data = None
    
    def __next__(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        
        data = self.next_data
        if data is None:
            raise StopIteration
        
        self.preload()
        return data
    
    def __len__(self):
        return len(self.loader)


# Example usage function
def create_optimized_dataloaders(config, weak_transform, strong_transform, val_transform):
    """
    Create optimized dataloaders for training
    """
    print("Creating optimized datasets...")
    
    





if __name__ == "__main__":
    print('buddy...')

    # sanity check, example usage 

    # Create datasets
    # train_dataset = OptimizedTrainDataset(
    #     csv_file="train.csv", 
    #     root_dir="./data/train",
    #     weak_transform=weak_transform, 
    #     strong_transform=strong_transform,
    #     use_threading=True,
    #     num_threads=8
    # )
    
    # val_dataset = ImageDataset(
    #     csv_file="val.csv", 
    #     root_dir="./data/val", 
    #     transform=val_transform
    # )
    
    
    # # Create optimized dataloaders
    # train_loader = create_optimized_dataloader(
    #     train_dataset,
    #     batch_size=config.batch_size,
    #     shuffle=True,
    #     num_workers=0,  # 0 for Windows compatibility
    #     prefetch_factor=2,
    #     use_prefetch_wrapper=True,
    #     prefetch_size=300,  # Queue size
    #     prefetch_workers=6,  # Background threads
    #     pin_memory=True
    # )
    
    # val_loader = create_optimized_dataloader(
    #     val_dataset,
    #     batch_size=config.batch_size,
    #     shuffle=False,
    #     num_workers=0,
    #     use_prefetch_wrapper=False,  # Val dataset is cached
    #     pin_memory=True
    # )
    
    # # Wrap with CUDA prefetcher
    # if torch.cuda.is_available():
    #     train_loader = DataPrefetcher(train_loader, config.device)
    #     val_loader = DataPrefetcher(val_loader, config.device)
    
    # train_loader, val_loader