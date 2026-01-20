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
from torchvision import models


class ResNet50Classifier(nn.Module):
    """ResNet50-based classifier with additional FC layers"""
    def __init__(self, num_classes=3, dropout_rate=0.2):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.dropout = nn.Dropout(dropout_rate)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
    
class ResNet152Classifier(nn.Module):
    """ResNet50-based classifier with additional FC layers"""
    def __init__(self, num_classes=3, dropout_rate=0.2):
        super().__init__()
        self.backbone = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        # self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.dropout = nn.Dropout(dropout_rate)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def load_for_inference(checkpoint_path, device='cuda'):
    """
    Load model for inference only (simpler version)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    num_classes = checkpoint.get('config', {}).get('num_classes', 3)
    
    # Load teacher model (usually performs better)
    model = ResNet152Classifier(num_classes=num_classes).to(device)
    
    if 'teacher_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['teacher_state_dict'])
        print("Loaded teacher model for inference")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded student model for inference")
    else:
        raise ValueError("No model weights found in checkpoint")
    
    model.eval()
    return model