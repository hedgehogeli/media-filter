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



# training and validation logic for mean teacher method# Mean Teacher utilities
def update_ema_variables(model, ema_model, alpha, global_step):
    """Update EMA variables with warmup"""
    # alpha = min(1 - 1 / (global_step + 1), alpha)
    # with torch.no_grad():
    #     for ema_param, param in zip(ema_model.parameters(), model.parameters()):
    #         ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    with torch.no_grad():
        for s_param, t_param in zip(model.parameters(), ema_model.parameters()):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=(1.0 - alpha))

    # Copy buffers (important for BatchNorm running_mean/var etc.)
    # Note: buffers are not parameters but they matter for forward behavior.
    for t_buf, s_buf in zip(model.buffers(), ema_model.buffers()):
        t_buf.data.copy_(s_buf.data)

def compute_consistency_loss(student_outputs, teacher_outputs, temperature=4.0, entropy_threshold=0.4):
    """Compute consistency loss using KL divergence with per-sample reduction"""

    # student_log_probs = F.log_softmax(student_outputs / temperature, dim=1)
    # teacher_probs = F.softmax(teacher_outputs / temperature, dim=1)
    # per_sample_loss = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=1) * (temperature ** 2)

    # if False:
    #     # Compute teacher's entropy (uncertainty)
    #     # Use teacher_outputs without temperature for entropy calculation
    #     teacher_probs_for_entropy = F.softmax(teacher_outputs, dim=1)
    #     teacher_entropy = -torch.sum(teacher_probs_for_entropy * torch.log(teacher_probs_for_entropy + 1e-8), dim=1)
        
    #     # Normalize entropy by maximum possible entropy (log(num_classes))
    #     num_classes = teacher_outputs.shape[1]
    #     max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))
    #     normalized_entropy = teacher_entropy / max_entropy  # Range: [0, 1]
        
    #     # Compute confidence weight: low entropy = high confidence = high weight
    #     # Using exponential decay for smooth weighting
    #     entropy_threshold = 0.5
    #     confidence_weights = torch.exp(-normalized_entropy / entropy_threshold)
        
    #     # Alternative weighting schemes you could try:
        
    #     # Option 2: Linear weighting
    #     # confidence_weights = 1.0 - normalized_entropy
        
    #     # Option 3: Threshold-based (binary)
    #     # confidence_weights = (normalized_entropy < entropy_threshold).float()
        
    #     # Option 4: Sigmoid-based smooth transition
    #     # confidence_weights = torch.sigmoid((entropy_threshold - normalized_entropy) * 10)
        
    #     # Apply confidence weights to the loss
    #     per_sample_loss = per_sample_loss * confidence_weights
    
    # return per_sample_loss  # Returns tensor of shape [batch_size]

    


    
    # 1. clamp logits to avoid huge exponentials (safety)
    student_outputs = student_outputs.clamp(min=-1e2, max=1e2)
    teacher_outputs = teacher_outputs.clamp(min=-1e2, max=1e2)

    # 2. compute log-probs for both (stable)
    student_logp = F.log_softmax(student_outputs / temperature, dim=1)
    teacher_logp = F.log_softmax(teacher_outputs / temperature, dim=1)

    # 3. Use log_target=True to pass log-prob target directly (avoids 0*log0 issues)
    # reduction='none' keeps per-sample results; sum across classes to get per-sample.
    per_sample = F.kl_div(student_logp, teacher_logp, reduction='none', log_target=True).sum(dim=1)

    # 4. temperature scaling factor (common in distillation)
    per_sample = per_sample * (temperature ** 2)

    # 5. exponential thresholding:
    # compute teacher probs (without temperature for entropy)
    teacher_probs = F.softmax(teacher_outputs, dim=1)
    # Shannon entropy per sample
    teacher_entropy = -torch.sum(
        teacher_probs * torch.log(teacher_probs + 1e-8),
        dim=1
    )
    # normalize entropy to [0, 1]
    num_classes = teacher_outputs.shape[1]
    max_entropy = torch.log(torch.tensor(num_classes, dtype=teacher_entropy.dtype, device=teacher_entropy.device))
    normalized_entropy = teacher_entropy / max_entropy
    # Exponential decay weighting
    confidence_weights = torch.exp(-normalized_entropy / entropy_threshold)
    per_sample = per_sample * confidence_weights

    # 6. clamp/nan-safety: replace NaN/inf with large finite values instead of letting them propagate
    per_sample = torch.nan_to_num(per_sample, nan=1e3, posinf=1e3, neginf=1e3)

    return per_sample  # shape [B]



