import torchvision.transforms.v2 as transforms
import torch


to_tensor = transforms.Compose([
    transforms.ToImage(), 
    transforms.ToDtype(torch.float32, scale=True),
])

weak_transform = transforms.Compose([
    # transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), # move this out
    
    transforms.RandomRotation(degrees=(-5, 5), interpolation=transforms.InterpolationMode.BILINEAR, expand=True, fill=0),
    transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True), # Resize maintaining aspect ratio, then pad to square
    transforms.RandomCrop(224), 

    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.005), transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

strong_transform = transforms.Compose([
    # transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
    
    transforms.RandomRotation(degrees=(-15, 15), interpolation=transforms.InterpolationMode.BILINEAR, expand=True, fill=0),
    transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True), # Resize maintaining aspect ratio, then pad to square
    transforms.RandomCrop(224), 

    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01), transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
    
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    
    transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True), # Resize maintaining aspect ratio, then pad to square
    transforms.CenterCrop(224), 
])
