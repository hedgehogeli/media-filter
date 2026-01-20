import os
import torchvision.transforms.v2 as transforms
from transformers import CLIPVisionModel, CLIPImageProcessor
from ultralytics import YOLO
import torch

from PIL import Image
from torchvision import models
import torch.nn as nn

# THIS FILE MUST: 
# export function load_for_inference, returning a model
# export function transform_image, returning a single transformed obj (can be tuple) 
# said model must support forward() on the returned obj ^
# ^ is currently not true because I have ModelBatch doing logic for me
# it really shouldnt be doing this 


class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-3])

    def forward(self, x):
        features = self.feature_extractor(x)  # [B, 1024, H/16, W/16]
        return features


class YOLOv11(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = YOLO("/home/hedge/Workspace/Code/img-classifier/model/yolov11l-face.pt").model
        # self.backbone = torch.nn.Sequential(*list(self.model.model.children())[:7])  # Stops after C3k2 (layer 6)
        self.feature_model = torch.nn.Sequential(
            *list(self.model.model.children())[:10]
        )  # Stops after SPPF (layer 9)

    def forward(self, x):
        return self.feature_model(x)


class CLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        # CLIP's final hidden state before projection (not the projection itself)
        self.clip_output_dim = self.clip_model.config.hidden_size

    def forward(self, x):

        outputs = self.clip_model(**x)
        pooled_output = outputs.pooler_output  # shape: [batch_size, 512]
        return pooled_output


class BiggerClassifier(torch.nn.Module):
    def __init__(self, output_dim=3):
        super().__init__()
        self.clip = CLIP()  # CLIP outputs: [B, 768]
        self.yolo = YOLOv11()  # YOLO outputs: [B, 512, 20, 20]
        self.resnet = ResNet()  # ResNet outputs: [B, 1024, H/16, W/16]

        # Global average pooling for feature maps
        self.yolo_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.resnet_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # self.fc1 = torch.nn.Linear(768 + 512 + 1024, 2048)
        self.fc1 = torch.nn.Linear(768 + 512, 2048)
        self.activation1 = torch.nn.GELU()
        self.dropout1 = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(2048, 1024)
        self.activation2 = torch.nn.GELU()
        self.dropout2 = torch.nn.Dropout(0.3)
        self.fc3 = torch.nn.Linear(1024, output_dim)

    def forward(self, clip_inputs, img_tensor):
        clip_features = self.clip(clip_inputs)  # [B, 768]
        yolo_features = self.yolo(img_tensor)  # [B, 512, 20, 20]
        # resnet_features = self.resnet(img_tensor) # [B, 1024, _, _]

        # Pool YOLO features to [B, 512, 1, 1] then to [B, 512]
        yolo_features = self.yolo_pool(yolo_features).flatten(1)
        # resnet_features = self.resnet_pool(resnet_features).flatten(1)

        # combined_features = torch.cat([clip_features, yolo_features, resnet_features], dim=1)  # [B, 2304]
        combined_features = torch.cat([clip_features, yolo_features], dim=1)

        x = self.fc1(combined_features)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.fc3(x)

        return x


def load_for_inference(checkpoint_path, device="cuda"):
    """
    Load model for inference from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        model: Loaded model in eval mode
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model for inference from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize model
    model = BiggerClassifier(output_dim=3)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    print(f"Model loaded")

    return model


CLIP_PROCESSOR = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")


to_tensor = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

yolo_val_transform = transforms.Compose(
    [
        to_tensor,
        transforms.Resize(
            size=700,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),  # Resize maintaining aspect ratio, then pad to square
        transforms.CenterCrop(640),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
    ]
)

clip_val_transform = transforms.Compose(
    [
        to_tensor,
        transforms.Resize(
            size=256,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
    ]
)


def transform_image(image: Image.Image):
    """Transform image using the provided transforms, return CPU tensor"""

    yolo_image = yolo_val_transform(image)
    clip_image = clip_val_transform(image)
    clip_image = CLIP_PROCESSOR(
        images=clip_image, return_tensors="pt", do_rescale=False
    )
    clip_image["pixel_values"] = clip_image["pixel_values"].squeeze(0)
    return (clip_image, yolo_image)
