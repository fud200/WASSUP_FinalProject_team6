import torch
import torch.nn as nn
from torchvision.models import densenet121
from vit_pytorch import ViT

# Hybrid Model 정의: DenseNet-121 + ViT
class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridModel, self).__init__()
        self.densenet = densenet121(pretrained=True)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()  # Remove the fully connected layer
        
        self.vit = ViT(
            image_size=224,
            patch_size=32,
            num_classes=1024,  # Use 1024 as an intermediate dimension
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        
        self.fc = nn.Linear(num_ftrs + 1024, num_classes)  # Adjusted for combined features

    def forward(self, x):
        densenet_features = self.densenet(x)  # Shape: (batch_size, num_ftrs)
        vit_features = self.vit(x)  # Shape: (batch_size, 1024)
        combined_features = torch.cat((densenet_features, vit_features), dim=1)  # Shape: (batch_size, num_ftrs + 1024)
        out = self.fc(combined_features)
        return out

# 모델 초기화 및 GPU로 이동
def initialize_model(num_classes, device):
    model = HybridModel(num_classes=num_classes).to(device)
    return model
