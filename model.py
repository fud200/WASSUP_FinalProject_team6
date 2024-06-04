import torch
import torch.nn as nn
from torchvision.models import efficientnet_b1, efficientnet_b2
from vit_pytorch import ViT

class HybridModel(nn.Module):
    def __init__(self, num_classes, model_version='b1'):
        super(HybridModel, self).__init__()
        if model_version == 'b1':
            self.efficientnet = efficientnet_b1(pretrained=True)
        elif model_version == 'b2':
            self.efficientnet = efficientnet_b2(pretrained=True)
        else:
            raise ValueError("model_version should be 'b1' or 'b2'")
        
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()
        
        self.vit = ViT(
            image_size=224,
            patch_size=32,
            num_classes=1024,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        
        self.fc = nn.Linear(num_ftrs + 1024, num_classes)

    def forward(self, x):
        efficientnet_features = self.efficientnet(x)
        vit_features = self.vit(x)
        combined_features = torch.cat((efficientnet_features, vit_features), dim=1)
        out = self.fc(combined_features)
        return out
