import torch
import torch.nn as nn
import torchvision.models as models

class CottonGuardCNN(nn.Module):
    def __init__(
        self,
        num_classes : int   = 10,
        dropout     : float = 0.3,
        hidden_dim  : int   = 256,
        freeze_until: int   = 8,
        pretrained  : bool  = True,
    ):
        super().__init__()
        
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        # Freeze early layers
        for i, layer in enumerate(backbone.features):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad_(False)

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, hidden_dim),
            nn.Hardswish(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

class CottonResNet18Refined(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Use Dropout + Linear head for regularisation (matches v3 training)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
