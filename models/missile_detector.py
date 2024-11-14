import torch
import torch.nn as nn

class MissileDetector(nn.Module):
    def __init__(self, num_classes=1):
        super(MissileDetector, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(128, num_classes, 1)
        self.regressor = nn.Conv2d(128, 4, 1)

    def forward(self, x):
        x = self.features(x)
        cls = self.classifier(x)
        reg = self.regressor(x)
        return cls, reg
