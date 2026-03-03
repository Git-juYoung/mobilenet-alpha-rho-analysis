import torch
import torch.nn as nn


class SC(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DSC(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class StandardCNN(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()

        self.features = nn.Sequential(
            SC(3, 32, stride=2),
            SC(32, 64),
            SC(64, 128, stride=2),
            SC(128, 128),
            SC(128, 256, stride=2),
            SC(256, 256),
            SC(256, 512, stride=2),
            *[SC(512, 512) for _ in range(5)],
            SC(512, 1024, stride=2),
            SC(1024, 1024, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class MobileNet(nn.Module):
    def __init__(self, alpha=1.0, num_classes=200):
        super().__init__()

        def c(ch):
            return int(ch * alpha)

        self.features = nn.Sequential(
            SC(3, c(32), stride=2),
            DSC(c(32), c(64)),
            DSC(c(64), c(128), stride=2),
            DSC(c(128), c(128)),
            DSC(c(128), c(256), stride=2),
            DSC(c(256), c(256)),
            DSC(c(256), c(512), stride=2),
            *[DSC(c(512), c(512)) for _ in range(5)],
            DSC(c(512), c(1024), stride=2),
            DSC(c(1024), c(1024), stride=1),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c(1024), num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)