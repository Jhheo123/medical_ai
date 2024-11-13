import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        se = torch.mean(x, dim=-1, keepdim=True)
        se = torch.mean(se, dim=-2, keepdim=True).view(se.size(0), -1)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).view(x.size(0), x.size(1), 1, 1)
        return x * se

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.downsample = downsample
        self.dropout = nn.Dropout(0.2)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.dropout(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        return F.relu(out)


class ResNetECG(nn.Module):
    def __init__(self, num_classes=26):
        super(ResNetECG, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 추가: age와 gender 정보를 결합할 수 있도록 입력 크기를 2만큼 증가
        self.fc = nn.Linear(512 + 2, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        layers = [ResidualBlock(in_channels, out_channels, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, age_gender):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        
        # age와 gender 정보를 결합하여 입력에 추가
        x = torch.cat([x, age_gender], dim=1)  # [batch_size, 512 + 2]
        return torch.sigmoid(self.fc(x))
