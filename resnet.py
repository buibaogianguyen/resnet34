import torch
import torch.nn as nn
from residual import ResidualBlock

class ResNet34(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNet34, self).__init__()

        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer1 = self._make_layer(64, ResidualBlock, 3, stride=1)
        self.layer2 = self._make_layer(128, ResidualBlock, 4)
        self.layer3 = self._make_layer(256, ResidualBlock, 6)
        self.layer4 = self._make_layer(512, ResidualBlock, 3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels, block, blocks, stride=2):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        for i in range (1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
