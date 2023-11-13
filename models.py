import torch
import torch.nn as nn
import torchvision.models
import torchvision.models as models
import torch.nn.functional as F
import torchinfo


class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        self.resnet = models.resnet34()
        modules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.r_fc = nn.Sequential(nn.Linear(1024, 2), torch.nn.Softmax(dim=1))
        self.d_fc = nn.Sequential(nn.Linear(512, 2), torch.nn.Softmax(dim=1))
        self.r2_fc = nn.Sequential(nn.Linear(512, 2), torch.nn.Softmax(dim=1))

    def forward(self, x):
        x = self.resnet(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

    def extract(self, x):
        x = self.resnet(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)

        return x
