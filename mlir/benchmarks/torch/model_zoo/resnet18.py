import torch
import torch.nn as nn
import torchvision.models as models
from functools import partial

import docc.torch

class Resnet18BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if in_channels == out_channels:
            self.downsample = nn.Sequential()
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
    def forward(self, x: torch.Tensor):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y += self.downsample(x)
        return y

class Resnet18Layer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.layer = nn.Sequential(
            Resnet18BasicBlock(in_channels, out_channels, stride),
            Resnet18BasicBlock(out_channels, out_channels)
        )
    def forward(self, x: torch.Tensor):
        y = self.layer(x)
        return y

class Resnet18Begin(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    def forward(self, x: torch.Tensor):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)
        return y

class Resnet18End(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=1000, bias=True)
    def forward(self, x: torch.Tensor):
        y = self.avgpool(x)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y

class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.begin = Resnet18Begin()
        self.layer1 = Resnet18Layer(64, 64, 1)
        self.layer2 = Resnet18Layer(64, 128, 2)
        self.layer3 = Resnet18Layer(128, 256, 2)
        self.layer4 = Resnet18Layer(256, 512, 2)
        self.end = Resnet18End()
    def forward(self, x: torch.Tensor):
        y = self.begin(x)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.end(y)
        return y

def setup_basicblock(in_channels: int, out_channels: int, stride: int, h: int, w: int):
    model = Resnet18BasicBlock(in_channels, out_channels, stride)
    model.eval()
    x = torch.randn(32, in_channels, h, w)
    return model, x

def setup_layer(in_channels: int, out_channels: int, stride: int, h: int, w: int):
    model = Resnet18Layer(in_channels, out_channels, stride)
    model.eval()
    x = torch.randn(32, in_channels, h, w)
    return model, x

def setup_begin():
    model = Resnet18Begin()
    model.eval()
    x = torch.randn(32, 3, 224, 224)
    return model, x

def setup_end():
    model = Resnet18End()
    model.eval()
    x = torch.randn(32, 512, 7, 7)
    return model, x

def setup_all():
    model = Resnet18()
    model.eval()
    x = torch.randn(32, 3, 224, 224)
    return model, x

def setup():
    model = models.resnet18(weights=None)
    model.eval()
    x = torch.randn(32, 3, 224, 224)
    return model, x

BENCHMARKS = {
    "default": setup,
    "begin": setup_begin,
    "layer1_begin": partial(setup_basicblock, 64, 64, 1, 58, 58),
    "layer1_end": partial(setup_basicblock, 64, 64, 1, 58, 58),
    "layer1": partial(setup_layer, 64, 64, 1, 58, 58),
    "layer2_begin": partial(setup_basicblock, 64, 128, 2, 58, 58),
    "layer2_end": partial(setup_basicblock, 128, 128, 1, 30, 30),
    "layer2": partial(setup_layer, 64, 128, 2, 58, 58),
    "layer3_begin": partial(setup_basicblock, 128, 256, 2, 30, 30),
    "layer3_end": partial(setup_basicblock, 256, 256, 1, 16, 16),
    "layer3": partial(setup_layer, 128, 256, 2, 30, 30),
    "layer4_begin": partial(setup_basicblock, 256, 512, 2, 16, 16),
    "layer4_end": partial(setup_basicblock, 512, 512, 1, 9, 9),
    "layer4": partial(setup_layer, 256, 512, 2, 16, 16),
    "end": setup_end,
    "all": setup_all,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="resnet18 benchmark")
    parser.add_argument("--variant", type=str, choices=list(BENCHMARKS.keys()), default="default")
    args, remaining = parser.parse_known_args()

    import sys

    sys.argv = [sys.argv[0]] + remaining

    from benchmarks.harness import run_benchmark

    run_benchmark(BENCHMARKS[args.variant], f"resnet18 {args.variant}")
