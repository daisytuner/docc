import torch
import torch.nn as nn

from benchmarks.harness import run_benchmark


class MaxPool2dNet(nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


class AvgPool2dNet(nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


class AdaptiveAvgPool2dNet(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=output_size)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


# batch=64, channels=256, height=224, width=224 — large feature map stress test
def setup_maxpool():
    model = MaxPool2dNet(kernel_size=3, stride=2)
    x = torch.randn(64, 256, 224, 224)
    return model, x


def setup_avgpool():
    model = AvgPool2dNet(kernel_size=3, stride=2)
    x = torch.randn(64, 256, 224, 224)
    return model, x


def setup_adaptive_avgpool():
    model = AdaptiveAvgPool2dNet(output_size=(14, 14))
    x = torch.randn(64, 1024, 56, 56)
    return model, x


BENCHMARKS = {
    "maxpool2d": setup_maxpool,
    "avgpool2d": setup_avgpool,
    "adaptive_avgpool2d": setup_adaptive_avgpool,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pooling layer benchmarks")
    parser.add_argument(
        "--variant",
        type=str,
        choices=list(BENCHMARKS.keys()),
        default="maxpool2d",
        help="Pooling variant to benchmark",
    )
    args, remaining = parser.parse_known_args()

    import sys

    sys.argv = [sys.argv[0]] + remaining

    run_benchmark(BENCHMARKS[args.variant], args.variant)
