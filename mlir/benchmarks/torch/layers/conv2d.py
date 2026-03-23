import torch
import torch.nn as nn

from benchmarks.harness import run_benchmark


class Conv2dNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, bias=False, padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


# batch=32, in_channels=64, out_channels=128, kernel=3, height=56, width=56
def setup():
    model = Conv2dNet(64, 128, kernel_size=3)
    x = torch.randn(32, 64, 56, 56)
    return model, x


if __name__ == "__main__":
    run_benchmark(setup, "conv2d")
