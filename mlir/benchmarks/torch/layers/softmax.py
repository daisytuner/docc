import torch
import torch.nn as nn

from benchmarks.harness import run_benchmark


class SoftmaxNet(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x: torch.Tensor):
        return self.softmax(x)


class LogSoftmaxNet(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=dim)

    def forward(self, x: torch.Tensor):
        return self.log_softmax(x)


# batch=64, classes=1000 — classifier output
def setup_softmax_classifier():
    model = SoftmaxNet(dim=1)
    x = torch.randn(64, 1000)
    return model, x


# batch=64, seq_len=512, features=768 — transformer-style attention scores
def setup_softmax_attention():
    model = SoftmaxNet(dim=-1)
    x = torch.randn(64, 512, 768)
    return model, x


# batch=64, classes=1000 — log-softmax for NLLLoss
def setup_log_softmax():
    model = LogSoftmaxNet(dim=1)
    x = torch.randn(64, 1000)
    return model, x


BENCHMARKS = {
    "softmax_classifier": setup_softmax_classifier,
    "softmax_attention": setup_softmax_attention,
    "log_softmax": setup_log_softmax,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Softmax layer benchmarks")
    parser.add_argument(
        "--variant",
        type=str,
        choices=list(BENCHMARKS.keys()),
        default="softmax_classifier",
        help="Softmax variant to benchmark",
    )
    args, remaining = parser.parse_known_args()

    import sys

    sys.argv = [sys.argv[0]] + remaining

    run_benchmark(BENCHMARKS[args.variant], args.variant)
