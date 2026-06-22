import torch
import torch.nn as nn
import pytest

from benchmarks.harness import run_benchmark
from integration.torch.check import check_backend


class SoftmaxModel(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        return self.softmax(x)


# ──────────────────────────────────────────────────────────────────────
# Segformer attention_scores shapes:
#   4D: (batch, num_heads, seq_len, key_len)
#   2D: (batch * num_heads * seq_len, key_len)  — flattened
#
# After sequence reduction (SR), key_len = seq_len / sr_ratio^2
# attention_head_size = hidden_size / num_heads = 32 for all blocks
#
# Batch sizes: 1, 4, 16
# ──────────────────────────────────────────────────────────────────────

# --- Block 0: hidden=32, heads=1, sr=8, seq=16384, key=256 ---

def setup_block0_4d_b1():
    """Block 0 4D batch=1: (1, 1, 16384, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(1, 1, 16384, 256)


def setup_block0_4d_b4():
    """Block 0 4D batch=4: (4, 1, 16384, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(4, 1, 16384, 256)


def setup_block0_4d_b16():
    """Block 0 4D batch=16: (16, 1, 16384, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(16, 1, 16384, 256)


def setup_block0_2d_b1():
    """Block 0 2D batch=1: (1*1*16384, 256) = (16384, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(16384, 256)


def setup_block0_2d_b4():
    """Block 0 2D batch=4: (4*1*16384, 256) = (65536, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(65536, 256)


def setup_block0_2d_b16():
    """Block 0 2D batch=16: (16*1*16384, 256) = (262144, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(262144, 256)


# --- Block 1: hidden=64, heads=2, sr=4, seq=4096, key=256 ---

def setup_block1_4d_b1():
    """Block 1 4D batch=1: (1, 2, 4096, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(1, 2, 4096, 256)


def setup_block1_4d_b4():
    """Block 1 4D batch=4: (4, 2, 4096, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(4, 2, 4096, 256)


def setup_block1_4d_b16():
    """Block 1 4D batch=16: (16, 2, 4096, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(16, 2, 4096, 256)


def setup_block1_2d_b1():
    """Block 1 2D batch=1: (1*2*4096, 256) = (8192, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(8192, 256)


def setup_block1_2d_b4():
    """Block 1 2D batch=4: (4*2*4096, 256) = (32768, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(32768, 256)


def setup_block1_2d_b16():
    """Block 1 2D batch=16: (16*2*4096, 256) = (131072, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(131072, 256)


# --- Block 2: hidden=160, heads=5, sr=2, seq=1024, key=256 ---

def setup_block2_4d_b1():
    """Block 2 4D batch=1: (1, 5, 1024, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(1, 5, 1024, 256)


def setup_block2_4d_b4():
    """Block 2 4D batch=4: (4, 5, 1024, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(4, 5, 1024, 256)


def setup_block2_4d_b16():
    """Block 2 4D batch=16: (16, 5, 1024, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(16, 5, 1024, 256)


def setup_block2_2d_b1():
    """Block 2 2D batch=1: (1*5*1024, 256) = (5120, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(5120, 256)


def setup_block2_2d_b4():
    """Block 2 2D batch=4: (4*5*1024, 256) = (20480, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(20480, 256)


def setup_block2_2d_b16():
    """Block 2 2D batch=16: (16*5*1024, 256) = (81920, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(81920, 256)


# --- Block 3: hidden=256, heads=8, sr=1, seq=256, key=256 ---

def setup_block3_4d_b1():
    """Block 3 4D batch=1: (1, 8, 256, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(1, 8, 256, 256)


def setup_block3_4d_b4():
    """Block 3 4D batch=4: (4, 8, 256, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(4, 8, 256, 256)


def setup_block3_4d_b16():
    """Block 3 4D batch=16: (16, 8, 256, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(16, 8, 256, 256)


def setup_block3_2d_b1():
    """Block 3 2D batch=1: (1*8*256, 256) = (2048, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(2048, 256)


def setup_block3_2d_b4():
    """Block 3 2D batch=4: (4*8*256, 256) = (8192, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(8192, 256)


def setup_block3_2d_b16():
    """Block 3 2D batch=16: (16*8*256, 256) = (32768, 256)"""
    model = SoftmaxModel(dim=-1)
    model.eval()
    return model, torch.randn(32768, 256)


# ──────────────────────────────────────────────────────────────────────
# 4D softmax on non-last axis (reduction over seq_len dimension)
# ──────────────────────────────────────────────────────────────────────

def setup_4d_dim2_b1():
    """4D softmax over dim=2: (1, 8, 256, 256)"""
    model = SoftmaxModel(dim=2)
    model.eval()
    return model, torch.randn(1, 8, 256, 256)


def setup_4d_dim2_b4():
    """4D softmax over dim=2: (4, 8, 256, 256)"""
    model = SoftmaxModel(dim=2)
    model.eval()
    return model, torch.randn(4, 8, 256, 256)


def setup_4d_dim2_b16():
    """4D softmax over dim=2: (16, 8, 256, 256)"""
    model = SoftmaxModel(dim=2)
    model.eval()
    return model, torch.randn(16, 8, 256, 256)


# ──────────────────────────────────────────────────────────────────────
# 2D softmax on non-last axis (reduction over dim=0)
# ──────────────────────────────────────────────────────────────────────

def setup_2d_dim0():
    """2D softmax over dim=0: (256, 256)"""
    model = SoftmaxModel(dim=0)
    model.eval()
    return model, torch.randn(256, 256)


BENCHMARKS = {
    # 4D block variants
    "block0_4d_b1": setup_block0_4d_b1,
    "block0_4d_b4": setup_block0_4d_b4,
    "block0_4d_b16": setup_block0_4d_b16,
    "block1_4d_b1": setup_block1_4d_b1,
    "block1_4d_b4": setup_block1_4d_b4,
    "block1_4d_b16": setup_block1_4d_b16,
    "block2_4d_b1": setup_block2_4d_b1,
    "block2_4d_b4": setup_block2_4d_b4,
    "block2_4d_b16": setup_block2_4d_b16,
    "block3_4d_b1": setup_block3_4d_b1,
    "block3_4d_b4": setup_block3_4d_b4,
    "block3_4d_b16": setup_block3_4d_b16,
    # 2D block variants
    "block0_2d_b1": setup_block0_2d_b1,
    "block0_2d_b4": setup_block0_2d_b4,
    "block0_2d_b16": setup_block0_2d_b16,
    "block1_2d_b1": setup_block1_2d_b1,
    "block1_2d_b4": setup_block1_2d_b4,
    "block1_2d_b16": setup_block1_2d_b16,
    "block2_2d_b1": setup_block2_2d_b1,
    "block2_2d_b4": setup_block2_2d_b4,
    "block2_2d_b16": setup_block2_2d_b16,
    "block3_2d_b1": setup_block3_2d_b1,
    "block3_2d_b4": setup_block3_2d_b4,
    "block3_2d_b16": setup_block3_2d_b16,
    # Non-last-axis variants
    "4d_dim2_b1": setup_4d_dim2_b1,
    "4d_dim2_b4": setup_4d_dim2_b4,
    "4d_dim2_b16": setup_4d_dim2_b16,
    "2d_dim0": setup_2d_dim0,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Softmax layer benchmarks")
    parser.add_argument(
        "--variant",
        type=str,
        choices=list(BENCHMARKS.keys()),
        default="block1_4d_b1",
        help="Softmax variant to benchmark",
    )
    args, remaining = parser.parse_known_args()

    import sys

    sys.argv = [sys.argv[0]] + remaining

    run_benchmark(BENCHMARKS[args.variant], args.variant)


# ──────────────────────────────────────────────────────────────────────
# Pytest correctness tests
# ──────────────────────────────────────────────────────────────────────

TARGETS = ["sequential", "openmp", "cuda", "rocm"]
BATCH_SIZES = [1, 4, 16]


# --- Block 0: 4D ---

@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_softmax_block0_4d(target, batch_size) -> None:
    model = SoftmaxModel(dim=-1)
    model.eval()
    x = torch.randn(batch_size, 1, 16384, 256)
    check_backend(model, x, target=target)


# --- Block 0: 2D ---

@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_softmax_block0_2d(target, batch_size) -> None:
    model = SoftmaxModel(dim=-1)
    model.eval()
    x = torch.randn(batch_size * 1 * 16384, 256)
    check_backend(model, x, target=target)


# --- Block 1: 4D ---

@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_softmax_block1_4d(target, batch_size) -> None:
    model = SoftmaxModel(dim=-1)
    model.eval()
    x = torch.randn(batch_size, 2, 4096, 256)
    check_backend(model, x, target=target)


# --- Block 1: 2D ---

@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_softmax_block1_2d(target, batch_size) -> None:
    model = SoftmaxModel(dim=-1)
    model.eval()
    x = torch.randn(batch_size * 2 * 4096, 256)
    check_backend(model, x, target=target)


# --- Block 2: 4D ---

@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_softmax_block2_4d(target, batch_size) -> None:
    model = SoftmaxModel(dim=-1)
    model.eval()
    x = torch.randn(batch_size, 5, 1024, 256)
    check_backend(model, x, target=target)


# --- Block 2: 2D ---

@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_softmax_block2_2d(target, batch_size) -> None:
    model = SoftmaxModel(dim=-1)
    model.eval()
    x = torch.randn(batch_size * 5 * 1024, 256)
    check_backend(model, x, target=target)


# --- Block 3: 4D ---

@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_softmax_block3_4d(target, batch_size) -> None:
    model = SoftmaxModel(dim=-1)
    model.eval()
    x = torch.randn(batch_size, 8, 256, 256)
    check_backend(model, x, target=target)


# --- Block 3: 2D ---

@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_softmax_block3_2d(target, batch_size) -> None:
    model = SoftmaxModel(dim=-1)
    model.eval()
    x = torch.randn(batch_size * 8 * 256, 256)
    check_backend(model, x, target=target)


# --- 4D softmax over non-last axis (dim=2) ---

@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_softmax_4d_dim2(target, batch_size) -> None:
    model = SoftmaxModel(dim=2)
    model.eval()
    x = torch.randn(batch_size, 8, 256, 256)
    check_backend(model, x, target=target)


# --- 2D softmax over non-last axis (dim=0) ---

@pytest.mark.parametrize("target", TARGETS)
def test_softmax_2d_dim0(target) -> None:
    model = SoftmaxModel(dim=0)
    model.eval()
    x = torch.randn(256, 256)
    check_backend(model, x, target=target)
