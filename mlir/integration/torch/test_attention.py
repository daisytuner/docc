import pytest

import torch
import torch.nn as nn

from integration.torch.check import check_backend, check_compile

# --- Self-attention: 8 heads, dim 64 ---


def test_self_attention_compile():
    class SelfAttn8h64dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=64, num_heads=8, batch_first=True
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    check_compile(SelfAttn8h64dCompile().eval(), torch.randn(2, 16, 64))


def test_self_attention_backend():
    class SelfAttn8h64dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=64, num_heads=8, batch_first=True
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    check_backend(SelfAttn8h64dBackend().eval(), torch.randn(2, 16, 64))


# --- Self-attention: 4 heads, dim 128 ---


def test_self_attention_4h128d_compile():
    class SelfAttn4h128dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=128, num_heads=4, batch_first=True
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    check_compile(SelfAttn4h128dCompile().eval(), torch.randn(2, 32, 128))


def test_self_attention_4h128d_backend():
    class SelfAttn4h128dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=128, num_heads=4, batch_first=True
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    check_backend(SelfAttn4h128dBackend().eval(), torch.randn(2, 32, 128))


# --- Self-attention: 1 head (degenerate) ---


def test_self_attention_1head_compile():
    class SelfAttn1hCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=32, num_heads=1, batch_first=True
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    check_compile(SelfAttn1hCompile().eval(), torch.randn(4, 8, 32))


def test_self_attention_1head_backend():
    class SelfAttn1hBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=32, num_heads=1, batch_first=True
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    check_backend(SelfAttn1hBackend().eval(), torch.randn(4, 8, 32))


# --- Self-attention: 16 heads, dim 256 ---


def test_self_attention_16h256d_compile():
    class SelfAttn16h256dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=256, num_heads=16, batch_first=True
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    check_compile(SelfAttn16h256dCompile().eval(), torch.randn(1, 16, 256))


def test_self_attention_16h256d_backend():
    class SelfAttn16h256dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=256, num_heads=16, batch_first=True
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    check_backend(SelfAttn16h256dBackend().eval(), torch.randn(1, 16, 256))


# --- Self-attention: batch_first=False ---


def test_self_attention_seq_first_compile():
    class SelfAttnSeqFirstCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=64, num_heads=8, batch_first=False
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    # (seq, batch, embed)
    check_compile(SelfAttnSeqFirstCompile().eval(), torch.randn(16, 2, 64))


def test_self_attention_seq_first_backend():
    class SelfAttnSeqFirstBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=64, num_heads=8, batch_first=False
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    check_backend(SelfAttnSeqFirstBackend().eval(), torch.randn(16, 2, 64))


# --- Cross-attention ---


def test_cross_attention_compile():
    class CrossAttn8h64dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=64, num_heads=8, batch_first=True
            )

        def forward(self, q: torch.Tensor, kv: torch.Tensor):
            out, _ = self.attn(q, kv, kv)
            return out

    check_compile(
        CrossAttn8h64dCompile().eval(), torch.randn(2, 8, 64), torch.randn(2, 16, 64)
    )


def test_cross_attention_backend():
    class CrossAttn8h64dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=64, num_heads=8, batch_first=True
            )

        def forward(self, q: torch.Tensor, kv: torch.Tensor):
            out, _ = self.attn(q, kv, kv)
            return out

    check_backend(
        CrossAttn8h64dBackend().eval(), torch.randn(2, 8, 64), torch.randn(2, 16, 64)
    )


def test_cross_attention_large_compile():
    class CrossAttn4h128dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=128, num_heads=4, batch_first=True
            )

        def forward(self, q: torch.Tensor, kv: torch.Tensor):
            out, _ = self.attn(q, kv, kv)
            return out

    check_compile(
        CrossAttn4h128dCompile().eval(), torch.randn(1, 4, 128), torch.randn(1, 64, 128)
    )


def test_cross_attention_large_backend():
    class CrossAttn4h128dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=128, num_heads=4, batch_first=True
            )

        def forward(self, q: torch.Tensor, kv: torch.Tensor):
            out, _ = self.attn(q, kv, kv)
            return out

    check_backend(
        CrossAttn4h128dBackend().eval(), torch.randn(1, 4, 128), torch.randn(1, 64, 128)
    )
