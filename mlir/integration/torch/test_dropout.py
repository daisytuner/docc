import torch
import torch.nn as nn

from integration.torch.check import check_backend, check_compile


def test_dropout_backend():
    class DropoutNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(p=0.2)

        def forward(self, x: torch.Tensor):
            return self.dropout(x)

    check_backend(DropoutNet().eval(), torch.randn(20, 16))


def test_dropout_compile():
    class DropoutNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(p=0.2)

        def forward(self, x: torch.Tensor):
            return self.dropout(x)

    check_compile(DropoutNet().eval(), torch.randn(20, 16))


def test_dropout1d_backend():
    class Dropout1dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout1d(p=0.2)

        def forward(self, x: torch.Tensor):
            return self.dropout(x)

    check_backend(Dropout1dNet().eval(), torch.randn(20, 16, 32))


def test_dropout1d_compile():
    class Dropout1dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout1d(p=0.2)

        def forward(self, x: torch.Tensor):
            return self.dropout(x)

    check_compile(Dropout1dNet().eval(), torch.randn(20, 16, 32))


def test_dropout2d_backend():
    class Dropout2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout2d(p=0.2)

        def forward(self, x: torch.Tensor):
            return self.dropout(x)

    check_backend(Dropout2dNet().eval(), torch.randn(20, 16, 32, 32))


def test_dropout2d_compile():
    class Dropout2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout2d(p=0.2)

        def forward(self, x: torch.Tensor):
            return self.dropout(x)

    check_compile(Dropout2dNet().eval(), torch.randn(20, 16, 32, 32))


def test_dropout3d_backend():
    class Dropout3dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout3d(p=0.2)

        def forward(self, x: torch.Tensor):
            return self.dropout(x)

    check_backend(Dropout3dNet().eval(), torch.randn(20, 16, 4, 32, 32))


def test_dropout3d_compile():
    class Dropout3dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout3d(p=0.2)

        def forward(self, x: torch.Tensor):
            return self.dropout(x)

    check_compile(Dropout3dNet().eval(), torch.randn(20, 16, 4, 32, 32))


def test_alpha_dropout_backend():
    class AlphaDropoutNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.AlphaDropout(p=0.2)

        def forward(self, x: torch.Tensor):
            return self.dropout(x)

    check_backend(AlphaDropoutNet().eval(), torch.randn(20, 16))


def test_alpha_dropout_compile():
    class AlphaDropoutNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.AlphaDropout(p=0.2)

        def forward(self, x: torch.Tensor):
            return self.dropout(x)

    check_compile(AlphaDropoutNet().eval(), torch.randn(20, 16))


def test_feature_alpha_dropout_backend():
    class FeatureAlphaDropoutNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.FeatureAlphaDropout(p=0.2)

        def forward(self, x: torch.Tensor):
            return self.dropout(x)

    check_backend(FeatureAlphaDropoutNet().eval(), torch.randn(20, 16, 4, 32, 32))


def test_feature_alpha_dropout_compile():
    class FeatureAlphaDropoutNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.FeatureAlphaDropout(p=0.2)

        def forward(self, x: torch.Tensor):
            return self.dropout(x)

    check_compile(FeatureAlphaDropoutNet().eval(), torch.randn(20, 16, 4, 32, 32))
