import torch
import torch.nn as nn

import docc.torch


# --- Flatten 2D input (batch, features) -> already flat ---


def test_flatten_2d_compile():
    class FlatNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.flatten(x, start_dim=1)

    model = FlatNet()
    model_ref = FlatNet()
    example_input = torch.randn(4, 8)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref)


def test_flatten_2d_backend():
    class FlatNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.flatten(x, start_dim=1)

    model = FlatNet()
    model_ref = FlatNet()
    example_input = torch.randn(4, 8)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref)


# --- Flatten 4D tensor (N, C, H, W) -> (N, C*H*W), as in ResNet-18 ---


def test_flatten_4d_compile():
    class FlatNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.flatten(x, start_dim=1)

    model = FlatNet()
    model_ref = FlatNet()
    example_input = torch.randn(2, 8, 4, 4)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref)


def test_flatten_4d_backend():
    class FlatNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.flatten(x, start_dim=1)

    model = FlatNet()
    model_ref = FlatNet()
    example_input = torch.randn(2, 8, 4, 4)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref)


# --- Flatten 3D tensor (N, S, D) -> (N*S, D) ---


def test_flatten_3d_compile():
    class FlatNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.flatten(x, start_dim=0, end_dim=1)

    model = FlatNet()
    model_ref = FlatNet()
    example_input = torch.randn(3, 5, 16)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref)


def test_flatten_3d_backend():
    class FlatNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.flatten(x, start_dim=0, end_dim=1)

    model = FlatNet()
    model_ref = FlatNet()
    example_input = torch.randn(3, 5, 16)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref)


# --- Flatten all dims -> 1D ---


def test_flatten_all_compile():
    class FlatAllNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.flatten(x)

    model = FlatAllNet()
    model_ref = FlatAllNet()
    example_input = torch.randn(2, 3, 4)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref)


def test_flatten_all_backend():
    class FlatAllNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.flatten(x)

    model = FlatAllNet()
    model_ref = FlatAllNet()
    example_input = torch.randn(2, 3, 4)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref)


# --- AdaptiveAvgPool2d -> flatten -> Linear (ResNet-18 classifier pattern) ---


def test_flatten_after_avgpool_compile():
    class PoolFlatLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512, 10, bias=True)

        def forward(self, x: torch.Tensor):
            x = self.pool(x)  # (N, 512, 1, 1)
            x = torch.flatten(x, start_dim=1)  # (N, 512)
            return self.fc(x)

    model = PoolFlatLinear()
    model_ref = PoolFlatLinear()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(2, 512, 7, 7)

    program = docc.torch.compile_torch(model, example_input)
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4, atol=1e-5)


def test_flatten_after_avgpool_backend():
    class PoolFlatLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512, 10, bias=True)

        def forward(self, x: torch.Tensor):
            x = self.pool(x)  # (N, 512, 1, 1)
            x = torch.flatten(x, start_dim=1)  # (N, 512)
            return self.fc(x)

    model = PoolFlatLinear()
    model_ref = PoolFlatLinear()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(2, 512, 7, 7)

    docc.torch.set_backend_options(target="none", category="server")
    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)

    assert torch.allclose(res, res_ref, rtol=1e-4, atol=1e-5)
