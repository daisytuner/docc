import pytest

import torch
import torch.nn as nn

import docc.torch


def test_single_output():
    docc.torch.set_backend_options(target="none", category="server")

    class LinearNet(nn.Module):
        def __init__(self, in_features=4, out_features=2):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = LinearNet()
    model.eval()
    model_ref = LinearNet()
    model_ref.eval()
    model_ref.load_state_dict(model.state_dict())

    program = torch.compile(model, backend="docc")

    example_input = torch.randn(2, 4)

    # Force dynamo (inference) backend
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert res.shape == (2, 2)
    assert torch.allclose(res, ref, rtol=1e-5)


def test_multi_output_float32():
    """Test model returning two float32 tensors."""
    docc.torch.set_backend_options(target="none", category="server")

    class TwoOutputNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4, 2, bias=False)
            self.linear2 = nn.Linear(4, 3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear1(x), self.linear2(x)

    model = TwoOutputNet()
    model.eval()
    model_ref = TwoOutputNet()
    model_ref.eval()
    model_ref.load_state_dict(model.state_dict())

    program = torch.compile(model, backend="docc")
    example_input = torch.randn(2, 4)

    with torch.no_grad():
        res1, res2 = program(example_input)
        ref1, ref2 = model_ref(example_input)

    assert res1.shape == (2, 2)
    assert res2.shape == (2, 3)
    assert torch.allclose(res1, ref1, rtol=1e-5)
    assert torch.allclose(res2, ref2, rtol=1e-5)


def test_multi_output_float64():
    """Test model returning two float64 tensors."""
    docc.torch.set_backend_options(target="none", category="server")

    class TwoOutputNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4, 2, bias=False, dtype=torch.float64)
            self.linear2 = nn.Linear(4, 3, bias=False, dtype=torch.float64)

        def forward(self, x: torch.Tensor):
            return self.linear1(x), self.linear2(x)

    model = TwoOutputNet()
    model.eval()
    model_ref = TwoOutputNet()
    model_ref.eval()
    model_ref.load_state_dict(model.state_dict())

    program = torch.compile(model, backend="docc")
    example_input = torch.randn(2, 4, dtype=torch.float64)

    with torch.no_grad():
        res1, res2 = program(example_input)
        ref1, ref2 = model_ref(example_input)

    assert res1.dtype == torch.float64
    assert res2.dtype == torch.float64
    assert res1.shape == (2, 2)
    assert res2.shape == (2, 3)
    assert torch.allclose(res1, ref1, rtol=1e-5)
    assert torch.allclose(res2, ref2, rtol=1e-5)


def test_multi_output_different_shapes():
    """Test model returning tensors with different shapes."""
    docc.torch.set_backend_options(target="none", category="server")

    class DiffShapeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(8, 6, bias=False)
            self.linear2 = nn.Linear(8, 4, bias=False)

        def forward(self, x: torch.Tensor):
            out1 = self.linear1(x)  # (3, 6)
            out2 = self.linear2(x)  # (3, 4)
            return out1, out2

    model = DiffShapeNet()
    model.eval()
    model_ref = DiffShapeNet()
    model_ref.eval()
    model_ref.load_state_dict(model.state_dict())

    program = torch.compile(model, backend="docc")
    example_input = torch.randn(3, 8)

    with torch.no_grad():
        res1, res2 = program(example_input)
        ref1, ref2 = model_ref(example_input)

    assert res1.shape == (3, 6)
    assert res2.shape == (3, 4)
    assert torch.allclose(res1, ref1, rtol=1e-5)
    assert torch.allclose(res2, ref2, rtol=1e-5)


def test_multi_output_three_outputs():
    """Test model returning three tensors."""
    docc.torch.set_backend_options(target="none", category="server")

    class ThreeOutputNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4, 2, bias=False)
            self.linear2 = nn.Linear(4, 3, bias=False)
            self.linear3 = nn.Linear(4, 5, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear1(x), self.linear2(x), self.linear3(x)

    model = ThreeOutputNet()
    model.eval()
    model_ref = ThreeOutputNet()
    model_ref.eval()
    model_ref.load_state_dict(model.state_dict())

    program = torch.compile(model, backend="docc")
    example_input = torch.randn(2, 4)

    with torch.no_grad():
        res1, res2, res3 = program(example_input)
        ref1, ref2, ref3 = model_ref(example_input)

    assert res1.shape == (2, 2)
    assert res2.shape == (2, 3)
    assert res3.shape == (2, 5)
    assert torch.allclose(res1, ref1, rtol=1e-5)
    assert torch.allclose(res2, ref2, rtol=1e-5)
    assert torch.allclose(res3, ref3, rtol=1e-5)


def test_output_c_contiguous():
    """Verify all outputs are C-contiguous."""
    docc.torch.set_backend_options(target="none", category="server")

    class TwoOutputNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4, 3, bias=False)
            self.linear2 = nn.Linear(4, 5, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear1(x), self.linear2(x)

    model = TwoOutputNet()
    model.eval()

    program = torch.compile(model, backend="docc")
    example_input = torch.randn(2, 4)

    with torch.no_grad():
        res1, res2 = program(example_input)

    assert res1.is_contiguous(), "Output 1 should be C-contiguous"
    assert res2.is_contiguous(), "Output 2 should be C-contiguous"


def test_output_stride_verification():
    """Verify output strides match expected C-order strides."""
    docc.torch.set_backend_options(target="none", category="server")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 6, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = Net()
    model.eval()

    program = torch.compile(model, backend="docc")
    example_input = torch.randn(4, 8)

    with torch.no_grad():
        res = program(example_input)

    # Expected C-order strides for shape (4, 6): [6, 1]
    assert res.shape == (4, 6)
    expected_strides = (6, 1)
    assert (
        res.stride() == expected_strides
    ), f"Expected strides {expected_strides}, got {res.stride()}"


def test_multi_output_strides():
    """Verify multi-output strides are all C-order."""
    docc.torch.set_backend_options(target="none", category="server")

    class MultiNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(8, 6, bias=False)
            self.linear2 = nn.Linear(8, 4, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear1(x), self.linear2(x)

    model = MultiNet()
    model.eval()

    program = torch.compile(model, backend="docc")
    example_input = torch.randn(3, 8)

    with torch.no_grad():
        res1, res2 = program(example_input)

    # Check C-order strides for (3, 6) and (3, 4)
    assert res1.stride() == (
        6,
        1,
    ), f"res1 strides should be (6, 1), got {res1.stride()}"
    assert res2.stride() == (
        4,
        1,
    ), f"res2 strides should be (4, 1), got {res2.stride()}"


def test_training():
    """Verify gradient descent learns a known linear function."""
    docc.torch.set_backend_options(target="none", category="server")

    # Target: learn the 2x2 identity matrix
    target_weights = torch.eye(2)

    class LinearNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    torch.manual_seed(42)
    model = LinearNet()

    program = torch.compile(model, backend="docc")
    optimizer = torch.optim.SGD(program.parameters(), lr=0.5)
    criterion = nn.MSELoss()

    # Train on random inputs, target = input (identity function)
    for _ in range(20):
        x = torch.randn(32, 2)
        target = x  # Identity: output should equal input

        optimizer.zero_grad()
        res = program(x)
        loss = criterion(res, target)
        loss.backward()
        optimizer.step()

    # Verify learned weights converged to identity matrix
    learned_weights = model.linear.weight.detach()
    print(learned_weights)
    assert torch.allclose(
        learned_weights, target_weights, atol=0.05
    ), f"Expected identity matrix, got:\n{learned_weights}"
