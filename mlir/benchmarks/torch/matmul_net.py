import torch
import torch.nn as nn

import docc.torch

def test_pytorch():
    class MatmulNet(nn.Module):
        def __init__(self, weight1: torch.Tensor, weight2: torch.Tensor):
            super().__init__()
            self.W1 = nn.Parameter(weight1)
            self.W2 = nn.Parameter(weight2)

        def forward(self, x: torch.Tensor):
            h1 = torch.matmul(x, self.W1)
            h2 = torch.matmul(h1, self.W2)
            return h2

    weight1 = torch.randn(10, 16)
    weight2 = torch.randn(16, 3)
    model = MatmulNet(weight1, weight2)
    example_input = torch.randn(8, 10)

    program = torch.compile(model)
    res = program(example_input)

    res_ref = torch.matmul(torch.matmul(example_input, weight1), weight2)
    assert torch.allclose(res, res_ref)

# def test_backend():
#     class MatmulNet(nn.Module):
#         def __init__(self, weight1: torch.Tensor, weight2: torch.Tensor):
#             super().__init__()
#             self.W1 = nn.Parameter(weight1)
#             self.W2 = nn.Parameter(weight2)

#         def forward(self, x: torch.Tensor):
#             h1 = torch.matmul(x, self.W1)
#             h2 = torch.matmul(h1, self.W2)
#             return h2

#     weight1 = torch.randn(10, 16)
#     weight2 = torch.randn(16, 3)
#     model = MatmulNet(weight1, weight2)
#     example_input = torch.randn(8, 10)

#     docc.torch.set_backend_options(target="none", category="server")
#     program = torch.compile(model, backend="docc")
#     res = program(example_input)

#     res_ref = torch.matmul(torch.matmul(example_input, weight1), weight2)
#     assert torch.allclose(res, res_ref)

# def test_compile():
#     class MatmulNet(nn.Module):
#         def __init__(self, weight1: torch.Tensor, weight2: torch.Tensor):
#             super().__init__()
#             self.W1 = nn.Parameter(weight1)
#             self.W2 = nn.Parameter(weight2)

#         def forward(self, x: torch.Tensor):
#             h1 = torch.matmul(x, self.W1)
#             h2 = torch.matmul(h1, self.W2)
#             return h2

#     weight1 = torch.randn(10, 16)
#     weight2 = torch.randn(16, 3)
#     model = MatmulNet(weight1, weight2)
#     example_input = torch.randn(8, 10)

#     program = docc.torch.compile_torch(model, example_input)
#     res = program(example_input)

#     res_ref = torch.matmul(torch.matmul(example_input, weight1), weight2)
#     assert torch.allclose(res, res_ref)