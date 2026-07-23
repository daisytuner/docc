import torch
import copy
from math import isnan

import docc.pytorch


def compare_shapes(res_shape: torch.Size, ref_shape: torch.Size) -> None:
    if ref_shape == torch.Size([]) or ref_shape == torch.Size([1]):
        assert res_shape == torch.Size([]) or res_shape == torch.Size([1])
    else:
        assert res_shape == ref_shape


def compare(
    res: None | int | float | torch.Tensor | tuple,
    ref: None | int | float | torch.Tensor | tuple,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    equal_nan: bool = False,
):
    if res is None and ref is None:
        pass  # This is valid
    if type(res) == int and type(ref) == int:
        assert res == ref
    elif type(res) == float and type(ref) == float:
        if isnan(res):
            assert equal_nan and isnan(ref)
        elif isnan(ref):
            assert equal_nan and isnan(res)
        else:
            assert abs(res - ref) <= atol + rtol * abs(ref)
    elif type(res) == torch.Tensor and type(ref) == torch.Tensor:
        assert res.dtype == ref.dtype
        compare_shapes(res.shape, ref.shape)
        if torch.is_floating_point(res):
            assert torch.allclose(res, ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
        else:
            assert torch.all(res == ref)
    elif type(res) == tuple and type(ref) == tuple:
        assert len(res) == len(ref)
        for res_elem, ref_elem in zip(res, ref):
            compare(res_elem, ref_elem, rtol=rtol, atol=atol, equal_nan=equal_nan)
    else:
        assert False, f"Unsupported result types: {type(res)} and {type(ref)}"


def check(
    model,
    *inputs,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    equal_nan: bool = False,
    target: str = "none",
    category: str = "server",
    remote_tuning: bool = False,
):
    model_ref = copy.deepcopy(model)
    program = torch.compile(
        model,
        backend="docc",
        options={
            "target": target,
            "category": category,
            "remote_tuning": remote_tuning,
        },
    )
    with torch.no_grad():
        ref = model_ref(*inputs)
        res = program(*inputs)
    compare(res, ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
