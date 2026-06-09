import torch
import copy
from math import isnan

import docc.torch


def compare(
    res: None | int | float | torch.Tensor | tuple,
    ref: None | int | float | torch.Tensor | tuple,
    rtol=1e-4,
    atol=1e-5,
    equal_nan: bool = False,
):
    if type(res) == None and type(ref) == None:
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
        assert res.shape == ref.shape
        assert torch.allclose(res, ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
    elif type(res) == tuple and type(ref) == tuple:
        assert len(res) == len(ref)
        for res_elem, ref_elem in zip(res, ref):
            compare(res_elem, ref_elem, rtol=rtol, atol=atol, equal_nan=equal_nan)
    else:
        assert False, f"Unsupported result types: {type(res)} and {type(ref)}"


def check_backend(
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
        res = program(*inputs)
        ref = model_ref(*inputs)
    compare(res, ref, rtol=rtol, atol=atol, equal_nan=equal_nan)


def check_compile(
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
    example_input = inputs[0] if len(inputs) == 1 else inputs
    program = docc.torch.compile_torch(
        model,
        example_input,
        target=target,
        category=category,
        remote_tuning=remote_tuning,
    )
    with torch.no_grad():
        res = program(*inputs)
        ref = model_ref(*inputs)
    compare(res, ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
