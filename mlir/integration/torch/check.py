import torch
import copy

import docc.torch

def check_backend(model, *inputs, rtol=1e-4, atol=1e-5, target="none", category="server", remote_tuning=False):
    model_ref = copy.deepcopy(model)
    program = torch.compile(model, backend="docc", options={"target": target, "category": category, "remote_tuning": remote_tuning})
    with torch.no_grad():
        res = program(*inputs)
        ref = model_ref(*inputs)
    assert torch.allclose(res, ref, rtol=rtol, atol=atol)


def check_compile(model, *inputs, rtol=1e-4, atol=1e-5, target="none", category="server", remote_tuning=False):
    model_ref = copy.deepcopy(model)
    example_input = inputs[0] if len(inputs) == 1 else inputs
    program = docc.torch.compile_torch(model, example_input, target=target, category=category, remote_tuning=remote_tuning)
    with torch.no_grad():
        res = program(*inputs)
        ref = model_ref(*inputs)
    assert torch.allclose(res, ref, rtol=rtol, atol=atol)