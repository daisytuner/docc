import torch

from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from tests import check


def test_qwen2_mlp(target: str) -> None:
    model = Qwen2ForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float32
    )
    mlp = model.model.layers[0].mlp
    check(mlp, torch.randn(1, 39, 1536), target=target, atol=1e-4)


def test_qwen2_mlp_fixed(target: str) -> None:
    model = Qwen2ForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float32
    )
    mlp = model.model.layers[5].mlp
    torch.manual_seed(42)
    check(mlp, torch.randn(1, 39, 1536), target=target, atol=2e-5)
