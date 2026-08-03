import torch
import transformers
import pytest

from tests import check


@pytest.mark.skip(reason="Mismatching output shapes")
def test_segformer_b0_finetuned_cityscapes_1024_1024_simple(target: str) -> None:
    model = transformers.SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    )
    model.eval()
    check(
        model.segformer,
        torch.randn(8, 3, 512, 512),
        kwargs={
            "output_hidden_states": True,
            "return_dict": False,
        },
        target=target,
    )
