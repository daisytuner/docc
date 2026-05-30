"""Layerwise test for SegFormer-b0.

Tests each encoder stage and the decode head individually with the docc backend,
checking the output of each against a pure-PyTorch reference.

Structure of SegFormer-b0:
  Encoder:
    Stage 0: OverlapPatchEmbedding (stride=4) + 2x TransformerBlock + LayerNorm -> (B, 32, H/4,  W/4)
    Stage 1: OverlapPatchEmbedding (stride=2) + 2x TransformerBlock + LayerNorm -> (B, 64, H/8,  W/8)
    Stage 2: OverlapPatchEmbedding (stride=2) + 2x TransformerBlock + LayerNorm -> (B,160, H/16, W/16)
    Stage 3: OverlapPatchEmbedding (stride=2) + 2x TransformerBlock + LayerNorm -> (B,256, H/32, W/32)
  Decode head:
    4x Linear projection + upsample to stage-0 resolution + concat + fuse Conv+BN + classifier Conv
"""

import time

import pytest
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

import docc.torch

MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
INPUT_SHAPE = (1, 3, 512, 512)
RTOL = 1e-2
ATOL = 1e-4


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class EncoderStageWrapper(nn.Module):
    """One encoder stage (SegformerStage): patch embedding + transformer blocks + layer norm.

    In newer HuggingFace versions the stage is a self-contained SegformerStage module
    whose forward accepts and returns a spatial feature map (B, C, H, W).
    """

    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    def forward(self, x):
        return self.stage(x)


class DecodeHeadWrapper(nn.Module):
    """Decode head: takes 4 stage feature maps, returns logits (B, num_classes, H/4, W/4).

    Accepts stage outputs as individual positional arguments (not a tuple) so that
    torch.compile / docc can trace through without dynamic container unpacking.
    """

    def __init__(self, decode_head):
        super().__init__()
        self.decode_head = decode_head

    def forward(self, s0, s1, s2, s3):
        return self.decode_head((s0, s1, s2, s3))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_diff(result: torch.Tensor, reference: torch.Tensor, label: str) -> bool:
    diff = (result - reference).abs()
    rel = diff / reference.abs().clamp(min=1e-8)
    n_total = diff.numel()
    n_fail = (~torch.isclose(result, reference, rtol=RTOL, atol=ATOL)).sum().item()
    print(
        f"  {label}: "
        f"abs max={diff.max().item():.6f} mean={diff.mean().item():.6f} | "
        f"rel max={rel.max().item():.6f} mean={rel.mean().item():.6f} | "
        f"failing {n_fail}/{n_total} ({100 * n_fail / n_total:.2f}%)"
    )
    return n_fail == 0


def _compile(module: nn.Module) -> nn.Module:
    return torch.compile(
        module,
        backend="docc",
        options={"target": "sequential", "category": "server"},
        dynamic=False,  # keep height/width as concrete ints, not SymInts
    )


# ---------------------------------------------------------------------------
# Shared fixture: load model + compute reference outputs for all stages once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def segformer_refs():
    """Load the pretrained model and run the reference forward pass stage by stage."""
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).eval()
    stages = model.segformer.stages

    example_input = torch.randn(*INPUT_SHAPE)

    stage_inputs = []   # input to each encoder stage
    stage_outputs = []  # output of each encoder stage (2-D spatial feature map)

    x = example_input
    with torch.no_grad():
        for stage in stages:
            stage_inputs.append(x.clone())
            x = stage(x)
            stage_outputs.append(x.clone())

        # Reference logits from the full model (using reference stage outputs)
        ref_logits = model.decode_head(tuple(stage_outputs))

    return {
        "model": model,
        "example_input": example_input,
        "stage_inputs": stage_inputs,
        "stage_outputs": stage_outputs,
        "ref_logits": ref_logits,
    }


# ---------------------------------------------------------------------------
# Encoder stage tests
# ---------------------------------------------------------------------------

def _test_encoder_stage(segformer_refs, stage_idx: int):
    refs = segformer_refs
    stage = refs["model"].segformer.stages[stage_idx]

    wrapper = EncoderStageWrapper(stage)

    compiled = _compile(wrapper)
    stage_input = refs["stage_inputs"][stage_idx]

    t0 = time.perf_counter()
    with torch.no_grad():
        result = compiled(stage_input)
    t1 = time.perf_counter()
    print(f"\nEncoderStage{stage_idx} inference: {(t1 - t0) * 1000:.2f} ms")

    reference = refs["stage_outputs"][stage_idx]
    ok = _print_diff(result, reference, f"EncoderStage{stage_idx}")
    assert ok, f"EncoderStage{stage_idx} output mismatch (see diff above)"


def test_encoder_stage_0(segformer_refs):
    _test_encoder_stage(segformer_refs, 0)


def test_encoder_stage_1(segformer_refs):
    _test_encoder_stage(segformer_refs, 1)


def test_encoder_stage_2(segformer_refs):
    _test_encoder_stage(segformer_refs, 2)


def test_encoder_stage_3(segformer_refs):
    _test_encoder_stage(segformer_refs, 3)


# ---------------------------------------------------------------------------
# Individual transformer block tests (finer granularity within a stage)
# ---------------------------------------------------------------------------

class SingleBlockWrapper(nn.Module):
    """A single SegformerLayer (attention + FFN) with fixed height/width."""

    def __init__(self, block, height: int, width: int):
        super().__init__()
        self.block = block
        self.height = height
        self.width = width

    def forward(self, hidden_states):
        return self.block(hidden_states, self.height, self.width)[0]


def _test_transformer_block(segformer_refs, stage_idx: int, block_idx: int):
    """Test one transformer block inside an encoder stage.

    Uses the actual intermediate hidden states at that block's input by running
    the patch embedding (and preceding blocks) in reference mode.
    """
    refs = segformer_refs
    stage = refs["model"].segformer.stages[stage_idx]
    # SegformerStage stores its transformer blocks as 'layers' in newer HF versions
    blocks = getattr(stage, "layers", None) or getattr(stage, "blocks", None)
    if blocks is None:
        pytest.skip(f"Cannot find transformer blocks in SegformerStage (stage {stage_idx})")

    stage_input = refs["stage_inputs"][stage_idx]

    with torch.no_grad():
        hidden_states, height, width = stage.patch_embeddings(stage_input)
        for j in range(block_idx):
            hidden_states = blocks[j](hidden_states, height, width)[0]
        block_input = hidden_states.clone()
        block_ref_output = blocks[block_idx](block_input, height, width)[0]

    wrapper = SingleBlockWrapper(blocks[block_idx], height, width)
    compiled = _compile(wrapper)

    with torch.no_grad():
        result = compiled(block_input)

    label = f"Stage{stage_idx}/Block{block_idx}"
    ok = _print_diff(result, block_ref_output, label)
    assert ok, f"{label} output mismatch"


def test_stage0_block0(segformer_refs):
    _test_transformer_block(segformer_refs, 0, 0)


def test_stage0_block1(segformer_refs):
    _test_transformer_block(segformer_refs, 0, 1)


def test_stage1_block0(segformer_refs):
    _test_transformer_block(segformer_refs, 1, 0)


def test_stage1_block1(segformer_refs):
    _test_transformer_block(segformer_refs, 1, 1)


def test_stage2_block0(segformer_refs):
    _test_transformer_block(segformer_refs, 2, 0)


def test_stage2_block1(segformer_refs):
    _test_transformer_block(segformer_refs, 2, 1)


def test_stage3_block0(segformer_refs):
    _test_transformer_block(segformer_refs, 3, 0)


def test_stage3_block1(segformer_refs):
    _test_transformer_block(segformer_refs, 3, 1)


# ---------------------------------------------------------------------------
# Decode head test
# ---------------------------------------------------------------------------

def test_decode_head(segformer_refs):
    """Test the decode head in isolation using the reference stage outputs as input."""
    refs = segformer_refs
    decode_head = refs["model"].decode_head
    s0, s1, s2, s3 = refs["stage_outputs"]

    wrapper = DecodeHeadWrapper(decode_head)
    compiled = _compile(wrapper)

    t0 = time.perf_counter()
    with torch.no_grad():
        result = compiled(s0, s1, s2, s3)
    t1 = time.perf_counter()
    print(f"\nDecodeHead inference: {(t1 - t0) * 1000:.2f} ms")

    ok = _print_diff(result, refs["ref_logits"], "DecodeHead")
    assert ok, "DecodeHead output mismatch"


# ---------------------------------------------------------------------------
# End-to-end composed test: use compiled stages in sequence
# ---------------------------------------------------------------------------

def test_end_to_end_composed(segformer_refs):
    """Run all 4 compiled encoder stages + compiled decode head in sequence.

    This is the same as test_backend in segformer_test.py but with the model
    manually decomposed so that the first failing stage is immediately visible.
    """
    refs = segformer_refs
    stages = refs["model"].segformer.stages

    compiled_stages = [
        _compile(EncoderStageWrapper(stage))
        for stage in stages
    ]
    compiled_head = _compile(DecodeHeadWrapper(refs["model"].decode_head))

    x = refs["example_input"]
    stage_outputs = []
    with torch.no_grad():
        for i, stage in enumerate(compiled_stages):
            t0 = time.perf_counter()
            x = stage(x)
            t1 = time.perf_counter()
            print(f"\nComposed Stage{i}: {(t1 - t0) * 1000:.2f} ms, shape={tuple(x.shape)}")

            ok = _print_diff(x, refs["stage_outputs"][i], f"ComposedStage{i}")
            assert ok, f"Composed encoder stage {i} output mismatch"
            stage_outputs.append(x)

        t0 = time.perf_counter()
        logits = compiled_head(*stage_outputs)
        t1 = time.perf_counter()
        print(f"Composed DecodeHead: {(t1 - t0) * 1000:.2f} ms")

    ok = _print_diff(logits, refs["ref_logits"], "ComposedLogits")
    assert ok, "End-to-end composed output mismatch"
