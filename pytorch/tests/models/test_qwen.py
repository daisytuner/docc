import pytest
import torch
import torch.nn as nn
import torch._dynamo
from tests import check

# Define a small batch size matrix for fast testing
BATCH_SIZES = [1, 2, 8]


def get_qwen2_classes():
    try:
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2Model,
            Qwen2MLP,
            Qwen2Attention,
            Qwen2DecoderLayer,
            Qwen2RotaryEmbedding,
        )

        return (
            Qwen2Config,
            Qwen2Model,
            Qwen2MLP,
            Qwen2Attention,
            Qwen2DecoderLayer,
            Qwen2RotaryEmbedding,
        )
    except ImportError:
        pytest.skip("Transformers library not available or Qwen2 not supported.")


def get_dummy_config():
    Qwen2Config, _, _, _, _, _ = get_qwen2_classes()
    return Qwen2Config(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=1024,
        max_position_embeddings=512,
        use_cache=False,
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_qwen_mlp(target: str, batch_size: int) -> None:
    torch._dynamo.reset()
    _, _, Qwen2MLP, _, _, _ = get_qwen2_classes()

    class QwenMLPWrapper(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.mlp = Qwen2MLP(config)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.mlp(hidden_states)

    config = get_dummy_config()
    model = QwenMLPWrapper(config)
    model.eval()

    seq_length = 16
    x = torch.randn(batch_size, seq_length, config.hidden_size)
    check(model, x, target=target)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_qwen_qkv(target: str, batch_size: int) -> None:
    torch._dynamo.reset()
    _, _, _, Qwen2Attention, _, _ = get_qwen2_classes()

    class QwenQKVWrapper(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.attn = Qwen2Attention(config, layer_idx=0)

        def forward(
            self, hidden_states: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                self.attn.q_proj(hidden_states),
                self.attn.k_proj(hidden_states),
                self.attn.v_proj(hidden_states),
            )

    config = get_dummy_config()
    model = QwenQKVWrapper(config)
    model.eval()

    seq_length = 16
    x = torch.randn(batch_size, seq_length, config.hidden_size)
    check(model, x, target=target)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_qwen_attention(target: str, batch_size: int) -> None:
    torch._dynamo.reset()
    _, _, _, Qwen2Attention, _, Qwen2RotaryEmbedding = get_qwen2_classes()

    class QwenAttentionWrapper(nn.Module):
        def __init__(self, config):
            super().__init__()
            # Qwen2Attention requires layer_idx for RoPE cache handling in recent versions
            self.attn = Qwen2Attention(config, layer_idx=0)
            self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        def forward(
            self, hidden_states: torch.Tensor, position_ids: torch.Tensor
        ) -> torch.Tensor:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            return self.attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
            )[0]

    config = get_dummy_config()
    model = QwenAttentionWrapper(config)
    model.eval()

    seq_length = 16
    x = torch.randn(batch_size, seq_length, config.hidden_size)
    position_ids = (
        torch.arange(seq_length, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .clone()
    )

    check(model, x, position_ids, target=target)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_qwen_decoder_layer(target: str, batch_size: int) -> None:
    torch._dynamo.reset()
    _, _, _, _, Qwen2DecoderLayer, Qwen2RotaryEmbedding = get_qwen2_classes()

    class QwenDecoderLayerWrapper(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.layer = Qwen2DecoderLayer(config, layer_idx=0)
            self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        def forward(
            self, hidden_states: torch.Tensor, position_ids: torch.Tensor
        ) -> torch.Tensor:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            return self.layer(
                hidden_states=hidden_states,
                attention_mask=None,
                position_embeddings=position_embeddings,
            )[0]

    config = get_dummy_config()
    model = QwenDecoderLayerWrapper(config)
    model.eval()

    seq_length = 16
    x = torch.randn(batch_size, seq_length, config.hidden_size)
    position_ids = (
        torch.arange(seq_length, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .clone()
    )

    check(model, x, position_ids, target=target)


@pytest.mark.xfail(reason="Requires aten.embedding.default")
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_qwen_model(target: str, batch_size: int) -> None:
    torch._dynamo.reset()
    _, Qwen2Model, _, _, _, _ = get_qwen2_classes()

    class QwenModelWrapper(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.model = Qwen2Model(config)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.model(input_ids)[0]

    config = get_dummy_config()
    model = QwenModelWrapper(config)
    model.eval()

    seq_length = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))

    check(model, input_ids, target=target)
