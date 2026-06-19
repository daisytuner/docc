import pytest

import torch
import torch.nn as nn
import transformers
from transformers.modeling_outputs import BaseModelOutput
from functools import partial
import math

from integration.torch.check import check_backend


class SegformerOverlapPatchEmbeddings(nn.Module):
    def __init__(
        self, patch_size: int, stride: int, num_channels: int, hidden_size: int
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        embeddings = self.proj(pixel_values)
        _, _, height, width = embeddings.shape
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.layer_norm(embeddings)
        return embeddings, height, width


def setup_segformer_overlap_patch_embeddings(
    patch_size: int, stride: int, num_channels: int, hidden_size: int, h: int, w: int
) -> tuple[SegformerOverlapPatchEmbeddings, torch.Tensor]:
    model = SegformerOverlapPatchEmbeddings(
        patch_size, stride, num_channels, hidden_size
    )
    model.eval()
    x = torch.randn(1, num_channels, 512, 512)
    return model, x


class SegformerEfficientSelfAttention(nn.Module):
    def __init__(
        self, hidden_size: int, num_attention_heads: int, sequence_reduction_ratio: int
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=0.0)

        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sr = nn.Conv2d(
                hidden_size,
                hidden_size,
                kernel_size=sequence_reduction_ratio,
                stride=sequence_reduction_ratio,
            )
            self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        height: int,
        width: int,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
        batch_size = hidden_states.shape[0]
        query_layer = (
            self.query(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        if self.sr_ratio > 1:
            batch_size, _, num_channels = hidden_states.shape
            hidden_states = hidden_states.permute(0, 2, 1).reshape(
                batch_size, num_channels, height, width
            )
            hidden_states = self.sr(hidden_states)
            hidden_states = hidden_states.reshape(batch_size, num_channels, -1).permute(
                0, 2, 1
            )
            hidden_states = self.layer_norm(hidden_states)

        key_layer = (
            self.key(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        value_layer = (
            self.value(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


def setup_segformer_efficient_self_attention(
    hidden_size: int,
    num_attention_heads: int,
    sequence_reduction_ratio: int,
    h: int,
    w: int,
    height: int,
    width: int,
) -> tuple[SegformerEfficientSelfAttention, tuple[torch.Tensor, int, int]]:
    model = SegformerEfficientSelfAttention(
        hidden_size, num_attention_heads, sequence_reduction_ratio
    )
    model.eval()
    x = torch.randn(1, h, w)
    return model, (x, height, width)


class SegformerSelfOutput(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.0)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


def setup_segformer_self_output(
    h: int, w: int
) -> tuple[SegformerSelfOutput, tuple[torch.Tensor, torch.Tensor]]:
    model = SegformerSelfOutput(w)
    model.eval()
    x = torch.randn(1, h, w)
    y = torch.randn(1, h, w)
    return model, (x, y)


class SegformerAttention(nn.Module):
    def __init__(
        self, hidden_size: int, num_attention_heads: int, sequence_reduction_ratio: int
    ) -> None:
        super().__init__()
        self.self = SegformerEfficientSelfAttention(
            hidden_size, num_attention_heads, sequence_reduction_ratio
        )
        self.output = SegformerSelfOutput(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        height: int,
        width: int,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, height, width, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


def setup_segformer_attention(
    hidden_size: int,
    num_attention_heads: int,
    sequence_reduction_ratio: int,
    h: int,
    height: int,
    width: int,
) -> tuple[SegformerAttention, tuple[torch.Tensor, int, int]]:
    model = SegformerAttention(
        hidden_size, num_attention_heads, sequence_reduction_ratio
    )
    model.eval()
    x = torch.randn(1, h, hidden_size)
    return model, (x, height, width)


class SegformerDropPath(nn.Module):
    def __init__(self, drop_prob: float | None = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob if drop_prob else 0.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return hidden_states
        keep_prop = 1 - self.drop_prob
        shape = (hidden_states.shape[0],) + (1,) * (hidden_states.ndim - 1)
        random_tensor = keep_prop + torch.rand(
            shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        random_tensor.floor_()
        return hidden_states.div(keep_prop) * random_tensor

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


def setup_segformer_drop_path(drop_prob: float, h: int, w: int):
    model = SegformerDropPath(drop_prob)
    model.eval()
    x = torch.randn(1, h, w)
    return model, x


class SegformerDWConv(nn.Module):
    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(
        self, hidden_states: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        batch_size, _, num_channels = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2).view(
            batch_size, num_channels, height, width
        )
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        return hidden_states


def setup_segformer_dw_conv(
    dim: int, h: int, height: int, width: int
) -> tuple[SegformerDWConv, tuple[torch.Tensor, int, int]]:
    model = SegformerDWConv(dim)
    model.eval()
    x = torch.randn(1, h, dim)
    return model, (x, height, width)


class SegformerMixFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        out_features = in_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.dwconv = SegformerDWConv(hidden_features)
        self.intermediate_act_fn = nn.functional.gelu
        self.dense2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(p=0.0)

    def forward(
        self, hidden_states: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.dwconv(hidden_states, height, width)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


def setup_segformer_mix_ffn(
    in_features: int, hidden_features: int, h: int, height: int, width: int
) -> tuple[SegformerMixFFN, tuple[torch.Tensor, int, int]]:
    model = SegformerMixFFN(in_features, hidden_features)
    model.eval()
    x = torch.randn(1, h, in_features)
    return model, (x, height, width)


class SegformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        drop_path: float,
        sequence_reduction_ratio: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.attention = SegformerAttention(
            hidden_size,
            num_attention_heads,
            sequence_reduction_ratio,
        )
        self.drop_path = (
            SegformerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = SegformerMixFFN(hidden_size, mlp_hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        height: int,
        width: int,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
        self_attention_outputs = self.attention(
            self.layer_norm_1(hidden_states),
            height,
            width,
            output_attentions=output_attentions,
        )

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        attention_output = self.drop_path(attention_output)
        hidden_states = attention_output + hidden_states

        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)

        mlp_output = self.drop_path(mlp_output)
        layer_output = mlp_output + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


def setup_segformer_layer(
    hidden_size: int,
    num_attention_heads: int,
    drop_path: float,
    sequence_reduction_ratio: int,
    mlp_ratio: int,
    h: int,
    height: int,
    width: int,
) -> tuple[SegformerLayer, tuple[torch.Tensor, int, int]]:
    model = SegformerLayer(
        hidden_size, num_attention_heads, drop_path, sequence_reduction_ratio, mlp_ratio
    )
    model.eval()
    x = torch.randn(1, h, hidden_size)
    return model, (x, height, width)


class SegformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_embeddings = nn.ModuleList(
            [
                SegformerOverlapPatchEmbeddings(7, 4, 3, 32),
                SegformerOverlapPatchEmbeddings(3, 2, 32, 64),
                SegformerOverlapPatchEmbeddings(3, 2, 64, 160),
                SegformerOverlapPatchEmbeddings(3, 2, 160, 256),
            ]
        )

        self.block = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        SegformerLayer(32, 1, 0 * (0.1 / 7), 8, 4),
                        SegformerLayer(32, 1, 1 * (0.1 / 7), 8, 4),
                    ]
                ),
                nn.ModuleList(
                    [
                        SegformerLayer(64, 2, 2 * (0.1 / 7), 4, 4),
                        SegformerLayer(64, 2, 3 * (0.1 / 7), 4, 4),
                    ]
                ),
                nn.ModuleList(
                    [
                        SegformerLayer(160, 5, 4 * (0.1 / 7), 2, 4),
                        SegformerLayer(160, 5, 5 * (0.1 / 7), 2, 4),
                    ]
                ),
                nn.ModuleList(
                    [
                        SegformerLayer(256, 8, 6 * (0.1 / 7), 1, 4),
                        SegformerLayer(256, 8, 7 * (0.1 / 7), 1, 4),
                    ]
                ),
            ]
        )

        self.layer_norm = nn.ModuleList(
            [
                nn.LayerNorm(32),
                nn.LayerNorm(64),
                nn.LayerNorm(160),
                nn.LayerNorm(256),
            ]
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
    ) -> tuple | BaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size = pixel_values.shape[0]

        hidden_states = pixel_values
        for idx, x in enumerate(
            zip(self.patch_embeddings, self.block, self.layer_norm)
        ):
            embedding_layer, block_layer, norm_layer = x
            hidden_states, height, width = embedding_layer(hidden_states)
            for blk in block_layer:  # type: ignore
                layer_outputs = blk(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)  # type: ignore
            hidden_states = norm_layer(hidden_states)
            if (
                idx != len(self.patch_embeddings) - 1
                or idx == len(self.patch_embeddings) - 1
            ):
                hidden_states = (
                    hidden_states.reshape(batch_size, height, width, -1)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


def setup_segformer_encoder() -> (
    tuple[SegformerEncoder, tuple[torch.Tensor, bool, bool, bool]]
):
    model = SegformerEncoder()
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    return model, (x, False, True, False)


class SegformerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = SegformerEncoder()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutput:
        output_attentions = (
            output_attentions if output_attentions is not None else False
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        return_dict = return_dict if return_dict is not None else True

        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def setup_segformer_model() -> (
    tuple[SegformerModel, tuple[torch.Tensor, bool | None, bool | None, bool | None]]
):
    model = SegformerModel()
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    return model, (x, None, True, False)


class SegformerMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 256)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


def setup_segformer_mlp(
    input_dim: int, height: int, width: int
) -> tuple[SegformerMLP, torch.Tensor]:
    model = SegformerMLP(input_dim)
    model.eval()
    x = torch.randn(1, input_dim, height, width)
    return model, x


class SegformerDecodeHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_c = nn.ModuleList(
            [
                SegformerMLP(32),
                SegformerMLP(64),
                SegformerMLP(160),
                SegformerMLP(256),
            ]
        )

        self.linear_fuse = nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(256)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Conv2d(256, 19, kernel_size=1)

    def forward(
        self,
        encoder_hidden_states: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(
                batch_size, -1, height, width
            )

            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state,
                size=encoder_hidden_states[0].size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        logits = self.classifier(hidden_states)

        return logits


def setup_segformer_decode_head() -> tuple[
    SegformerDecodeHead,
    tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    model = SegformerDecodeHead()
    model.eval()
    w = torch.randn(1, 32, 128, 128)
    x = torch.randn(1, 64, 64, 64)
    y = torch.randn(1, 160, 32, 32)
    z = torch.randn(1, 256, 16, 16)
    return model, ((w, x, y, z),)


class SegformerForSemanticSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.segformer = SegformerModel()
        self.decode_head = SegformerDecodeHead()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> tuple:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=False,
        )

        encoder_hidden_states = outputs[1]

        logits = self.decode_head(encoder_hidden_states)

        loss = None
        if labels is not None:
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            loss_fct = nn.CrossEntropyLoss(ignore_index=255)
            loss = loss_fct(upsampled_logits, labels)

        if output_hidden_states:
            output = (logits,) + outputs[1:]
        else:
            output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


def setup_segformer_for_semantic_segmentation() -> (
    tuple[SegformerForSemanticSegmentation, torch.Tensor]
):
    model = SegformerForSemanticSegmentation()
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    return model, x


def setup() -> tuple[transformers.SegformerForSemanticSegmentation, torch.Tensor]:
    model = transformers.SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    )
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    return model, x


TARGETS = ["sequential", "openmp", "cuda", "rocm"]
BATCH_SIZES = [1, 4, 16]


BENCHMARKS = {
    "default": setup,
    "segformer.encoder.path_embeddings.0": partial(
        setup_segformer_overlap_patch_embeddings, 7, 4, 3, 32, 512, 512
    ),
    "segformer.encoder.path_embeddings.1": partial(
        setup_segformer_overlap_patch_embeddings, 3, 2, 32, 64, 128, 128
    ),
    "segformer.encoder.path_embeddings.2": partial(
        setup_segformer_overlap_patch_embeddings, 3, 2, 64, 160, 64, 64
    ),
    "segformer.encoder.path_embeddings.3": partial(
        setup_segformer_overlap_patch_embeddings, 3, 2, 160, 256, 32, 32
    ),
    "segformer.encoder.block.0.0.attention.self": partial(
        setup_segformer_efficient_self_attention, 32, 1, 8, 16384, 32, 128, 128
    ),
    "segformer.encoder.block.1.0.attention.self": partial(
        setup_segformer_efficient_self_attention, 64, 2, 4, 4096, 64, 64, 64
    ),
    "segformer.encoder.block.2.0.attention.self": partial(
        setup_segformer_efficient_self_attention, 160, 5, 2, 1024, 160, 32, 32
    ),
    "segformer.encoder.block.3.0.attention.self": partial(
        setup_segformer_efficient_self_attention, 256, 8, 1, 256, 256, 16, 16
    ),
    "segformer.encoder.block.0.0.attention.output": partial(
        setup_segformer_self_output, 16384, 32
    ),
    "segformer.encoder.block.1.0.attention.output": partial(
        setup_segformer_self_output, 4096, 64
    ),
    "segformer.encoder.block.2.0.attention.output": partial(
        setup_segformer_self_output, 1024, 160
    ),
    "segformer.encoder.block.3.0.attention.output": partial(
        setup_segformer_self_output, 256, 256
    ),
    "segformer.encoder.block.0.0.attention": partial(
        setup_segformer_attention, 32, 1, 8, 16384, 128, 128
    ),
    "segformer.encoder.block.1.0.attention": partial(
        setup_segformer_attention, 64, 2, 4, 4096, 64, 64
    ),
    "segformer.encoder.block.2.0.attention": partial(
        setup_segformer_attention, 160, 5, 2, 1024, 32, 32
    ),
    "segformer.encoder.block.3.0.attention": partial(
        setup_segformer_attention, 256, 8, 1, 256, 16, 16
    ),
    "segformer.encoder.block.0.0.drop_path": partial(
        setup_segformer_drop_path, 0 * (0.1 / 7), 16384, 32
    ),
    "segformer.encoder.block.0.1.drop_path": partial(
        setup_segformer_drop_path, 1 * (0.1 / 7), 16384, 32
    ),
    "segformer.encoder.block.1.0.drop_path": partial(
        setup_segformer_drop_path, 2 * (0.1 / 7), 4096, 64
    ),
    "segformer.encoder.block.1.1.drop_path": partial(
        setup_segformer_drop_path, 3 * (0.1 / 7), 4096, 64
    ),
    "segformer.encoder.block.2.0.drop_path": partial(
        setup_segformer_drop_path, 4 * (0.1 / 7), 1024, 160
    ),
    "segformer.encoder.block.2.1.drop_path": partial(
        setup_segformer_drop_path, 5 * (0.1 / 7), 1024, 160
    ),
    "segformer.encoder.block.3.0.drop_path": partial(
        setup_segformer_drop_path, 6 * (0.1 / 7), 256, 256
    ),
    "segformer.encoder.block.3.1.drop_path": partial(
        setup_segformer_drop_path, 7 * (0.1 / 7), 256, 256
    ),
    "segformer.encoder.block.0.0.mlp.dwconv": partial(
        setup_segformer_dw_conv, 128, 16384, 128, 128
    ),
    "segformer.encoder.block.1.0.mlp.dwconv": partial(
        setup_segformer_dw_conv, 256, 4096, 64, 64
    ),
    "segformer.encoder.block.2.0.mlp.dwconv": partial(
        setup_segformer_dw_conv, 640, 1024, 32, 32
    ),
    "segformer.encoder.block.3.0.mlp.dwconv": partial(
        setup_segformer_dw_conv, 1024, 256, 16, 16
    ),
    "segformer.encoder.block.0.0.mlp": partial(
        setup_segformer_mix_ffn, 32, 128, 16384, 128, 128
    ),
    "segformer.encoder.block.1.0.mlp": partial(
        setup_segformer_mix_ffn, 64, 256, 4096, 64, 64
    ),
    "segformer.encoder.block.2.0.mlp": partial(
        setup_segformer_mix_ffn, 160, 640, 1024, 32, 32
    ),
    "segformer.encoder.block.3.0.mlp": partial(
        setup_segformer_mix_ffn, 256, 1024, 256, 16, 16
    ),
    "segformer.encoder.block.0.0": partial(
        setup_segformer_layer, 32, 1, 0 * (0.1 / 7), 8, 4, 16384, 128, 128
    ),
    "segformer.encoder.block.0.1": partial(
        setup_segformer_layer, 32, 1, 1 * (0.1 / 7), 8, 4, 16384, 128, 128
    ),
    "segformer.encoder.block.1.0": partial(
        setup_segformer_layer, 64, 2, 2 * (0.1 / 7), 4, 4, 4096, 64, 64
    ),
    "segformer.encoder.block.1.1": partial(
        setup_segformer_layer, 64, 2, 3 * (0.1 / 7), 4, 4, 4096, 64, 64
    ),
    "segformer.encoder.block.2.0": partial(
        setup_segformer_layer, 160, 5, 4 * (0.1 / 7), 2, 4, 1024, 32, 32
    ),
    "segformer.encoder.block.2.1": partial(
        setup_segformer_layer, 160, 5, 5 * (0.1 / 7), 2, 4, 1024, 32, 32
    ),
    "segformer.encoder.block.3.0": partial(
        setup_segformer_layer, 256, 8, 6 * (0.1 / 7), 1, 4, 256, 16, 16
    ),
    "segformer.encoder.block.3.1": partial(
        setup_segformer_layer, 256, 8, 7 * (0.1 / 7), 1, 4, 256, 16, 16
    ),
    "segformer.encoder": setup_segformer_encoder,
    "segformer": setup_segformer_model,
    "decode_head.linear_c.0": partial(setup_segformer_mlp, 32, 128, 128),
    "decode_head.linear_c.1": partial(setup_segformer_mlp, 64, 64, 64),
    "decode_head.linear_c.2": partial(setup_segformer_mlp, 160, 32, 32),
    "decode_head.linear_c.3": partial(setup_segformer_mlp, 256, 16, 16),
    "decode_head": setup_segformer_decode_head,
    "all": setup_segformer_for_semantic_segmentation,
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="segformer benchmark")
    parser.add_argument(
        "--variant", type=str, choices=list(BENCHMARKS.keys()), default="default"
    )
    args, remaining = parser.parse_known_args()

    import sys

    sys.argv = [sys.argv[0]] + remaining

    from benchmarks.harness import run_benchmark

    run_benchmark(BENCHMARKS[args.variant], f"segformer {args.variant}")


@pytest.mark.parametrize("target", TARGETS)
def test_segformer_encoder_path_embeddings_0(target) -> None:
    model, x = BENCHMARKS["segformer.encoder.path_embeddings.0"]()
    check_backend(model, x, target=target)


@pytest.mark.parametrize("target", TARGETS)
def test_segformer_encoder_path_embeddings_1(target) -> None:
    model, x = BENCHMARKS["segformer.encoder.path_embeddings.1"]()
    check_backend(model, x, target=target)


@pytest.mark.parametrize("target", TARGETS)
def test_segformer_encoder_path_embeddings_2(target) -> None:
    model, x = BENCHMARKS["segformer.encoder.path_embeddings.2"]()
    check_backend(model, x, target=target)


@pytest.mark.parametrize("target", TARGETS)
def test_segformer_encoder_path_embeddings_3(target) -> None:
    model, x = BENCHMARKS["segformer.encoder.path_embeddings.3"]()
    check_backend(model, x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_0_0_attention_self(target, batch_size) -> None:
    class SegformerEncoderBlock00AttentionSelf(SegformerEfficientSelfAttention):
        def __init__(self) -> None:
            super().__init__(32, 1, 8)

        def forward(
            self,
            hidden_states: torch.Tensor,
            height: int,
            width: int,
            output_attentions: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
            return SegformerEfficientSelfAttention.forward(
                self, hidden_states, height, width, output_attentions
            )

    model = SegformerEncoderBlock00AttentionSelf()
    model.eval()
    x = (torch.randn(batch_size, 16384, 32), 128, 128)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_1_0_attention_self(target, batch_size) -> None:
    class SegformerEncoderBlock10AttentionSelf(SegformerEfficientSelfAttention):
        def __init__(self) -> None:
            super().__init__(64, 2, 4)

        def forward(
            self,
            hidden_states: torch.Tensor,
            height: int,
            width: int,
            output_attentions: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
            return SegformerEfficientSelfAttention.forward(
                self, hidden_states, height, width, output_attentions
            )

    model = SegformerEncoderBlock10AttentionSelf()
    model.eval()
    x = (torch.randn(batch_size, 4096, 64), 64, 64)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_2_0_attention_self(target, batch_size) -> None:
    class SegformerEncoderBlock20AttentionSelf(SegformerEfficientSelfAttention):
        def __init__(self) -> None:
            super().__init__(160, 5, 2)

        def forward(
            self,
            hidden_states: torch.Tensor,
            height: int,
            width: int,
            output_attentions: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
            return SegformerEfficientSelfAttention.forward(
                self, hidden_states, height, width, output_attentions
            )

    model = SegformerEncoderBlock20AttentionSelf()
    model.eval()
    x = (torch.randn(batch_size, 1024, 160), 32, 32)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_3_0_attention_self(target, batch_size) -> None:
    class SegformerEncoderBlock30AttentionSelf(SegformerEfficientSelfAttention):
        def __init__(self) -> None:
            super().__init__(256, 8, 1)

        def forward(
            self,
            hidden_states: torch.Tensor,
            height: int,
            width: int,
            output_attentions: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
            return SegformerEfficientSelfAttention.forward(
                self, hidden_states, height, width, output_attentions
            )

    model = SegformerEncoderBlock30AttentionSelf()
    model.eval()
    x = (torch.randn(batch_size, 256, 256), 16, 16)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_0_0_attention_output(target, batch_size) -> None:
    class SegformerEncoderBlock00AttentionOutput(SegformerSelfOutput):
        def __init__(self) -> None:
            super().__init__(32)

        def forward(
            self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
        ) -> torch.Tensor:
            return SegformerSelfOutput.forward(self, hidden_states, input_tensor)

    model = SegformerEncoderBlock00AttentionOutput()
    model.eval()
    x = (torch.randn(batch_size, 16384, 32), torch.randn(batch_size, 16384, 32))
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_1_0_attention_output(target, batch_size) -> None:
    class SegformerEncoderBlock10AttentionOutput(SegformerSelfOutput):
        def __init__(self) -> None:
            super().__init__(64)

        def forward(
            self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
        ) -> torch.Tensor:
            return SegformerSelfOutput.forward(self, hidden_states, input_tensor)

    model = SegformerEncoderBlock10AttentionOutput()
    model.eval()
    x = (torch.randn(batch_size, 4096, 64), torch.randn(batch_size, 4096, 64))
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_2_0_attention_output(target, batch_size) -> None:
    class SegformerEncoderBlock20AttentionOutput(SegformerSelfOutput):
        def __init__(self) -> None:
            super().__init__(160)

        def forward(
            self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
        ) -> torch.Tensor:
            return SegformerSelfOutput.forward(self, hidden_states, input_tensor)

    model = SegformerEncoderBlock20AttentionOutput()
    model.eval()
    x = (torch.randn(batch_size, 1024, 160), torch.randn(batch_size, 1024, 160))
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_3_0_attention_output(target, batch_size) -> None:
    class SegformerEncoderBlock30AttentionOutput(SegformerSelfOutput):
        def __init__(self) -> None:
            super().__init__(256)

        def forward(
            self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
        ) -> torch.Tensor:
            return SegformerSelfOutput.forward(self, hidden_states, input_tensor)

    model = SegformerEncoderBlock30AttentionOutput()
    model.eval()
    x = (torch.randn(batch_size, 256, 256), torch.randn(batch_size, 256, 256))
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_0_0_attention(target, batch_size) -> None:
    class SegformerEncoderBlock00Attention(SegformerAttention):
        def __init__(self) -> None:
            super().__init__(32, 1, 8)

        def forward(
            self,
            hidden_states: torch.Tensor,
            height: int,
            width: int,
            output_attentions: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
            return SegformerAttention.forward(
                self, hidden_states, height, width, output_attentions
            )

    model = SegformerEncoderBlock00Attention()
    model.eval()
    x = (torch.randn(batch_size, 16384, 32), 128, 128)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_1_0_attention(target, batch_size) -> None:
    class SegformerEncoderBlock10Attention(SegformerAttention):
        def __init__(self) -> None:
            super().__init__(64, 2, 4)

        def forward(
            self,
            hidden_states: torch.Tensor,
            height: int,
            width: int,
            output_attentions: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
            return SegformerAttention.forward(
                self, hidden_states, height, width, output_attentions
            )

    model = SegformerEncoderBlock10Attention()
    model.eval()
    x = (torch.randn(batch_size, 4096, 64), 64, 64)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_2_0_attention(target, batch_size) -> None:
    class SegformerEncoderBlock20Attention(SegformerAttention):
        def __init__(self) -> None:
            super().__init__(160, 5, 2)

        def forward(
            self,
            hidden_states: torch.Tensor,
            height: int,
            width: int,
            output_attentions: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
            return SegformerAttention.forward(
                self, hidden_states, height, width, output_attentions
            )

    model = SegformerEncoderBlock20Attention()
    model.eval()
    x = (torch.randn(batch_size, 1024, 160), 32, 32)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_3_0_attention(target, batch_size) -> None:
    class SegformerEncoderBlock30Attention(SegformerAttention):
        def __init__(self) -> None:
            super().__init__(256, 8, 1)

        def forward(
            self,
            hidden_states: torch.Tensor,
            height: int,
            width: int,
            output_attentions: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
            return SegformerAttention.forward(
                self, hidden_states, height, width, output_attentions
            )

    model = SegformerEncoderBlock30Attention()
    model.eval()
    x = (torch.randn(batch_size, 256, 256), 16, 16)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_0_0_drop_path(target, batch_size) -> None:
    class SegformerEncoderBlock00DropPath(SegformerDropPath):
        def __init__(self) -> None:
            super().__init__(0 * (0.1 / 7))

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return SegformerDropPath.forward(self, hidden_states)

    model = SegformerEncoderBlock00DropPath()
    model.eval()
    x = torch.randn(batch_size, 16384, 32)
    check_backend(model, x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_0_1_drop_path(target, batch_size) -> None:
    class SegformerEncoderBlock01DropPath(SegformerDropPath):
        def __init__(self) -> None:
            super().__init__(1 * (0.1 / 7))

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return SegformerDropPath.forward(self, hidden_states)

    model = SegformerEncoderBlock01DropPath()
    model.eval()
    x = torch.randn(batch_size, 16384, 32)
    check_backend(model, x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_1_0_drop_path(target, batch_size) -> None:
    class SegformerEncoderBlock10DropPath(SegformerDropPath):
        def __init__(self) -> None:
            super().__init__(2 * (0.1 / 7))

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return SegformerDropPath.forward(self, hidden_states)

    model = SegformerEncoderBlock10DropPath()
    model.eval()
    x = torch.randn(batch_size, 4096, 64)
    check_backend(model, x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_1_1_drop_path(target, batch_size) -> None:
    class SegformerEncoderBlock11DropPath(SegformerDropPath):
        def __init__(self) -> None:
            super().__init__(3 * (0.1 / 7))

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return SegformerDropPath.forward(self, hidden_states)

    model = SegformerEncoderBlock11DropPath()
    model.eval()
    x = torch.randn(batch_size, 4096, 64)
    check_backend(model, x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_2_0_drop_path(target, batch_size) -> None:
    class SegformerEncoderBlock20DropPath(SegformerDropPath):
        def __init__(self) -> None:
            super().__init__(4 * (0.1 / 7))

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return SegformerDropPath.forward(self, hidden_states)

    model = SegformerEncoderBlock20DropPath()
    model.eval()
    x = torch.randn(batch_size, 1024, 160)
    check_backend(model, x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_2_1_drop_path(target, batch_size) -> None:
    class SegformerEncoderBlock21DropPath(SegformerDropPath):
        def __init__(self) -> None:
            super().__init__(5 * (0.1 / 7))

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return SegformerDropPath.forward(self, hidden_states)

    model = SegformerEncoderBlock21DropPath()
    model.eval()
    x = torch.randn(batch_size, 1024, 160)
    check_backend(model, x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_3_0_drop_path(target, batch_size) -> None:
    class SegformerEncoderBlock30DropPath(SegformerDropPath):
        def __init__(self) -> None:
            super().__init__(6 * (0.1 / 7))

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return SegformerDropPath.forward(self, hidden_states)

    model = SegformerEncoderBlock30DropPath()
    model.eval()
    x = torch.randn(batch_size, 256, 256)
    check_backend(model, x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_3_1_drop_path(target, batch_size) -> None:
    class SegformerEncoderBlock31DropPath(SegformerDropPath):
        def __init__(self) -> None:
            super().__init__(7 * (0.1 / 7))

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return SegformerDropPath.forward(self, hidden_states)

    model = SegformerEncoderBlock31DropPath()
    model.eval()
    x = torch.randn(batch_size, 256, 256)
    check_backend(model, x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_0_0_mlp_dwconv(target, batch_size) -> None:
    class SegformerEncoderBlock00MlpDwconv(SegformerDWConv):
        def __init__(self) -> None:
            super().__init__(128)

        def forward(
            self, hidden_states: torch.Tensor, height: int, width: int
        ) -> torch.Tensor:
            return SegformerDWConv.forward(self, hidden_states, height, width)

    model = SegformerEncoderBlock00MlpDwconv()
    model.eval()
    x = (torch.randn(batch_size, 16384, 128), 128, 128)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_1_0_mlp_dwconv(target, batch_size) -> None:
    class SegformerEncoderBlock10MlpDwconv(SegformerDWConv):
        def __init__(self) -> None:
            super().__init__(256)

        def forward(
            self, hidden_states: torch.Tensor, height: int, width: int
        ) -> torch.Tensor:
            return SegformerDWConv.forward(self, hidden_states, height, width)

    model = SegformerEncoderBlock10MlpDwconv()
    model.eval()
    x = (torch.randn(batch_size, 4096, 256), 64, 64)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_2_0_mlp_dwconv(target, batch_size) -> None:
    class SegformerEncoderBlock20MlpDwconv(SegformerDWConv):
        def __init__(self) -> None:
            super().__init__(640)

        def forward(
            self, hidden_states: torch.Tensor, height: int, width: int
        ) -> torch.Tensor:
            return SegformerDWConv.forward(self, hidden_states, height, width)

    model = SegformerEncoderBlock20MlpDwconv()
    model.eval()
    x = (torch.randn(batch_size, 1024, 640), 32, 32)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_3_0_mlp_dwconv(target, batch_size) -> None:
    class SegformerEncoderBlock30MlpDwconv(SegformerDWConv):
        def __init__(self) -> None:
            super().__init__(1024)

        def forward(
            self, hidden_states: torch.Tensor, height: int, width: int
        ) -> torch.Tensor:
            return SegformerDWConv.forward(self, hidden_states, height, width)

    model = SegformerEncoderBlock30MlpDwconv()
    model.eval()
    x = (torch.randn(batch_size, 256, 1024), 16, 16)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_0_0_mlp(target, batch_size) -> None:
    class SegformerEncoderBlock00Mlp(SegformerMixFFN):
        def __init__(self) -> None:
            super().__init__(32, 128)

        def forward(
            self, hidden_states: torch.Tensor, height: int, width: int
        ) -> torch.Tensor:
            return SegformerMixFFN.forward(self, hidden_states, height, width)

    model = SegformerEncoderBlock00Mlp()
    model.eval()
    x = (torch.randn(batch_size, 16384, 32), 128, 128)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_1_0_mlp(target, batch_size) -> None:
    class SegformerEncoderBlock10Mlp(SegformerMixFFN):
        def __init__(self) -> None:
            super().__init__(64, 256)

        def forward(
            self, hidden_states: torch.Tensor, height: int, width: int
        ) -> torch.Tensor:
            return SegformerMixFFN.forward(self, hidden_states, height, width)

    model = SegformerEncoderBlock10Mlp()
    model.eval()
    x = (torch.randn(batch_size, 4096, 64), 64, 64)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_2_0_mlp(target, batch_size) -> None:
    class SegformerEncoderBlock20Mlp(SegformerMixFFN):
        def __init__(self) -> None:
            super().__init__(160, 640)

        def forward(
            self, hidden_states: torch.Tensor, height: int, width: int
        ) -> torch.Tensor:
            return SegformerMixFFN.forward(self, hidden_states, height, width)

    model = SegformerEncoderBlock20Mlp()
    model.eval()
    x = (torch.randn(batch_size, 1024, 160), 32, 32)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_3_0_mlp(target, batch_size) -> None:
    class SegformerEncoderBlock30Mlp(SegformerMixFFN):
        def __init__(self) -> None:
            super().__init__(256, 1024)

        def forward(
            self, hidden_states: torch.Tensor, height: int, width: int
        ) -> torch.Tensor:
            return SegformerMixFFN.forward(self, hidden_states, height, width)

    model = SegformerEncoderBlock30Mlp()
    model.eval()
    x = (torch.randn(batch_size, 256, 256), 16, 16)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_0_0(target, batch_size) -> None:
    class SegformerEncoderBlock00(SegformerLayer):
        def __init__(self) -> None:
            super().__init__(32, 1, 0 * (0.1 / 7), 8, 4)

        def forward(
            self,
            hidden_states: torch.Tensor,
            height: int,
            width: int,
            output_attentions: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
            return SegformerLayer.forward(
                self, hidden_states, height, width, output_attentions
            )

    model = SegformerEncoderBlock00()
    model.eval()
    x = (torch.randn(batch_size, 16384, 32), 128, 128)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_0_1(target, batch_size) -> None:
    class SegformerEncoderBlock01(SegformerLayer):
        def __init__(self) -> None:
            super().__init__(32, 1, 1 * (0.1 / 7), 8, 4)

        def forward(
            self,
            hidden_states: torch.Tensor,
            height: int,
            width: int,
            output_attentions: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
            return SegformerLayer.forward(
                self, hidden_states, height, width, output_attentions
            )

    model = SegformerEncoderBlock01()
    model.eval()
    x = (torch.randn(batch_size, 16384, 32), 128, 128)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_1_0(target, batch_size) -> None:
    class SegformerEncoderBlock10(SegformerLayer):
        def __init__(self) -> None:
            super().__init__(64, 2, 2 * (0.1 / 7), 4, 4)

        def forward(
            self,
            hidden_states: torch.Tensor,
            height: int,
            width: int,
            output_attentions: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
            return SegformerLayer.forward(
                self, hidden_states, height, width, output_attentions
            )

    model = SegformerEncoderBlock10()
    model.eval()
    x = (torch.randn(batch_size, 4096, 64), 64, 64)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_1_1(target, batch_size) -> None:
    class SegformerEncoderBlock11(SegformerLayer):
        def __init__(self) -> None:
            super().__init__(64, 2, 3 * (0.1 / 7), 4, 4)

        def forward(
            self,
            hidden_states: torch.Tensor,
            height: int,
            width: int,
            output_attentions: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
            return SegformerLayer.forward(
                self, hidden_states, height, width, output_attentions
            )

    model = SegformerEncoderBlock11()
    model.eval()
    x = (torch.randn(batch_size, 4096, 64), 64, 64)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_2_0(target, batch_size) -> None:
    class SegformerEncoderBlock20(SegformerLayer):
        def __init__(self) -> None:
            super().__init__(160, 5, 4 * (0.1 / 7), 2, 4)

        def forward(
            self,
            hidden_states: torch.Tensor,
            height: int,
            width: int,
            output_attentions: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
            return SegformerLayer.forward(
                self, hidden_states, height, width, output_attentions
            )

    model = SegformerEncoderBlock20()
    model.eval()
    x = (torch.randn(batch_size, 1024, 160), 32, 32)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_2_1(target, batch_size) -> None:
    class SegformerEncoderBlock21(SegformerLayer):
        def __init__(self) -> None:
            super().__init__(160, 5, 5 * (0.1 / 7), 2, 4)

        def forward(
            self,
            hidden_states: torch.Tensor,
            height: int,
            width: int,
            output_attentions: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
            return SegformerLayer.forward(
                self, hidden_states, height, width, output_attentions
            )

    model = SegformerEncoderBlock21()
    model.eval()
    x = (torch.randn(batch_size, 1024, 160), 32, 32)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_3_0(target, batch_size) -> None:
    class SegformerEncoderBlock30(SegformerLayer):
        def __init__(self) -> None:
            super().__init__(256, 8, 6 * (0.1 / 7), 1, 4)

        def forward(
            self,
            hidden_states: torch.Tensor,
            height: int,
            width: int,
            output_attentions: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
            return SegformerLayer.forward(
                self, hidden_states, height, width, output_attentions
            )

    model = SegformerEncoderBlock30()
    model.eval()
    x = (torch.randn(batch_size, 256, 256), 16, 16)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_segformer_encoder_block_3_1(target, batch_size) -> None:
    class SegformerEncoderBlock31(SegformerLayer):
        def __init__(self) -> None:
            super().__init__(256, 8, 7 * (0.1 / 7), 1, 4)

        def forward(
            self,
            hidden_states: torch.Tensor,
            height: int,
            width: int,
            output_attentions: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
            return SegformerLayer.forward(
                self, hidden_states, height, width, output_attentions
            )

    model = SegformerEncoderBlock31()
    model.eval()
    x = (torch.randn(batch_size, 256, 256), 16, 16)
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.skip(reason="This test is too slow to run in CI")
def test_segformer_encoder(target) -> None:
    model, x = BENCHMARKS["segformer.encoder"]()
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.skip(reason="This test is too slow to run in CI")
def test_segformer(target) -> None:
    model, x = BENCHMARKS["segformer"]()
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_decode_head_linear_c_0(target, batch_size) -> None:
    class DecodeHeadLinearC0(SegformerMLP):
        def __init__(self) -> None:
            super().__init__(32)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return SegformerMLP.forward(self, hidden_states)

    model = DecodeHeadLinearC0()
    model.eval()
    x = torch.randn(batch_size, 32, 128, 128)
    check_backend(model, x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_decode_head_linear_c_1(target, batch_size) -> None:
    class DecodeHeadLinearC1(SegformerMLP):
        def __init__(self) -> None:
            super().__init__(64)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return SegformerMLP.forward(self, hidden_states)

    model = DecodeHeadLinearC1()
    model.eval()
    x = torch.randn(batch_size, 64, 64, 64)
    check_backend(model, x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_decode_head_linear_c_2(target, batch_size) -> None:
    class DecodeHeadLinearC2(SegformerMLP):
        def __init__(self) -> None:
            super().__init__(160)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return SegformerMLP.forward(self, hidden_states)

    model = DecodeHeadLinearC2()
    model.eval()
    x = torch.randn(batch_size, 160, 32, 32)
    check_backend(model, x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_decode_head_linear_c_3(target, batch_size) -> None:
    class DecodeHeadLinearC3(SegformerMLP):
        def __init__(self) -> None:
            super().__init__(256)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return SegformerMLP.forward(self, hidden_states)

    model = DecodeHeadLinearC3()
    model.eval()
    x = torch.randn(batch_size, 256, 16, 16)
    check_backend(model, x, target=target)


@pytest.mark.parametrize("target", TARGETS)
def test_decode_head(target) -> None:
    model, x = BENCHMARKS["decode_head"]()
    check_backend(model, *x, target=target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.skip(reason="This test is too slow to run in CI")
def test_all(target) -> None:
    model, x = BENCHMARKS["all"]()
    check_backend(model, x, target=target)
