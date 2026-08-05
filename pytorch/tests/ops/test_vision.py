import torch
import torch.nn as nn
import torch.nn.functional as F

from tests import check

# ---------------------------------------------------------------------------
# aten::upsample_bilinear2d.vec(
#     Tensor input, SymInt[]? output_size, bool align_corners,
#     float[]? scale_factors) -> Tensor
#
# The tests below drive the operator through two entry points:
#   * an explicit output ``size``      -> output_size set, scale_factors None
#   * an explicit ``scale_factor``     -> output_size None, scale_factors set
# and exercise both ``align_corners`` settings, up-/down-sampling, integer and
# non-integer ratios, symmetric and asymmetric spatial dimensions as well as a
# variety of input shapes (batch, channels, height, width).
# ---------------------------------------------------------------------------


# --- Upsample by explicit output size, align_corners=False ---


def test_upsample_bilinear_size_upscale(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(32, 32), mode="bilinear", align_corners=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_size_downscale(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(8, 8), mode="bilinear", align_corners=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_size_asymmetric(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(24, 12), mode="bilinear", align_corners=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_size_non_integer_ratio(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(23, 19), mode="bilinear", align_corners=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(1, 4, 16, 16), target=target)


def test_upsample_bilinear_size_identity(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(16, 16), mode="bilinear", align_corners=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


# --- Upsample by explicit output size, align_corners=True ---


def test_upsample_bilinear_size_upscale_align_corners(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(32, 32), mode="bilinear", align_corners=True)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_size_downscale_align_corners(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(8, 8), mode="bilinear", align_corners=True)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_size_asymmetric_align_corners(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(30, 10), mode="bilinear", align_corners=True)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_size_non_integer_ratio_align_corners(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(21, 27), mode="bilinear", align_corners=True)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(1, 4, 16, 16), target=target)


# --- Upsample by scale factor, align_corners=False ---


def test_upsample_bilinear_scale_2x(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(
                scale_factor=2.0, mode="bilinear", align_corners=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_scale_half(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(
                scale_factor=0.5, mode="bilinear", align_corners=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_scale_non_integer(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(
                scale_factor=1.5, mode="bilinear", align_corners=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_scale_asymmetric(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(
                scale_factor=(2.0, 3.0), mode="bilinear", align_corners=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


# --- Upsample by scale factor, align_corners=True ---


def test_upsample_bilinear_scale_2x_align_corners(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_scale_non_integer_align_corners(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(scale_factor=2.5, mode="bilinear", align_corners=True)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_scale_asymmetric_align_corners(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(
                scale_factor=(3.0, 2.0), mode="bilinear", align_corners=True
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


# --- Different input shapes (batch / channels / rectangular inputs) ---


def test_upsample_bilinear_single_batch_single_channel(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(20, 20), mode="bilinear", align_corners=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(1, 1, 8, 8), target=target)


def test_upsample_bilinear_many_channels(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(
                scale_factor=2.0, mode="bilinear", align_corners=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 32, 8, 8), target=target)


def test_upsample_bilinear_rectangular_input(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(16, 40), mode="bilinear", align_corners=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 8, 20), target=target)


def test_upsample_bilinear_rectangular_input_align_corners(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Upsample(size=(40, 16), mode="bilinear", align_corners=True)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.up(input)

    check(Net(), torch.randn(2, 3, 20, 8), target=target)


# --- Functional interface ---


def test_upsample_bilinear_functional_size(target: str) -> None:
    class Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.interpolate(
                input, size=(28, 28), mode="bilinear", align_corners=False
            )

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


def test_upsample_bilinear_functional_scale(target: str) -> None:
    class Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.interpolate(
                input, scale_factor=2.0, mode="bilinear", align_corners=True
            )

    check(Net(), torch.randn(2, 3, 16, 16), target=target)


# ---------------------------------------------------------------------------
# Combined multi-node integration tests.
#
# These progressively assemble a vision pipeline
# (reshape -> upsample -> cat -> 1x1 conv -> batch-norm -> relu -> 1x1 conv)
# at small sizes. They isolate layout-sensitive interactions between the
# upsample node and its neighbours, where a tensor produced in one physical
# layout is consumed assuming another.
# ---------------------------------------------------------------------------


def test_upsample_then_conv1x1(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=1, bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            x = F.interpolate(
                input, size=(16, 16), mode="bilinear", align_corners=False
            )
            return self.conv(x)

    check(Net().eval(), torch.randn(2, 3, 8, 8), target=target)


def test_reshape_then_upsample(target: str) -> None:
    class Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            b, c, h, w = input.shape
            t = input.flatten(2).transpose(1, 2)
            t = t.permute(0, 2, 1).reshape(b, c, h, w)
            return F.interpolate(t, size=(16, 16), mode="bilinear", align_corners=False)

    check(Net().eval(), torch.randn(2, 4, 8, 8), target=target)


def test_upsample_cat(target: str) -> None:
    class Net(nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a = F.interpolate(a, size=(16, 16), mode="bilinear", align_corners=False)
            b = F.interpolate(b, size=(16, 16), mode="bilinear", align_corners=False)
            return torch.cat([a, b], dim=1)

    check(
        Net().eval(),
        torch.randn(2, 3, 16, 16),
        torch.randn(2, 5, 8, 8),
        target=target,
    )


def test_upsample_cat_conv(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fuse = nn.Conv2d(8, 6, kernel_size=1, bias=False)

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a = F.interpolate(a, size=(16, 16), mode="bilinear", align_corners=False)
            b = F.interpolate(b, size=(16, 16), mode="bilinear", align_corners=False)
            return self.fuse(torch.cat([b, a], dim=1))

    check(
        Net().eval(),
        torch.randn(2, 3, 16, 16),
        torch.randn(2, 5, 8, 8),
        target=target,
    )


def test_conv_bn_relu_conv(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fuse = nn.Conv2d(8, 6, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(6)
            self.act = nn.ReLU()
            self.classifier = nn.Conv2d(6, 4, kernel_size=1)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.act(self.bn(self.fuse(input))))

    check(Net().eval(), torch.randn(2, 8, 16, 16), target=target)


def test_mini_decode_head(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fuse = nn.Conv2d(8, 6, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(6)
            self.act = nn.ReLU()
            self.classifier = nn.Conv2d(6, 4, kernel_size=1)

        def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            a, b = inputs
            size = a.shape[2:]
            ua = F.interpolate(a, size=size, mode="bilinear", align_corners=False)
            ub = F.interpolate(b, size=size, mode="bilinear", align_corners=False)
            fused = self.fuse(torch.cat([ub, ua], dim=1))
            x = self.act(self.bn(fused))
            return self.classifier(x)

    a = torch.randn(2, 3, 16, 16)
    b = torch.randn(2, 5, 8, 8)
    check(Net().eval(), (a, b), target=target)


def test_mlp_reshape_upsample(target: str) -> None:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 8)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            batch_size, _, height, width = input.shape
            t = self.proj(input.flatten(2).transpose(1, 2))
            t = t.permute(0, 2, 1).reshape(batch_size, -1, height, width)
            return F.interpolate(t, size=(16, 16), mode="bilinear", align_corners=False)

    check(Net().eval(), torch.randn(2, 4, 8, 8), target=target)


def test_decode_head_scaled(target: str) -> None:
    class MLP(nn.Module):
        def __init__(self, input_dim: int, hidden: int) -> None:
            super().__init__()
            self.proj = nn.Linear(input_dim, hidden)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x.flatten(2).transpose(1, 2))

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = 8
            self.linear_c = nn.ModuleList(
                [
                    MLP(4, hidden),
                    MLP(8, hidden),
                    MLP(16, hidden),
                    MLP(32, hidden),
                ]
            )
            self.fuse = nn.Conv2d(4 * hidden, 8, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(8)
            self.act = nn.ReLU()
            self.classifier = nn.Conv2d(8, 5, kernel_size=1)

        def forward(
            self,
            inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> torch.Tensor:
            batch_size = inputs[-1].shape[0]
            size = inputs[0].shape[2:]
            all_hidden_states = ()
            for state, mlp in zip(inputs, self.linear_c):
                height, width = state.shape[2], state.shape[3]
                state = mlp(state)
                state = state.permute(0, 2, 1).reshape(batch_size, -1, height, width)
                state = F.interpolate(
                    state, size=size, mode="bilinear", align_corners=False
                )
                all_hidden_states += (state,)
            fused = self.fuse(torch.cat(all_hidden_states[::-1], dim=1))
            x = self.act(self.bn(fused))
            return self.classifier(x)

    inputs = (
        torch.randn(2, 4, 16, 16),
        torch.randn(2, 8, 8, 8),
        torch.randn(2, 16, 4, 4),
        torch.randn(2, 32, 2, 2),
    )
    check(Net().eval(), inputs, target=target)
