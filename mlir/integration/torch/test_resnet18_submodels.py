import copy

import torch
import torch.nn as nn
import pytest

import docc.torch


# ---------------------------------------------------------------------------
# Submodels matching ResNet18-v2 (pre-activation) substructures
# ---------------------------------------------------------------------------


class BNReluConv(nn.Module):
    """BN -> ReLU -> Conv(3x3): smallest repeating unit."""

    def __init__(self, in_ch=64, out_ch=64, stride=1, padding=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=padding, bias=False
        )

    def forward(self, x):
        return self.conv(self.relu(self.bn(x)))


class BasicBlock(nn.Module):
    """Basic residual block (no downsampling): BN-ReLU-Conv-BN-ReLU-Conv + skip."""

    def __init__(self, ch=64):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(ch)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)

    def forward(self, x):
        identity = x
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        return out + identity


class DownsampleBlock(nn.Module):
    """Downsample block: main path (3x3 s2 + 3x3 s1) with 1x1 shortcut."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, 2, 0, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)
        main = self.conv2(self.relu2(self.bn2(self.conv1(x))))
        return main + self.shortcut(x)


class Stage1(nn.Module):
    """Stage 1: two basic blocks (64 channels, no downsampling)."""

    def __init__(self):
        super().__init__()
        self.block1 = BasicBlock(64)
        self.block2 = BasicBlock(64)

    def forward(self, x):
        return self.block2(self.block1(x))


class Stage2(nn.Module):
    """Stage 2: downsample block (64->128) + basic block (128)."""

    def __init__(self):
        super().__init__()
        self.down = DownsampleBlock(64, 128)
        self.block = BasicBlock(128)

    def forward(self, x):
        return self.block(self.down(x))


class StemPlusStage1(nn.Module):
    """Stem (BN-Conv7x7-BN-ReLU-MaxPool) followed by stage 1."""

    def __init__(self):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(3)
        self.conv0 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.stage1 = Stage1()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv0(self.bn0(x)))))
        return self.stage1(x)


class GlobalAvgPoolFC(nn.Module):
    """AvgPool -> Flatten -> FC classifier head."""

    def __init__(self, in_ch=512, num_classes=1000):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        return self.fc(torch.flatten(self.avgpool(x), 1))


class Tail(nn.Module):
    """BN -> ReLU -> AvgPool -> Flatten -> FC."""

    def __init__(self, in_ch=512, num_classes=1000):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        return self.fc(torch.flatten(self.avgpool(self.relu(self.bn(x))), 1))


# ---------------------------------------------------------------------------
# Test matrix: (name, model_factory, input_shape)
# ---------------------------------------------------------------------------

SUBMODELS = [
    ("bn_relu_conv", lambda: BNReluConv(64, 64), (1, 64, 56, 56)),
    ("basic_block", lambda: BasicBlock(64), (1, 64, 56, 56)),
    ("downsample_64_128", lambda: DownsampleBlock(64, 128), (1, 64, 56, 56)),
    ("stage1", lambda: Stage1(), (1, 64, 56, 56)),
    ("stage2", lambda: Stage2(), (1, 64, 56, 56)),
    ("stem_stage1", lambda: StemPlusStage1(), (1, 3, 224, 224)),
    ("avgpool_fc", lambda: GlobalAvgPoolFC(512, 1000), (1, 512, 7, 7)),
    ("tail", lambda: Tail(512, 1000), (1, 512, 7, 7)),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,model_fn,shape",
    SUBMODELS,
    ids=[s[0] for s in SUBMODELS],
)
def test_resnet18_submodel_backend(name, model_fn, shape):
    """docc backend (target='none') matches PyTorch eager for each substructure."""
    torch._dynamo.reset()

    model = model_fn()
    model.eval()
    x = torch.randn(*shape)
    model_ref = copy.deepcopy(model)

    docc.torch.set_backend_options(target="none", category="server")
    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(x)
        res_ref = model_ref(x)

    assert torch.allclose(res, res_ref, rtol=1e-3, atol=1e-5), (
        f"[{name}] max abs diff: {(res - res_ref).abs().max().item():.6e}, "
        f"max rel diff: {((res - res_ref).abs() / (res_ref.abs() + 1e-8)).max().item():.6e}"
    )
