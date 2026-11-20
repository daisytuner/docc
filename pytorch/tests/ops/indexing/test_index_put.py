import torch
import torch.nn as nn

from tests import check


def test_static_cache_update(target: str) -> None:
    """A static KV cache stores the keys of the current tokens at their positions (index_copy_ along the tokens)."""

    class CacheUpdateNet(nn.Module):
        def forward(
            self, cache: torch.Tensor, positions: torch.Tensor, keys: torch.Tensor
        ) -> torch.Tensor:
            updated = cache.clone()
            updated.index_copy_(2, positions, keys)
            return updated

    check(
        CacheUpdateNet(),
        torch.randn(1, 2, 16, 8),
        torch.tensor([5, 9]),
        torch.randn(1, 2, 2, 8),
        target=target,
    )


def test_first_dim(target: str) -> None:
    class RowUpdateNet(nn.Module):
        def forward(
            self, x: torch.Tensor, rows: torch.Tensor, values: torch.Tensor
        ) -> torch.Tensor:
            return torch.index_put(x, (rows,), values)

    check(
        RowUpdateNet(),
        torch.randn(6, 4),
        torch.tensor([4, 1]),
        torch.randn(2, 4),
        target=target,
    )


def test_buffers_updated_in_place_and_a_result(target: str) -> None:
    """A static KV cache: two buffers updated in place at a position, and a result computed from them."""

    class KVCacheNet(nn.Module):
        # register_buffer leaves the attributes typed as Tensor | Module.
        keys: torch.Tensor
        values: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("keys", torch.zeros(1, 2, 16, 8))
            self.register_buffer("values", torch.zeros(1, 2, 16, 8))

        def forward(
            self, positions: torch.Tensor, k: torch.Tensor, v: torch.Tensor
        ) -> torch.Tensor:
            self.keys.index_copy_(2, positions, k)
            self.values.index_copy_(2, positions, v)
            return (self.keys * self.values).sum(dim=2)

    check(
        KVCacheNet(),
        torch.tensor([3]),
        torch.randn(1, 2, 1, 8),
        torch.randn(1, 2, 1, 8),
        target=target,
    )


def _updated(model, *inputs, target):
    """``model``'s result on ``inputs``, eager and compiled by docc, and the tensors each updated in place: the
    inputs and the model's buffers."""
    import copy

    eager_model, eager_inputs = copy.deepcopy(model), copy.deepcopy(inputs)
    program = torch.compile(
        model, backend="docc", dynamic=False, options={"target": target}
    )
    with torch.no_grad():
        expected = eager_model(*eager_inputs)
        actual = program(*inputs)
    return (
        (actual, list(inputs) + list(model.buffers())),
        (expected, list(eager_inputs) + list(eager_model.buffers())),
    )


def test_inputs_updated_in_place(target: str) -> None:
    """A KV cache passed in, as Hugging Face's StaticCache passes its tensors: updated in place at a position, and read
    updated."""

    class CacheNet(nn.Module):
        def forward(self, keys, positions, k):
            keys.index_copy_(2, positions, k)
            return keys.sum(dim=2)

    (actual, updated), (expected, reference) = _updated(
        CacheNet(),
        torch.randn(1, 2, 16, 8),
        torch.tensor([3]),
        torch.randn(1, 2, 1, 8),
        target=target,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(updated[0], reference[0])


def test_buffers_keep_their_updates(target: str) -> None:
    class KVCacheNet(nn.Module):
        keys: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("keys", torch.zeros(1, 2, 16, 8))

        def forward(self, positions, k):
            self.keys.index_copy_(2, positions, k)
            return self.keys.sum(dim=2)

    (actual, updated), (expected, reference) = _updated(
        KVCacheNet(), torch.tensor([3]), torch.randn(1, 2, 1, 8), target=target
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(updated[-1], reference[-1])


def test_other_updates_are_written_back(target: str) -> None:
    """An input updated by more than a write at positions, which the program does not compute in place."""

    class ScaleNet(nn.Module):
        def forward(self, x):
            x.mul_(2.0)
            return x + 1.0

    (actual, updated), (expected, reference) = _updated(
        ScaleNet(), torch.randn(4, 8), target=target
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(updated[0], reference[0])
