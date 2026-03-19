import pytest
import torch
import docc.torch


@pytest.fixture(autouse=True)
def seed_rng():
    """Set a fixed random seed before each test for reproducibility."""
    torch.manual_seed(815)


@pytest.fixture(autouse=True, scope="function")
def clear_global_state():
    yield None
    docc.torch.reset_backend_options()
