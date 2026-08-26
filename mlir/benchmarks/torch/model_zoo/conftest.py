import pytest

import docc.torch
import torch


@pytest.fixture(autouse=True)
def seed_rng():
    """Set a fixed random seed before each test for reproducibility."""
    torch.manual_seed(815)
