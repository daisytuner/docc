import pytest

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import docc.torch

docc.torch.set_backend_options(target="none", category="server")


def test_inference():
    class LinearRegression(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear1 = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear1(x)

    # ---- Dummy Dataset ----
    class DummyDataset(Dataset):
        def __init__(self, n_samples=10, dim=4):
            self.data = torch.randn(n_samples, dim)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # ---- Create dataset & dataloader ----
    dataset = DummyDataset(n_samples=8, dim=4)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # ---- Model ----
    model = LinearRegression(in_features=4, out_features=1).eval()

    model_ref = LinearRegression(in_features=4, out_features=1).eval()
    model_ref.load_state_dict(
        model.state_dict()
    )  # Ensure both models have the same weights

    model = torch.compile(model, backend="docc")
    model_ref = torch.compile(model_ref)  # Compile reference model with default backend

    # ---- Inference loop ----
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            output = model(batch)
            print(f"Batch {batch_idx}")
            print("Input shape :", batch.shape)
            print("Output shape:", output.shape)
            print("-" * 30)
            # Verify output
            output_ref = model_ref(batch)
            assert torch.allclose(output, output_ref, rtol=1e-5)


@pytest.mark.skip()
def test_training():
    class LinearRegression(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear1 = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear1(x)

    # ---- Dummy Dataset ----
    class DummyDataset(Dataset):
        def __init__(self, n_samples=1, in_dim=4):
            self.x = torch.randn(n_samples, in_dim)
            self.y = torch.randn(n_samples, 1)

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    # ---- Setup ----
    batch_size = 1
    lr = 0.01
    epochs = 5

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LinearRegression(in_features=4, out_features=1)
    model = torch.compile(model, backend="docc", mode="default")

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # ---- Training Loop ----
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()

            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
