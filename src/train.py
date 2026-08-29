"""Train the CIFAR-10 classifier and emit machine-readable metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from src.dataset import get_dataloaders
from src.model import get_model


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * inputs.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss_sum += criterion(outputs, targets).item() * inputs.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)
    return loss_sum / total, correct / total


def resolve_config_path(cli_path: str | None) -> Path:
    """Select explicit CLI config, mounted/env config, or the local default."""
    candidates = [
        cli_path,
        os.environ.get("TRAINING_CONFIG"),
        "/app/configs/training_config.yaml",
        "configs/training_config.yaml",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("No training config found; use --config or TRAINING_CONFIG.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 classifier.")
    parser.add_argument("--config", help="Path to a YAML training configuration.")
    arguments = parser.parse_args()
    config = load_config(resolve_config_path(arguments.config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(**config["model"]).to(device)
    train_loader, validation_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = Path(config["output"]["checkpoint_dir"]) / config["output"]["model_name"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_validation_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]
    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
        validation_loss, validation_accuracy = evaluate(model, validation_loader, criterion, device)
        print(json.dumps({
            "event": "epoch_complete", "epoch": epoch, "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_accuracy, 4), "val_loss": round(validation_loss, 4),
            "val_accuracy": round(validation_accuracy, 4),
        }), flush=True)
        if validation_loss < best_validation_loss:
            best_validation_loss, patience_counter = validation_loss, 0
            torch.save({
                "architecture": config["model"]["architecture"],
                "num_classes": config["model"]["num_classes"],
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": validation_loss,
                "val_accuracy": validation_accuracy,
            }, checkpoint_path)
            print(json.dumps({"event": "checkpoint_saved", "path": str(checkpoint_path)}), flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(json.dumps({"event": "early_stopping", "epoch": epoch}), flush=True)
                break
    print(json.dumps({"event": "training_complete", "best_val_loss": round(best_validation_loss, 4)}), flush=True)


if __name__ == "__main__":
    main()
