"""FastAPI service for CIFAR-10 model inference."""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from src.dataset import CIFAR10_MEAN, CIFAR10_STD
from src.model import get_model

CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
MODEL: torch.nn.Module | None = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREPROCESS = transforms.Compose((
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
))


def load_model(checkpoint_path: str | Path) -> torch.nn.Module:
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = get_model(checkpoint.get("architecture", "resnet18"), checkpoint.get("num_classes", len(CLASS_NAMES)))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(DEVICE).eval()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL
    checkpoint = Path(os.environ.get("MODEL_PATH", "/app/checkpoints/classifier_v1.pt"))
    if checkpoint.is_file():
        MODEL = load_model(checkpoint)
    yield
    MODEL = None


app = FastAPI(title="CIFAR-10 Classifier", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model checkpoint is not loaded")
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, object]:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model checkpoint is not loaded")
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, or WebP image")
    try:
        rgb_image = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception as error:
        raise HTTPException(status_code=400, detail="Unable to decode image") from error
    with torch.inference_mode():
        output = MODEL(PREPROCESS(rgb_image).unsqueeze(0).to(DEVICE))
        probabilities = torch.softmax(output, dim=1)[0].cpu().tolist()
    predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "probabilities": dict(zip(CLASS_NAMES, probabilities)),
    }
