from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from torchvision import models
from torchvision.models import EfficientNet_B3_Weights


# ============================================================
# DERMAVISION - EFFICIENTNET-B3 CLASSIFIER
# ============================================================
#
# Pipeline:
#
# Image
#   ↓
# EfficientNet-B3
#   ↓
# 1536 features
#   ↓
# Linear(1536 → 512)
#   ↓
# ReLU
#   ↓
# Dropout
#   ↓
# Linear(512 → 3)
#   ↓
# Prediction
#
# The architecture is based on the supplied checkpoint:
#
# classifier.1.weight -> (512, 1536)
# classifier.4.weight -> (3, 512)
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BASE_DIR.parent

MODEL_PATH = (
    PROJECT_DIR
    / "EfficientNet"
    / "Skin-imperfection-9"
    / "best_efficientnet_b3_skin_tone_best_epoch_1.pth"
)

# Class names are NOT stored in the checkpoint.
#
# Create:
#
# RAG ASSISTANT/
# └── knowledge_base/
#     └── classes.json
#
CLASSES_PATH = (
    BASE_DIR
    / "knowledge_base"
    / "classes.json"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMAGE SIZE
# ============================================================

IMAGE_SIZE = 300


# ============================================================
# IMAGE TRANSFORM
# ============================================================
#
# EfficientNet-B3 standard ImageNet normalization.
#
# IMPORTANT:
# If your original training notebook used different
# normalization/augmentation, use that exact preprocessing
# for production inference.
#
# ============================================================

weights = EfficientNet_B3_Weights.DEFAULT

transform = weights.transforms()


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names() -> List[str]:

    if not CLASSES_PATH.exists():

        raise FileNotFoundError(
            f"""
classes.json was not found.

Expected location:

{CLASSES_PATH}

Your checkpoint contains 3 output classes, but the checkpoint
does NOT contain their names.

Create classes.json with the EXACT class order used during
training.

Example format:

[
    "Class_A",
    "Class_B",
    "Class_C"
]

DO NOT guess the class order.
"""
        )

    with open(
        CLASSES_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        classes = json.load(file)

    if not isinstance(classes, list):

        raise ValueError(
            "classes.json must contain a JSON list."
        )

    if len(classes) != 3:

        raise ValueError(
            f"""
The checkpoint has 3 output classes,
but classes.json contains {len(classes)} classes.

Expected exactly 3 classes.
"""
        )

    return classes


# ============================================================
# CREATE EFFICIENTNET-B3 MODEL
# ============================================================

def create_model(
    num_classes: int = 3,
) -> nn.Module:

    # Create EfficientNet-B3 architecture.
    #
    # We don't load ImageNet weights because the supplied
    # checkpoint contains the trained weights.

    model = models.efficientnet_b3(
        weights=None
    )

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # The checkpoint contains:
    #
    # classifier.1.weight -> (512, 1536)
    # classifier.1.bias   -> (512)
    #
    # classifier.4.weight -> (3, 512)
    # classifier.4.bias   -> (3)
    #
    # Therefore the classifier must contain:
    #
    # Dropout
    # Linear(1536, 512)
    # ReLU
    # Dropout
    # Linear(512, 3)
    #
    # --------------------------------------------------------

    model.classifier = nn.Sequential(

        nn.Dropout(
            p=0.4
        ),

        nn.Linear(
            1536,
            512
        ),

        nn.ReLU(
            inplace=True
        ),

        nn.Dropout(
            p=0.3
        ),

        nn.Linear(
            512,
            num_classes
        ),
    )

    return model


# ============================================================
# EXTRACT STATE DICT
# ============================================================

def extract_state_dict(
    checkpoint
):

    # The inspection showed that your checkpoint itself
    # is the state dictionary.
    #
    # Still, this function supports common checkpoint formats.

    if isinstance(
        checkpoint,
        dict
    ):

        possible_keys = [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
        ]

        for key in possible_keys:

            if key in checkpoint:

                candidate = checkpoint[key]

                if isinstance(
                    candidate,
                    dict
                ):

                    return candidate

        # Your checkpoint is expected to reach this branch.

        if all(
            isinstance(
                value,
                torch.Tensor
            )
            for value in checkpoint.values()
        ):

            return checkpoint

    raise ValueError(
        "Could not find model state dictionary."
    )


# ============================================================
# CLEAN STATE DICT
# ============================================================

def clean_state_dict(
    state_dict: Dict
):

    cleaned = {}

    for key, value in state_dict.items():

        # Remove DataParallel prefix if present.

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned[key] = value

    return cleaned


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    print(
        "\nLoading DermaVision EfficientNet-B3..."
    )

    print(
        "Checkpoint:",
        MODEL_PATH
    )

    print(
        "Device:",
        DEVICE
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"""
Model checkpoint not found:

{MODEL_PATH}

Check the filename inside:

C:\\projects\\DERMAVISION\\EfficientNet\\Skin-imperfection-9
"""
        )

    # --------------------------------------------------------
    # Load classes
    # --------------------------------------------------------

    class_names = load_class_names()

    print(
        "Classes:",
        class_names
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # --------------------------------------------------------
    # Extract state dictionary
    # --------------------------------------------------------

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = clean_state_dict(
        state_dict
    )

    # --------------------------------------------------------
    # Create exact architecture
    # --------------------------------------------------------

    model = create_model(
        num_classes=len(class_names)
    )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=True
        )
    )

    # strict=True means that if architecture does not match,
    # the program stops instead of silently producing bad
    # predictions.

    if missing_keys:

        raise RuntimeError(
            f"Missing model keys: {missing_keys}"
        )

    if unexpected_keys:

        raise RuntimeError(
            f"Unexpected model keys: {unexpected_keys}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    print(
        "Model loaded successfully."
    )

    return model, class_names


# ============================================================
# LOAD MODEL ONCE
# ============================================================

MODEL, CLASS_NAMES = load_model()


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(
    image_path: str | Path
):

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )

    return image, tensor


# ============================================================
# PREDICT
# ============================================================

def predict(
    image_path: str | Path
) -> Dict:

    image, tensor = preprocess_image(
        image_path
    )

    MODEL.eval()

    with torch.no_grad():

        logits = MODEL(
            tensor
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )

        confidence, class_index = torch.max(
            probabilities,
            dim=1
        )

    index = int(
        class_index.item()
    )

    confidence_value = float(
        confidence.item()
    )

    predicted_class = CLASS_NAMES[
        index
    ]

    return {

        "predicted_class":
            predicted_class,

        "class_index":
            index,

        "confidence":
            confidence_value,

        "confidence_percent":
            confidence_value * 100.0,

    }


# ============================================================
# TOP-K PREDICTIONS
# ============================================================

def predict_top_k(
    image_path: str | Path,
    k: int = 3
) -> List[Dict]:

    _, tensor = preprocess_image(
        image_path
    )

    MODEL.eval()

    with torch.no_grad():

        logits = MODEL(
            tensor
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )

    k = min(
        k,
        len(CLASS_NAMES)
    )

    values, indices = torch.topk(
        probabilities,
        k=k,
        dim=1
    )

    results = []

    for probability, index in zip(
        values[0],
        indices[0]
    ):

        idx = int(
            index.item()
        )

        probability_value = float(
            probability.item()
        )

        results.append({

            "predicted_class":
                CLASS_NAMES[idx],

            "class_index":
                idx,

            "confidence":
                probability_value,

            "confidence_percent":
                probability_value * 100.0,

        })

    return results


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("DERMAVISION SKIN CLASSIFIER")
    print("=" * 70)

    print(
        "Model: EfficientNet-B3"
    )

    print(
        "Output classes:",
        len(CLASS_NAMES)
    )

    print(
        "Class names:",
        CLASS_NAMES
    )

    print(
        "Device:",
        DEVICE
    )

    print(
        "Checkpoint:",
        MODEL_PATH
    )

    print("=" * 70)

    image_path = input(
        "\nEnter skin image path: "
    ).strip()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    result = predict(
        image_path
    )

    print()
    print("=" * 70)
    print("PREDICTION")
    print("=" * 70)

    print(
        "Predicted class:",
        result["predicted_class"]
    )

    print(
        "Class index:",
        result["class_index"]
    )

    print(
        "Confidence:",
        f"{result['confidence_percent']:.2f}%"
    )

    # --------------------------------------------------------
    # Top predictions
    # --------------------------------------------------------

    print()
    print("TOP PREDICTIONS")
    print("-" * 70)

    top_predictions = predict_top_k(
        image_path,
        k=3
    )

    for rank, item in enumerate(
        top_predictions,
        start=1
    ):

        print(
            f"{rank}. "
            f"{item['predicted_class']} "
            f"({item['confidence_percent']:.2f}%)"
        )

    print()
    print("=" * 70)
    print("CLASSIFICATION COMPLETE")
    print("=" * 70)