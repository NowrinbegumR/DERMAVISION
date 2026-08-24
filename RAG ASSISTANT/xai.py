from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from classifier import (
    MODEL,
    CLASS_NAMES,
    DEVICE,
    transform,
)


# ============================================================
# DERMAVISION - GRAD-CAM
# ============================================================


class GradCAM:
    """
    Grad-CAM implementation for the EfficientNet-B3 classifier.
    """

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self._backward_hook
            )
        )

    # --------------------------------------------------------
    # Forward hook
    # --------------------------------------------------------

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):

        self.activations = output

    # --------------------------------------------------------
    # Backward hook
    # --------------------------------------------------------

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output,
    ):

        self.gradients = grad_output[0]

    # --------------------------------------------------------
    # Generate CAM
    # --------------------------------------------------------

    def generate(
        self,
        image_tensor,
        class_index,
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        output = self.model(
            image_tensor
        )

        target_score = output[
            0,
            class_index
        ]

        target_score.backward()

        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:

            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        activations = self.activations

        gradients = self.gradients

        # ----------------------------------------------------
        # Global average pooling of gradients
        # ----------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ----------------------------------------------------
        # Weighted activation maps
        # ----------------------------------------------------

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        # ReLU
        cam = F.relu(
            cam
        )

        # ----------------------------------------------------
        # Resize CAM to image size
        # ----------------------------------------------------

        cam = F.interpolate(
            cam,
            size=(
                image_tensor.shape[2],
                image_tensor.shape[3],
            ),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[
            0,
            0
        ]

        # ----------------------------------------------------
        # Normalize 0 → 1
        # ----------------------------------------------------

        cam -= cam.min()

        max_value = cam.max()

        if max_value > 0:

            cam /= max_value

        return cam.detach().cpu().numpy()

    # --------------------------------------------------------
    # Remove hooks
    # --------------------------------------------------------

    def close(self):

        self.forward_handle.remove()

        self.backward_handle.remove()


# ============================================================
# HEATMAP CREATION
# ============================================================

def create_heatmap(
    cam: np.ndarray,
) -> Image.Image:

    """
    Convert normalized CAM into an RGB heatmap.

    No external plotting library is required.
    """

    cam = np.clip(
        cam,
        0.0,
        1.0
    )

    values = np.uint8(
        cam * 255
    )

    height, width = values.shape

    heatmap = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    # Simple blue → red representation.

    heatmap[:, :, 0] = values

    heatmap[:, :, 1] = (
        255 - values
    )

    heatmap[:, :, 2] = (
        255 - values
    )

    return Image.fromarray(
        heatmap,
        mode="RGB"
    )


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(
    original: Image.Image,
    heatmap: Image.Image,
    alpha: float = 0.40,
) -> Image.Image:

    original = original.convert(
        "RGB"
    )

    heatmap = heatmap.resize(
        original.size,
        Image.Resampling.BILINEAR,
    )

    overlay = Image.blend(
        original,
        heatmap,
        alpha,
    )

    return overlay


# ============================================================
# EXPLAIN
# ============================================================

def explain(
    image_path: str | Path,
    output_path: str | Path | None = None,
):

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # --------------------------------------------------------
    # Load original image
    # --------------------------------------------------------

    original_image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    image_tensor = transform(
        original_image
    ).unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Get prediction
    # --------------------------------------------------------

    MODEL.eval()

    with torch.no_grad():

        logits = MODEL(
            image_tensor
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )

        confidence, class_index = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = int(
        class_index.item()
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence_value = float(
        confidence.item()
    )

    # --------------------------------------------------------
    # Grad-CAM target
    # --------------------------------------------------------

    target_layer = MODEL.features[-1]

    gradcam = GradCAM(
        MODEL,
        target_layer
    )

    try:

        cam = gradcam.generate(
            image_tensor,
            predicted_index,
        )

    finally:

        gradcam.close()

    # --------------------------------------------------------
    # Create heatmap
    # --------------------------------------------------------

    heatmap = create_heatmap(
        cam
    )

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = create_overlay(
        original_image,
        heatmap,
        alpha=0.40,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if output_path is None:

        output_path = (
            image_path.parent
            / f"{image_path.stem}_gradcam.png"
        )

    output_path = Path(
        output_path
    )

    overlay.save(
        output_path
    )

    return {

        "predicted_class":
            predicted_class,

        "class_index":
            predicted_index,

        "confidence":
            confidence_value,

        "confidence_percent":
            confidence_value * 100.0,

        "xai_method":
            "Grad-CAM",

        "heatmap":
            heatmap,

        "overlay":
            overlay,

        "output_path":
            str(output_path),

    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("DERMAVISION XAI - GRAD-CAM")
    print("=" * 70)

    image_path = input(
        "Enter skin image path: "
    ).strip()

    result = explain(
        image_path
    )

    print()
    print("=" * 70)
    print("XAI RESULT")
    print("=" * 70)

    print(
        "Predicted class:",
        result["predicted_class"]
    )

    print(
        "Confidence:",
        f"{result['confidence_percent']:.2f}%"
    )

    print(
        "Method:",
        result["xai_method"]
    )

    print(
        "Saved heatmap:",
        result["output_path"]
    )

    print("=" * 70)