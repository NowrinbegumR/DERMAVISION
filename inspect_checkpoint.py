import torch
from pathlib import Path


CHECKPOINT_PATH = Path(
    r"C:\projects\DERMAVISION\EfficientNet\Skin-imperfection-9"
    r"\best_efficientnet_b3_skin_tone_best_epoch_1.pth"
)


print("=" * 70)
print("DERMAVISION CHECKPOINT INSPECTION")
print("=" * 70)

print("\nCheckpoint:")
print(CHECKPOINT_PATH)

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )

print("\nCheckpoint found.")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

print("\nCheckpoint type:")
print(type(checkpoint))


# ============================================================
# CHECK CHECKPOINT CONTENT
# ============================================================

if isinstance(checkpoint, dict):

    print("\nCheckpoint keys:")

    for key in checkpoint.keys():
        print("  ", key)

    # --------------------------------------------------------
    # Look for class information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLASS INFORMATION")
    print("=" * 70)

    possible_class_keys = [
        "classes",
        "class_names",
        "class_to_idx",
        "idx_to_class",
        "labels",
        "label_names",
        "num_classes",
    ]

    found_class_info = False

    for key in possible_class_keys:

        if key in checkpoint:

            found_class_info = True

            print(f"\n{key}:")
            print(checkpoint[key])

    if not found_class_info:
        print(
            "\nNo class information was stored directly "
            "inside the checkpoint."
        )

    # --------------------------------------------------------
    # Look for training information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING INFORMATION")
    print("=" * 70)

    possible_training_keys = [
        "epoch",
        "best_acc",
        "best_accuracy",
        "val_acc",
        "val_accuracy",
        "accuracy",
        "loss",
    ]

    for key in possible_training_keys:

        if key in checkpoint:

            print(
                f"{key}: {checkpoint[key]}"
            )

    # --------------------------------------------------------
    # Find state dictionary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL STATE DICTIONARY")
    print("=" * 70)

    state_dict = None
    state_source = None

    possible_state_keys = [
        "model_state_dict",
        "state_dict",
        "model",
        "net",
    ]

    for key in possible_state_keys:

        if key in checkpoint:

            candidate = checkpoint[key]

            if isinstance(candidate, dict):

                state_dict = candidate
                state_source = key

                break

    # Sometimes checkpoint itself is state_dict

    if state_dict is None:

        if all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):

            state_dict = checkpoint
            state_source = "checkpoint itself"

    if state_dict is None:

        print(
            "\nCould not identify a state dictionary."
        )

    else:

        print(
            f"\nState dictionary source: {state_source}"
        )

        print(
            f"Number of parameters: {len(state_dict)}"
        )

        print("\nFirst 20 parameter names:")

        for i, key in enumerate(
            state_dict.keys()
        ):

            if i >= 20:
                break

            value = state_dict[key]

            shape = (
                tuple(value.shape)
                if hasattr(value, "shape")
                else "N/A"
            )

            print(
                f"{key}  shape={shape}"
            )

        # ----------------------------------------------------
        # Find final classification layer
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("POSSIBLE CLASSIFICATION LAYERS")
        print("=" * 70)

        found_final = False

        for key, value in state_dict.items():

            key_lower = key.lower()

            if any(
                name in key_lower
                for name in [
                    "classifier",
                    "fc",
                    "head",
                    "linear",
                ]
            ):

                found_final = True

                shape = (
                    tuple(value.shape)
                    if hasattr(value, "shape")
                    else "N/A"
                )

                print(
                    f"{key}  shape={shape}"
                )

        if not found_final:

            print(
                "No obvious classifier/fc/head layer found."
            )

else:

    print(
        "\nCheckpoint is not a dictionary."
    )


print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)