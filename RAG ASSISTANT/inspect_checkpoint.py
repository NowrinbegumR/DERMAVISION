import torch
from pathlib import Path

checkpoint_path = Path(
    r"C:\projects\DERMAVISION\EfficientNet\Skin-imperfection-9"
    r"\best_efficientnet_b3_skin_tone_best_epoch_1.pth"
)

print("=" * 70)
print("CHECKPOINT INSPECTION")
print("=" * 70)

print("Path:")
print(checkpoint_path)

print("Exists:", checkpoint_path.exists())

if not checkpoint_path.exists():
    raise FileNotFoundError(checkpoint_path)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)

print("\nCheckpoint type:")
print(type(checkpoint))

if isinstance(checkpoint, dict):

    print("\nCheckpoint keys:")

    for key in checkpoint.keys():
        print("  ", key)

    print("\nPossible metadata:")

    for key in [
        "classes",
        "class_names",
        "class_to_idx",
        "idx_to_class",
        "labels",
        "num_classes",
        "epoch",
        "best_acc",
        "val_acc",
        "accuracy",
    ]:
        if key in checkpoint:
            print(f"\n{key}:")
            print(checkpoint[key])

else:
    print("\nCheckpoint is not a dictionary.")

print("\n" + "=" * 70)
print("STATE DICT INSPECTION")
print("=" * 70)

if isinstance(checkpoint, dict):

    state_dict = None

    for key in [
        "model_state_dict",
        "state_dict",
        "model",
        "net",
    ]:
        if key in checkpoint:
            possible = checkpoint[key]

            if isinstance(possible, dict):
                state_dict = possible
                print(f"\nUsing state dictionary from key: {key}")
                break

    if state_dict is None:

        # Sometimes the checkpoint itself is the state dict
        if all(isinstance(v, torch.Tensor)
               for v in checkpoint.values()):
            state_dict = checkpoint
            print("\nCheckpoint itself appears to be a state_dict.")

    if state_dict is not None:

        print("\nNumber of parameters:", len(state_dict))

        print("\nFirst 20 parameter names:")

        for i, key in enumerate(state_dict.keys()):

            if i >= 20:
                break

            print(" ", key)

        print("\nLast layer candidates:")

        for key, value in state_dict.items():

            key_lower = key.lower()

            if any(
                x in key_lower
                for x in [
                    "classifier",
                    "fc",
                    "head",
                    "linear",
                ]
            ):
                print(
                    key,
                    "shape=",
                    tuple(value.shape)
                    if hasattr(value, "shape")
                    else type(value)
                )

print("\nDONE")