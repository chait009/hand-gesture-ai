import os
import sys

import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# =========================================================
# Project root
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
    )
)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


from src.gesture_ai.data.dataset import (
    GestureDataset,
)

from src.gesture_ai.models.model import (
    create_model,
)


# =========================================================
# Configuration
# =========================================================

MODEL_PATH = (
    "models/best_model.pth"
)

TEST_SAMPLES_PER_CLASS = 60

BATCH_SIZE = 8


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "\n================================"
)

print(
    "Model Evaluation"
)

print(
    "================================"
)

print(
    f"Device: {device}"
)


# =========================================================
# Model checkpoint
# =========================================================

if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(
        f"Model not found: "
        f"{MODEL_PATH}"
    )


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)


class_names = checkpoint[
    "class_names"
]

num_classes = checkpoint[
    "num_classes"
]


print(
    f"\nClasses: "
    f"{class_names}"
)

print(
    f"Saved validation accuracy: "
    f"{checkpoint['validation_accuracy']:.2f}%"
)


if "stage" in checkpoint:

    print(
        f"Best stage: "
        f"{checkpoint['stage']}"
    )


if "epoch" in checkpoint:

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )


# =========================================================
# Model
# =========================================================

model = create_model(
    num_classes=num_classes,
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model = model.to(
    device
)


model.eval()


# =========================================================
# Test dataset
# =========================================================

test_dataset = GestureDataset(
    split="test",
    samples_per_class=(
        TEST_SAMPLES_PER_CLASS
    ),
    augment=False,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


print(
    f"\nTest samples: "
    f"{len(test_dataset)}"
)


# =========================================================
# Prediction
# =========================================================

all_labels = []

all_predictions = []


with torch.no_grad():

    for images, labels in (
        test_loader
    ):

        images = images.to(
            device
        )


        outputs = model(
            images
        )


        predictions = torch.argmax(
            outputs,
            dim=1,
        )


        all_labels.extend(
            labels.numpy().tolist()
        )


        all_predictions.extend(
            predictions
            .cpu()
            .numpy()
            .tolist()
        )


# =========================================================
# Accuracy
# =========================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions,
)


print(
    "\n================================"
)

print(
    "TEST RESULTS"
)

print(
    "================================"
)


print(
    f"\nTest Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# =========================================================
# Classification report
# =========================================================

print(
    "\nClassification Report:\n"
)


report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0,
)


print(
    report
)


# =========================================================
# Confusion matrix
# =========================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
)


print(
    "\nConfusion Matrix:"
)

print(
    cm
)


# =========================================================
# Save results
# =========================================================

os.makedirs(
    "results",
    exist_ok=True,
)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)


fig, ax = plt.subplots(
    figsize=(8, 6)
)


display.plot(
    ax=ax,
    values_format="d",
)


plt.title(
    "Hand Gesture Recognition "
    "Confusion Matrix"
)


plt.tight_layout()


output_path = (
    "results/confusion_matrix.png"
)


plt.savefig(
    output_path,
    dpi=200,
)


print(
    f"\nConfusion matrix saved:"
)

print(
    output_path
)


plt.show()


print(
    "\n================================"
)

print(
    "Evaluation complete"
)

print(
    "================================"
)