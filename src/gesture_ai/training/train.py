import os
import sys

import torch
import torch.nn as nn

from torch.utils.data import DataLoader


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
    freeze_features,
    unfreeze_last_blocks,
    count_parameters,
)


# =========================================================
# Configuration
# =========================================================

BATCH_SIZE = 8

NUM_CLASSES = 4

TRAIN_SAMPLES_PER_CLASS = 300

VAL_SAMPLES_PER_CLASS = 60


# ---------------------------------------------------------
# Stage 1
# Train only classifier
# ---------------------------------------------------------

STAGE1_EPOCHS = 4

STAGE1_LR = 0.001


# ---------------------------------------------------------
# Stage 2
# Fine tune last MobileNet blocks
# ---------------------------------------------------------

STAGE2_EPOCHS = 7

STAGE2_LR = 0.0001

UNFREEZE_BLOCKS = 3


# ---------------------------------------------------------
# Early stopping
# ---------------------------------------------------------

EARLY_STOPPING_PATIENCE = 4


# =========================================================
# Reproducibility
# =========================================================

torch.manual_seed(
    42
)


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "\n========================================"
)

print(
    "Hand Gesture Recognition Training"
)

print(
    "========================================"
)

print(
    f"Device: {device}"
)


# =========================================================
# Dataset
# =========================================================

train_dataset = GestureDataset(
    split="train",
    samples_per_class=(
        TRAIN_SAMPLES_PER_CLASS
    ),
    augment=True,
)


val_dataset = GestureDataset(
    split="validation",
    samples_per_class=(
        VAL_SAMPLES_PER_CLASS
    ),
    augment=False,
)


# =========================================================
# DataLoaders
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


print(
    f"\nTraining samples: "
    f"{len(train_dataset)}"
)

print(
    f"Validation samples: "
    f"{len(val_dataset)}"
)


# =========================================================
# Model
# =========================================================

model = create_model(
    num_classes=NUM_CLASSES,
)

model = model.to(
    device
)


# =========================================================
# Loss
# =========================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)


# =========================================================
# Save directory
# =========================================================

os.makedirs(
    "models",
    exist_ok=True,
)


MODEL_PATH = (
    "models/best_model.pth"
)


# =========================================================
# Validation function
# =========================================================

def validate_model():

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in (
            val_loader
        ):

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels,
            )


            running_loss += (
                loss.item()
            )


            predictions = torch.argmax(
                outputs,
                dim=1,
            )


            total += (
                labels.size(0)
            )


            correct += (
                predictions
                == labels
            ).sum().item()


    validation_loss = (
        running_loss
        / len(val_loader)
    )


    validation_accuracy = (
        100.0
        * correct
        / total
    )


    return (
        validation_loss,
        validation_accuracy,
    )


# =========================================================
# Train one epoch
# =========================================================

def train_one_epoch(
    optimizer,
    stage,
):

    model.train()

    # -----------------------------------------------------
    # Important:
    # Keep frozen BatchNorm statistics frozen too.
    # -----------------------------------------------------

    if stage == 1:

        model.features.eval()

    else:

        # Earlier frozen feature blocks remain eval mode.

        frozen_end = (
            len(model.features)
            - UNFREEZE_BLOCKS
        )

        for index, block in enumerate(
            model.features
        ):

            if index < frozen_end:

                block.eval()

            else:

                block.train()


    running_loss = 0.0

    correct = 0

    total = 0


    for batch_number, (
        images,
        labels,
    ) in enumerate(
        train_loader,
        start=1,
    ):

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels,
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item()
        )


        predictions = torch.argmax(
            outputs,
            dim=1,
        )


        total += (
            labels.size(0)
        )


        correct += (
            predictions
            == labels
        ).sum().item()


        if (
            batch_number
            % 25
            == 0
        ):

            current_accuracy = (
                100.0
                * correct
                / total
            )

            print(
                f"Batch "
                f"{batch_number}/"
                f"{len(train_loader)}"
                f" | Accuracy: "
                f"{current_accuracy:.2f}%"
            )


    training_loss = (
        running_loss
        / len(train_loader)
    )


    training_accuracy = (
        100.0
        * correct
        / total
    )


    return (
        training_loss,
        training_accuracy,
    )


# =========================================================
# Save checkpoint
# =========================================================

def save_checkpoint(
    validation_accuracy,
    stage,
    epoch,
):

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "class_names":
            train_dataset.class_names,

        "num_classes":
            NUM_CLASSES,

        "validation_accuracy":
            validation_accuracy,

        "stage":
            stage,

        "epoch":
            epoch,
    }


    torch.save(
        checkpoint,
        MODEL_PATH,
    )


# =========================================================
# Stage 1
# =========================================================

print(
    "\n========================================"
)

print(
    "STAGE 1"
)

print(
    "Train classifier only"
)

print(
    "========================================"
)


freeze_features(
    model
)


total_parameters, trainable_parameters = (
    count_parameters(
        model
    )
)


print(
    f"Total parameters: "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)


stage1_optimizer = torch.optim.AdamW(
    filter(
        lambda parameter:
            parameter.requires_grad,
        model.parameters(),
    ),
    lr=STAGE1_LR,
    weight_decay=0.0001,
)


best_val_accuracy = 0.0


for epoch in range(
    1,
    STAGE1_EPOCHS + 1,
):

    print(
        "\n----------------------------------------"
    )

    print(
        f"Stage 1 | Epoch "
        f"{epoch}/"
        f"{STAGE1_EPOCHS}"
    )

    print(
        "----------------------------------------"
    )


    train_loss, train_accuracy = (
        train_one_epoch(
            stage1_optimizer,
            stage=1,
        )
    )


    val_loss, val_accuracy = (
        validate_model()
    )


    print(
        f"\nTrain Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.2f}%"
    )


    if (
        val_accuracy
        > best_val_accuracy
    ):

        best_val_accuracy = (
            val_accuracy
        )

        save_checkpoint(
            validation_accuracy=(
                val_accuracy
            ),
            stage=1,
            epoch=epoch,
        )

        print(
            "Best model saved."
        )


# =========================================================
# Load best Stage 1 checkpoint
# =========================================================

print(
    "\nLoading best Stage 1 model..."
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


# =========================================================
# Stage 2
# =========================================================

print(
    "\n========================================"
)

print(
    "STAGE 2"
)

print(
    "Fine tune final MobileNet blocks"
)

print(
    "========================================"
)


unfreeze_last_blocks(
    model,
    num_blocks=UNFREEZE_BLOCKS,
)


total_parameters, trainable_parameters = (
    count_parameters(
        model
    )
)


print(
    f"Total parameters: "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)


stage2_optimizer = torch.optim.AdamW(
    filter(
        lambda parameter:
            parameter.requires_grad,
        model.parameters(),
    ),
    lr=STAGE2_LR,
    weight_decay=0.0001,
)


scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        stage2_optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
)


epochs_without_improvement = 0


for epoch in range(
    1,
    STAGE2_EPOCHS + 1,
):

    print(
        "\n----------------------------------------"
    )

    print(
        f"Stage 2 | Epoch "
        f"{epoch}/"
        f"{STAGE2_EPOCHS}"
    )

    print(
        "----------------------------------------"
    )


    train_loss, train_accuracy = (
        train_one_epoch(
            stage2_optimizer,
            stage=2,
        )
    )


    val_loss, val_accuracy = (
        validate_model()
    )


    current_lr = (
        stage2_optimizer
        .param_groups[0]["lr"]
    )


    print(
        f"\nTrain Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.2f}%"
    )

    print(
        f"Learning Rate: "
        f"{current_lr:.8f}"
    )


    scheduler.step(
        val_accuracy
    )


    if (
        val_accuracy
        > best_val_accuracy
    ):

        best_val_accuracy = (
            val_accuracy
        )

        epochs_without_improvement = 0


        save_checkpoint(
            validation_accuracy=(
                val_accuracy
            ),
            stage=2,
            epoch=epoch,
        )


        print(
            "Best model saved."
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement for "
            f"{epochs_without_improvement} "
            f"epoch(s)."
        )


    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        break


# =========================================================
# Complete
# =========================================================

print(
    "\n========================================"
)

print(
    "Training complete"
)

print(
    "========================================"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Best model: "
    f"{MODEL_PATH}"
)