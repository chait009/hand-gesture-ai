import torch.nn as nn

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


# =========================================================
# Create MobileNetV3
# =========================================================

def create_model(
    num_classes=4,
):

    weights = (
        MobileNet_V3_Small_Weights.DEFAULT
    )

    model = mobilenet_v3_small(
        weights=weights
    )

    # =====================================================
    # Replace ImageNet classifier
    # =====================================================

    in_features = (
        model.classifier[3].in_features
    )

    model.classifier[3] = nn.Linear(
        in_features,
        num_classes,
    )

    # =====================================================
    # Start with feature extractor frozen
    # =====================================================

    freeze_features(
        model
    )

    return model


# =========================================================
# Freeze entire feature extractor
# =========================================================

def freeze_features(
    model,
):

    for parameter in (
        model.features.parameters()
    ):

        parameter.requires_grad = False

    # Classifier must remain trainable

    for parameter in (
        model.classifier.parameters()
    ):

        parameter.requires_grad = True


# =========================================================
# Unfreeze last feature blocks
# =========================================================

def unfreeze_last_blocks(
    model,
    num_blocks=3,
):

    freeze_features(
        model
    )

    for block in (
        model.features[
            -num_blocks:
        ]
    ):

        for parameter in (
            block.parameters()
        ):

            parameter.requires_grad = True


# =========================================================
# Count parameters
# =========================================================

def count_parameters(
    model,
):

    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    return (
        total,
        trainable,
    )