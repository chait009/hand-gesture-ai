# Real-Time Hand Gesture Recognition

A computer vision project that recognizes hand gestures from images and a live webcam feed using PyTorch, MobileNetV3, Hugging Face datasets, and OpenCV.

I built this project to practice a complete computer vision workflow: working with a large image dataset, building a reusable PyTorch data pipeline, applying transfer learning, evaluating the model, and connecting the trained model to real-time webcam inference.

The current version recognizes four gestures:

- Fist
- Like
- Peace
- Stop

## Project Highlights

- Built an end-to-end image classification pipeline in PyTorch
- Streamed selected images directly from Hugging Face instead of downloading the full dataset
- Used MobileNetV3-Small with ImageNet pretrained weights
- Implemented two-stage transfer learning and fine-tuning
- Added image augmentation to improve generalization
- Evaluated the model using accuracy, precision, recall, F1-score, and a confusion matrix
- Built real-time webcam inference using OpenCV
- Added prediction smoothing to reduce unstable frame-by-frame predictions

## Architecture

```text
Hugging Face Dataset
        |
        v
Class-based Image Streaming
        |
        v
Image Preprocessing + Augmentation
        |
        v
PyTorch DataLoader
        |
        v
MobileNetV3-Small
        |
        v
Transfer Learning
        |
        v
Fine-Tuning
        |
        v
Model Evaluation
        |
        v
Real-Time OpenCV Inference
```

## Dataset

The project uses a subset of the HaGRID hand gesture dataset available on Hugging Face.

```text
ntsrigaud/hagrid-subset
```

The full dataset is much larger than what I needed for the first version of the project, so instead of downloading everything locally, the data pipeline streams only the selected gesture classes.

The current model uses:

```text
fist
like
peace
stop
```

This approach reduced local disk usage and made it easier to iterate on the training pipeline.

## Model

The classifier uses `MobileNetV3-Small` with pretrained ImageNet weights.

I chose MobileNetV3 because it provides a good balance between model size, inference speed, and image classification performance, which also makes it suitable for real-time
